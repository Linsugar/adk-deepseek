// Package deepseek 为 Google ADK Go v2 提供 DeepSeek 模型适配器。
//
// DeepSeek API 兼容 OpenAI Chat Completion 协议，本包实现 model.LLM 接口，
// 在 ADK 内部格式（genai.Content）与 OpenAI 格式之间自动转换，
// 支持流式和非流式两种调用模式。
//
// # 快速开始
//
// 设置环境变量：
//
//	export DEEPSEEK_API_KEY="sk-xxx"
//
// 一行接入：
//
//	import deepseek "github.com/Linsugar/adk-deepseek"
//
//	llm := deepseek.New("deepseek-v4-flash")
//	agent, _ := llmagent.New(llmagent.Config{Model: llm, ...})
//
// # 支持的模型
//
//	deepseek-v4-pro      — 旗舰模型（最强推理能力）
//	deepseek-v4-flash    — 快速推理（推荐日常使用）
//
// # 自定义配置
//
//	llm := deepseek.New(deepseek.Config{
//	    ModelName: "deepseek-v4-pro",
//	    APIKey:    os.Getenv("DEEPSEEK_API_KEY"),
//	    BaseURL:   "https://api.deepseek.com/v1",
//	})
//
// # 实现细节
//
// 本包将以下 ADK 类型转换为 OpenAI Chat Completion 格式：
//   - genai.Content（对话历史）         → messages 数组
//   - genai.Schema（工具 JSON Schema）  → tools 数组
//   - genai.FunctionCall（工具调用）     ← tool_calls 字段
//   - genai.FunctionResponse（工具结果） → tool 角色消息
//   - model.LLMRequest.Config            → temperature, max_tokens 等参数
//
// 流式模式下，工具调用的函数名和参数可能分多个 SSE chunk 返回，
// 本包自动累积碎片，流结束后合并为完整的 FunctionCall 事件。
package deepseek

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"os"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// Config 配置 DeepSeek 客户端。
type Config struct {
	// ModelName DeepSeek 模型名称，如 "deepseek-v4-flash"、"deepseek-v4-pro"。
	ModelName string

	// APIKey 从 DeepSeek 平台获取的 API Key。
	// 若为空，自动从环境变量 DEEPSEEK_API_KEY 读取。
	APIKey string

	// BaseURL API 端点地址，默认 "https://api.deepseek.com/v1"。
	// 可用于代理、私有部署等场景。
	BaseURL string

	// HTTPClient 自定义 HTTP 客户端，nil 使用 http.DefaultClient。
	HTTPClient *http.Client
}

func (c *Config) defaults() {
	if c.APIKey == "" {
		c.APIKey = os.Getenv("DEEPSEEK_API_KEY")
	}
	if c.BaseURL == "" {
		c.BaseURL = "https://api.deepseek.com/v1"
	}
	if c.HTTPClient == nil {
		c.HTTPClient = http.DefaultClient
	}
}

// LLM 实现 model.LLM 接口，连接 DeepSeek API。
type LLM struct {
	name      string
	apiKey    string
	baseURL   string
	client    *http.Client
	chatPath  string // Chat Completion 路径，默认 "/chat/completions"
}

// New 创建一个 DeepSeek 模型客户端（便捷方法）。
//
// API Key 读取优先级：参数传入 > 环境变量 DEEPSEEK_API_KEY
//
//	// 方式 1：自动从环境变量读取（Linux/macOS/Windows 均支持）
//	llm := deepseek.New("deepseek-v4-flash")
//
//	// 方式 2：直接传入 API Key（无需设环境变量，Windows 友好）
//	llm := deepseek.New("deepseek-v4-flash", "sk-xxxxxxxxxxxxxxxx")
//
//	// 方式 3：完整配置
//	llm := deepseek.NewWithConfig(deepseek.Config{...})
func New(modelName string, fallbackKey ...string) *LLM {
	cfg := Config{ModelName: modelName}
	if len(fallbackKey) > 0 {
		cfg.APIKey = fallbackKey[0]
	}
	return NewWithConfig(cfg)
}

// NewWithConfig 使用完整配置创建 DeepSeek 模型客户端。
func NewWithConfig(cfg Config) *LLM {
	cfg.defaults()
	return &LLM{
		name:     cfg.ModelName,
		apiKey:   cfg.APIKey,
		baseURL:  strings.TrimRight(cfg.BaseURL, "/"),
		client:   cfg.HTTPClient,
		chatPath: "/chat/completions",
	}
}

// Name 实现 model.LLM 接口。
func (m *LLM) Name() string {
	return m.name
}

// GenerateContent 实现 model.LLM 接口。
//
// 将 ADK 的 LLMRequest 转换为 OpenAI Chat Completion 格式，
// 调用 DeepSeek API，然后将响应转回 LLMResponse。
//
// stream=false: 一次性返回完整响应
// stream=true:  通过 iter.Seq2 逐个返回 SSE chunk
func (m *LLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	if stream {
		return m.generateStream(ctx, req)
	}
	return m.generate(ctx, req)
}

// ── 非流式调用 ──

func (m *LLM) generate(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		body := m.buildChatRequest(req, false)
		httpReq, err := m.buildHTTPRequest(ctx, body)
		if err != nil {
			yield(nil, err)
			return
		}

		resp, err := m.client.Do(httpReq)
		if err != nil {
			yield(nil, fmt.Errorf("deepseek: 请求失败: %w", err))
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != 200 {
			errBody, _ := io.ReadAll(resp.Body)
			yield(nil, fmt.Errorf("deepseek: API 错误 (status=%d): %s", resp.StatusCode, string(errBody)))
			return
		}

		var chatResp chatCompletionResponse
		if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
			yield(nil, fmt.Errorf("deepseek: 解析响应失败: %w", err))
			return
		}

		llmResp := m.chatResponseToLLMResponse(&chatResp)
		yield(llmResp, nil)
	}
}

// ── 流式调用 ──

func (m *LLM) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		body := m.buildChatRequest(req, true)
		httpReq, err := m.buildHTTPRequest(ctx, body)
		if err != nil {
			yield(nil, err)
			return
		}

		resp, err := m.client.Do(httpReq)
		if err != nil {
			yield(nil, fmt.Errorf("deepseek: 流式请求失败: %w", err))
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != 200 {
			errBody, _ := io.ReadAll(resp.Body)
			yield(nil, fmt.Errorf("deepseek: API 错误 (status=%d): %s", resp.StatusCode, string(errBody)))
			return
		}

		// 解析 SSE 流
		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

		// 累积工具调用（流式模式下工具调用可能分多个 chunk 返回）
		tcAccum := make(map[int]*tcAccumulator)

		for scanner.Scan() {
			line := scanner.Text()
			if line == "" || line == "data: [DONE]" {
				continue
			}
			if !strings.HasPrefix(line, "data: ") {
				continue
			}

			data := strings.TrimPrefix(line, "data: ")

			var chunk chatCompletionChunk
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				yield(nil, fmt.Errorf("deepseek: 解析流式 chunk 失败: %w", err))
				return
			}

			if len(chunk.Choices) == 0 {
				continue
			}
			choice := chunk.Choices[0]
			delta := choice.Delta

			var parts []*genai.Part

			if delta.Content != "" {
				parts = append(parts, &genai.Part{Text: delta.Content})
			}

			// 流式工具调用片段累积
			for _, tc := range delta.ToolCalls {
				idx := tc.Index
				if idx == nil {
					continue
				}
				acc, ok := tcAccum[*idx]
				if !ok {
					acc = &tcAccumulator{}
					tcAccum[*idx] = acc
				}

				if tc.ID != nil {
					acc.id = *tc.ID
				}
				if tc.Function != nil {
					if tc.Function.Name != nil {
						acc.name += *tc.Function.Name
					}
					if tc.Function.Arguments != nil {
						acc.args += *tc.Function.Arguments
					}
				}
			}

			if len(parts) == 0 && len(delta.ToolCalls) == 0 {
				continue
			}

			var content *genai.Content
			if len(parts) > 0 {
				content = &genai.Content{
					Role:  "model",
					Parts: parts,
				}
			}

			finishReason := genai.FinishReasonUnspecified
			if choice.FinishReason != nil {
				finishReason = mapFinishReason(*choice.FinishReason)
			}

			llmResp := &model.LLMResponse{
				Content:      content,
				Partial:      true,
				TurnComplete: finishReason != genai.FinishReasonUnspecified,
				FinishReason: finishReason,
			}

			if !yield(llmResp, nil) {
				return
			}
		}

		// 流结束后，发送累积的工具调用
		if len(tcAccum) > 0 {
			var parts []*genai.Part
			for i := 0; i < len(tcAccum); i++ {
				acc, ok := tcAccum[i]
				if !ok {
					continue
				}
				var argsMap map[string]any
				json.Unmarshal([]byte(acc.args), &argsMap)
				parts = append(parts, &genai.Part{
					FunctionCall: &genai.FunctionCall{
						ID:   acc.id,
						Name: acc.name,
						Args: argsMap,
					},
				})
			}

			if len(parts) > 0 {
				llmResp := &model.LLMResponse{
					Content: &genai.Content{
						Role:  "model",
						Parts: parts,
					},
					TurnComplete: true,
					FinishReason: genai.FinishReasonStop,
				}
				yield(llmResp, nil)
			}
		}

		if err := scanner.Err(); err != nil {
			yield(nil, fmt.Errorf("deepseek: 读取流式响应出错: %w", err))
		}
	}
}

// ── 请求构建 ──

func (m *LLM) buildChatRequest(req *model.LLMRequest, stream bool) chatCompletionRequest {
	chatReq := chatCompletionRequest{
		Model:    m.name,
		Stream:   stream,
		Messages: m.convertMessages(req),
	}

	if req.Config != nil {
		chatReq.Temperature = req.Config.Temperature
		if req.Config.MaxOutputTokens > 0 {
			chatReq.MaxTokens = &req.Config.MaxOutputTokens
		}
		chatReq.TopP = req.Config.TopP
		chatReq.Stop = req.Config.StopSequences
		chatReq.Tools = m.convertTools(req.Config.Tools)
	}

	return chatReq
}

// convertMessages 将 genai.Content 列表转换为 OpenAI 消息格式。
func (m *LLM) convertMessages(req *model.LLMRequest) []chatMessage {
	var messages []chatMessage

	// 系统指令
	if req.Config != nil && req.Config.SystemInstruction != nil {
		for _, part := range req.Config.SystemInstruction.Parts {
			if part.Text != "" {
				messages = append(messages, chatMessage{
					Role:    "system",
					Content: part.Text,
				})
			}
		}
	}

	for _, content := range req.Contents {
		if content == nil {
			continue
		}

		switch content.Role {
		case "user":
			msg := chatMessage{Role: "user"}
			textParts, toolResults := splitParts(content.Parts)
			if len(textParts) > 0 {
				msg.Content = strings.Join(textParts, "\n")
			}
			// 工具返回结果以独立消息形式发送
			if len(toolResults) > 0 {
				for _, fr := range toolResults {
					respJSON, _ := json.Marshal(fr.Response)
					messages = append(messages, chatMessage{
						Role:       "tool",
						ToolCallID: fr.ID,
						Content:    string(respJSON),
					})
				}
				if msg.Content == "" {
					continue
				}
			}
			messages = append(messages, msg)

		case "model":
			msg := chatMessage{Role: "assistant"}
			var textParts []string
			var toolCalls []toolCall
			for _, part := range content.Parts {
				if part.Text != "" && !part.Thought {
					textParts = append(textParts, part.Text)
				}
				if part.FunctionCall != nil {
					argsJSON, _ := json.Marshal(part.FunctionCall.Args)
					toolCalls = append(toolCalls, toolCall{
						ID:   part.FunctionCall.ID,
						Type: "function",
						Function: toolCallFunction{
							Name:      part.FunctionCall.Name,
							Arguments: string(argsJSON),
						},
					})
				}
			}
			if len(textParts) > 0 {
				msg.Content = strings.Join(textParts, "\n")
			}
			if len(toolCalls) > 0 {
				msg.ToolCalls = toolCalls
			}
			messages = append(messages, msg)
		}
	}

	return messages
}

// splitParts 将 parts 分为文本和工具返回结果。
func splitParts(parts []*genai.Part) (texts []string, toolResults []*genai.FunctionResponse) {
	for _, p := range parts {
		if p == nil {
			continue
		}
		if p.Text != "" && !p.Thought {
			texts = append(texts, p.Text)
		}
		if p.FunctionResponse != nil {
			toolResults = append(toolResults, p.FunctionResponse)
		}
	}
	return
}

// convertTools 将 genai.Tool 列表转换为 OpenAI 工具定义。
func (m *LLM) convertTools(tools []*genai.Tool) []toolDef {
	var result []toolDef
	for _, t := range tools {
		if t == nil {
			continue
		}
		for _, fd := range t.FunctionDeclarations {
			if fd == nil {
				continue
			}
			def := toolDef{
				Type: "function",
				Function: functionDef{
					Name:        fd.Name,
					Description: fd.Description,
				},
			}
			if fd.Parameters != nil {
				def.Function.Parameters = schemaToAny(fd.Parameters)
			}
			result = append(result, def)
		}
	}
	return result
}

// schemaToAny 将 genai.Schema 递归转为 map[string]any，供 JSON 序列化。
func schemaToAny(s *genai.Schema) map[string]any {
	if s == nil {
		return nil
	}
	m := make(map[string]any)
	if s.Type != "" {
		m["type"] = strings.ToLower(string(s.Type))
	}
	if s.Description != "" {
		m["description"] = s.Description
	}
	if len(s.Properties) > 0 {
		props := make(map[string]any)
		for k, v := range s.Properties {
			props[k] = schemaToAny(v)
		}
		m["properties"] = props
	}
	if len(s.Required) > 0 {
		m["required"] = s.Required
	}
	if s.Items != nil {
		m["items"] = schemaToAny(s.Items)
	}
	if len(s.Enum) > 0 {
		m["enum"] = s.Enum
	}
	return m
}

// ── 响应转换 ──

func (m *LLM) chatResponseToLLMResponse(resp *chatCompletionResponse) *model.LLMResponse {
	if len(resp.Choices) == 0 {
		return &model.LLMResponse{
			TurnComplete: true,
			FinishReason: genai.FinishReasonStop,
		}
	}

	choice := resp.Choices[0]
	msg := choice.Message

	var parts []*genai.Part

	if msg.Content != "" {
		parts = append(parts, &genai.Part{Text: msg.Content})
	}

	for _, tc := range msg.ToolCalls {
		var argsMap map[string]any
		json.Unmarshal([]byte(tc.Function.Arguments), &argsMap)
		parts = append(parts, &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   tc.ID,
				Name: tc.Function.Name,
				Args: argsMap,
			},
		})
	}

	finishReason := genai.FinishReasonUnspecified
	if choice.FinishReason != nil {
		finishReason = mapFinishReason(*choice.FinishReason)
	}

	llmResp := &model.LLMResponse{
		TurnComplete: true,
		FinishReason: finishReason,
	}

	if len(parts) > 0 {
		llmResp.Content = &genai.Content{
			Role:  "model",
			Parts: parts,
		}
	}

	if resp.Usage != nil {
		llmResp.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     int32(resp.Usage.PromptTokens),
			CandidatesTokenCount: int32(resp.Usage.CompletionTokens),
			TotalTokenCount:      int32(resp.Usage.TotalTokens),
		}
	}

	return llmResp
}

func (m *LLM) buildHTTPRequest(ctx context.Context, body any) (*http.Request, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("deepseek: 序列化请求失败: %w", err)
	}

	url := m.baseURL + m.chatPath
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.apiKey)
	return req, nil
}

// ── OpenAI 类型定义 ──

type chatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Stream      bool          `json:"stream,omitempty"`
	Temperature *float32      `json:"temperature,omitempty"`
	MaxTokens   *int32        `json:"max_tokens,omitempty"`
	TopP        *float32      `json:"top_p,omitempty"`
	Stop        []string      `json:"stop,omitempty"`
	Tools       []toolDef     `json:"tools,omitempty"`
}

type chatMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type toolCall struct {
	ID       string          `json:"id,omitempty"`
	Type     string          `json:"type"`
	Function toolCallFunction `json:"function"`
}

type toolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type toolDef struct {
	Type     string      `json:"type"`
	Function functionDef `json:"function"`
}

type functionDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type chatCompletionResponse struct {
	Choices []struct {
		Message struct {
			Content   string     `json:"content"`
			ToolCalls []toolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// ── 流式响应类型 ──

type chatCompletionChunk struct {
	Choices []struct {
		Delta struct {
			Content   string           `json:"content"`
			ToolCalls []streamToolCall `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
}

type streamToolCall struct {
	Index    *int    `json:"index"`
	ID       *string `json:"id"`
	Type     *string `json:"type"`
	Function *struct {
		Name      *string `json:"name"`
		Arguments *string `json:"arguments"`
	} `json:"function"`
}

type tcAccumulator struct {
	id   string
	name string
	args string
}

// ── 辅助函数 ──

func mapFinishReason(reason string) genai.FinishReason {
	switch reason {
	case "stop":
		return genai.FinishReasonStop
	case "length":
		return genai.FinishReasonMaxTokens
	case "tool_calls":
		return genai.FinishReasonStop
	case "content_filter":
		return genai.FinishReasonSafety
	default:
		return genai.FinishReasonOther
	}
}
