# adk-deepseek

[![Go Reference](https://pkg.go.dev/badge/github.com/Linsugar/adk-deepseek.svg)](https://pkg.go.dev/github.com/Linsugar/adk-deepseek)
[![Go Version](https://img.shields.io/badge/Go-%3E%3D1.23-blue)](https://go.dev/)

DeepSeek 模型适配器 for [Google ADK Go v2](https://github.com/google/adk-go)。

实现 `google.golang.org/adk/v2/model.LLM` 接口，一行代码将 DeepSeek 接入 ADK Agent。

## 安装

```bash
go get github.com/Linsugar/adk-deepseek
```

## 快速开始

```go
package main

import (
    deepseek "github.com/Linsugar/adk-deepseek"
    "google.golang.org/adk/v2/agent/llmagent"
)

func main() {
    // 方式 1：自动读环境变量
    llm := deepseek.New("deepseek-v4-flash")

    // 方式 2：直接传 Key（无需配环境变量，Windows 友好）
    // llm := deepseek.New("deepseek-v4-flash", "sk-xxxxxxxxxxxxxxxx")

    agent, _ := llmagent.New(llmagent.Config{
        Name:  "my-assistant",
        Model: llm,
        Instruction: "你是一个有用的AI助手。",
    })
}
```

## 环境变量（可选）

如果你不想每次传 Key，可以设环境变量：

```bash
# Linux / macOS
export DEEPSEEK_API_KEY="sk-xxx"

# Windows PowerShell
$env:DEEPSEEK_API_KEY = "sk-xxx"

# Windows CMD
set DEEPSEEK_API_KEY=sk-xxx
```

**API Key 优先级：函数参数 > 环境变量**

> [获取 API Key](https://platform.deepseek.com/)

## 配置选项

```go
// 方式 1：简洁模式
llm := deepseek.New("deepseek-v4-pro")

// 方式 2：完整配置
llm := deepseek.NewWithConfig(deepseek.Config{
    ModelName: "deepseek-v4-pro",
    APIKey:    os.Getenv("DEEPSEEK_API_KEY"),
    BaseURL:   "https://api.deepseek.com/v1",    // 代理或私有部署
    HTTPClient: &http.Client{Timeout: 30 * time.Second},
})
```

## 支持的模型

| 模型名 | 说明 |
|--------|------|
| `deepseek-v4-flash` | 快速推理（推荐日常使用） |
| `deepseek-v4-pro` | 旗舰模型（最强推理能力） |

## 工作原理

```
ADK 内部格式                     DeepSeek API
──────────                       ──────────────
genai.Content (对话)      ──▶    messages
genai.Schema  (工具定义)   ──▶    tools
genai.FunctionCall        ◀──    tool_calls
iter.Seq2 (流式)          ◀──    SSE chunk →
```

本包在 `model.LLM` 接口层完成所有格式转换，对 ADK 上层完全透明。

## 与 Python ADK 对比

```python
# Python ADK
from google.adk.models.lite_llm import LiteLlm
model = LiteLlm(model="deepseek/deepseek-v4-pro", api_base="...")
```

```go
// Go ADK + adk-deepseek（同等体验）
import deepseek "github.com/Linsugar/adk-deepseek"
model := deepseek.New("deepseek-v4-pro")
```

## 许可证

MIT
