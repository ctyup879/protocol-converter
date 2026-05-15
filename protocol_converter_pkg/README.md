# Protocol Converter

协议转换器 — 支持 **OpenAI Chat**、**OpenAI Responses** 和 **Anthropic Messages** 三种协议的相互转换。

可让使用不同 API 格式的客户端统一接入同一个后端，或让同一客户端灵活对接不同格式的后端服务。

## 功能特性

- 自动检测请求协议类型（OpenAI Chat / OpenAI Responses / Anthropic）
- 请求转换：任意协议 → OpenAI Chat / OpenAI Responses / Anthropic
- 响应转换：后端响应 → 客户端协议格式
- 支持流式（SSE）和非流式响应
- 完整支持 function calling / tool_use 转换
- 模型名称映射
- 三种后端模式：OpenAI Chat / OpenAI Responses / Anthropic

## 安装

```bash
pip install -e .
```

依赖：Python 3.9+，httpx

## 快速开始

### 1. 协议检测

```python
from protocol_converter import ProtocolDetector, Protocol

detector = ProtocolDetector()

# OpenAI Chat
detector.detect({"model": "gpt-4o", "messages": [...]})
# → Protocol.OPENAI_CHAT

# Anthropic
detector.detect({"model": "claude-sonnet-4-20250514", "max_tokens": 1024, "messages": [...]})
# → Protocol.ANTHROPIC

# OpenAI Responses
detector.detect({"model": "gpt-4o", "input": "Hello"})
# → Protocol.OPENAI_RESPONSES
```

### 2. 请求转换

```python
from protocol_converter import AnthropicConverter, OpenAIResponsesConverter

# Anthropic → OpenAI Chat
anthropic_req = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "system": "You are helpful.",
    "messages": [{"role": "user", "content": "Hello"}],
    "tools": [{"name": "get_weather", "description": "Get weather", "input_schema": {...}}]
}
chat_req = AnthropicConverter.to_openai_chat(anthropic_req)
# → {"model": "claude-sonnet-4-20250514", "messages": [...], "tools": [...], ...}

# OpenAI Chat → OpenAI Responses
chat_req = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}
responses_req = OpenAIResponsesConverter.from_openai_chat_request(chat_req)
# → {"model": "gpt-4o", "input": "Hello"}
```

### 3. 响应转换

```python
from protocol_converter import AnthropicConverter, OpenAIResponsesConverter

# OpenAI Chat 响应 → Anthropic 格式
chat_resp = {"id": "...", "model": "gpt-4o", "choices": [...], "usage": {...}}
anthropic_resp = AnthropicConverter.from_openai_chat(chat_resp)
# → {"id": "...", "type": "message", "role": "assistant", "content": [...], "stop_reason": "end_turn", ...}

# OpenAI Chat 响应 → Responses 格式
responses_resp = OpenAIResponsesConverter.from_openai_chat(chat_resp)
# → {"id": "...", "object": "response", "status": "completed", "output": [...], ...}
```

### 4. 使用引擎（推荐）

```python
from protocol_converter import ProtocolConverterEngine, ConverterConfig, Protocol

config = ConverterConfig(
    backend_type="openai",                # "openai" | "anthropic" | "openrouter"
    backend_url="https://api.minimaxi.com/v1/chat/completions",
    api_key="sk-xxx",
    default_model="MiniMax-M2.7",
    model_mapping={
        "claude-sonnet-4-20250514": "MiniMax-M2.7",
        "gpt-4o": "MiniMax-M2.7",
    }
)

engine = ProtocolConverterEngine(config)

# 检测 + 转换请求
request = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}]
}
protocol = engine.detect_protocol(request)   # Protocol.ANTHROPIC
converted = engine.convert_request(request)  # → OpenAI Chat 格式，模型名已映射

# 转换响应
response = engine.convert_response(backend_response, Protocol.ANTHROPIC)
```

## 后端配置

### OpenAI Chat Completions 后端

```python
config = ConverterConfig(
    backend_type="openai",
    backend_url="https://api.minimaxi.com/v1/chat/completions",
    api_key="sk-xxx",
)
# 认证头: Authorization: Bearer sk-xxx
```

### Anthropic 兼容后端

```python
config = ConverterConfig(
    backend_type="anthropic",
    backend_url="https://api.minimaxi.com/anthropic/v1/messages",
    api_key="sk-xxx",
)
# 认证头: x-api-key: sk-xxx, anthropic-version: 2023-06-01
```

### OpenAI Responses 后端

```python
config = ConverterConfig(
    backend_type="openai_responses",
    backend_url="https://openrouter.ai/api/v1/responses",  # 或任意 OpenAI Responses 兼容端点
    api_key="sk-xxx",
    default_model="openai/gpt-oss-120b:free",
)
# 认证头: Authorization: Bearer sk-xxx
```

## 协议对照

### 请求格式对比

| 字段 | OpenAI Chat | OpenAI Responses | Anthropic |
|------|------------|------------------|-----------|
| 模型 | `model` | `model` | `model` |
| 输入 | `messages[]` | `input` (string/array) | `messages[]` |
| 系统提示 | `messages[0].role=system` | `instructions` | `system` (顶级参数) |
| 最大 token | `max_tokens` / `max_completion_tokens` | `max_output_tokens` | `max_tokens` |
| 工具定义 | `tools[].function` | `tools[]` | `tools[].input_schema` |
| 工具选择 | `tool_choice` | `tool_choice` | `tool_choice` |
| 停止序列 | `stop` | - | `stop_sequences` |
| 温度 | `temperature` | `temperature` | `temperature` |

### 响应格式对比

| 字段 | OpenAI Chat | OpenAI Responses | Anthropic |
|------|------------|------------------|-----------|
| 类型 | `chat.completion` | `response` | `message` |
| 内容 | `choices[].message.content` | `output[].content[]` | `content[]` |
| 工具调用 | `choices[].message.tool_calls[]` | `output[].type=function_call` | `content[].type=tool_use` |
| 停止原因 | `finish_reason: stop/length/tool_calls` | `status: completed/incomplete` | `stop_reason: end_turn/max_tokens/tool_use` |
| 用量 | `usage.prompt_tokens` | `usage.input_tokens` | `usage.input_tokens` |

### tool_choice 映射

| Anthropic | OpenAI Chat | OpenAI Responses |
|-----------|-------------|-----------------|
| `auto` | `auto` | `auto` |
| `any` | `required` | `required` |
| `none` | `none` | `none` |
| `{"type":"tool","name":"x"}` | `{"type":"function","function":{"name":"x"}}` | `{"type":"function","name":"x"}` |

### 停止原因映射

| OpenAI Chat | OpenAI Responses | Anthropic |
|-------------|------------------|-----------|
| `stop` | `completed` | `end_turn` |
| `length` | `incomplete` (max_output_tokens) | `max_tokens` |
| `tool_calls` | `completed` | `tool_use` |
| `content_filter` | `incomplete` (content_filter) | `refusal` |

## 项目结构

```
protocol_converter_pkg/
├── protocol_converter/
│   ├── __init__.py              # 包入口
│   ├── protocol_detector.py     # 协议检测器
│   ├── anthropic.py             # Anthropic Messages 转换器
│   ├── openai_chat.py           # OpenAI Chat Completions 转换器
│   ├── openai_responses.py     # OpenAI Responses 转换器
│   └── engine.py                # 核心转换引擎
├── tests/
│   └── test_protocol_converter.py
├── examples/
│   ├── integration_test.py                      # MiniMax OpenAI Chat 后端测试
│   ├── integration_test_anthropic_backend.py   # MiniMax Anthropic 后端测试
│   ├── test_openrouter_responses.py            # OpenAI Responses 协议测试 (OpenRouter 端点)
│   └── unified_test.py                         # 统一测试
└── pyproject.toml
```

## 运行测试

```bash
# 单元测试
python -m pytest tests/ -v

# 集成测试（需要配置 API Key）
python examples/integration_test.py
python examples/integration_test_anthropic_backend.py
python examples/test_openrouter_responses.py
```

## License

MIT
