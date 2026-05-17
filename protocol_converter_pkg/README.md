# Protocol Converter

协议转换器 — 支持 **OpenAI Chat Completions**、**OpenAI Responses** 和 **Anthropic Messages** 三种 API 协议的相互转换。

可让使用不同 API 格式的客户端统一接入同一个后端，或让同一客户端灵活对接不同格式的后端服务。

## 功能特性

- **自动检测**请求协议类型（OpenAI Chat / OpenAI Responses / Anthropic）
- **请求转换**：任意协议 → 任意协议（6 种转换路径）+ 直接转换路径（Anthropic↔Responses、Chat↔Responses）
- **响应转换**：后端响应 → 客户端协议格式
- **流式支持**：完整的 SSE 流式转换，严格遵循各协议事件序列（含 `content_block_start/stop`、`output_item.added/done`、`reasoning_summary_text.delta/done`、`refusal.delta/done` 等）
- **工具调用**：完整支持 function calling / tool_use / tool_result 跨协议转换
- **扩展思考**：Anthropic `thinking` ↔ OpenAI `reasoning_effort` / `reasoning_content` 双向映射（含 `adaptive`、`display`），`redacted_thinking` 通过 `[redacted_thinking: data]` 格式实现三方完整往返转换
- **Responses 推理输出**：`reasoning` 类型输出项（含 `reasoning_text`、`encrypted_content`）→ Chat `reasoning_content` 字段，支持 `thinking.display` ↔ `reasoning.summary` 映射
- **developer 角色降级**：目标后端不支持 `developer` 角色时自动降级为 `system`
- **多模态**：图片（base64 / URL）、文档（PDF）、文件输入转换
- **Structured Outputs 往返**：Responses `text.format` ↔ Chat `response_format` ↔ Anthropic `output_format` 的 `json_schema` 格式正确互转，支持 `name`/`schema`/`strict` 字段完整映射
- **annotations 保留**：Chat `message.annotations` ↔ Responses `output_text.annotations` ↔ Anthropic `text.citations` 跨协议保留
- **`is_error` 字段处理**：Anthropic `tool_result.is_error` 在转换时添加 `[Error]` 前缀标记
- **模型映射**：自定义模型名称映射表，支持请求正向映射与响应反向映射
- **同协议透传优化**：源协议与后端协议一致时，请求直接浅拷贝透传（无需修改时）或最小修改（模型映射 / developer 降级 / extra_body 合并），流式响应直接转发原始 SSE，避免不必要的 JSON 解析再序列化
- **三种后端**：OpenAI Chat / OpenAI Responses / Anthropic

## 安装

```bash
pip install -e .
```

依赖：Python 3.9+，httpx（可选，用于转发请求）

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

# 带 thinking 参数自动识别为 Anthropic
detector.detect({"model": "claude-opus-4-6", "max_tokens": 1024,
                  "thinking": {"type": "enabled", "budget_tokens": 10000}, "messages": [...]})
# → Protocol.ANTHROPIC
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

# OpenAI Responses 响应 → Chat 格式
responses_backend_resp = {"id": "...", "object": "response", "output": [...], "usage": {...}}
chat_from_responses = OpenAIResponsesConverter.to_chat_response(responses_backend_resp)
# → {"id": "...", "object": "chat.completion", "choices": [...], ...}
# 注：Responses 中的 reasoning 输出项会转为 Chat 的 reasoning_content 字段
```

### 4. 使用引擎（推荐）

引擎自动检测源协议、转换为目标格式、处理响应回转，无需手动判断。

```python
from protocol_converter import ProtocolConverterEngine, ConverterConfig, Protocol

# --- 场景 A：所有客户端请求统一转发到 OpenAI Chat 后端 ---
config = ConverterConfig(
    backend_type="openai",                # "openai" | "openai_responses" | "anthropic"
    backend_url="https://api.openai.com/v1/chat/completions",
    api_key="sk-xxx",
    model_mapping={
        "claude-sonnet-4-20250514": "gpt-4o",  # 客户端用 claude，后端用 gpt
    }
)
engine = ProtocolConverterEngine(config)

# 客户端发来 Anthropic 请求 → 自动转换为 Chat 格式发给后端
request = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}]
}
protocol = engine.detect_protocol(request)   # Protocol.ANTHROPIC
converted = engine.convert_request(request)  # → OpenAI Chat 格式，模型名已映射

# 后端返回 Chat 响应 → 自动转回 Anthropic 格式返回给客户端
response = engine.convert_response(backend_response, Protocol.ANTHROPIC)

# --- 场景 B：客户端请求转发到 Anthropic 后端 ---
config_b = ConverterConfig(
    backend_type="anthropic",
    backend_url="https://api.anthropic.com/v1/messages",
    api_key="sk-ant-xxx",
    model_mapping={"gpt-4o": "claude-sonnet-4-20250514"},
)
engine_b = ProtocolConverterEngine(config_b)

# 客户端发来 Chat 请求 → 自动转换为 Anthropic 格式
chat_req = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}
converted_b = engine_b.convert_request(chat_req)
# → {"model": "claude-sonnet-4-20250514", "max_tokens": 4096, "messages": [...]}
```

## 后端配置

### OpenAI Chat Completions 后端

```python
config = ConverterConfig(
    backend_type="openai",
    backend_url="https://api.openai.com/v1/chat/completions",
    api_key="sk-xxx",
)
# 认证头: Authorization: Bearer sk-xxx
```

### Anthropic 兼容后端

```python
config = ConverterConfig(
    backend_type="anthropic",
    backend_url="https://api.anthropic.com/v1/messages",
    api_key="sk-ant-xxx",
    anthropic_version="2023-06-01",       # 可选，默认 2023-06-01
    inference_geo="us",                    # 可选，指定推理地理区域
)
# 认证头: x-api-key: sk-ant-xxx, anthropic-version: 2023-06-01
```

### OpenAI Responses 后端

```python
config = ConverterConfig(
    backend_type="openai_responses",
    backend_url="https://api.openai.com/v1/responses",  # 或任意 Responses 兼容端点
    api_key="sk-xxx",
)
# 认证头: Authorization: Bearer sk-xxx
```

### 完整配置项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `backend_type` | str | `"openai"` | 后端类型：`"openai"` / `"openai_responses"` / `"anthropic"` |
| `backend_url` | str | — | 后端 API 地址 |
| `api_key` | str | `None` | API 密钥 |
| `default_model` | str | `"gpt-4o"` | 默认模型 |
| `timeout` | float | `60.0` | 超时时间（秒） |
| `stream` | bool | `False` | 是否启用流式响应 |
| `extra_headers` | dict | `{}` | 额外请求头 |
| `extra_body` | dict | `{}` | 额外请求体参数 |
| `model_mapping` | dict | `{}` | 模型名称映射表（请求：原始模型 → 后端模型） |
| `reverse_model_mapping_in_stream` | bool | `False` | 响应中是否将 model 反向替换为原始请求模型名（`True` 时隐藏后端实际模型） |
| `anthropic_version` | str | `"2023-06-01"` | Anthropic API 版本 |
| `inference_geo` | str | `None` | Anthropic 推理地理区域 |
| `prompt_cache_key` | str | `None` | OpenAI 提示缓存键（替代 `user`） |
| `prompt_cache_retention` | str | `None` | 缓存保留策略（`"in-memory"` / `"24h"`） |

## 参数映射详情

### 请求格式对比

| 字段 | OpenAI Chat | OpenAI Responses | Anthropic |
|------|------------|------------------|-----------|
| 模型 | `model` | `model` | `model` |
| 输入 | `messages[]` | `input` (string/array) | `messages[]` |
| 系统提示 | `messages[0].role=system/developer` | `instructions` | `system`（顶级参数，支持 str 或 TextBlock[]） |
| 最大 token | `max_tokens` / `max_completion_tokens` | `max_output_tokens` | `max_tokens`（必填，0=仅预热缓存） |
| 工具定义 | `tools[].function` | `tools[]` | `tools[].input_schema` |
| 工具选择 | `tool_choice` | `tool_choice` | `tool_choice` |
| 停止序列 | `stop` | — | `stop_sequences` |
| 温度 | `temperature` (0-2) | `temperature` (0-2) | `temperature` (0-1) |
| 推理控制 | `reasoning_effort` | `reasoning` | `thinking` |
| 响应格式 | `response_format`（text/json_object/json_schema） | `text.format`（text/json_object/json_schema） | — |
| 并行工具调用 | `parallel_tool_calls` | `parallel_tool_calls` | — |
| 服务层级 | `service_tier` | `service_tier` | `service_tier` |
| 存储响应 | `store` | `store` | — |
| 元数据 | `metadata` | `metadata` | `metadata` |
| 安全标识 | `safety_identifier` | `safety_identifier` | — |
| 缓存键 | `prompt_cache_key` | `prompt_cache_key` | — |
| 缓存保留 | `prompt_cache_retention` | `prompt_cache_retention` | — |

### Anthropic 特有参数

| 参数 | 说明 | 转换处理 |
|------|------|----------|
| `system` | 顶级系统提示参数（str 或 TextBlock[]） | → Chat `messages[0].role=system` |
| `top_k` | Top-K 采样 | → `extra_body.top_k` |
| `stop_sequences` | 自定义停止序列 | → Chat `stop` |
| `thinking` | 扩展思考配置（含 `type`/`budget_tokens`/`display`） | → Chat `reasoning_effort`，原始值保留在 `extra_body` |
| `cache_control` | 缓存控制 | → `extra_body.cache_control` |
| `container` | 容器标识 | → `extra_body.container` |
| `output_config` | 输出配置 | → `extra_body.output_config` |
| `output_format` | 结构化输出格式（`json_schema`/`json_object`） | → Chat `response_format` |
| `inference_geo` | 推理地理区域 | → `extra_body.inference_geo` |

### OpenAI Responses 特有参数

| 参数 | 说明 | 转换处理 |
|------|------|----------|
| `instructions` | 系统/开发者提示 | → Chat `messages[0].role=developer` |
| `max_output_tokens` | 最大输出 token | → Chat `max_completion_tokens` |
| `previous_response_id` | 多轮对话 ID | → `extra_body` |
| `reasoning` | 推理配置（含 `effort`/`summary`） | → Chat `reasoning_effort` |
| `text.format` | 结构化输出配置 | → Chat `response_format` |
| `truncation` | 截断策略 | → `extra_body` |
| `background` | 后台运行 | → `extra_body` |
| `max_tool_calls` | 最大工具调用次数 | → `extra_body` |
| `conversation` | 对话参数 | → `extra_body` |
| `context_management` | 上下文管理 | → `extra_body` |
| `include` | 额外输出数据 | → `extra_body` |
| `prompt` | 提示模板 | → `extra_body` |
| `stream_options` | 流式选项（含 `include_obfuscation`） | → Chat `stream_options`（顶层参数） |
| `top_logprobs` | 返回 top logprobs 数量 | → Chat `top_logprobs`（顶层参数） |

### OpenAI Chat 特有参数

| 参数 | 说明 | 转换处理（→ Anthropic） | 转换处理（→ Responses） |
|------|------|------------------------|----------------------|
| `verbosity` | 输出详细程度（low/medium/high） | → `extra_body.verbosity` | → `text.verbosity` |
| `modalities` | 输出类型（text/audio） | → `extra_body.modalities` | — |
| `audio` | 音频输出参数 | → `extra_body.audio` | — |
| `prediction` | 预测内容 | → `extra_body.prediction` | — |
| `safety_identifier` | 安全标识符 | → `extra_body.safety_identifier` | → `safety_identifier` |
| `prompt_cache_key` | 缓存键 | → `extra_body.prompt_cache_key` | → `prompt_cache_key` |
| `prompt_cache_retention` | 缓存保留策略 | → `extra_body.prompt_cache_retention` | → `prompt_cache_retention` |
| `frequency_penalty` | 频率惩罚 | → `extra_body.frequency_penalty` | → `extra_body.frequency_penalty` |
| `presence_penalty` | 存在惩罚 | → `extra_body.presence_penalty` | → `extra_body.presence_penalty` |
| `seed` | 随机种子 | → `extra_body.seed` | → `extra_body.seed` |
| `logprobs` | 返回 logprobs | → `extra_body.logprobs` | — |
| `logit_bias` | token 偏置 | → `extra_body.logit_bias` | → `extra_body.logit_bias` |
| `web_search_options` | 网页搜索选项 | → `extra_body.web_search_options` | — |

### thinking ↔ reasoning_effort 映射

**Anthropic → OpenAI Chat：**

| Anthropic `thinking` | OpenAI Chat `reasoning_effort` |
|----------------------|-------------------------------|
| `{"type": "disabled"}` | `"none"` |
| `{"type": "enabled", "budget_tokens": <10000}` | `"low"` |
| `{"type": "enabled", "budget_tokens": 10000~31999}` | `"medium"` |
| `{"type": "enabled", "budget_tokens": ≥32000}` | `"high"` |
| `{"type": "adaptive"}` | `"medium"`（SDK 规范：adaptive 无 budget_tokens，默认 medium） |
| `{"type": "adaptive", "budget_tokens": N}` | 按预算映射 |

**OpenAI Chat → Anthropic：**

| OpenAI Chat `reasoning_effort` | Anthropic `thinking` |
|-------------------------------|----------------------|
| `"none"` / `"minimal"` | `{"type": "disabled"}` |
| `"low"` | `{"type": "enabled", "budget_tokens": 1024}` |
| `"medium"` | `{"type": "enabled", "budget_tokens": 10000}` |
| `"high"` | `{"type": "enabled", "budget_tokens": 32000}` |
| `"xhigh"` | `{"type": "enabled", "budget_tokens": 64000}` |

> `thinking.display` 字段（`"summarized"` | `"omitted"`）会保留在 `extra_body` 中，供支持该参数的后端使用。`reasoning_effort` 支持值：`none` / `minimal` / `low` / `medium` / `high` / `xhigh`。gpt-5.1 默认 `none`，gpt-5-pro 默认且仅支持 `high`，`xhigh` 支持 gpt-5.1-codex-max 之后的模型。

### 响应格式对比

| 字段 | OpenAI Chat | OpenAI Responses | Anthropic |
|------|------------|------------------|-----------|
| 类型标识 | `object: "chat.completion"` | `object: "response"` | `type: "message"` |
| 内容 | `choices[].message.content` | `output[].content[]` | `content[]` |
| 工具调用 | `choices[].message.tool_calls[]` | `output[].type=function_call` | `content[].type=tool_use` |
| 扩展思考 | `choices[].message.reasoning_content` | `output[].type=reasoning`（含 `reasoning_text`） | `content[].type=thinking` |
| 停止原因 | `finish_reason: stop/length/tool_calls` | `status: completed/incomplete` | `stop_reason: end_turn/max_tokens/tool_use` |
| 用量 | `usage.prompt_tokens` | `usage.input_tokens` | `usage.input_tokens` |
| 缓存用量 | `usage.prompt_tokens_details.cached_tokens` | `usage.input_tokens_details.cached_tokens` | `usage.cache_read_input_tokens` |
| 推理用量 | `usage.completion_tokens_details.reasoning_tokens` | `usage.output_tokens_details.reasoning_tokens` | — |
| 服务器工具用量 | `usage.server_tool_use`（保留） | — | `usage.server_tool_use` |
| 缓存创建详情 | `usage.cache_creation`（保留） | — | `usage.cache_creation` |
| 引用/注解 | `message.annotations` | `output_text.annotations` | `text.citations` |
| 完成时间 | — | `completed_at` | — |

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
| `content_filter` | `incomplete` (content_filter) / `failed` | `refusal` |

### service_tier 映射

**请求参数映射：**

|| Anthropic 请求 | OpenAI Chat / Responses 请求 |
|-----------|---------------------------|
| `"auto"` | `"auto"` |
| `"standard_only"` | `"default"` |

**响应 usage 映射：**

|| Anthropic 响应 | OpenAI Chat / Responses 响应 |
|-----------|----------------------------|
| `"standard"` | `"default"` |
| `"priority"` | `"priority"` |
| `"batch"` | `"default"` |

### 内容块类型映射

| Anthropic | → OpenAI Chat | → OpenAI Responses |
|-----------|--------------|-------------------|
| `text` | `{"type":"text","text":"..."}` | `{"type":"output_text","text":"..."}` |
| `text` (with citations) | `{"type":"text","text":"...","citations":[...]}` | `{"type":"output_text","text":"...","annotations":[...]}` |
| `image` (base64) | `{"type":"image_url","image_url":{"url":"data:..."}}` | `{"type":"input_image","image_url":"data:..."}` |
| `image` (url) | `{"type":"image_url","image_url":{"url":"https://..."}}` | `{"type":"input_image","image_url":"https://..."}` |
| `document` (base64) | `{"type":"file","file":{...}}` | `{"type":"input_file","file_data":"..."}` |
| `tool_use` | `tool_calls[{"id","function":{"name","arguments"}}]` | `{"type":"function_call","call_id",...}` |
| `tool_result` | `{"role":"tool","tool_call_id","content"}` | `{"type":"function_call_output","call_id","output"}` |
| `tool_result(is_error)` | `{"role":"tool","content":"[Error]..."}` | — |
| `thinking` | `reasoning_content` | `{"type":"reasoning","content":[{"type":"reasoning_text"}]}` |
| `redacted_thinking` | `reasoning_content` = `[redacted_thinking: data]` | `{"type":"reasoning","content":[],"encrypted_content":"..."}` |
| `server_tool_use` | `tool_calls[...]` | `{"type":"function_call",...}` |

## 流式转换

流式转换严格遵循各协议的事件序列，支持一对多事件映射（例如一个 OpenAI Chat chunk 可能生成 `content_block_start` + `content_block_delta` 两个 Anthropic 事件）。

### Anthropic 流式事件序列

```
message_start → 
  [content_block_start → content_block_delta* → content_block_stop]* → 
message_delta → message_stop
```

| 事件 | 说明 |
|------|------|
| `message_start` | 包含完整 message 对象（含 usage） |
| `content_block_start` | 新内容块开始（text / thinking / tool_use） |
| `content_block_delta` | 内容增量（text_delta / thinking_delta / input_json_delta / signature_delta / citations_delta） |
| `content_block_stop` | 内容块结束 |
| `message_delta` | 消息级增量（stop_reason + usage.output_tokens） |
| `message_stop` | 消息结束 |
| `ping` | 心跳 |
| `error` | 错误（如 `content_filter` 触发时生成） |

> `signature_delta` 事件在 `thinking_delta` 后自动追加，用于多轮对话的 thinking 连续性（符合 Anthropic SDK `ThinkingBlock` 的 `signature` 字段要求）。

### OpenAI Responses 流式事件序列

```
response.created → response.in_progress → 
  [response.output_item.added → 
    [response.content_part.added → response.output_text.delta* → response.output_text.done → response.content_part.done]?
    [response.reasoning_text.delta* → response.reasoning_text.done]?
    [response.function_call_arguments.delta* → response.function_call_arguments.done]?
  → response.output_item.done]* → 
response.completed | response.failed
```

### 流式状态管理

转换器内部维护流式状态，确保事件序列的完整性：

```python
# 每次新流式请求前应重置状态
AnthropicConverter.reset_stream_state()
OpenAIResponsesConverter.reset_stream_state()
```

## 协议检测规则

检测器按优先级依次检查（Anthropic > Responses > Chat）：

### Anthropic 检测规则

1. 必须有 `max_tokens`（Anthropic 必填参数）和 `messages`
2. 模型名以 `claude-` 开头
3. 存在 Anthropic 特有参数：`system`、`stop_sequences`、`thinking`、`cache_control`、`top_k`、`container`、`output_config`、`output_format`、`inference_geo`
4. `tool_choice` 为 `"any"` 或 `{"type":"tool"}`
5. 消息内容包含 Anthropic 特有内容类型：`tool_result`、`tool_use`、`thinking` 等

### OpenAI Responses 检测规则

1. 使用 `input` 参数而非 `messages`
2. 存在 Responses 特有参数：`instructions`、`max_output_tokens`、`previous_response_id`、`reasoning`、`text`、`truncation`、`background`、`max_tool_calls` 等

### OpenAI Chat 检测规则

1. 使用 `messages` 参数
2. 消息角色包含 `system`、`user`、`assistant`、`tool`、`developer`

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
│   └── test_protocol_converter.py   # 372 个单元测试
├── examples/
│   ├── integration_test_chat_backend.py       # OpenAI Chat 后端集成测试
│   ├── integration_test_anthropic_backend.py  # Anthropic 后端集成测试
│   ├── integration_test_responses_backend.py  # OpenAI Responses 后端集成测试
│   └── integration_test_all_9_paths.py       # 3×3 全量 9 路集成测试
└── pyproject.toml
```

## 运行测试

### 环境准备

```bash
# 安装项目及开发依赖
pip install -e ".[dev]"

# 或手动安装核心依赖
pip install pytest pytest-asyncio httpx
```

### 核心依赖说明

| 依赖 | 版本要求 | 用途 | 是否必需 |
|------|---------|------|---------|
| Python | ≥ 3.9 | 运行时 | 必需 |
| httpx | ≥ 0.25.0 | HTTP 客户端（用于转发请求） | 可选 |
| pytest | ≥ 7.0.0 | 单元测试框架 | 开发必需 |
| pytest-asyncio | ≥ 0.21.0 | 异步测试支持 | 开发必需 |
| pytest-cov | ≥ 4.0.0 | 测试覆盖率 | 开发可选 |

**协议参考与验证来源**：

- OpenAI Chat Completions API — `openai-python` SDK v2.11 源码（`ChatCompletionReasoningEffort`、`ChatCompletionStreamOptions`、`prompt_cache_retention` 等）
- OpenAI Responses API — 官方文档 `developers.openai.com`（`reasoning.summary`、`text.format`、流式事件序列）
- Anthropic Messages API — `anthropic-sdk-python` 源码（`ThinkingConfigParam`、`RedactedThinkingBlock.data`、`OutputConfigParam`、`Container` 等）
- Context7 实时文档检索验证（三方 API 参数规范与类型定义交叉比对）

### 执行单元测试

```bash
# 1. 进入项目目录
cd /root/repos/ai-proxy/protocol_converter_pkg

# 2. 运行全部 372 个单元测试
python3 -m pytest tests/test_protocol_converter.py -v

# 3. 运行带覆盖率的测试
python3 -m pytest tests/test_protocol_converter.py -v --cov=protocol_converter

# 4. 运行特定测试类
python3 -m pytest tests/test_protocol_converter.py -v -k "TestAnthropicThinkingWithText"
```

### 执行集成测试

集成测试需要配置有效的 API Key（位于 `examples/` 下的各测试文件中），支持以下后端：

```bash
# 1. 进入项目目录
cd /root/repos/ai-proxy/protocol_converter_pkg

# 2. OpenAI Chat 后端集成测试（MiniMax API，8 项测试）
python3 examples/integration_test_chat_backend.py

# 3. Anthropic 兼容后端集成测试（MiniMax API，9 项测试，含流式）
python3 examples/integration_test_anthropic_backend.py

# 4. OpenAI Responses 后端集成测试（OpenRouter API，9 项测试，含流式）
python3 examples/integration_test_responses_backend.py

# 5. 3×3 全量 9 路集成测试（覆盖所有协议 × 后端组合，非流式 + 流式）
python3 examples/integration_test_all_9_paths.py
```

> 集成测试自动检测 API 连接，如果 API Key 无效或网络不可达，仅 API 调用部分会跳过，协议检测和转换逻辑的验证仍然正常运行。

### 测试验证清单

| 测试类型 | 测试文件 | 测试数量 | 通过标准 |
|---------|---------|---------|---------|
| 单元测试 | `tests/test_protocol_converter.py` | 372 | 全部通过 |
| Chat 后端集成测试 | `examples/integration_test_chat_backend.py` | 10 | 全部通过 |
| Anthropic 后端集成测试 | `examples/integration_test_anthropic_backend.py` | 9 | 全部通过 |
| Responses 后端集成测试 | `examples/integration_test_responses_backend.py` | 9 | 全部通过 |
| 3×3 全量集成测试 | `examples/integration_test_all_9_paths.py` | 9 (非流式+流式) | 全部通过 |

## 参考

- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
- [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses)
- [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)
- [openai-python SDK](https://github.com/openai/openai-python)
- [anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python)

## 版本信息

| 版本 | 日期 | 说明 |
|------|------|------|
| **v1.26.0** | 2026-05-17 | 集成测试流式覆盖补全：4 个集成测试文件均补齐跨协议流式测试，9 路全量测试支持非流式+流式双模式 |
| **v1.25.0** | 2026-05-17 | 7 项深度审查修复：instructions 顺序修正、redacted_thinking 往返转换完善、reasoning_content→Responses encrypted_content 映射、thinking.display→reasoning.summary 映射、user→metadata.user_id 映射、流式完成事件转换 |
| v1.24.0 | — | 6 轮深度审查修复：`minimal` 推理映射修正、`redacted_thinking` 往返转换、空 input 容错、`stop` 空值过滤、工具 `input_schema` 默认值 |
| v1.23.0 | — | 6 轮审查修复：`reasoning_effort` 映射、`output_format` 逆向映射、`service_tier` 映射、流式 reasoning 转换等 |
| v1.17.0 | — | 3 轮审查修复：`text.verbosity` 双向映射、`generate_summary` 往返、`cache_control` 保留等 |
| v1.0.0 | — | 初始版本：9 路转换矩阵、流式 SSE、工具调用、多模态支持 |

**当前版本**：`1.26.0`（同步更新于 `pyproject.toml` 和 `protocol_converter/__init__.py`）

**核心依赖**：
- Python ≥ 3.9
- httpx ≥ 0.25.0（可选，用于 `convert_and_forward` 转发请求）
- pytest ≥ 7.0.0 + pytest-asyncio ≥ 0.21.0（开发依赖，用于测试）

**测试执行**：

```bash
# 单元测试（372 个测试用例）
cd protocol_converter_pkg
PYTHONPATH=. python3 -m pytest tests/test_protocol_converter.py -v

# 集成测试（9 路全量，需要网络访问和 API Key）
PYTHONPATH=. python3 examples/integration_test_all_9_paths.py

# 单后端集成测试
PYTHONPATH=. python3 examples/integration_test_chat_backend.py
PYTHONPATH=. python3 examples/integration_test_anthropic_backend.py
PYTHONPATH=. python3 examples/integration_test_responses_backend.py
```

## 更新日志

### v1.26.0

集成测试流式覆盖补全，4 个集成测试文件均补齐跨协议流式测试，9 路全量测试支持非流式+流式双模式：

**集成测试流式覆盖补全：**

- **`integration_test_all_9_paths.py`**：9 条路径（3 协议 × 3 后端）均新增流式测试子项，每条路径报告 `(非流式=✓, 流式=✓)` 双模式结果
  - 新增 `call_backend_stream()` 统一流式调用函数，自动适配 Chat/Responses/Anthropic 三种后端 SSE 格式
  - Chat 后端：`data:` 行 + `choices[0].delta.content` 解析
  - Responses 后端：`event:` + `data:` 行 + `response.output_text.delta` 解析
  - Anthropic 后端：`event:` + `data:` 行 + `content_block_delta`/`text_delta` 解析

- **`integration_test_chat_backend.py`**：新增 2 项跨协议流式测试（Responses→Chat 流式、Anthropic→Chat 流式），总计 10 项测试

- **`integration_test_anthropic_backend.py`**：新增 2 项跨协议流式测试（Chat→Anthropic 流式、Responses→Anthropic 流式），总计 9 项测试

- **`integration_test_responses_backend.py`**：新增 2 项跨协议流式测试（Chat→Responses 流式、Anthropic→Responses 流式），总计 9 项测试

**流式测试修复：**

- 修复所有跨协议流式测试中 `engine.convert_request()` 不自动映射 model 的问题，统一添加 `converted["model"] = CONFIG.default_model`
- 修复 Responses 后端流式测试中错误响应体未输出导致难以调试的问题

**测试验证：**

- 372 个单元测试全部通过
- 9 路全量集成测试全部通过（非流式+流式双模式）
- Chat 后端集成测试 10 项全部通过
- Anthropic 后端集成测试 9 项全部通过
- Responses 后端集成测试 9 项全部通过

### v1.25.0

基于官方文档、Context7 实时检索和 Python SDK（`openai-python` v2.11、`anthropic-sdk-python`）深度审查后修复 7 项关键缺陷：

**协议解析与字段映射修复：**

- **Fix #1: instructions 数组顺序修正**：`OpenAIResponsesConverter.to_openai_chat` 中 `instructions` 为数组时，`reversed()` 调用导致项顺序翻转（最后一条变为第一条）。已移除 `reversed()`，保持原始顺序
- **Fix #2: redacted_thinking 块数据保留**：`AnthropicConverter._convert_message` 中 `redacted_thinking` 块之前被 `pass` 静默丢弃，脱敏思考数据完全丢失。现已保留 `data` 字段（fallback `signature`）到 `reasoning_content`，使用 `[redacted_thinking: data]` 格式标记，确保三方往返转换完整
- **Fix #3: reasoning_content→Responses 输入项 encrypted_content 映射**：新增 `_convert_reasoning_content_to_responses_input()` 方法，识别 `[redacted_thinking: data]` 格式并映射为 Responses `reasoning` 输入项的 `encrypted_content`（而非错误地映射为 `summary_text`）
- **Fix #4: Chat→Responses 响应转换中 redacted_thinking 检测**：`from_openai_chat()` 响应转换中 `reasoning_content` 为 `[redacted_thinking: ...]` 格式时，正确映射为 Responses reasoning 输出项的 `encrypted_content`

**参数映射修复：**

- **Fix #5: thinking.display→reasoning.summary 映射**：`from_openai_chat_request()` 在 Chat→Responses 请求转换中，当 `extra_body.thinking.display` 存在时，正确映射为 Responses `reasoning.summary`（`display="summarized"` → `summary="concise"`，`display="omitted"` → `summary="omitted"`）
- **Fix #6: user→metadata.user_id 映射**：`to_anthropic_request()` 中 Responses 请求的 `user` 参数正确映射为 Anthropic `metadata.user_id`（之前错误地放入 `extra_body`，丢失语义）

**流式转换修复（关键）：**

- **Fix #7: Responses 后端流式完成/失败事件转换**：`_handle_stream_response()` 中 `response.completed`/`response.failed`/`response.incomplete` 事件之前被 `pass` 静默丢弃，导致 Responses 后端流式响应永远不会终止。现已正确转换为 Chat 格式流式块（含 `finish_reason`），确保流式转换完整结束

**测试验证：**

- 372 个单元测试全部通过（新增 23 个覆盖上述修复，含完整往返转换测试）
- 3×3 全量 9 路集成测试全部通过
- Chat 后端集成测试全部通过（8 项）
- Anthropic 后端集成测试全部通过（7 项）
- Responses 后端集成测试全部通过（7 项）

### v1.24.0

基于 OpenAI Python SDK v2.11、Anthropic SDK 最新规范及 Context7 文档交叉验证，执行 6 轮深度审查修复：

**协议解析与字段映射修复：**
- **`reasoning_effort="minimal"` 映射修正**：原映射 `budget=0` → `thinking.disabled`（语义错误），修正为 `budget=1024` → `thinking.enabled`，与 Anthropic 最小可用 budget 一致。影响文件：`engine.py`、`openai_responses.py`
- **`redacted_thinking` 往返转换支持**：Anthropic `redacted_thinking` → Chat `[redacted_thinking: data]` 格式 → 恢复为 `redacted_thinking` 块，实现脱敏思考内容的完整往返转换。影响文件：`engine.py`、`anthropic.py`、`openai_responses.py`
- **Responses `encrypted_content` 传递**：当 `reasoning` 输出项仅含 `encrypted_content` 无可见文本时，正确标记为 `[redacted_thinking: ...]` 格式传递至 Chat 格式

**异常处理与边界条件修复：**
- **空 `input` 容错**：Responses `input=[]`、`input=""` 不再产生无效请求，自动填充空用户消息。影响路径：`to_openai_chat`、`to_anthropic_request`、`from_openai_chat_request`
- **空 `messages` 容错**：Chat→Anthropic 转换时空消息列表自动补充空用户消息
- **`stop=[]`/`stop=""` 过滤**：空停止序列不再传递给 Anthropic `stop_sequences`（原代码会将空列表传入）
- **`max_completion_tokens=0` 支持**：正确映射为 Anthropic `max_tokens=0`（缓存预热场景），已使用 `is not None` 判断

**数据类型转换修复：**
- **Anthropic 工具 `input_schema` 缺失默认值**：原代码条件性添加 `parameters`，修正为始终提供默认值 `{"type": "object", "properties": {}}`，符合 Anthropic SDK 规范
- **新增 7 个单元测试**覆盖以上修复场景（总计 354 个测试用例，全部通过）

### v1.23.0

基于官方文档、Context7 实时检索和 Python SDK（`openai-python` v2.11、`anthropic-sdk-python`）进行6轮深度审查后修复关键缺陷：

**协议解析与字段映射修复：**

- **`tool_choice` dict `"none"` 类型处理修复**：Anthropic `tool_choice={"type":"none"}` 格式之前未被处理，正确添加 `type=="none"` → `return "none"` 映射（之前仅处理 "auto"/"any"/"tool" 三种类型，遗漏 "none"）

**数据类型转换修复：**

- **工具输入 JSON 双重序列化修复**：Anthropic→Chat 转换中，tool_use 块的 `input` 字段若已是 JSON 字符串（往返转换场景），`json.dumps()` 会将其双重序列化为 `"\\\"...\\\""` 格式导致解析失败。现添加类型检查：input 为字符串时直接使用，为 None 时输出 `"{}"`，为 dict 时才序列化

**测试覆盖：**

- 全部 347 个单元测试通过
- 3×3 全量 9 路集成测试全部通过

### v1.22.0

基于官方文档、Context7 实时检索和 Python SDK（`openai-python` v2.11、`anthropic-sdk-python`）进行6轮深度审查后修复防御性改进：

**第1-3轮 — 协议解析与字段映射逻辑缺陷：**

- **`summary="omitted"` 映射完善**：engine._chat_to_anthropic_request 中之前仅处理 `concise/detailed` → `summarized`，遗漏了 `omitted` → `omitted` 的情况。现已补全

**第4-6轮 — 异常处理遗漏与边界条件漏洞：**

- **`_anthropic_to_chat_response` 类型防护**：添加 `isinstance(block, dict)` 检查，防止非字典元素导致 `AttributeError`

- **`_convert_message` None 处理**：content 为 None 时不再转换为 `"None"` 字符串，而是直接返回空结果避免数据污染

- **`audio_tokens` 字段名修复**：Anthropic usage 映射中之前错误使用 `audio_output_tokens`，现已修正为 `audio_tokens`（与官方 SDK 字段名对齐）

- **`_anthropic_to_chat_response` content 迭代防护**：添加 `isinstance(block, dict)` 检查，确保安全迭代

**测试覆盖：**

- 全部 347 个单元测试通过
- 3×3 全量 9 路集成测试全部通过

### v1.21.0

基于官方文档、Context7 实时检索和 Python SDK（`openai-python` v2.11、`anthropic-sdk-python`）进行第3轮深度审查后修复异常处理遗漏与类型安全缺陷：

**第1轮 — 协议解析与字段映射逻辑缺陷：**

- **`_convert_system_blocks` 非 text 类型 system 块降级保留**：Anthropic→Chat 转换中，system 消息的 `thinking`/`image` 等非 text 类型块之前被静默丢弃（仅设置 `has_complex=True` 但不处理内容）。现降级为文本占位 `[block_type]` 追加到 text_parts 和 content_parts，避免内容丢失

- **`_convert_message` assistant 多模态内容完整保留**：Anthropic→Chat 转换中，assistant 消息同时含 text 和 image_url/file 等多模态块时，之前错误地将 `content_parts` 数组降级为纯文本字符串或置为 None，导致多模态内容丢失。现保留完整 `content_parts` 数组

**第2轮 — 异常处理遗漏与边界条件漏洞：**

- **`_convert_tools` 类型防护修复**：`tools` 参数若返回 `None`（请求中显式设为 `null`），直接迭代 `None` 会抛出 `TypeError`。现添加 `if not isinstance(tools, list): return []` 防护

- **`from_openai_chat` choice/tool_calls 元素防护修复**：Chat 响应 `choices` 列表中若含非字典元素，`choice.get("message", {})` 返回 `None` 后调用 `.get()` 会抛出 `AttributeError`。现添加 `if not isinstance(choice, dict): continue` 和 `if not isinstance(tc, dict): continue` 防护

- **`to_openai_responses` content 块元素防护修复**：Chat 响应 content 列表中若含非字典元素，`block.get("type")` 会抛出 `AttributeError`。现添加 `if not isinstance(block, dict): continue` 防护

**第3轮 — 数据类型转换校验及交叉复查：**

- **`_convert_message` content 列表块元素防护修复**：Anthropic 消息 content 列表中若含非字典元素，`block.get("type", "")` 会抛出 `AttributeError`。现添加 `if not isinstance(block, dict): continue` 防护

**本轮补充修复 — 深度审查遗漏项：**

- **`reasoning_effort` 新增 xhigh 级别支持**：`enabled` 类型 `budget_tokens >= 64000` 时现在正确映射为 `"xhigh"`，之前最高只能映射到 `"high"`

- **`redacted_thinking` 加密内容判断修复**：Responses→Anthropic 转换中，之前 `if encrypted_content and not item.get("content")` 对空列表 `[]` 返回 False，导致应该转换为 `redacted_thinking` 的情况被错误处理。现正确检查是否有可见内容

- **`thinking.display` 往返转换完整保留**：Anthropic→Chat→Anthropic 往返转换中，`thinking.display` 字段之前在某些路径丢失。现已在以下位置修复

- **`adaptive` 类型处理修复**：`thinking.type="adaptive"` 且带 `budget_tokens` 时，现在按预算值正确映射到对应 `reasoning_effort` 级别（xhigh/high/medium/low），而非错误地默认 medium

**测试覆盖：**

- 全部 347 个单元测试通过
- 3×3 全量 9 路集成测试全部通过

### v1.19.0

- **同协议快速路径优化**：`convert_request` 在源协议与目标后端协议一致时，若无需任何修改直接浅拷贝透传；仅需模型映射 / developer 降级 / extra_body 合并时只做最小修改，避免完整 deepcopy+转换的性能损耗
- **流式同协议直接转发**：`_handle_stream_response` 中 Chat→Chat、Anthropic→Anthropic、Responses→Responses 同协议时直接转发原始 SSE，跳过 `json.loads` + `convert_stream_chunk_multi` + `json.dumps` 的冗余处理
- **流式 Responses event 丢失修复**：`_handle_stream_response` 中 Responses 后端的 `event:` 行之前因 `pending_event_type` 被提前清空而丢失，现已修复
- **响应反向模型映射**：新增 `ConverterConfig.reverse_model_mapping_in_stream` 参数，控制非流式和流式响应中是否将 `model` 字段替换回原始请求模型名（默认 `False`，保持后端透传行为）
- **新增 15 个单元测试**覆盖同协议透传、反向模型映射（含非流式与流式场景）
- **全部 347 个单元测试通过**

### v1.18.0

基于官方文档、Context7 实时检索和 Python SDK（`openai-python` v2.11、`anthropic-sdk-python`）进行3轮深度审查后修复协议转换缺陷：

**第1轮 — 协议解析与字段映射逻辑缺陷：**

- **`protocol_detector` model 为 None 时崩溃修复**：`request.get("model")` 可能返回 `None`，直接调用 `.startswith()` 会抛出 `AttributeError`。现统一添加 `isinstance(model, str)` 校验后再调用字符串方法
- **`engine._anthropic_to_chat_response` 多个 text 块合并修复**：Anthropic 响应含多个 `text` 内容块时，后一个块会覆盖前一个块的内容，导致文本丢失。现改为追加合并，citations 同样追加
- **`engine._chat_to_anthropic_request` assistant 多模态内容块支持补全**：assistant 消息 content 为列表时，之前仅处理 `type="text"` 块，`image_url`/`file` 等多模态块被静默丢弃。现补充 image_url → Anthropic `image` 块、file → Anthropic `document` 块的映射
- **`openai_responses.from_openai_chat_request` tool 消息列表 content 结构保留修复**：Chat `tool` 消息 `content` 为列表（多模态工具结果）时，之前使用 `str(content)` 转为字符串，丢失结构。现调用 `_convert_chat_content_to_responses` 正确转换为 Responses 内容块列表
- **`anthropic.to_openai_chat` metadata 非字典容错修复**：`request.get("metadata")` 返回非字典值时，直接调用 `.get("user_id")` 会抛出 `AttributeError`。现添加 `isinstance(metadata, dict)` 校验
- **`anthropic.to_openai_responses` / `openai_responses.from_openai_chat` `created_at` 类型修复**：`time.time()` 返回浮点数，不符合 Responses API 整数时间戳规范。现统一使用 `int(time.time())`
- **`openai_responses.to_openai_chat` instructions 列表多模态支持**：instructions 数组中的 `input_image`/`input_file` 类型项之前被静默忽略。现正确转换为 Chat `image_url`/`file` 内容块，合并为单条 developer 消息
- **`openai_responses._convert_input_item_to_message` input_image/input_file 支持**：Responses `input` 数组中的独立 `input_image`/`input_file` 项之前返回 `None` 被丢弃。现正确转换为 Chat 多模态内容块

**第2轮 — 异常处理遗漏与边界条件漏洞：**

- **`engine._handle_stream_response` SSE 解析健壮性修复**：之前使用 `response.content` 迭代原始字节块，若服务器一次发送多行 SSE，解析会出错（event 与 data 混杂）。现优先使用 `response.aiter_lines()` 正确按行解析，兼容非 httpx 客户端回退到原逻辑
- **`anthropic.from_openai_chat` 响应含 error 字段处理**：Chat 响应若包含顶层 `error` 字段（如后端报错），之前 `stop_reason` 仍映射为 `"end_turn"`。现正确映射为 `"refusal"` 并填充 `stop_details`
- **`engine._chat_to_anthropic_request` system 消息非字典块降级修复**：system/developer 消息 `content` 为列表时，若元素为非字典类型（如字符串列表），之前被 `continue` 跳过导致内容丢失。现降级为字符串追加到 system_parts
- **全模块 `model=None` 边界统一修复**：所有 `request.get("model", default)` 在 `model` 显式为 `None` 时不会使用默认值。现统一改为 `request.get("model") or default`，确保 None 时回退到默认模型
- **`engine._chat_to_anthropic_request` 空 assistant 消息保留**：assistant 消息 `content=None` 且无 `tool_calls` 时，之前不生成任何消息导致消息序列断裂。现生成空 `content: []` 的 assistant 消息保持序列完整

**第3轮 — 数据类型转换校验及交叉复查：**

- **`openai_responses._convert_input_item_to_message` fallback 逻辑增强**：未知类型输入项的 fallback 不再仅检查 `"role" in item and "content" in item`，而是同时支持 `content` 为列表时的 `_convert_content_to_chat` 转换
- **`openai_responses.to_openai_chat` instructions message 类型 `text` 字段兼容**：message 类型的 instructions 项若使用 `text` 字段（非标准但可能出现），之前因只读取 `content` 而丢失文本。现已在重构后的 instructions 合并逻辑中覆盖
- **交叉复查验证**：逐文件检查所有 `request.get(...)` 的默认值逻辑、所有 `isinstance(...)` 的类型守卫、所有 `.startswith()` 的前置类型校验，确保边界条件全覆盖

**测试覆盖：**

- 全部 347 个单元测试通过
- 4 项集成测试全部通过（Chat 后端、Anthropic 后端、Responses 后端、3×3 全量 9 路）

### v1.17.0

基于官方文档、Context7 实时检索和 Python SDK（`openai-python` v2.11、`anthropic-sdk-python`）进行3轮深度审查后修复协议转换缺陷：

**第1轮 — 协议解析与字段映射逻辑缺陷：**

- **`text.verbosity` ↔ Chat `verbosity` 双向映射修复**：Responses API 的 `text.verbosity`（`ResponseTextConfigParam.verbosity`，嵌套在 `text` 配置内）之前错误地放入 `extra_body` 而非映射为 Chat 顶层 `verbosity` 参数。现修正：
  - Responses→Chat：`text.verbosity` 正确映射为 Chat 顶层 `verbosity`
  - Chat→Responses：`verbosity` 正确映射为 Responses `text.verbosity`（而非放入 `extra_body`）
- **`reasoning.generate_summary` 往返保留修复**：OpenAI SDK v2.11 新增 `reasoning.generate_summary` 参数（控制推理摘要生成），之前在 Responses→Chat→Responses 往返转换中丢失。现修正：
  - Responses→Chat：`reasoning.generate_summary` 与 `reasoning.summary` 一起保留在 `extra_body.reasoning` 中
  - Chat→Responses：从 `extra_body.reasoning` 中恢复 `generate_summary` 到目标 `reasoning` 配置
- **Responses→Anthropic `text.verbosity` 保留**：`to_anthropic_request()` 现将 `text.verbosity` 保留到 `extra_body.verbosity`，供 Anthropic 兼容后端使用
- **Responses→Anthropic `reasoning.generate_summary` 保留**：`to_anthropic_request()` 现将 `reasoning.generate_summary` 保留到 `extra_body` 中

**第2轮 — 异常处理遗漏与边界条件漏洞：**

- **Chat→Anthropic system `cache_control` 保留修复**：Chat 请求的 system 消息内容块带 `cache_control` 时（如 `[{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}]`），之前在转换中丢失 `cache_control` 字段。现修正：
  - `engine.py` 的 `_chat_to_anthropic_request()` 检测 system 消息中的 `cache_control`，使用 Anthropic `TextBlockParam[]` 格式（而非扁平字符串）构造 `system` 参数
  - `anthropic.py` 的 `_convert_system_blocks` 在 Anthropic→Chat 反向转换时保留 `cache_control`
- **`_convert_content_to_anthropic` 空内容返回类型修复**：`openai_responses.py` 中当内容为空时之前返回 `""`（字符串），但调用方 `_convert_input_item_to_anthropic` 在 `tool_result` 场景中期望内容块列表。现修正为返回 `[{"type": "text", "text": ""}]`

**第3轮 — 数据类型转换校验及交叉复查：**

- **`openai_responses.py` 局部 `import json as _json` 不一致修复**：方法内使用 `import json as _json` 与模块级 `import json` 不一致，导致不同代码路径使用不同的 `json` 引用。统一使用模块级 `import json`
- **参数映射表修正**：README 中 Chat→Responses 的 `verbosity` 映射目标从 `extra_body.verbosity` 修正为 `text.verbosity`

**测试覆盖：**

- 新增 11 个单元测试（4 个 `TestVerbosityMapping`、3 个 `TestReasoningGenerateSummary`、2 个 `TestSystemCacheControl`、2 个 `TestConvertContentToAnthropicEmpty`）
- 全部 347 个单元测试通过

### v1.16.0

本次审查修复以下缺陷（基于官方文档、Context7 实时检索和 Python SDK `openai-python` v2.11、`anthropic-sdk-python` 源码）：

- **Chat→Responses assistant 消息 `content=None` + `reasoning_content` 转换修复**：`from_openai_chat_request` 在 assistant 消息有 `reasoning_content` 但 `content=None`（且无 `tool_calls`）时，之前将 None 转为字符串 `"None"` 创建 `input_text` 消息。现正确处理：`content=None` 时创建 reasoning 输入项而非包含 "None" 文本的消息
- **Chat→Responses assistant 消息 `content=None` 无 reasoning 无 tool_calls 修复**：assistant 消息 `content=None` 且无推理内容和工具调用时，现创建空 `content` 的 message 输入项保持消息序列完整性，而非生成 "None" 文本
- **`_convert_content_to_chat` 多模态空内容返回类型一致性修复**：`openai_responses.py` 中当 `has_multimodal=True` 但 `content_parts` 和 `text_parts` 均为空时，之前返回 `""`（空字符串），破坏了 `has_multimodal=True` 时返回列表类型的约定。现修正为返回 `[]`（空列表），保持类型一致性
- **Responses→Anthropic `container` 字段保留**：`OpenAIResponsesConverter.to_anthropic` 响应转换现正确保留 Responses 响应中的 `container` 字段（Anthropic API 支持 container 语义）
- **Responses→Anthropic `function_call_output` 非字符串 output 转换修复**：`_convert_input_item_to_anthropic` 在 Responses `function_call_output` 的 `output` 为列表（多内容块工具结果）时，现正确转换为 Anthropic `tool_result.content` 的内容块列表（之前直接传递原始值，可能导致类型不匹配）
- **`ConverterConfig` 文档注释修正**：`prompt_cache_retention` 值从 `"in_memory"` 修正为 `"in-memory"`，与 OpenAI Python SDK `Literal["in-memory", "24h"]` 规范一致
- **321 个单元测试**（新增 9 个覆盖上述修复）
- **9 路集成测试全部通过**

### v1.15.0

基于官方文档、Context7 和 Python SDK（`openai-python` v2.11、`anthropic-sdk-python`）全面审查后修复协议转换缺陷并完善逻辑：

- **Anthropic→Chat thinking 块内容丢失修复**：`AnthropicConverter._convert_message` 在 assistant 消息同时包含 `thinking` 和 `text` 块时，`thinking` 内容现正确映射为 Chat `reasoning_content` 字段（之前仅在 `content=None` 且无 `tool_calls` 时才保留，导致有文本内容时推理信息完全丢失）
- **Chat→Responses assistant 列表内容 + reasoning_content 转换修复**：`from_openai_chat_request` 在 assistant 消息有列表类型 `content` 和 `reasoning_content`（无 `tool_calls`）时，`reasoning_content` 现正确映射为 Responses `reasoning` 输入项（之前仅处理字符串类型 content 的 reasoning_content，列表类型场景遗漏）
- **Chat→Anthropic tool 消息列表内容转换修复**：`_chat_to_anthropic_request` 在 Chat `tool` 消息的 `content` 为列表（多模态工具结果）时，现正确转换为 Anthropic `tool_result` 的 content 块列表（之前使用 `str(content)` 将列表转为字符串表示，丢失内容结构）
- **Chat→Anthropic tool 列表内容 `[Error]` 前缀检测**：列表类型 tool content 中的 `[Error]` 前缀现正确检测并映射为 Anthropic `tool_result.is_error` 字段
- **Responses→Chat `reasoning_summary_text.delta` 事件处理**：`_convert_responses_event_to_chat_chunk` 新增 `response.reasoning_summary_text.delta` 事件到 Chat `reasoning_content` 的映射（之前该事件被丢弃，导致推理摘要内容丢失）
- **Responses→Chat `response.refusal.delta` 事件处理**：`_convert_responses_event_to_chat_chunk` 新增 `response.refusal.delta` 事件到 Chat `content` delta 的映射（之前拒绝内容增量被丢弃）
- **312 个单元测试**（新增 12 个覆盖上述修复）
- **9 路集成测试全部通过**

### v1.14.0

基于官方文档、context7 和 Python SDK（`openai-python` v2.11、`anthropic-sdk-python`）全面审查后修复协议转换缺陷并完善逻辑：

- **协议检测器误判修复**：`ANTHROPIC_CONTENT_TYPES` 中移除 `"text"` 类型。之前包含 `"text"` 导致 Chat API 的多模态内容（`{"type": "text", "text": "..."}` 块）被错误检测为 Anthropic 协议
- **refusal 停止原因优先级修复**：`AnthropicConverter.from_openai_chat` 中当 Chat 响应同时包含 `refusal` 和 `finish_reason` 时，`refusal` 现正确优先于 `finish_reason` 映射为 `stop_reason="refusal"`（之前 `finish_reason` 会覆盖 `refusal`）
- **Anthropic→Responses `container` 字段保留**：`AnthropicConverter.to_openai_responses` 现正确保留 Anthropic 响应中的 `container` 字段
- **Anthropic `stop_sequence` 停止原因映射补充**：`to_openai_responses` 中 `stop_sequence` 停止原因正确映射为 Responses `completed` 状态（非 `incomplete`）
- **Responses `completed_at` 字段修复**：`OpenAIResponsesConverter.from_openai_chat` 仅在 `status="completed"` 时设置 `completed_at`，`incomplete` 状态不再包含该字段（符合 Responses API 规范）
- **Chat→Responses assistant+tool_calls 多模态内容转换修复**：`from_openai_chat_request` 中当 assistant 消息同时含 `tool_calls` 和列表类型 `content` 时，内容正确转换为 Responses 输入项（之前仅处理字符串类型 content）
- **Chat→Responses `reasoning_content` 转换补全**：`from_openai_chat_request` 中 assistant 消息的 `reasoning_content` 现正确映射为 Responses `reasoning` 输入项（含 `summary_text`），覆盖有/无 `tool_calls` 两种场景
- **Responses 后端流式事件转换实现**：新增 `OpenAIResponsesConverter._convert_responses_event_to_chat_chunk()` 方法，将 Responses SSE 事件（`response.output_text.delta`、`response.reasoning_text.delta`、`response.function_call_arguments.delta`、`response.output_item.added`、`response.completed` 等）正确转换为 Chat 格式流式块，解决真正的 Responses 格式后端流式转换问题
- **引擎 Responses 后端流式处理修复**：`ProtocolConverterEngine._handle_stream_response` 中 Responses 后端返回的命名 SSE 事件（`event: response.output_text.delta`）现正确解析并转换为目标协议格式（之前直接丢弃事件类型信息，导致转换失败）
- **reasoning 内容空值处理优化**：`to_chat_response` 中 reasoning 输出项的 `reasoning_text` 和 `summary_text` 现过滤空字符串，避免生成无意义的换行符
- **300 个单元测试**（新增 21 个覆盖上述修复）
- **9 路集成测试全部通过**

基于官方文档、context7 和 Python SDK 全面审查转换逻辑，修复缺陷并完善往返转换：

- **Chat→Anthropic 往返参数恢复**：`_chat_to_anthropic_request` 现正确提取并恢复 Anthropic 特有参数（`top_k`、`container`、`cache_control`、`output_config`），这些参数在 Anthropic→Chat 转换时存入 `extra_body`，经引擎合并到顶层后，之前在反向转换时丢失
- **Anthropic 服务器工具恢复**：`anthropic_server_tools`（`web_search_*`、`bash_*` 等）现正确合并回 Anthropic 请求的 `tools` 列表，且自动去重
- **Responses→Anthropic `stop`→`stop_sequences` 映射**：`OpenAIResponsesConverter.to_anthropic_request` 现正确将 `stop` 参数映射为 Anthropic 的 `stop_sequences`（之前仅放入 `extra_body`，未映射为 Anthropic 原生参数）
- **assistant 消息仅含 thinking 块时的处理**：`AnthropicConverter._convert_message` 在 assistant 消息只包含 `thinking`/`redacted_thinking` 块时，现保留 `reasoning_content` 字段（OpenAI o 系列格式），避免生成 `content: None` 的空消息
- **`_convert_content_to_chat` 多模态返回类型一致性修复**：当 `has_multimodal=True` 但 `content_parts` 为空时，返回空列表 `[]` 而非字符串，确保多模态场景返回类型一致
- **279 个单元测试**（新增 13 个覆盖上述修复）
- **9 路集成测试全部通过**

### v1.13.0

基于 context7 官方文档和 Python SDK 审查后修复转换逻辑缺陷并完善核心转换功能：

- **`to_anthropic_request` 的 `max_output_tokens=0` 边界修复**：Responses→Anthropic 请求转换中，`max_output_tokens=0`（缓存预热场景）之前被错误默认为 `4096`，现使用 `is not None` 判断正确保留 0 值
- **`to_anthropic_request` 的 `metadata` 覆盖 bug 修复**：之前先设置 `metadata.user_id` 再无条件覆盖为完整 metadata dict，导致 `user_id` 提取逻辑成为死代码。现修正为：仅 `user_id` 传递给 Anthropic `metadata` 参数，其他非 `user_id` 字段保留在 `extra_body` 中（Anthropic API 仅支持 `metadata.user_id` 字段）
- **`to_anthropic_request` 新增 `parallel_tool_calls` → `disable_parallel_tool_use` 映射**：Responses→Anthropic 请求转换中，`parallel_tool_calls=False` 现正确映射到 Anthropic `tool_choice` 中的 `disable_parallel_tool_use: True`（与 Chat→Anthropic 路径行为一致）
- **`to_anthropic_request` 新增 `reasoning.summary` → `thinking.display` 映射**：Responses API 的 `reasoning.summary`（`"concise"`/`"detailed"`）现映射到 Anthropic `thinking.display`（`"summarized"`），`summary="auto"/None` 不设置 display（使用 Anthropic 默认行为）
- **Chat→Anthropic 路径同步 `reasoning.summary` → `thinking.display`**：当 Chat 请求带 `reasoning_effort` 且 `extra_body.reasoning.summary` 存在时，`summary` 值正确映射到 `thinking.display`
- **协议检测器增加 `tool_choice` dict `type="any"` 识别**：Anthropic SDK 发送 `{"type": "any", "disable_parallel_tool_use": true}` 格式的 `tool_choice`，现正确识别为 Anthropic 协议
- **Anthropic 后端流式转换实现**：新增 `AnthropicConverter.convert_anthropic_event_to_chat()` 方法，当后端为 Anthropic 且目标协议为 Chat/Responses 时，Anthropic SSE 事件正确转换为 Chat 格式流式块（之前直接透传导致格式错误）
- **Responses 后端流式直通支持**：当后端为 Responses 且目标协议同为 Responses 时，原始 SSE 事件直接转发
- **`extra_body` 合并逻辑修复**：`to_anthropic_request` 中 `extra_body` 不再被后续参数覆盖，改为正确合并
- **266 个单元测试**（新增 25 个覆盖上述修复）
- **9 路集成测试全部通过**

### v1.11.0

基于 context7 官方文档和 Python SDK（`openai-python` v2.33.0、`anthropic-sdk-python`）审查后完善转换逻辑：

- **修复 `_convert_chat_content_to_responses` 方法缺失**：该方法之前作为死代码存在于 `_convert_tool_choice_to_anthropic` 内部，调用时会抛出 `AttributeError`。现已独立为正确的 `@classmethod`，Chat 多模态内容（图片、文件等）转 Responses 格式正常工作
- **Anthropic `output_format` 参数映射**：新增 Anthropic 结构化输出 `output_format` 参数的三方映射：
  - Anthropic `output_format`（`json_schema`/`json_object`）→ Chat `response_format`
  - Chat `response_format` → Anthropic `output_format`（从仅存入 `extra_body` 改为同时映射到 `output_format` 顶层参数）
  - Responses `text.format` → Anthropic `output_format`（直接转换路径）
- **Responses API `response.incomplete` 流式事件**：新增 `response.incomplete` 事件类型，`finish_reason=length` 时生成 `response.incomplete` 而非仅 `response.completed`
- **Responses API 流式事件文档补全**：补充完整的流式事件列表，包括：
  - `response.reasoning_summary_part.added/done`
  - `response.reasoning_summary_text.delta/done`
  - `response.refusal.delta/done`
  - `response.code_interpreter_call_code.delta/done`
  - `response.computer_call.in_progress/completed`
  - `response.incomplete`
- **协议检测器增加 `output_format` 识别**：带 `output_format` 参数的请求正确识别为 Anthropic 协议
- **241 个单元测试**（新增 12 个覆盖上述改进）
- **9 路集成测试全部通过**

### v1.10.0

基于官方文档和 Python SDK 进一步审查后修复转换逻辑缺陷并完善功能：

- **`top_logprobs` 自动设置 `logprobs=True`**：Responses→Chat 请求转换中，设置 `top_logprobs` 时自动添加 `logprobs=True`（Chat API 要求 `logprobs=True` 才能使用 `top_logprobs`，否则会报错）
- **`input_image` base64 回退逻辑修复**：Responses→Chat 内容转换中，`input_image` 的 `image_url` 为空时的 base64 回退逻辑之前始终得到空值，现正确处理 `file_id` 字段（降级为文本占位）
- **`stream_options` 跨协议映射修正**：
  - Responses `stream_options.include_obfuscation` 不再错误传递为 Chat 顶层参数（Chat API 不支持该字段），放入 `extra_body` 保留
  - Chat `stream_options.include_usage` 不再传递给 Responses API（Responses 自动在 `response.completed` 事件中返回 usage），放入 `extra_body` 保留
  - Chat 中来自 Responses 的 `include_obfuscation` 正确恢复为 Responses `stream_options`
- **Chat `web_search_options` → Responses 工具映射修复**：Chat 请求的 `web_search_options`（顶层参数）现正确映射到 Responses `web_search` 工具的 `search_context_size` 和 `user_location` 字段，不再错误地从 Chat 工具定义内部读取
- **Responses→Anthropic 请求直接转换路径**：新增 `OpenAIResponsesConverter.to_anthropic_request()` 方法，Responses 请求转 Anthropic 后端不再经 Chat 中转，保留 `reasoning`→`thinking`、`instructions`→`system` 等完整参数映射
- **Responses→Anthropic 响应 `service_tier` 映射**：`to_anthropic()` 响应转换现包含 `service_tier` 在 usage 中的映射
- **输入类型校验**：所有主要转换方法（`to_openai_chat`、`from_openai_chat_request`、`from_openai_chat`）新增 `TypeError` 校验，传入非字典参数时抛出明确错误
- **229 个单元测试**（新增 11 个覆盖上述修复）
- **9 路集成测试全部通过**

### v1.9.1

基于官方文档和 Python SDK 进一步审查后修复边界问题并完善代码健壮性：

- **`ChatCompletionRequest` / `ChatCompletionResponse` 新增 `to_dict()` 方法**：数据类对象现可直接调用 `.to_dict()` 转换为字典，支持 `json.dumps` 序列化，解决示例代码与外部集成中的 `TypeError: Object of type ChatCompletionRequest is not JSON serializable` 问题
- **示例代码 JSON 序列化修复**：`examples/__init__.py` 中 `OpenAIChatConverter.to_openai_chat` 的返回结果现通过 `.to_dict()` 序列化，示例可直接运行
- **代码风格优化**：合并 `openai_responses.py` 中 `from_openai_chat` 方法内重复的 `if refusal:` 判断块
- **218 个单元测试全部通过**
- **9 路集成测试全部通过**

### v1.9.0

基于官方文档和 Python SDK 深度审查后修复转换逻辑缺陷并完善核心转换逻辑：

- **`OpenAIChatConverter._from_anthropic` 字段丢失修复**：Anthropic→Chat 转换路径中 `max_completion_tokens`、`reasoning_effort`、`parallel_tool_calls`、`service_tier` 等关键字段之前未从转换结果中读取，现已全部补齐（`max_completion_tokens` 之前错误读取 `max_tokens` 而实际值存储在 `max_completion_tokens` 键下）
- **`ChatCompletionRequest` 新增 `extra_body` 字段**：数据类新增 `extra_body` 字段，Anthropic 特有参数（如 `top_k`、`thinking`、`cache_control` 等）不再在 `_from_anthropic` 路径中丢失
- **Responses `text.format` ↔ Chat `response_format` `json_schema` 结构互转修复**：
  - Responses→Chat：Responses API 格式 `{"type": "json_schema", "name": "...", "schema": {...}}` 现正确转换为 Chat 格式 `{"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}`（之前错误将 `type` 字段包含在 `json_schema` 内部）
  - Chat→Responses：Chat 格式 `json_schema` 子对象现正确展开为 Responses 顶层格式（之前错误保留嵌套的 `json_schema` 键）
  - 兼容旧版嵌套格式（`format.json_schema.name`）自动识别
- **Responses 流式 reasoning→tool_calls 转换修复**：当工具调用紧跟推理内容出现（无文本内容过渡）时，reasoning 项现正确关闭（发送 `response.reasoning_text.done` + `response.output_item.done`），避免事件序列错误
- **Responses 响应 `incomplete_details` 字段始终存在**：`from_openai_chat` 转换结果始终包含 `incomplete_details` 字段（`null` 表示完成），符合 Responses API 规范
- **218 个单元测试**（新增 10 个覆盖上述修复）
- **9 路集成测试全部通过**

### v1.8.0

基于官方文档和 Python SDK 深度审查后修复逻辑缺陷并完善转换逻辑：

- **`parallel_tool_calls: False` 无显式 `tool_choice` 时的语义丢失修复**：Chat→Anthropic 转换中，当 `parallel_tool_calls: False` 但未设置 `tool_choice` 时，现自动设置 `tool_choice: {"type": "auto", "disable_parallel_tool_use": True}` 以保留禁用并行工具调用的语义（Anthropic API 的 `disable_parallel_tool_use` 是 `tool_choice` 的子字段，无法独立设置）
- **Responses 流式 `response.failed` 事件 `error_detail` 变量初始化修复**：`openai_responses.py` 中 `'error_detail' in dir()` 不可靠检查替换为显式 `error_detail = None` 初始化，避免 `content_filter` 场景下错误详情丢失
- **Anthropic→Anthropic 流式直通**：引擎流式响应处理中，当后端和客户端协议均为 Anthropic 时，直接转发原始 SSE 事件数据，不再经 Chat 格式中转，避免事件格式错误
- **`_convert_message` 死代码清理**：移除 `and role != "user"` 冗余条件（`"user"` 已在 `ROLE_MAP` 中）
- **`pyproject.toml` 版本号同步**：包版本从 `1.0.0` 修正为 `1.7.0`，与 `__init__.py` 的 `__version__` 保持一致
- **208 个单元测试**（新增 5 个覆盖上述修复）
- **9 路集成测试全部通过**

### v1.7.0

基于官方 Python SDK（`anthropic-sdk-python`、`openai-python`）最新源码深度审查后修复关键逻辑缺陷：

- **`RedactedThinkingBlock.data` 字段名修正**：Anthropic SDK `RedactedThinkingBlock` 使用 `data` 字段（非 `signature`），现正确读取 `data` 并兼容旧版 `signature` fallback，脱敏思考块转换不再丢失数据
- **Responses `reasoning` 输出项补充 `summary` 必填字段**：OpenAI SDK `ResponseReasoningItem.summary` 为必填字段，所有 reasoning 输出项现包含 `"summary": []`
- **Anthropic `tool_choice.disable_parallel_tool_use` 双向映射**：
  - Anthropic→Chat：`disable_parallel_tool_use: true` → `parallel_tool_calls: false`
  - Chat→Anthropic：`parallel_tool_calls: false` → `disable_parallel_tool_use: true`
- **Responses→Anthropic 直接转换路径**：新增 `OpenAIResponsesConverter.to_anthropic()` 方法，不经 Chat 中转直接转换，保留 `encrypted_content` → `redacted_thinking.data`、`reasoning_text` → `thinking.thinking`、`summary_text` → 纳入 thinking 内容
- **`max_output_tokens=0` 边界情况修复**：Responses↔Chat 双向转换中 `max_output_tokens=0` / `max_completion_tokens=0`（缓存预热场景）不再被错误跳过
- **Chat `[Error]` 前缀 → Anthropic `is_error` 反向映射**：Chat→Anthropic 请求转换中 `[Error] ` 前缀标记正确反向映射为 `tool_result.is_error: true`，同时去掉前缀
- **Responses→Chat `summary_text` 内容纳入 `reasoning_content`**：reasoning 项中 `summary` 字段的 `summary_text` 内容正确纳入 Chat `reasoning_content`
- **192 个单元测试**（新增 17 个覆盖上述修复）
- **9 路集成测试全部通过**

### v1.6.0

基于官方 Python SDK 和 API 文档深度审查后修复逻辑缺陷、兼容性问题及未覆盖场景：

- **`top_logprobs`、`safety_identifier`、`prompt_cache_key`、`prompt_cache_retention`、`stream_options` 参数位置修复**：从 `extra_body` 移至 Chat 顶层参数（Responses→Chat 转换）
- **`max_completion_tokens=0` 处理修复**：Anthropic 缓存预热场景 `max_tokens=0` 不再被错误默认为 4096
- **多条 `system`/`developer` 消息合并**：Chat→Responses 转换中所有 system/developer 消息正确合并为 `instructions`，不再只保留最后一条
- **`redacted_thinking` 转换补全 `content` 字段**：Responses reasoning 输出项必需的 `content` 字段不再缺失
- **Responses 错误信息保留**：`status=failed` 的 `error` 字段在转换为 Chat 格式时正确保留
- **`search_result` 多模态内容修复**：Anthropic `search_result` 块在多模态用户消息中正确保留在 `content_parts` 中
- **空字符串 `content` 正确处理**：Chat 响应 `content=""` 与 `content=None` 正确区分，空字符串创建 text 块
- **Anthropic 流式 `error` 事件**：`content_filter` finish_reason 生成 Anthropic SSE `error` 事件
- **Responses 流式 `response.failed` 事件**：`content_filter` finish_reason 生成 `response.failed` 事件（而非 `response.completed`）
- **Anthropic `container` 字段保留**：顶层 `container` 字段在 Anthropic→Chat 响应转换中正确保留
- **175 个单元测试**（新增 32 个覆盖上述修复）
- **9 路集成测试全部通过**

基于官方 Python SDK（`anthropic-sdk-python`、`openai-python`）最新源码深度对比审查后完善转换逻辑：

### v1.5.0

- **Anthropic 流式 `message_stop` 事件补全**：流式转换严格遵循 Anthropic SSE 事件序列，`message_delta` 后追加必需的 `message_stop` 事件
- **Anthropic 流式 thinking→text 块顺序修复**：thinking 块在 text 块开始前正确关闭（`content_block_stop`），避免事件序列错误
- **Anthropic 流式块索引计算修复**：thinking 块和 text 块使用独立索引跟踪（`next_block_index`），避免 thinking 和 text 块索引冲突
- **Anthropic 流式 thinking→tool_use 块顺序修复**：thinking 块在 tool_use 块开始前正确关闭
- **Responses 流式 reasoning 项关闭修复**：reasoning 项在 message 项开始前正确关闭（发送 `response.reasoning_text.done` + `response.output_item.done`）
- **Responses 流式 `output_idx` 更新修复**：reasoning 项关闭后 `output_index` 正确递增，避免后续事件索引错误
- **Responses 流式 `response.completed` 补充 `model` 字段**：完成事件包含模型名称信息
- **Chat `message.annotations` → Anthropic `text.citations` 映射**：Chat 响应中的 annotations 在转换为 Anthropic 格式时正确映射为 text 块的 citations 字段
- **Chat `developer` 角色 → Responses `instructions` 映射修复**：developer 角色消息的多模态内容（content 数组）正确提取文本作为 instructions
- **Responses `input_audio` 内容类型支持**：新增 `input_audio` 输入类型检测和转换处理
- **Responses → Chat metadata 保留**：Responses 响应中的 `metadata` 字段在转换为 Chat 格式时正确保留
- **143 个单元测试**（新增 10 个覆盖上述改进）
- **9 路集成测试全部通过**

### v1.4.0

基于官方 Python SDK（`anthropic-sdk-python`、`openai-python`）最新源码深度对比审查后完善转换逻辑：

- **`thinking` adaptive 类型映射修复**：Anthropic SDK `ThinkingConfigAdaptiveParam` 无 `budget_tokens` 字段，现无 `budget_tokens` 时默认映射为 `"medium"` 而非错误的 `"low"`
- **Anthropic `citations` 内容支持**：text 块中的 `citations` 字段在跨协议转换中正确保留（Anthropic text.citations ↔ Chat annotations ↔ Responses output_text.annotations）
- **Anthropic `tool_result.is_error` 字段处理**：`is_error=True` 的工具结果在转换为 Chat 格式时标记 `[Error]` 前缀
- **`response_format` text 类型支持**：Chat API 新增 `ResponseFormatText`（`{"type": "text"}`）与 Responses API 的 `text.format` 双向转换
- **Responses 响应补充 `completed_at` 字段**：Chat→Responses 转换结果包含 `completed_at` 时间戳
- **`cache_creation_input_tokens` 往返保留**：Anthropic→Chat→Anthropic 往返转换中 `cache_creation_input_tokens` 和 `cache_read_input_tokens` 不再丢失
- **`stop` 参数字符串格式修复**：Chat `stop` 字符串正确转换为 Anthropic `stop_sequences` 列表
- **Chat→Anthropic 请求新增参数传递**：`verbosity`、`modalities`、`audio`、`prediction`、`safety_identifier`、`prompt_cache_key`、`prompt_cache_retention`、`store` 等新增参数正确放入 `extra_body`
- **Anthropic 响应 `container` 字段保留**：从 Chat usage 额外字段中恢复 `container`
- **133 个单元测试**（新增 15 个覆盖上述改进）

### v1.3.0

基于官方 Python SDK（`anthropic-sdk-python`、`openai-python`）最新源码深度对比审查后完善转换逻辑：

- **`service_tier` 双向映射修复**：
  - Anthropic 响应 `usage.service_tier`（`"standard"`/`"priority"`/`"batch"`）↔ Chat（`"default"`/`"priority"`/`"auto"`）之前直接透传导致值不匹配，现正确映射
  - Anthropic 请求 `service_tier`（`"auto"`/`"standard_only"`）↔ Chat（`"auto"`/`"default"`）双向映射
- **Anthropic 流式 `signature_delta` 事件**：thinking 流现在在 `thinking_delta` 后追加 `signature_delta` 事件，符合 SDK 多轮对话 thinking 连续性要求
- **Responses 流式使用官方 `reasoning_text` 事件类型**：
  - 推理增量事件从自定义 `response.reasoning.delta` 改为官方 `response.reasoning_text.delta`
  - 推理完成事件新增 `response.reasoning_text.done`
- **`thinking.display` 在 Chat→Anthropic 转换中保留**：当 Chat 请求带 `reasoning_effort` 和 `extra_body.thinking.display` 时，`display` 字段正确保留到目标 `thinking` 配置
- **`reasoning.summary` 在 Chat→Responses 转换中保留**：从 `extra_body.reasoning` 中提取 `summary` 字段保留到目标 `reasoning` 配置
- **Responses 响应补充 `parallel_tool_calls` 必需字段**：Chat→Responses 转换结果包含 `parallel_tool_calls: True`
- **Anthropic `server_tool_use` 和 `cache_creation` 在 Chat usage 中保留**：这两个 Anthropic 特有 usage 字段不再丢失
- **文件内容转换保留 `mime_type`**：Chat→Responses 文件转换保留原始 `mime_type` 字段
- **118 个单元测试**（新增 15 个覆盖上述改进）

### v1.2.0

基于官方 Python SDK（`anthropic-sdk-python`、`openai-python`）最新规范深度审查后完善转换逻辑：

- **ThinkingBlock `signature` 字段补全**：Chat `reasoning_content` → Anthropic `thinking` 块现在包含必需的 `signature` 字段（符合 SDK `ThinkingBlock` 模型定义）
- **流式 thinking 块 `signature` 字段**：流式 `content_block_start` 事件中 thinking 块包含 `signature` 空字符串
- **修复 `reasoning_tokens` 错误映射**：之前 `reasoning_tokens` 被错误映射到 `server_tool_use`，现已修正
- **`pause_turn` 停止原因映射**：Anthropic `pause_turn` 现在正确映射为 Responses `incomplete`（原因 `max_output_tokens`）和 Chat `stop`
- **Chat→Anthropic 参数映射完善**：
  - `user` 字段 → Anthropic `metadata.user_id`
  - `metadata` 合并（支持 `user_id` + 自定义字段共存）
  - 不支持参数正确放入 `extra_body`：`parallel_tool_calls`、`frequency_penalty`、`presence_penalty`、`seed`、`n`、`response_format`、`logprobs`、`top_logprobs`、`logit_bias`、`web_search_options`
- **Anthropic→Chat 用量字段完善**：
  - `cache_creation_input_tokens` 正确保留在 `prompt_tokens_details` 中
  - `service_tier` 和 `inference_geo` 从 Anthropic usage 保留到 Chat usage
- **Responses `instructions` 数组格式支持**：`instructions` 参数现在支持 `string` 和 `ResponseInputItem[]` 两种格式
- **Responses 工具类型补全**：新增 `tool_search`、`namespace` 工具类型的处理
- **Responses `web_search` 工具转换修正**：使用 Chat API 原生的 `web_search_preview` 工具类型 + 顶层 `web_search_options` 参数
- **Responses 输出项类型补全**：新增 `tool_search_call`、`compaction_item` 输出项类型处理
- **协议检测器补全**：`OPENAI_RESPONSES_INPUT_TYPES` 增加 `local_shell_call_output`、`mcp_approval_response`
- **流式 Anthropic 后端 SSE 解析改进**：正确解析 `event:` 行，缓存事件类型
- **103 个单元测试**（新增 16 个覆盖上述改进）

### v1.0.0

初始版本，协议转换器核心功能：

- **三种协议支持**：OpenAI Chat Completions、OpenAI Responses、Anthropic Messages
- **9 路转换矩阵**：3×3 协议×后端组合全覆盖
- **请求转换**：任意协议 → 任意协议
- **响应转换**：后端响应 → 客户端协议格式
- **流式支持**：完整的 SSE 流式转换
- **工具调用**：function calling / tool_use / tool_result 跨协议转换
- **多模态**：图片、文档、文件输入转换
- **协议检测**：自动识别请求协议类型

### v1.1.0

基于官方 Python SDK（`anthropic-sdk-python`、`openai-python`）最新规范完善转换逻辑：

- **reasoning_content ↔ thinking 全链路双向转换**：Chat `reasoning_content` ↔ Anthropic `thinking` 块 ↔ Responses `reasoning` 输出项，三种协议间推理内容完整互通
- **redacted_thinking 支持**：Anthropic 脱敏思考块 → Responses `reasoning`（含 `encrypted_content`）
- **Usage 字段完善**：缓存 token（`cached_tokens` / `cache_read_input_tokens`）和推理 token（`reasoning_tokens`）跨协议映射
- **Anthropic 服务器工具识别**：`bash_*`、`web_search_*`、`code_execution_*`、`text_editor_*`、`memory_*`、`tool_search_*` 等类型自动识别并保留
- **Responses 输出项扩展**：`mcp_call`、`custom_tool_call`、`apply_patch_tool_call`、`shell_tool_call` 等转为 Chat `tool_call`
- **流式推理转换**：`reasoning_content` → Responses `response.reasoning_text.delta` 事件
- **thinking.display 保留**：`"summarized"` / `"omitted"` 字段在 `extra_body` 中保留
- **Chat file 块 → Anthropic document 块**：多模态文件输入跨协议转换
- **87 个单元测试**（新增 15 个覆盖上述改进，v1.2.0 新增 16 个至 103 个，v1.3.0 新增 15 个至 118 个，v1.4.0 新增 15 个至 133 个，v1.5.0 新增 10 个至 143 个）

## License

MIT
