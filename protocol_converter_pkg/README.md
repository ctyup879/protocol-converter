# Protocol Converter

协议转换器 — 支持 **OpenAI Chat Completions**、**OpenAI Responses** 和 **Anthropic Messages** 三种 API 协议的相互转换。

可让使用不同 API 格式的客户端统一接入同一个后端，或让同一客户端灵活对接不同格式的后端服务。

## 功能特性

- **自动检测**请求协议类型（OpenAI Chat / OpenAI Responses / Anthropic）
- **请求转换**：任意协议 → 任意协议（6 种转换路径）
- **响应转换**：后端响应 → 客户端协议格式
- **流式支持**：完整的 SSE 流式转换，严格遵循各协议事件序列（含 `content_block_start/stop`、`output_item.added/done`、`reasoning_summary_text.delta/done`、`refusal.delta/done` 等）
- **工具调用**：完整支持 function calling / tool_use / tool_result 跨协议转换
- **扩展思考**：Anthropic `thinking` ↔ OpenAI `reasoning_effort` / `reasoning_content` 双向映射（含 `adaptive`、`display`）
- **Responses 推理输出**：`reasoning` 类型输出项（含 `reasoning_text`）→ Chat `reasoning_content` 字段
- **developer 角色降级**：目标后端不支持 `developer` 角色时自动降级为 `system`
- **多模态**：图片（base64 / URL）、文档（PDF）、文件输入转换
- **Structured Outputs 往返**：Responses `text.format` ↔ Chat `response_format` ↔ Anthropic `output_format` 的 `json_schema` 格式正确互转，支持 `name`/`schema`/`strict` 字段完整映射
- **annotations 保留**：Chat `message.annotations` ↔ Responses `output_text.annotations` ↔ Anthropic `text.citations` 跨协议保留
- **`is_error` 字段处理**：Anthropic `tool_result.is_error` 在转换时添加 `[Error]` 前缀标记
- **模型映射**：自定义模型名称映射表
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
| `model_mapping` | dict | `{}` | 模型名称映射表 |
| `anthropic_version` | str | `"2023-06-01"` | Anthropic API 版本 |
| `inference_geo` | str | `None` | Anthropic 推理地理区域 |
| `prompt_cache_key` | str | `None` | OpenAI 提示缓存键（替代 `user`） |
| `prompt_cache_retention` | str | `None` | 缓存保留策略（`"in_memory"` / `"24h"`） |

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
| `verbosity` | 输出详细程度（low/medium/high） | → `extra_body.verbosity` | → `extra_body.verbosity` |
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
| `redacted_thinking` | （跳过） | `{"type":"reasoning","content":[],"encrypted_content":"..."}` |
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
│   └── test_protocol_converter.py   # 266 个单元测试
├── examples/
│   ├── integration_test_chat_backend.py       # OpenAI Chat 后端集成测试
│   ├── integration_test_anthropic_backend.py  # Anthropic 后端集成测试
│   ├── integration_test_responses_backend.py  # OpenAI Responses 后端集成测试
│   └── integration_test_all_9_paths.py       # 3×3 全量 9 路集成测试
└── pyproject.toml
```

## 运行测试

```bash
# 单元测试（266 个）
python -m pytest tests/ -v

# 集成测试（需要配置有效 API Key）
python -m examples.integration_test_chat_backend        # Chat 后端 (8 项)
python -m examples.integration_test_anthropic_backend    # Anthropic 后端 (7 项)
python -m examples.integration_test_responses_backend    # Responses 后端 (7 项)
python -m examples.integration_test_all_9_paths          # 3×3 全量 9 路测试
```

## 参考

- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
- [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses)
- [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)
- [openai-python SDK](https://github.com/openai/openai-python)
- [anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python)

## 更新日志

### v1.12.0

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
