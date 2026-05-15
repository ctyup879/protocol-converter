# Protocol Converter

协议转换器 — 支持 **OpenAI Chat Completions**、**OpenAI Responses** 和 **Anthropic Messages** 三种 API 协议的相互转换。

可让使用不同 API 格式的客户端统一接入同一个后端，或让同一客户端灵活对接不同格式的后端服务。

## 功能特性

- **自动检测**请求协议类型（OpenAI Chat / OpenAI Responses / Anthropic）
- **请求转换**：任意协议 → 任意协议（6 种转换路径）
- **响应转换**：后端响应 → 客户端协议格式
- **流式支持**：完整的 SSE 流式转换，严格遵循各协议事件序列（含 `content_block_start/stop`、`output_item.added/done` 等）
- **工具调用**：完整支持 function calling / tool_use / tool_result 跨协议转换
- **扩展思考**：Anthropic `thinking` ↔ OpenAI `reasoning_effort` / `reasoning_content` 双向映射（含 `adaptive`、`display`）
- **Responses 推理输出**：`reasoning` 类型输出项（含 `reasoning_text`）→ Chat `reasoning_content` 字段
- **developer 角色降级**：目标后端不支持 `developer` 角色时自动降级为 `system`
- **多模态**：图片（base64 / URL）、文档（PDF）、文件输入转换
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
| 响应格式 | `response_format` | `text.format` | — |
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
| `stream_options` | 流式选项（含 `include_obfuscation`） | → `extra_body` |

### thinking ↔ reasoning_effort 映射

**Anthropic → OpenAI Chat：**

| Anthropic `thinking` | OpenAI Chat `reasoning_effort` |
|----------------------|-------------------------------|
| `{"type": "disabled"}` | `"none"` |
| `{"type": "enabled", "budget_tokens": <10000}` | `"low"` |
| `{"type": "enabled", "budget_tokens": 10000~31999}` | `"medium"` |
| `{"type": "enabled", "budget_tokens": ≥32000}` | `"high"` |
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

### 内容块类型映射

| Anthropic | → OpenAI Chat | → OpenAI Responses |
|-----------|--------------|-------------------|
| `text` | `{"type":"text","text":"..."}` | `{"type":"output_text","text":"..."}` |
| `image` (base64) | `{"type":"image_url","image_url":{"url":"data:..."}}` | `{"type":"input_image","image_url":"data:..."}` |
| `image` (url) | `{"type":"image_url","image_url":{"url":"https://..."}}` | `{"type":"input_image","image_url":"https://..."}` |
| `document` (base64) | `{"type":"file","file":{...}}` | `{"type":"input_file","file_data":"..."}` |
| `tool_use` | `tool_calls[{"id","function":{"name","arguments"}}]` | `{"type":"function_call","call_id",...}` |
| `tool_result` | `{"role":"tool","tool_call_id","content"}` | `{"type":"function_call_output","call_id","output"}` |
| `thinking` | `reasoning_content` | `{"type":"reasoning","content":[{"type":"reasoning_text"}]}` |
| `redacted_thinking` | （跳过） | `{"type":"reasoning","encrypted_content":"..."}` |
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

### OpenAI Responses 流式事件序列

```
response.created → response.in_progress → 
  [response.output_item.added → 
    [response.content_part.added → response.output_text.delta* → response.output_text.done → response.content_part.done]?
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
3. 存在 Anthropic 特有参数：`system`、`stop_sequences`、`thinking`、`cache_control`、`top_k`、`container`、`output_config`、`inference_geo`
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
│   └── test_protocol_converter.py   # 103 个单元测试
├── examples/
│   ├── integration_test_chat_backend.py       # OpenAI Chat 后端集成测试
│   ├── integration_test_anthropic_backend.py  # Anthropic 后端集成测试
│   ├── integration_test_responses_backend.py  # OpenAI Responses 后端集成测试
│   └── integration_test_all_9_paths.py       # 3×3 全量 9 路集成测试
└── pyproject.toml
```

## 运行测试

```bash
# 单元测试（103 个）
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
- **流式推理转换**：`reasoning_content` → Responses `response.reasoning.delta` 事件
- **thinking.display 保留**：`"summarized"` / `"omitted"` 字段在 `extra_body` 中保留
- **Chat file 块 → Anthropic document 块**：多模态文件输入跨协议转换
- **87 个单元测试**（新增 15 个覆盖上述改进，v1.2.0 新增 16 个至 103 个）

## License

MIT
