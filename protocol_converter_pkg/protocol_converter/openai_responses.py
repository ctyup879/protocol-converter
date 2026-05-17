"""
OpenAI Responses API 协议转换器

参考: https://platform.openai.com/docs/api-reference/responses
参考 SDK: openai-python ResponseCreateParams

OpenAI Responses API 请求参数:
- model (必填): 模型名称
- input (必填): string | ResponseInputParam[] (文本/图片/文件输入)
- instructions: 系统提示词 (也可以是 ResponseInputItem[])
- tools: 工具定义 (function | web_search | file_search | computer | mcp | code_interpreter | image_generation | local_shell | custom | apply_patch | namespace | tool_search)
- tool_choice: 工具选择策略 (auto | none | required | function | mcp | custom | apply_patch | shell | allowed | types)
- temperature: 温度 (0-2)
- top_p: nucleus 采样
- max_output_tokens: 最大输出 token 数
- stream: 是否流式
- stream_options: 流式选项 {"include_obfuscation": bool}
- previous_response_id: 上一个响应 ID (用于多轮对话, 不能与 conversation 同时使用)
- store: 是否存储响应
- metadata: 元数据
- reasoning: 推理配置 (gpt-5和o系列模型, 含 effort + summary)
  - effort: "none" | "minimal" | "low" | "medium" | "high" | "xhigh"
  - summary: "auto" | "concise" | "detailed" | None
- text: 文本输出配置 (structured outputs, 含 format)
- parallel_tool_calls: 是否允许并行工具调用
- truncation: 截断策略 ("auto" | "disabled")
- service_tier: 服务层级 ("auto" | "default" | "flex" | "scale" | "priority")
- background: 是否后台运行
- max_tool_calls: 最大工具调用次数
- conversation: 对话参数 (conversation ID 或 ConversationParam, 不能与 previous_response_id 同时使用)
- context_management: 上下文管理配置
- include: 额外输出数据 (web_search_call.action.sources, code_interpreter_call.outputs, etc.)
- prompt: 提示模板
- safety_identifier: 安全标识符
- prompt_cache_key: 缓存键
- prompt_cache_retention: 缓存保留策略 ("in_memory" | "24h")
- user: 用户标识符 (被 safety_identifier/prompt_cache_key 替代)
- top_logprobs: 返回 top logprobs 数量

OpenAI Responses 输入项类型:
- message: 消息 {"type": "message", "role", "content": [...]}
- function_call: 函数调用 {"type": "function_call", "call_id", "name", "arguments"}
- function_call_output: 函数调用结果 {"type": "function_call_output", "call_id", "output"}
- computer_call_output: 计算机调用输出
- reasoning: 推理项

OpenAI Responses 输出项类型:
- message: 消息 {"type": "message", "role", "content": [...]}
- function_call: 函数调用 {"type": "function_call", "call_id", "name", "arguments", "status"}
- reasoning: 推理项 {"type": "reasoning", "id", "content": [...], "encrypted_content", "status"}
- web_search_call: 网页搜索调用
- file_search_call: 文件搜索调用
- computer_call: 计算机调用
- code_interpreter_call: 代码解释器调用
- image_generation_call: 图片生成调用
- local_shell_call: 本地终端调用
- mcp_call: MCP 调用
- custom_tool_call: 自定义工具调用
- apply_patch_tool_call: 补丁工具调用
- tool_search_call: 工具搜索调用
- compaction_item: 压缩项

OpenAI Responses 流式事件 (严格顺序):
  response.created -> response.in_progress -> [response.output_item.added -> [response.content_part.added -> response.output_text.delta* -> response.output_text.done -> response.content_part.done]? -> [response.reasoning_summary_part.added -> response.reasoning_summary_text.delta* -> response.reasoning_summary_text.done -> response.reasoning_summary_part.done]? -> [response.function_call_arguments.delta* -> response.function_call_arguments.done]? -> response.output_item.done]* -> response.completed | response.failed | response.incomplete
- response.created: 响应创建
- response.in_progress: 响应进行中
- response.output_item.added: 输出项添加
- response.content_part.added: 内容部分添加
- response.output_text.delta: 文本增量
- response.output_text.done: 文本完成
- response.content_part.done: 内容部分完成
- response.refusal.delta: 拒绝内容增量
- response.refusal.done: 拒绝内容完成
- response.reasoning_summary_part.added: 推理摘要部分添加
- response.reasoning_summary_part.done: 推理摘要部分完成
- response.reasoning_summary_text.delta: 推理摘要文本增量
- response.reasoning_summary_text.done: 推理摘要文本完成
- response.function_call_arguments.delta: 函数调用参数增量
- response.function_call_arguments.done: 函数调用参数完成
- response.output_item.done: 输出项完成
- response.completed: 响应完成
- response.failed: 响应失败
- response.incomplete: 响应不完整
- response.reasoning_text.delta: 推理文本增量
- response.reasoning_text.done: 推理文本完成
- response.web_search_call.in_progress/searching/completed: 网页搜索
- response.file_search_call.in_progress/searching/completed: 文件搜索
- response.code_interpreter_call.in_progress/completed: 代码解释器
- response.code_interpreter_call_code.delta/done: 代码解释器代码
- response.computer_call.in_progress/completed: 计算机调用
- response.ping: 心跳

OpenAI Responses Usage 字段:
- input_tokens: 输入 token 数
- output_tokens: 输出 token 数
- total_tokens: 总 token 数
- input_tokens_details: {cached_tokens: int}
- output_tokens_details: {reasoning_tokens: int}
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union, Iterator


class OpenAIResponsesConverter:
    """OpenAI Responses API 协议转换器"""
    
    PROTOCOL_NAME = "openai_responses"
    
    # 角色映射 (Responses -> Chat)
    ROLE_MAP = {
        "user": "user",
        "system": "system",
        "developer": "developer",
        "assistant": "assistant",
    }
    
    # OpenAI Chat finish_reason -> Responses 不完整原因
    INCOMPLETE_REASON_MAP = {
        "length": "max_output_tokens",
        "content_filter": "content_filter",
    }
    
    # Chat service_tier -> Responses service_tier
    SERVICE_TIER_MAP = {
        "auto": "auto",
        "default": "default",
        "flex": "flex",
        "scale": "scale",
        "priority": "priority",
    }
    
    # ================================================================
    # 请求转换：Responses -> Chat
    # ================================================================
    
    @classmethod
    def to_openai_chat(cls, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI Responses 请求转换为 OpenAI Chat 格式
        
        Args:
            request: OpenAI Responses 格式请求
            
        Returns:
            Dict: OpenAI Chat 格式请求
            
        Raises:
            TypeError: 如果 request 不是字典
        """
        if not isinstance(request, dict):
            raise TypeError(f"request must be a dict, got {type(request).__name__}")
        
        messages = []
        
        # 1. 处理 input（可以是字符串或数组）
        input_data = request.get("input", [])
        if isinstance(input_data, str):
            # 简单字符串输入 -> 单条 user 消息
            if input_data:
                messages.append({"role": "user", "content": input_data})
            else:
                messages.append({"role": "user", "content": ""})
        elif isinstance(input_data, list):
            for item in input_data:
                msg = cls._convert_input_item_to_message(item)
                if msg:
                    # 可能返回多条消息（例如 tool 消息）
                    if isinstance(msg, list):
                        messages.extend(msg)
                    else:
                        messages.append(msg)
        
        # 确保 messages 不为空（Chat API 要求至少一条消息）
        if not messages:
            messages.append({"role": "user", "content": ""})
        
        # 2. 处理 instructions（相当于 system/developer prompt）
        # instructions 可以是字符串或 ResponseInputItem 数组
        instructions = request.get("instructions")
        if instructions:
            if isinstance(instructions, str):
                messages.insert(0, {
                    "role": "developer",
                    "content": instructions
                })
            elif isinstance(instructions, list):
                # instructions 为数组时，逐项转换
                # 收集多模态内容块，最后合并为单条 developer 消息
                inst_content_parts = []
                inst_text_parts = []
                for inst_item in reversed(instructions):
                    if isinstance(inst_item, dict):
                        inst_type = inst_item.get("type", "")
                        if inst_type == "message":
                            role = inst_item.get("role", "developer")
                            content = inst_item.get("content", "")
                            if isinstance(content, list):
                                converted = cls._convert_content_to_chat(content)
                                if isinstance(converted, list):
                                    inst_content_parts.extend(converted)
                                else:
                                    inst_text_parts.append(str(converted))
                            else:
                                inst_text_parts.append(str(content))
                        elif inst_type == "input_text" or "text" in inst_item:
                            text_val = inst_item.get("text", "")
                            if text_val:
                                inst_text_parts.append(text_val)
                        elif inst_type == "input_image":
                            image_url = inst_item.get("image_url", "")
                            if image_url:
                                inst_content_parts.append({
                                    "type": "image_url",
                                    "image_url": {"url": image_url}
                                })
                        elif inst_type == "input_file":
                            file_data = inst_item.get("file_data", "")
                            file_id = inst_item.get("file_id")
                            filename = inst_item.get("filename", "")
                            if file_data:
                                file_obj = {"file_data": file_data}
                                if filename:
                                    file_obj["filename"] = filename
                                inst_content_parts.append({"type": "file", "file": file_obj})
                            elif file_id:
                                file_obj = {"file_id": file_id}
                                if filename:
                                    file_obj["filename"] = filename
                                inst_content_parts.append({"type": "file", "file": file_obj})
                    elif isinstance(inst_item, str):
                        inst_text_parts.append(inst_item)
                # 合并 instructions 为单条或多条 developer 消息
                if inst_content_parts or inst_text_parts:
                    if inst_text_parts:
                        text_block = {"type": "text", "text": "\n".join(inst_text_parts)}
                        if inst_content_parts:
                            messages.insert(0, {
                                "role": "developer",
                                "content": [text_block] + inst_content_parts
                            })
                        else:
                            messages.insert(0, {
                                "role": "developer",
                                "content": "\n".join(inst_text_parts)
                            })
                    elif inst_content_parts:
                        messages.insert(0, {
                            "role": "developer",
                            "content": inst_content_parts
                        })
        
        # 3. 转换 tools
        tools = cls._convert_tools(request.get("tools", []))
        
        # 4. 构建 Chat 请求
        chat_request = {
            "model": request.get("model") or "gpt-4o",
            "messages": messages,
        }
        
        # 4.5 从 Responses 的 web_search / web_search_preview 工具中提取配置
        # Chat API 的 web_search_options 是顶层请求参数，不是工具定义的一部分
        for tool in request.get("tools", []):
            if isinstance(tool, dict) and tool.get("type") in ("web_search", "web_search_preview"):
                ws_opts = tool.get("user_location", {})
                search_context_size = tool.get("search_context_size", "medium")
                chat_request["web_search_options"] = {
                    "search_context_size": search_context_size,
                    **({"user_location": ws_opts} if ws_opts else {})
                }
                break
        
        # 5. 映射可选参数
        if request.get("stream") is not None:
            chat_request["stream"] = request["stream"]
        if request.get("temperature") is not None:
            chat_request["temperature"] = request["temperature"]
        if request.get("top_p") is not None:
            chat_request["top_p"] = request["top_p"]
        if request.get("max_output_tokens") is not None:
            chat_request["max_completion_tokens"] = request["max_output_tokens"]
        if tools:
            chat_request["tools"] = tools
        if request.get("tool_choice"):
            chat_request["tool_choice"] = cls._convert_tool_choice(request["tool_choice"])
        if request.get("parallel_tool_calls") is not None:
            chat_request["parallel_tool_calls"] = request["parallel_tool_calls"]
        
        # 6. 映射元数据
        metadata = request.get("metadata")
        if metadata and isinstance(metadata, dict):
            user_id = metadata.get("user_id")
            if user_id:
                chat_request["user"] = user_id
            # 保留完整 metadata
            chat_request["metadata"] = metadata
        
        # 6.5 处理 user 参数 (被 safety_identifier/prompt_cache_key 替代)
        if request.get("user"):
            chat_request["user"] = request["user"]
        
        # 7. 处理 store 参数
        if request.get("store") is not None:
            chat_request["store"] = request["store"]
        
        # 8. 处理 reasoning 参数 -> reasoning_effort
        reasoning = request.get("reasoning")
        if reasoning and isinstance(reasoning, dict):
            effort = reasoning.get("effort")
            if effort:
                chat_request["reasoning_effort"] = effort
            # 保留 reasoning 的 summary 和 generate_summary 配置
            # SDK v2.11: reasoning 参数支持 effort + summary + generate_summary
            if reasoning.get("summary") is not None or reasoning.get("generate_summary") is not None:
                if "extra_body" not in chat_request:
                    chat_request["extra_body"] = {}
                chat_request["extra_body"]["reasoning"] = reasoning
        
        # 9. 处理 text 参数 (structured outputs + verbosity) -> response_format
        text_config = request.get("text")
        if text_config and isinstance(text_config, dict):
            # 9.1 text.verbosity -> Chat 顶层 verbosity
            # SDK: ResponseTextConfigParam.verbosity -> ChatCompletionRequest.verbosity
            verbosity = text_config.get("verbosity")
            if verbosity:
                chat_request["verbosity"] = verbosity
            # 9.2 text.format -> response_format
            format_config = text_config.get("format")
            if format_config and isinstance(format_config, dict):
                fmt_type = format_config.get("type")
                if fmt_type == "json_schema":
                    # Responses API: {"type": "json_schema", "name": "...", "schema": {...}, "strict": bool}
                    # Chat API: {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}, "strict": bool}}
                    # 兼容两种格式：标准 Responses 格式（name/schema 在顶层）和旧版嵌套格式
                    json_schema = {}
                    if format_config.get("name") or format_config.get("schema"):
                        # 标准 Responses 格式：name/schema 在顶层
                        if format_config.get("name"):
                            json_schema["name"] = format_config["name"]
                        if format_config.get("schema"):
                            json_schema["schema"] = format_config["schema"]
                        if format_config.get("strict") is not None:
                            json_schema["strict"] = format_config["strict"]
                    elif format_config.get("json_schema"):
                        # 旧版嵌套格式：name/schema 在 json_schema 子对象中
                        nested = format_config["json_schema"]
                        if isinstance(nested, dict):
                            if nested.get("name"):
                                json_schema["name"] = nested["name"]
                            if nested.get("schema"):
                                json_schema["schema"] = nested["schema"]
                            if nested.get("strict") is not None:
                                json_schema["strict"] = nested["strict"]
                    chat_request["response_format"] = {
                        "type": "json_schema",
                        "json_schema": json_schema
                    }
                elif fmt_type == "json_object":
                    chat_request["response_format"] = {"type": "json_object"}
                elif fmt_type == "text":
                    # Responses API text format type -> Chat ResponseFormatText
                    chat_request["response_format"] = {"type": "text"}
        
        # 10. 处理 service_tier
        if request.get("service_tier"):
            chat_request["service_tier"] = request["service_tier"]
        
        # 11. 处理 truncation - Chat API 没有直接等价参数
        # truncation="auto" 对应 Chat API 无直接映射，保留在 extra_body
        truncation = request.get("truncation")
        if truncation:
            if "extra_body" not in chat_request:
                chat_request["extra_body"] = {}
            chat_request["extra_body"]["truncation"] = truncation
        
        # 12. Responses 特有参数（Chat API 不支持的）
        extra = chat_request.get("extra_body", {})
        if request.get("previous_response_id"):
            extra["previous_response_id"] = request["previous_response_id"]
        if request.get("background") is not None:
            extra["background"] = request["background"]
        if request.get("max_tool_calls") is not None:
            extra["max_tool_calls"] = request["max_tool_calls"]
        if request.get("conversation"):
            extra["conversation"] = request["conversation"]
        if request.get("context_management"):
            extra["context_management"] = request["context_management"]
        if request.get("include"):
            extra["include"] = request["include"]
        if request.get("prompt"):
            extra["prompt"] = request["prompt"]
        # 以下参数在 Chat API 中为原生顶层参数，不应放入 extra_body
        if request.get("safety_identifier"):
            chat_request["safety_identifier"] = request["safety_identifier"]
        if request.get("prompt_cache_key"):
            chat_request["prompt_cache_key"] = request["prompt_cache_key"]
        if request.get("prompt_cache_retention"):
            chat_request["prompt_cache_retention"] = request["prompt_cache_retention"]
        # stream_options: Responses include_obfuscation 不等价于 Chat include_usage
        # Responses stream_options.include_obfuscation → Chat 不支持，放入 extra_body
        if request.get("stream_options"):
            stream_opts = request["stream_options"]
            if isinstance(stream_opts, dict) and stream_opts.get("include_obfuscation") is not None:
                if "extra_body" not in chat_request:
                    chat_request["extra_body"] = {}
                chat_request["extra_body"]["stream_options"] = stream_opts
            else:
                # 其他 stream_options 字段（如 include_usage）直接传递
                chat_request["stream_options"] = stream_opts
        if request.get("top_logprobs") is not None:
            chat_request["top_logprobs"] = request["top_logprobs"]
            # Chat API 要求 logprobs=True 时才能使用 top_logprobs
            chat_request["logprobs"] = True
        if extra:
            chat_request["extra_body"] = extra
        
        return chat_request
    
    @classmethod
    def _convert_input_item_to_message(cls, item: Dict[str, Any]) -> Optional[Union[Dict, List[Dict]]]:
        """转换 Responses 输入项到 Chat 消息格式
        
        Responses 输入项类型:
        - message: {"type": "message", "role", "content": [...]}
        - function_call: {"type": "function_call", "call_id", "name", "arguments"}
        - function_call_output: {"type": "function_call_output", "call_id", "output"}
        - computer_call_output: 计算机调用输出
        - reasoning: 推理项
        """
        if not isinstance(item, dict):
            return None
        
        item_type = item.get("type", "")
        
        if item_type == "message":
            role = cls.ROLE_MAP.get(item.get("role", ""), item.get("role", "user"))
            content = item.get("content", [])
            converted_content = cls._convert_content_to_chat(content)
            if isinstance(converted_content, str):
                return {"role": role, "content": converted_content}
            else:
                return {"role": role, "content": converted_content}
        
        elif item_type == "function_call_output":
            # 工具调用结果 -> role=tool
            output_val = item.get("output", "")
            return {
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": cls._convert_tool_output_to_chat_content(output_val)
            }
        
        elif item_type == "function_call":
            # assistant 的工具调用 -> role=assistant + tool_calls
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": item.get("call_id", item.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}")
                    }
                }]
            }
        
        elif item_type == "computer_call_output":
            return {
                "role": "user",
                "content": "[Computer screenshot output]"
            }
        
        elif item_type == "reasoning":
            # 推理项 - Chat API 不直接支持，跳过
            return None
        
        elif item_type == "local_shell_call_output":
            # 本地终端调用输出
            return {
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": str(item.get("output", ""))
            }
        
        elif item_type == "mcp_approval_response":
            # MCP 审批响应 - 跳过
            return None
        
        elif item_type == "input_audio":
            # 音频输入 - Chat API 不直接支持，转为文本占位
            return {
                "role": "user",
                "content": "[Audio input]"
            }
        
        elif item_type == "input_image":
            image_url = item.get("image_url", "")
            if image_url:
                return {
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }]
                }
            return None
        
        elif item_type == "input_file":
            file_data = item.get("file_data", "")
            file_id = item.get("file_id")
            filename = item.get("filename", "")
            file_obj = {}
            if file_data:
                file_obj["file_data"] = file_data
            elif file_id:
                file_obj["file_id"] = file_id
            if filename:
                file_obj["filename"] = filename
            if file_obj:
                return {
                    "role": "user",
                    "content": [{"type": "file", "file": file_obj}]
                }
            return None
        
        # 未知类型，尝试作为消息处理
        if "role" in item and "content" in item:
            role = cls.ROLE_MAP.get(item.get("role", ""), item.get("role", "user"))
            content = item.get("content", "")
            if isinstance(content, list):
                converted = cls._convert_content_to_chat(content)
                return {"role": role, "content": converted}
            return {"role": role, "content": str(content)}
        
        return None

    @classmethod
    def _convert_tool_output_to_chat_content(cls, output: Any) -> str:
        """转换 Responses 工具输出为 Chat tool 消息支持的文本内容"""
        if isinstance(output, str):
            return output
        if not isinstance(output, list):
            return str(output)

        text_parts = []
        for item in output:
            if not isinstance(item, dict):
                text_parts.append(str(item))
                continue

            item_type = item.get("type", "")
            if item_type in ("input_text", "output_text", "text"):
                text_parts.append(item.get("text", ""))
            elif item_type in ("input_image", "image"):
                image_ref = item.get("image_url") or item.get("file_id") or ""
                text_parts.append(f"[Image: {image_ref}]")
            elif item_type in ("input_file", "file"):
                file_ref = (
                    item.get("filename")
                    or item.get("file_id")
                    or item.get("file_url")
                    or "[inline file]"
                )
                text_parts.append(f"[File: {file_ref}]")
            else:
                text = item.get("text") or item.get("content")
                text_parts.append(str(text) if text is not None else str(item))

        return "\n".join(part for part in text_parts if part is not None)
    
    @classmethod
    def _convert_content_to_chat(cls, content: Union[str, List[Dict]]) -> Union[str, List[Dict]]:
        """转换 Responses 内容为 Chat 格式（可能是字符串或内容数组）"""
        if isinstance(content, str):
            return content
        
        if not isinstance(content, list):
            return str(content)
        
        # 检查是否需要多模态内容格式
        has_multimodal = False
        text_parts = []
        content_parts = []
        
        for item in content:
            if not isinstance(item, dict):
                text_parts.append(str(item))
                continue
                
            item_type = item.get("type", "")
            
            if item_type in ("text", "input_text"):
                text_parts.append(item.get("text", ""))
                content_parts.append({"type": "text", "text": item.get("text", "")})
            
            elif item_type == "input_image":
                has_multimodal = True
                image_url = item.get("image_url", "")
                file_id = item.get("file_id")
                detail = item.get("detail")  # Responses API 支持 detail 参数
                img_obj = {}
                if image_url:
                    img_obj["url"] = image_url
                elif file_id:
                    # Responses API input_image 支持 file_id，Chat API 无直接等价
                    # 降级为文本占位
                    text_parts.append(f"[Image file: {file_id}]")
                    content_parts.append({"type": "text", "text": f"[Image file: {file_id}]"})
                    has_multimodal = True
                if detail and img_obj:
                    img_obj["detail"] = detail
                if img_obj:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": img_obj
                    })
            
            elif item_type == "input_file":
                has_multimodal = True
                file_data = item.get("file_data", "")
                file_id = item.get("file_id")
                file_url = item.get("file_url", "")
                filename = item.get("filename", "")
                if file_id:
                    file_obj = {"file_id": file_id}
                    if filename:
                        file_obj["filename"] = filename
                    content_parts.append({
                        "type": "file",
                        "file": file_obj
                    })
                elif file_data:
                    file_obj = {"file_data": file_data}
                    if filename:
                        file_obj["filename"] = filename
                    content_parts.append({
                        "type": "file",
                        "file": file_obj
                    })
                elif file_url:
                    file_text = f"[File URL: {file_url}]"
                    text_parts.append(file_text)
                    content_parts.append({"type": "text", "text": file_text})
                elif filename:
                    text_parts.append(f"[File: {filename}]")
            
            elif item_type == "input_audio":
                # 音频输入 - Chat API 不直接支持，转为文本占位
                has_multimodal = True
                text_parts.append("[Audio input]")
                content_parts.append({"type": "text", "text": "[Audio input]"})
            
            elif item_type == "reasoning_text":
                # 推理文本 - Chat API 不直接支持，跳过
                pass
            
            elif item_type == "output_text":
                text_parts.append(item.get("text", ""))
                content_parts.append({"type": "text", "text": item.get("text", "")})
            
            else:
                # 未知内容类型，保留文本
                text = item.get("text", item.get("content", ""))
                if text:
                    text_parts.append(str(text))
                    content_parts.append({"type": "text", "text": str(text)})
        
        if has_multimodal:
            if content_parts:
                return content_parts
            elif text_parts:
                return [{"type": "text", "text": "\n".join(text_parts)}]
            else:
                return []
        elif text_parts:
            return "\n".join(text_parts)
        return ""
    
    @classmethod
    def _convert_tools(cls, tools: List[Dict]) -> List[Dict]:
        """
        转换 Responses 工具定义到 Chat 格式
        
        Responses 工具类型:
        - function: 函数工具
        - web_search: 网页搜索工具
        - file_search: 文件搜索工具
        - computer / computer_use_preview: 计算机使用工具
        - mcp: MCP 工具
        - code_interpreter: 代码解释器
        - image_generation: 图片生成
        - local_shell: 本地终端
        - custom: 自定义工具
        - apply_patch: 补丁工具
        - namespace: 命名空间工具
        - tool_search: 工具搜索
        
        Chat 工具类型: function, web_search_preview
        """
        converted = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_type = tool.get("type", "function")
            
            if tool_type == "function":
                func_def = {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                }
                if "parameters" in tool:
                    func_def["parameters"] = tool["parameters"]
                if "strict" in tool:
                    func_def["strict"] = tool["strict"]
                converted.append({
                    "type": "function",
                    "function": func_def
                })
            
            elif tool_type in ("web_search", "web_search_preview"):
                # Chat API 支持 web_search_preview 工具类型
                # 搜索上下文配置通过顶层 web_search_options 参数传递（在 to_openai_chat 中处理）
                converted.append({
                    "type": "web_search_preview"
                })
            
            elif tool_type == "file_search":
                # 文件搜索 - Chat API 不直接支持，保留在 extra_body
                pass
            
            elif tool_type in ("computer", "computer_use_preview"):
                # 计算机使用 - Chat API 不直接支持
                pass
            
            elif tool_type == "mcp":
                # MCP 工具 - Chat API 不直接支持
                pass
            
            elif tool_type == "code_interpreter":
                # 代码解释器 - Chat API 不直接支持
                pass
            
            elif tool_type == "image_generation":
                # 图片生成 - Chat API 不直接支持
                pass
            
            elif tool_type == "local_shell":
                # 本地终端 - Chat API 不直接支持
                pass
            
            elif tool_type == "custom":
                # 自定义工具 - 尝试转为 function
                func_def = {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                }
                if "parameters" in tool:
                    func_def["parameters"] = tool["parameters"]
                converted.append({
                    "type": "function",
                    "function": func_def
                })
            
            elif tool_type == "apply_patch":
                # 补丁工具 - Chat API 不直接支持
                pass
            
            elif tool_type == "namespace":
                # 命名空间工具 - Chat API 不直接支持
                pass
            
            elif tool_type == "tool_search":
                # 工具搜索 - Chat API 不直接支持
                pass
        
        return converted
    
    @classmethod
    def _convert_tool_choice(cls, tool_choice: Any) -> Any:
        """
        转换 Responses tool_choice 到 Chat 格式
        
        Responses: "auto" | "none" | "required" 
                    | {"type": "function", "name": "..."} 
                    | {"type": "mcp", ...}
                    | {"type": "custom", ...}
                    | {"type": "apply_patch", ...}
                    | {"type": "shell", ...}
        Chat: "auto" | "none" | "required" | {"type": "function", "function": {"name": "..."}}
        """
        if isinstance(tool_choice, str):
            return tool_choice  # auto, none, required 直接传递
        
        if isinstance(tool_choice, dict):
            tc_type = tool_choice.get("type", "")
            
            if tc_type == "function":
                return {
                    "type": "function",
                    "function": {"name": tool_choice.get("name", "")}
                }

            elif tc_type == "allowed_tools":
                return {
                    "type": "allowed_tools",
                    "allowed_tools": {
                        "mode": tool_choice.get("mode", "auto"),
                        "tools": [
                            cls._convert_allowed_tool_to_chat(tool)
                            for tool in tool_choice.get("tools", [])
                            if isinstance(tool, dict)
                        ],
                    },
                }
            
            elif tc_type in ("mcp", "custom", "apply_patch", "shell", "allowed", "types"):
                # 非 Chat 原生支持的 tool_choice 类型
                # 尝试提取最接近的语义
                name = tool_choice.get("name", "")
                if name:
                    return {
                        "type": "function",
                        "function": {"name": name}
                    }
                return "auto"
        
        return tool_choice

    @classmethod
    def _convert_allowed_tool_to_chat(cls, tool: Dict[str, Any]) -> Dict[str, Any]:
        """转换 Responses allowed_tools 中的工具引用为 Chat allowed_tools 形状"""
        if tool.get("type") == "function" and "function" not in tool:
            return {
                "type": "function",
                "function": {"name": tool.get("name", "")}
            }
        return tool
    
    # ================================================================
    # 请求转换：Chat -> Responses
    # ================================================================
    
    @classmethod
    def from_openai_chat_request(cls, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI Chat 请求转换为 OpenAI Responses 格式
        
        Args:
            request: OpenAI Chat 格式请求
            
        Returns:
            Dict: OpenAI Responses 格式请求
            
        Raises:
            TypeError: 如果 request 不是字典
        """
        if not isinstance(request, dict):
            raise TypeError(f"request must be a dict, got {type(request).__name__}")
        
        messages = request.get("messages", [])
        
        # 提取简单文本输入（用于 OpenRouter 等兼容 API）
        simple_texts = []
        has_complex = False
        instructions_parts = []  # 收集所有 system/developer 消息内容
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system" or role == "developer":
                # developer 角色映射为 instructions（Responses API 语义）
                # 多条 system/developer 消息应合并为 instructions
                if isinstance(content, str) and content:
                    instructions_parts.append(content)
                elif isinstance(content, list):
                    # 多模态 system/developer 内容，提取文本
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    if text_parts:
                        instructions_parts.append("\n".join(text_parts))
            elif role == "user" and isinstance(content, str):
                simple_texts.append(content)
            elif role == "assistant" and msg.get("tool_calls"):
                has_complex = True
            elif role == "tool":
                has_complex = True
            elif role not in ("user",) or not isinstance(content, str):
                has_complex = True
        
        # 合并所有 system/developer 消息为 instructions
        instructions = "\n\n".join(instructions_parts) if instructions_parts else None
        
        # 简单文本 -> 字符串 input；复杂 -> 数组 input
        if not has_complex and simple_texts:
            input_data = "\n".join(simple_texts)
        elif has_complex:
            input_items = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role in ("system", "developer"):
                    continue
                
                if role == "assistant" and msg.get("tool_calls"):
                    # assistant + tool_calls -> function_call items
                    # 处理 reasoning_content -> reasoning 输入项
                    reasoning_content = msg.get("reasoning_content")
                    if reasoning_content:
                        input_items.append({
                            "type": "reasoning",
                            "id": f"rs_{uuid.uuid4().hex[:24]}",
                            "summary": [{"type": "summary_text", "text": reasoning_content}]
                        })
                    if content:
                        if isinstance(content, str):
                            input_items.append({
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": content}]
                            })
                        elif isinstance(content, list):
                            content_items = cls._convert_chat_content_to_responses(content)
                            input_items.append({
                                "type": "message",
                                "role": "assistant",
                                "content": content_items
                            })
                    for tc in msg["tool_calls"]:
                        input_items.append({
                            "type": "function_call",
                            "call_id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": tc.get("function", {}).get("arguments", "{}")
                        })
                elif role == "tool":
                    if isinstance(content, str):
                        tool_output = content
                    elif isinstance(content, list):
                        tool_output = cls._convert_chat_content_to_responses(content)
                    else:
                        tool_output = str(content)
                    input_items.append({
                        "type": "function_call_output",
                        "call_id": msg.get("tool_call_id", ""),
                        "output": tool_output
                    })
                elif isinstance(content, str):
                    # 处理 assistant 消息的 reasoning_content
                    if role == "assistant" and msg.get("reasoning_content"):
                        input_items.append({
                            "type": "reasoning",
                            "id": f"rs_{uuid.uuid4().hex[:24]}",
                            "summary": [{"type": "summary_text", "text": msg["reasoning_content"]}]
                        })
                    if content:
                        input_items.append({
                            "type": "message",
                            "role": role,
                            "content": [{"type": "input_text", "text": content}]
                        })
                    elif role != "assistant":
                        # 非 assistant 角色的空字符串也创建消息
                        input_items.append({
                            "type": "message",
                            "role": role,
                            "content": [{"type": "input_text", "text": content}]
                        })
                    # assistant 角色且 content 为空字符串时跳过（reasoning 已单独处理）
                elif isinstance(content, list):
                    # 多模态内容
                    # 处理 assistant 消息的 reasoning_content（列表内容场景）
                    if role == "assistant" and msg.get("reasoning_content"):
                        input_items.append({
                            "type": "reasoning",
                            "id": f"rs_{uuid.uuid4().hex[:24]}",
                            "summary": [{"type": "summary_text", "text": msg["reasoning_content"]}]
                        })
                    content_items = cls._convert_chat_content_to_responses(content)
                    if content_items:
                        input_items.append({
                            "type": "message",
                            "role": role,
                            "content": content_items
                        })
                elif content is None:
                    # content 为 None（常见于 assistant + tool_calls 场景，但此处无 tool_calls）
                    if role == "assistant" and msg.get("reasoning_content"):
                        # 有推理内容 → 创建 reasoning 输入项
                        input_items.append({
                            "type": "reasoning",
                            "id": f"rs_{uuid.uuid4().hex[:24]}",
                            "summary": [{"type": "summary_text", "text": msg["reasoning_content"]}]
                        })
                    elif role == "assistant":
                        # assistant 无内容也无推理，创建空消息保持消息序列
                        input_items.append({
                            "type": "message",
                            "role": role,
                            "content": []
                        })
                    else:
                        # 非assistant角色的 None content 降级为空字符串
                        input_items.append({
                            "type": "message",
                            "role": role,
                            "content": [{"type": "input_text", "text": ""}]
                        })
                else:
                    # 其他类型（数字等）转为字符串
                    input_items.append({
                        "type": "message",
                        "role": role,
                        "content": [{"type": "input_text", "text": str(content)}]
                    })
            input_data = input_items
            # 确保 input_items 不为空
            if not input_data:
                input_data = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": ""}]}]
        else:
            input_data = ""
        
        # 构建 Responses 请求
        responses_request = {
            "model": request.get("model") or "gpt-4o",
            "input": input_data,
        }
        
        if instructions:
            responses_request["instructions"] = instructions
        
        # 可选参数映射
        if request.get("temperature") is not None:
            responses_request["temperature"] = request["temperature"]
        if request.get("top_p") is not None:
            responses_request["top_p"] = request["top_p"]
        if request.get("max_completion_tokens") is not None:
            responses_request["max_output_tokens"] = request["max_completion_tokens"]
        elif request.get("max_tokens") is not None:
            responses_request["max_output_tokens"] = request["max_tokens"]
        if request.get("stream"):
            responses_request["stream"] = request["stream"]
        if request.get("tools"):
            responses_request["tools"] = cls._convert_chat_tools_to_responses(request["tools"])
        if request.get("tool_choice"):
            responses_request["tool_choice"] = cls._convert_chat_tool_choice_to_responses(request["tool_choice"])
        if request.get("parallel_tool_calls") is not None:
            responses_request["parallel_tool_calls"] = request["parallel_tool_calls"]
        
        # Chat web_search_options → Responses web_search 工具内嵌配置
        # Chat API: 顶层 web_search_options 参数
        # Responses API: web_search 工具内的 search_context_size 和 user_location
        if request.get("web_search_options"):
            ws_opts = request["web_search_options"]
            if isinstance(ws_opts, dict):
                # 更新已有的 web_search 工具或添加新的
                tools_list = responses_request.get("tools", [])
                ws_tool_idx = None
                for i, t in enumerate(tools_list):
                    if isinstance(t, dict) and t.get("type") in ("web_search", "web_search_preview"):
                        ws_tool_idx = i
                        break
                ws_tool = tools_list[ws_tool_idx] if ws_tool_idx is not None else {"type": "web_search"}
                if ws_opts.get("search_context_size"):
                    ws_tool["search_context_size"] = ws_opts["search_context_size"]
                if ws_opts.get("user_location"):
                    ws_tool["user_location"] = ws_opts["user_location"]
                if ws_tool_idx is not None:
                    tools_list[ws_tool_idx] = ws_tool
                else:
                    tools_list.append(ws_tool)
                responses_request["tools"] = tools_list
        if request.get("store") is not None:
            responses_request["store"] = request["store"]
        if request.get("metadata"):
            responses_request["metadata"] = request["metadata"]
        
        # reasoning_effort -> reasoning (含 effort + summary + generate_summary)
        if request.get("reasoning_effort"):
            reasoning_config = {"effort": request["reasoning_effort"]}
            # 如果 extra_body 中有原始 reasoning 配置（含 summary/generate_summary），合并
            if isinstance(request.get("extra_body"), dict):
                original_reasoning = request["extra_body"].get("reasoning")
                if isinstance(original_reasoning, dict):
                    if original_reasoning.get("summary") is not None:
                        reasoning_config["summary"] = original_reasoning["summary"]
                    if original_reasoning.get("generate_summary") is not None:
                        reasoning_config["generate_summary"] = original_reasoning["generate_summary"]
            responses_request["reasoning"] = reasoning_config
        
        # response_format -> text
        if request.get("response_format"):
            fmt = request["response_format"]
            fmt_type = fmt.get("type", "")
            if fmt_type == "json_schema":
                # Chat API: {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}, "strict": bool}}
                # Responses API: {"type": "json_schema", "name": "...", "schema": {...}, "strict": bool}
                json_schema = fmt.get("json_schema", {})
                text_format = {"type": "json_schema"}
                if json_schema.get("name"):
                    text_format["name"] = json_schema["name"]
                if json_schema.get("schema"):
                    text_format["schema"] = json_schema["schema"]
                if json_schema.get("strict") is not None:
                    text_format["strict"] = json_schema["strict"]
                responses_request["text"] = {"format": text_format}
            elif fmt_type == "json_object":
                responses_request["text"] = {"format": {"type": "json_object"}}
            elif fmt_type == "text":
                # Chat API 新增 ResponseFormatText 类型
                responses_request["text"] = {"format": {"type": "text"}}
        
        # service_tier
        if request.get("service_tier"):
            responses_request["service_tier"] = request["service_tier"]
        
        # user -> safety_identifier (新字段)
        if request.get("user"):
            responses_request["user"] = request["user"]
        
        # safety_identifier
        if request.get("safety_identifier"):
            responses_request["safety_identifier"] = request["safety_identifier"]
        
        # prompt_cache_key
        if request.get("prompt_cache_key"):
            responses_request["prompt_cache_key"] = request["prompt_cache_key"]
        
        # prompt_cache_retention
        if request.get("prompt_cache_retention"):
            responses_request["prompt_cache_retention"] = request["prompt_cache_retention"]
        
        # verbosity -> text.verbosity
        # SDK: Chat 顶层 verbosity -> Responses text.verbosity (ResponseTextConfigParam)
        if request.get("verbosity"):
            if "text" not in responses_request:
                responses_request["text"] = {}
            if "format" not in responses_request["text"]:
                # 保留已有的 format 配置，仅添加 verbosity
                pass
            responses_request["text"]["verbosity"] = request["verbosity"]
        
        # top_logprobs
        if request.get("top_logprobs") is not None:
            responses_request["top_logprobs"] = request["top_logprobs"]
        
        # stream_options: Chat include_usage 不等价于 Responses include_obfuscation
        # Chat stream_options.include_usage → Responses 不支持，放入 extra_body
        # Responses stream_options.include_obfuscation → 如果有则恢复
        if request.get("stream_options"):
            stream_opts = request["stream_options"]
            if isinstance(stream_opts, dict) and stream_opts.get("include_obfuscation") is not None:
                responses_request["stream_options"] = {"include_obfuscation": stream_opts["include_obfuscation"]}
            elif isinstance(stream_opts, dict) and stream_opts.get("include_usage") is not None:
                # include_usage 在 Responses API 中不需要（Responses 自动在 completed 事件中返回 usage）
                if "extra_body" not in responses_request:
                    responses_request["extra_body"] = {}
                responses_request["extra_body"]["stream_options"] = stream_opts
        
        # logprobs
        if request.get("logprobs") is not None:
            if "extra_body" not in responses_request:
                responses_request["extra_body"] = {}
            responses_request["extra_body"]["logprobs"] = request["logprobs"]
        
        # Chat API 特有参数（Responses API 不直接支持的）
        extra = responses_request.get("extra_body", {})
        if request.get("frequency_penalty") is not None:
            extra["frequency_penalty"] = request["frequency_penalty"]
        if request.get("presence_penalty") is not None:
            extra["presence_penalty"] = request["presence_penalty"]
        if request.get("logit_bias") is not None:
            extra["logit_bias"] = request["logit_bias"]
        if request.get("seed") is not None:
            extra["seed"] = request["seed"]
        if request.get("n") is not None and request["n"] != 1:
            extra["n"] = request["n"]
        if request.get("stop"):
            extra["stop"] = request["stop"]
        if extra:
            responses_request["extra_body"] = extra
        
        return responses_request
    
    # ================================================================
    # 请求转换：Responses -> Anthropic (直接转换，避免经 Chat 中转丢失数据)
    # ================================================================
    
    @classmethod
    def to_anthropic_request(cls, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI Responses 请求直接转换为 Anthropic 格式
        
        直接转换路径，保留 Responses 特有的 reasoning/text 等参数，
        避免经 Chat 中转时的数据丢失。
        """
        from .anthropic import AnthropicConverter
        
        anthropic_messages = []
        
        # 1. 处理 input
        input_data = request.get("input", [])
        if isinstance(input_data, str):
            if input_data:
                anthropic_messages.append({"role": "user", "content": input_data})
            else:
                anthropic_messages.append({"role": "user", "content": ""})
        elif isinstance(input_data, list):
            for item in input_data:
                msg = cls._convert_input_item_to_anthropic(item)
                if msg:
                    if isinstance(msg, list):
                        anthropic_messages.extend(msg)
                    else:
                        anthropic_messages.append(msg)
        
        # 确保 messages 不为空（Anthropic API 要求至少一条消息）
        if not anthropic_messages:
            anthropic_messages.append({"role": "user", "content": ""})
        
        # 2. 处理 instructions → system
        system_prompt = None
        instructions = request.get("instructions")
        if instructions:
            if isinstance(instructions, str):
                system_prompt = instructions
            elif isinstance(instructions, list):
                inst_parts = []
                for inst_item in instructions:
                    if isinstance(inst_item, str):
                        inst_parts.append(inst_item)
                    elif isinstance(inst_item, dict):
                        inst_type = inst_item.get("type", "")
                        if inst_type == "message":
                            content = inst_item.get("content", "")
                            if isinstance(content, list):
                                texts = []
                                for c in content:
                                    if isinstance(c, dict) and c.get("type") in ("text", "input_text"):
                                        texts.append(c.get("text", ""))
                                inst_parts.append("\n".join(texts))
                            else:
                                inst_parts.append(str(content))
                        elif "text" in inst_item:
                            inst_parts.append(inst_item["text"])
                if inst_parts:
                    system_prompt = "\n\n".join(inst_parts)
        
        # 3. 转换 tools
        raw_tools = request.get("tools", [])
        anthropic_tools = cls._convert_tools_to_anthropic(raw_tools)
        
        # 4. 转换 tool_choice
        tool_choice = cls._convert_tool_choice_to_anthropic(request.get("tool_choice"))
        
        # 5. 构建 Anthropic 请求
        max_output_tokens = request.get("max_output_tokens")
        anthropic_request = {
            "model": request.get("model") or "claude-sonnet-4-20250514",
            "max_tokens": max_output_tokens if max_output_tokens is not None else 4096,
            "messages": anthropic_messages,
        }
        
        if system_prompt:
            anthropic_request["system"] = system_prompt
        if request.get("stream") is not None:
            anthropic_request["stream"] = request["stream"]
        if request.get("temperature") is not None:
            anthropic_request["temperature"] = request["temperature"]
        if request.get("top_p") is not None:
            anthropic_request["top_p"] = request["top_p"]
        if anthropic_tools:
            anthropic_request["tools"] = anthropic_tools
        if tool_choice is not None:
            anthropic_request["tool_choice"] = tool_choice
        
        # 5.5 处理 parallel_tool_calls → disable_parallel_tool_use
        # Anthropic API: disable_parallel_tool_use 是 tool_choice 的子字段，无法独立设置
        parallel_tool_calls = request.get("parallel_tool_calls")
        disable_parallel = parallel_tool_calls is False
        
        if disable_parallel:
            current_tc = anthropic_request.get("tool_choice")
            if isinstance(current_tc, dict):
                current_tc["disable_parallel_tool_use"] = True
            elif isinstance(current_tc, str) and current_tc in ("auto", "any"):
                anthropic_request["tool_choice"] = {"type": current_tc, "disable_parallel_tool_use": True}
            else:
                # 无显式 tool_choice 或 tool_choice="none"
                anthropic_request["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        
        # 5.6 处理 stop 参数 → Anthropic stop_sequences
        # Responses API 没有 stop 顶层参数，但可能从 Chat 转换时保留在 extra_body.stop 中
        stop_val = request.get("stop")
        if stop_val:
            if isinstance(stop_val, list):
                anthropic_request["stop_sequences"] = stop_val
            else:
                anthropic_request["stop_sequences"] = [stop_val]
        
        # 6. reasoning -> thinking
        reasoning = request.get("reasoning")
        if reasoning and isinstance(reasoning, dict):
            effort = reasoning.get("effort")
            if effort:
                effort_budget_map = {
                    "none": 0,
                    "minimal": 1024,
                    "low": 1024,
                    "medium": 10000,
                    "high": 32000,
                    "xhigh": 64000,
                }
                budget = effort_budget_map.get(effort, 10000)
                if budget > 0:
                    thinking_config = {"type": "enabled", "budget_tokens": budget}
                    # reasoning.summary → thinking.display 映射
                    # Responses: summary: "auto" | "concise" | "detailed" | None
                    # Anthropic: display: "summarized" | "omitted"
                    summary = reasoning.get("summary")
                    if summary is not None:
                        # "concise"/"detailed" → "summarized" (返回可见推理摘要)
                        # None/"auto" → 不设置 display (使用 Anthropic 默认行为)
                        # "omitted" → "omitted" (不显示推理摘要)
                        if summary in ("concise", "detailed"):
                            thinking_config["display"] = "summarized"
                        elif summary == "omitted":
                            thinking_config["display"] = "omitted"
                    anthropic_request["thinking"] = thinking_config
                else:
                    anthropic_request["thinking"] = {"type": "disabled"}
        
        # 7. text.format → Anthropic output_format (结构化输出映射)
        # Responses: {"type": "json_schema", "name": "...", "schema": {...}}
        # Anthropic: {"type": "json_schema", "name": "...", "schema": {...}}
        text_config = request.get("text")
        if text_config and isinstance(text_config, dict):
            format_config = text_config.get("format")
            if format_config and isinstance(format_config, dict):
                fmt_type = format_config.get("type")
                if fmt_type == "json_schema":
                    output_format = {"type": "json_schema"}
                    if format_config.get("name"):
                        output_format["name"] = format_config["name"]
                    if format_config.get("schema"):
                        output_format["schema"] = format_config["schema"]
                    if format_config.get("strict") is not None:
                        output_format["strict"] = format_config["strict"]
                    anthropic_request["output_format"] = output_format
                elif fmt_type == "json_object":
                    anthropic_request["output_format"] = {"type": "json_object"}
        
        # 8. metadata.user_id
        metadata = request.get("metadata")
        if metadata and isinstance(metadata, dict):
            user_id = metadata.get("user_id")
            if user_id:
                anthropic_request["metadata"] = {"user_id": user_id}
            # Anthropic API 仅支持 metadata.user_id 字段
            # 其他字段保留在 extra_body 中
            non_user_fields = {k: v for k, v in metadata.items() if k != "user_id"}
            if non_user_fields:
                if "extra_body" not in anthropic_request:
                    anthropic_request["extra_body"] = {}
                anthropic_request["extra_body"]["metadata"] = non_user_fields
        
        # 8.5 text.verbosity 保留（Anthropic 无直接等价参数，放入 extra_body）
        if isinstance(request.get("text"), dict) and request["text"].get("verbosity"):
            if "extra_body" not in anthropic_request:
                anthropic_request["extra_body"] = {}
            anthropic_request["extra_body"]["verbosity"] = request["text"]["verbosity"]
        
        # 9. service_tier 映射
        if request.get("service_tier"):
            tier_map = {"default": "standard_only", "auto": "auto"}
            anthropic_request["service_tier"] = tier_map.get(request["service_tier"], request["service_tier"])
        
        # 10. Responses 特有参数（Anthropic 不支持的，放入 extra_body）
        extra = {}
        if request.get("previous_response_id"):
            extra["previous_response_id"] = request["previous_response_id"]
        if request.get("truncation"):
            extra["truncation"] = request["truncation"]
        if request.get("parallel_tool_calls") is not None:
            extra["parallel_tool_calls"] = request["parallel_tool_calls"]
        if request.get("max_tool_calls") is not None:
            extra["max_tool_calls"] = request["max_tool_calls"]
        if request.get("background") is not None:
            extra["background"] = request["background"]
        if request.get("conversation"):
            extra["conversation"] = request["conversation"]
        if request.get("context_management"):
            extra["context_management"] = request["context_management"]
        if request.get("include"):
            extra["include"] = request["include"]
        if request.get("prompt"):
            extra["prompt"] = request["prompt"]
        if request.get("store") is not None:
            extra["store"] = request["store"]
        if request.get("safety_identifier"):
            extra["safety_identifier"] = request["safety_identifier"]
        if request.get("prompt_cache_key"):
            extra["prompt_cache_key"] = request["prompt_cache_key"]
        if request.get("prompt_cache_retention"):
            extra["prompt_cache_retention"] = request["prompt_cache_retention"]
        if request.get("user"):
            extra["user"] = request["user"]
        if request.get("top_logprobs") is not None:
            extra["top_logprobs"] = request["top_logprobs"]
        # reasoning.generate_summary 保留（请求参数，非 Anthropic 原生支持）
        if isinstance(request.get("reasoning"), dict) and request["reasoning"].get("generate_summary") is not None:
            extra["reasoning"] = request["reasoning"]
        if request.get("frequency_penalty") is not None:
            extra["frequency_penalty"] = request["frequency_penalty"]
        if request.get("presence_penalty") is not None:
            extra["presence_penalty"] = request["presence_penalty"]
        if request.get("logit_bias") is not None:
            extra["logit_bias"] = request["logit_bias"]
        if request.get("seed") is not None:
            extra["seed"] = request["seed"]
        # stop 已在上面映射为 stop_sequences，不再放入 extra_body
        if extra:
            if "extra_body" not in anthropic_request:
                anthropic_request["extra_body"] = {}
            anthropic_request["extra_body"].update(extra)
        
        return anthropic_request
    
    @classmethod
    def _convert_input_item_to_anthropic(cls, item: Dict[str, Any]) -> Optional[Union[Dict, List[Dict]]]:
        """转换 Responses 输入项到 Anthropic 消息格式"""
        if not isinstance(item, dict):
            return None
        
        item_type = item.get("type", "")
        
        if item_type == "message":
            role = item.get("role", "user")
            # Anthropic 只支持 user/assistant 角色
            if role not in ("user", "assistant"):
                role = "user"
            content = item.get("content", [])
            anthropic_content = cls._convert_content_to_anthropic(content)
            if isinstance(anthropic_content, str):
                return {"role": role, "content": anthropic_content}
            return {"role": role, "content": anthropic_content}
        
        elif item_type == "function_call_output":
            # 工具结果 → tool_result 块
            # Anthropic tool_result.content 可以是字符串或内容块列表
            output_val = item.get("output", "")
            if isinstance(output_val, str):
                tool_content = output_val
            elif isinstance(output_val, list):
                # 列表类型输出 → 转换为 Anthropic 内容块列表
                content_blocks = []
                for part in output_val:
                    if isinstance(part, dict):
                        part_type = part.get("type", "")
                        if part_type in ("text", "input_text", "output_text"):
                            content_blocks.append({"type": "text", "text": part.get("text", "")})
                        elif part_type == "input_image":
                            # 图片输出 → Anthropic image 块
                            image_url = part.get("image_url", "")
                            if image_url:
                                if image_url.startswith("data:"):
                                    parts = image_url.split(";", 1)
                                    media_type = parts[0].replace("data:", "") if len(parts) > 1 else "image/png"
                                    data = parts[1].replace("base64,", "") if len(parts) > 1 else ""
                                    content_blocks.append({
                                        "type": "image",
                                        "source": {"type": "base64", "media_type": media_type, "data": data}
                                    })
                                else:
                                    content_blocks.append({
                                        "type": "image",
                                        "source": {"type": "url", "url": image_url}
                                    })
                        else:
                            text = part.get("text", "")
                            if text:
                                content_blocks.append({"type": "text", "text": text})
                    else:
                        content_blocks.append({"type": "text", "text": str(part)})
                tool_content = content_blocks if content_blocks else ""
            else:
                tool_content = str(output_val) if output_val is not None else ""
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": item.get("call_id", ""),
                    "content": tool_content
                }]
            }
        
        elif item_type == "function_call":
            # assistant 的工具调用
            args_str = item.get("arguments", "{}")
            try:
                args_obj = json.loads(args_str) if isinstance(args_str, str) else args_str
            except (json.JSONDecodeError, TypeError):
                args_obj = {}
            return {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": item.get("call_id", item.get("id", "")),
                    "name": item.get("name", ""),
                    "input": args_obj
                }]
            }
        
        elif item_type == "reasoning":
            # 推理项 - Anthropic 不支持在输入中传递 reasoning，跳过
            return None
        
        elif item_type == "computer_call_output":
            return {
                "role": "user",
                "content": "[Computer screenshot output]"
            }
        
        elif item_type == "local_shell_call_output":
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": item.get("call_id", ""),
                    "content": str(item.get("output", ""))
                }]
            }
        
        elif item_type == "mcp_approval_response":
            return None
        
        elif item_type == "input_audio":
            return {
                "role": "user",
                "content": "[Audio input]"
            }
        
        # 未知类型，尝试作为消息处理
        if "role" in item and "content" in item:
            role = item.get("role", "user")
            if role not in ("user", "assistant"):
                role = "user"
            return {"role": role, "content": str(item.get("content", ""))}
        
        return None
    
    @classmethod
    def _convert_content_to_anthropic(cls, content: Union[str, List[Dict]]) -> Union[str, List[Dict]]:
        """转换 Responses 内容为 Anthropic 格式"""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)
        
        content_blocks = []
        for item in content:
            if not isinstance(item, dict):
                content_blocks.append({"type": "text", "text": str(item)})
                continue
            
            item_type = item.get("type", "")
            
            if item_type in ("text", "input_text"):
                content_blocks.append({"type": "text", "text": item.get("text", "")})
            elif item_type == "input_image":
                image_url = item.get("image_url", "")
                if image_url:
                    if image_url.startswith("data:"):
                        # data:image/png;base64,...
                        parts = image_url.split(";", 1)
                        media_type = parts[0].replace("data:", "") if len(parts) > 1 else "image/png"
                        data = parts[1].replace("base64,", "") if len(parts) > 1 else ""
                        content_blocks.append({
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": data}
                        })
                    else:
                        content_blocks.append({
                            "type": "image",
                            "source": {"type": "url", "url": image_url}
                        })
            elif item_type == "input_file":
                file_data = item.get("file_data", "")
                file_id = item.get("file_id")
                filename = item.get("filename", "")
                if file_data:
                    if file_data.startswith("data:"):
                        parts = file_data.split(";", 1)
                        media_type = parts[0].replace("data:", "") if len(parts) > 1 else "application/pdf"
                        data = parts[1].replace("base64,", "") if len(parts) > 1 else ""
                        content_blocks.append({
                            "type": "document",
                            "source": {"type": "base64", "media_type": media_type, "data": data}
                        })
                    else:
                        content_blocks.append({"type": "text", "text": f"[File: {filename or file_data}]"})
                elif file_id:
                    content_blocks.append({"type": "text", "text": f"[File ID: {file_id}]"})
            elif item_type == "output_text":
                content_blocks.append({"type": "text", "text": item.get("text", "")})
            else:
                text = item.get("text", item.get("content", ""))
                if text:
                    content_blocks.append({"type": "text", "text": str(text)})
        
        return content_blocks if content_blocks else [{"type": "text", "text": ""}]
    
    @classmethod
    def _convert_tools_to_anthropic(cls, tools: List[Dict]) -> List[Dict]:
        """转换 Responses 工具定义为 Anthropic 格式"""
        converted = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_type = tool.get("type", "function")
            
            if tool_type == "function":
                at = {"name": tool.get("name", ""), "input_schema": tool.get("parameters", {"type": "object", "properties": {}})}
                if tool.get("description"):
                    at["description"] = tool["description"]
                converted.append(at)
            # 其他 Responses 工具类型（web_search, file_search 等）Anthropic 不直接支持，跳过
        
        return converted
    
    @classmethod
    def _convert_tool_choice_to_anthropic(cls, tool_choice: Any) -> Any:
        """转换 Responses tool_choice 为 Anthropic 格式"""
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            tc_map = {"auto": "auto", "none": "none", "required": "any"}
            return tc_map.get(tool_choice, tool_choice)
        if isinstance(tool_choice, dict):
            tc_type = tool_choice.get("type", "")
            if tc_type == "function":
                return {"type": "tool", "name": tool_choice.get("name", "")}
            if tc_type in ("mcp", "custom", "apply_patch", "shell", "allowed", "types"):
                name = tool_choice.get("name", "")
                if name:
                    return {"type": "tool", "name": name}
                return "auto"
        return None

    @classmethod
    def _convert_chat_content_to_responses(cls, content: List[Dict]) -> List[Dict]:
        """转换 Chat 内容数组为 Responses 输入内容格式"""
        result = []
        for item in content:
            if not isinstance(item, dict):
                result.append({"type": "input_text", "text": str(item)})
                continue
            item_type = item.get("type", "")
            
            if item_type == "text":
                result.append({"type": "input_text", "text": item.get("text", "")})
            elif item_type == "image_url":
                url = item.get("image_url", {})
                if isinstance(url, dict):
                    result.append({"type": "input_image", "image_url": url.get("url", "")})
                else:
                    result.append({"type": "input_image", "image_url": str(url)})
            elif item_type == "file":
                file_data = item.get("file", {})
                input_file = {"type": "input_file"}
                if file_data.get("file_data"):
                    input_file["file_data"] = file_data["file_data"]
                if file_data.get("file_id"):
                    input_file["file_id"] = file_data["file_id"]
                if file_data.get("filename"):
                    input_file["filename"] = file_data["filename"]
                result.append(input_file)
            else:
                text = item.get("text", "")
                if text:
                    result.append({"type": "input_text", "text": text})
        
        return result
    
    @classmethod
    def _convert_chat_tools_to_responses(cls, tools: List[Dict]) -> List[Dict]:
        """转换 Chat 工具定义为 Responses 格式"""
        result = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_type = tool.get("type", "function")
            
            if tool_type == "function":
                func = tool.get("function", {})
                rt = {"type": "function", "name": func.get("name", "")}
                if func.get("description"):
                    rt["description"] = func["description"]
                if func.get("parameters"):
                    rt["parameters"] = func["parameters"]
                if func.get("strict") is not None:
                    rt["strict"] = func["strict"]
                result.append(rt)
            elif tool_type in ("web_search", "web_search_preview"):
                # Chat API: web_search_preview 工具 + 顶层 web_search_options 参数
                # Responses API: web_search 工具 + 工具内 user_location/search_context_size
                rt = {"type": "web_search"}
                # 注意：web_search_options 是顶层请求参数，不在 tool 定义内部
                # 此处仅处理工具本身，search_context_size/user_location 由调用方从请求中提取
                result.append(rt)
            else:
                # 其他类型直接传递
                result.append(tool)
        
        return result
    
    @classmethod
    def _convert_chat_tool_choice_to_responses(cls, tool_choice: Any) -> Any:
        """转换 Chat tool_choice 为 Responses 格式"""
        if isinstance(tool_choice, str):
            return tool_choice  # auto, none, required
        
        if isinstance(tool_choice, dict):
            tc_type = tool_choice.get("type", "")
            if tc_type == "function":
                func = tool_choice.get("function", {})
                return {"type": "function", "name": func.get("name", "")}
            if tc_type == "allowed_tools":
                allowed_tools = tool_choice.get("allowed_tools", {})
                return {
                    "type": "allowed_tools",
                    "mode": allowed_tools.get("mode", "auto"),
                    "tools": [
                        cls._convert_allowed_tool_to_responses(tool)
                        for tool in allowed_tools.get("tools", [])
                        if isinstance(tool, dict)
                    ],
                }
        
        return tool_choice

    @classmethod
    def _convert_allowed_tool_to_responses(cls, tool: Dict[str, Any]) -> Dict[str, Any]:
        """转换 Chat allowed_tools 中的工具引用为 Responses allowed_tools 形状"""
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            return {
                "type": "function",
                "name": tool["function"].get("name", "")
            }
        return tool
    
    # ================================================================
    # 响应转换：Chat Response -> Responses Response
    # ================================================================
    
    @classmethod
    def from_openai_chat(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI Chat 响应转换为 Responses 格式
        
        Chat: {"id", "model", "choices": [{"message": {"content", "tool_calls", "reasoning_content"}, "finish_reason"}], "usage"}
        Responses: {"id", "object", "status", "created_at", "model", "output": [...], "usage", 
                    "incomplete_details", "error", "metadata", "parallel_tool_calls", "service_tier"}
        """
        output = []
        choices = response.get("choices", [])
        status = "completed"
        incomplete_details = None
        
        for i, choice in enumerate(choices):
            message = choice.get("message", {})
            content = message.get("content")
            finish_reason = choice.get("finish_reason", "")
            reasoning_content = message.get("reasoning_content")
            
            # 处理推理内容 -> reasoning 输出项
            if reasoning_content:
                output.append({
                    "type": "reasoning",
                    "id": f"rs_{uuid.uuid4().hex[:24]}",
                    "content": [{
                        "type": "reasoning_text",
                        "text": reasoning_content
                    }],
                    "summary": [],  # Responses API: summary 是必填字段 (List[Summary])
                    "status": "completed"
                })
            
            # 处理工具调用（先输出 function_call，再输出 message）
            tool_calls = message.get("tool_calls", [])
            for tc in tool_calls:
                output.append({
                    "type": "function_call",
                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "call_id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", "{}"),
                    "status": "completed"
                })
            
            # 处理文本内容
            if content is not None:
                msg_content = []
                output_text_item = {
                    "type": "output_text",
                    "text": content,
                }
                # 保留 annotations（Chat message.annotations -> Responses output_text.annotations）
                annotations = message.get("annotations", [])
                if annotations:
                    output_text_item["annotations"] = annotations
                msg_content.append(output_text_item)
                
                output.append({
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "status": "completed",
                    "role": "assistant",
                    "content": msg_content
                })

            # 处理 refusal 内容
            refusal = message.get("refusal")
            if refusal:
                output.append({
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "status": "incomplete",
                    "role": "assistant",
                    "content": [{
                        "type": "refusal",
                        "refusal": refusal,
                    }]
                })
                status = "incomplete"
                incomplete_details = {"reason": "content_filter"}
            
            # 检查是否不完整
            if finish_reason == "length":
                status = "incomplete"
                incomplete_details = {"reason": "max_output_tokens"}
            elif finish_reason == "content_filter":
                status = "incomplete"
                incomplete_details = {"reason": "content_filter"}
        
        # 转换 usage - 包含 input_tokens_details 和 output_tokens_details
        usage = response.get("usage", {})
        
        response_usage = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        
        # input_tokens_details: 从 prompt_tokens_details.cached_tokens 提取
        prompt_tokens_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_tokens_details, dict):
            cached_tokens = prompt_tokens_details.get("cached_tokens", 0)
            if cached_tokens:
                response_usage["input_tokens_details"] = {"cached_tokens": cached_tokens}
        
        # output_tokens_details: 从 completion_tokens_details.reasoning_tokens 提取
        completion_tokens_details = usage.get("completion_tokens_details")
        if isinstance(completion_tokens_details, dict):
            reasoning_tokens = completion_tokens_details.get("reasoning_tokens", 0)
            if reasoning_tokens:
                response_usage["output_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
        
        result = {
            "id": response.get("id", f"resp_{uuid.uuid4().hex[:24]}"),
            "object": "response",
            "status": status,
            "created_at": response.get("created", int(time.time())),
            "model": response.get("model", ""),
            "output": output,
            "usage": response_usage,
            "incomplete_details": incomplete_details,  # Responses API: 始终存在，null 表示完成
            "error": None,
            "metadata": response.get("metadata"),
            "parallel_tool_calls": True,  # Chat API 默认允许并行工具调用
        }
        
        # 仅在 status 为 completed 时设置 completed_at
        if status == "completed":
            result["completed_at"] = int(time.time())
        
        # 添加可选字段（如果 Chat 响应中存在）
        if response.get("service_tier"):
            result["service_tier"] = response["service_tier"]
        if response.get("system_fingerprint"):
            result["system_fingerprint"] = response["system_fingerprint"]
        # 温度和 top_p 如果原始请求中有
        if response.get("temperature") is not None:
            result["temperature"] = response["temperature"]
        if response.get("top_p") is not None:
            result["top_p"] = response["top_p"]
        
        return result
    
    # ================================================================
    # 响应转换：Responses Response -> Chat Response
    # ================================================================
    
    @classmethod
    def to_chat_response(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI Responses 响应转换为 OpenAI Chat 响应格式
        
        Args:
            response: OpenAI Responses 格式响应 (含 output, status, usage 等)
            
        Returns:
            Dict: OpenAI Chat 格式响应
        """
        output = response.get("output", [])
        message_content = None
        message_annotations = None
        message_refusal = None
        tool_calls = []
        reasoning_content = None
        
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type", "")
            
            if item_type == "message":
                content_list = item.get("content", [])
                texts = []
                all_annotations = []
                for c in content_list:
                    if not isinstance(c, dict):
                        continue
                    c_type = c.get("type", "")
                    if c_type in ("output_text", "text"):
                        texts.append(c.get("text", ""))
                        # 保留 annotations
                        if c.get("annotations"):
                            all_annotations.extend(c["annotations"])
                    elif c_type == "refusal":
                        message_refusal = c.get("refusal", "")
                if texts:
                    message_content = "\n".join(texts)
                if all_annotations:
                    message_annotations = all_annotations
            
            elif item_type == "reasoning":
                # reasoning 输出项 -> reasoning_content (OpenAI o系列格式)
                reasoning_texts = []
                # 处理 content 中的 reasoning_text（实时推理内容）
                for c in item.get("content", []):
                    if isinstance(c, dict):
                        c_type = c.get("type", "")
                        if c_type == "reasoning_text":
                            t = c.get("text", "")
                            if t:
                                reasoning_texts.append(t)
                        elif c_type == "summary_text":
                            # Responses SDK: reasoning 项 content 中的摘要文本
                            # 也应纳入 reasoning_content 以保留推理信息
                            t = c.get("text", "")
                            if t:
                                reasoning_texts.append(t)
                # 处理 summary 字段 (Responses API: summary 是必填字段)
                for s in item.get("summary", []):
                    if isinstance(s, dict) and s.get("type") == "summary_text":
                        text = s.get("text", "")
                        if text and text not in reasoning_texts:
                            reasoning_texts.append(text)
                if reasoning_texts:
                    reasoning_content = "\n".join(reasoning_texts)
                elif item.get("encrypted_content"):
                    # 仅含 encrypted_content 无可见文本 → 标记为 redacted_thinking 以便往返转换
                    reasoning_content = f"[redacted_thinking: {item['encrypted_content']}]"
            
            elif item_type == "function_call":
                tool_calls.append({
                    "id": item.get("call_id", item.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}")
                    }
                })
            
            # 以下输出项类型在 Chat API 中无直接等价，忽略或部分转换
            
            elif item_type == "web_search_call":
                # 网页搜索调用 - Chat API 无等价项，跳过
                pass
            
            elif item_type == "file_search_call":
                # 文件搜索调用 - Chat API 无等价项，跳过
                pass
            
            elif item_type == "computer_call":
                # 计算机调用 - Chat API 无等价项，跳过
                pass
            
            elif item_type == "code_interpreter_call":
                # 代码解释器调用 - Chat API 无等价项，跳过
                pass
            
            elif item_type == "image_generation_call":
                # 图片生成调用 - Chat API 无等价项，跳过
                pass
            
            elif item_type == "local_shell_call":
                # 本地终端调用 - Chat API 无等价项，跳过
                pass
            
            elif item_type == "mcp_call":
                # MCP 调用 - 尝试转为工具调用
                name = item.get("name", "")
                if name:
                    tool_calls.append({
                        "id": item.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": item.get("arguments", "{}")
                        }
                    })
            
            elif item_type == "custom_tool_call":
                # 自定义工具调用 - 转为工具调用
                name = item.get("name", "")
                if name:
                    tool_calls.append({
                        "id": item.get("id", item.get("call_id", f"call_{uuid.uuid4().hex[:24]}")),
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": item.get("arguments", "{}")
                        }
                    })
            
            elif item_type == "apply_patch_tool_call":
                # 补丁工具调用 - 转为工具调用
                tool_calls.append({
                    "id": item.get("id", item.get("call_id", f"call_{uuid.uuid4().hex[:24]}")),
                    "type": "function",
                    "function": {
                        "name": "apply_patch",
                        "arguments": item.get("arguments", "{}")
                    }
                })
            
            elif item_type == "shell_tool_call":
                # Shell 工具调用 - 转为工具调用
                tool_calls.append({
                    "id": item.get("id", item.get("call_id", f"call_{uuid.uuid4().hex[:24]}")),
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "arguments": item.get("arguments", "{}")
                    }
                })
            
            elif item_type == "tool_search_call":
                # 工具搜索调用 - 转为工具调用
                tool_calls.append({
                    "id": item.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": "tool_search",
                        "arguments": item.get("arguments", "{}")
                    }
                })
            
            elif item_type == "compaction_item":
                # 压缩项 - Chat API 无等价项，跳过
                pass
        
        # 映射状态
        status = response.get("status", "completed")
        finish_reason = "stop"
        error_info = None
        if status == "incomplete":
            details = response.get("incomplete_details", {})
            reason = details.get("reason", "") if details else ""
            if reason == "max_output_tokens":
                finish_reason = "length"
            elif reason == "content_filter":
                finish_reason = "content_filter"
        elif status == "failed":
            finish_reason = "stop"
            # 保留错误信息
            error_info = response.get("error")
        
        usage = response.get("usage", {})
        
        # 构建 Chat usage (含 prompt_tokens_details 和 completion_tokens_details)
        chat_usage = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }
        
        # input_tokens_details.cached_tokens -> prompt_tokens_details.cached_tokens
        input_tokens_details = usage.get("input_tokens_details")
        if isinstance(input_tokens_details, dict):
            cached_tokens = input_tokens_details.get("cached_tokens", 0)
            if cached_tokens:
                chat_usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
        
        # output_tokens_details.reasoning_tokens -> completion_tokens_details.reasoning_tokens
        output_tokens_details = usage.get("output_tokens_details")
        if isinstance(output_tokens_details, dict):
            reasoning_tokens = output_tokens_details.get("reasoning_tokens", 0)
            if reasoning_tokens:
                chat_usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
        
        # 构建 message
        message = {
            "role": "assistant",
            "content": message_content,
        }
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        if message_refusal is not None:
            message["refusal"] = message_refusal
        if tool_calls:
            message["tool_calls"] = tool_calls
        # 保留 annotations（Responses output_text.annotations -> Chat message.annotations）
        if message_annotations is not None:
            message["annotations"] = message_annotations
        
        result = {
            "id": response.get("id", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
            "object": "chat.completion",
            "created": response.get("created_at", int(time.time())),
            "model": response.get("model", ""),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }],
            "usage": chat_usage
        }
        
        # 添加可选字段
        if response.get("service_tier"):
            result["service_tier"] = response["service_tier"]
        if response.get("system_fingerprint"):
            result["system_fingerprint"] = response["system_fingerprint"]
        # 保留 metadata
        if response.get("metadata"):
            result["metadata"] = response["metadata"]
        # 保留错误信息（Responses failed 状态）
        if error_info is not None:
            result["error"] = error_info
        
        return result
    
    # ================================================================
    # 响应转换：Anthropic -> Responses
    # ================================================================
    
    @classmethod
    def from_anthropic(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anthropic 响应转换为 OpenAI Responses 格式
        """
        from .anthropic import AnthropicConverter
        return AnthropicConverter.to_openai_responses(response)
    
    # ================================================================
    # 响应转换：Responses -> Anthropic (直接转换，避免经 Chat 中转丢失数据)
    # ================================================================
    
    @classmethod
    def to_anthropic(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI Responses 响应直接转换为 Anthropic 格式
        
        保留 reasoning 的 summary 和 encrypted_content，
        避免 Responses→Chat→Anthropic 中转丢失数据。
        """
        content_blocks = []
        stop_reason = "end_turn"
        stop_details = None
        
        output = response.get("output", [])
        
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type", "")
            
            if item_type == "reasoning":
                # reasoning 输出项 → thinking 块或 redacted_thinking 块
                encrypted_content = item.get("encrypted_content")

                # 检查是否有可见的推理内容
                content = item.get("content", [])
                summary = item.get("summary", [])
                has_visible_content = any(
                    c.get("type") == "reasoning_text" and c.get("text")
                    for c in content
                ) or any(
                    s.get("type") == "summary_text" and s.get("text")
                    for s in summary
                )

                if encrypted_content and not has_visible_content:
                    # 仅含 encrypted_content，无可见内容 → redacted_thinking
                    content_blocks.append({
                        "type": "redacted_thinking",
                        "data": encrypted_content,  # Anthropic SDK: RedactedThinkingBlock.data
                    })
                else:
                    # 有可见推理内容 → thinking 块
                    reasoning_texts = []
                    for c in item.get("content", []):
                        if isinstance(c, dict):
                            c_type = c.get("type", "")
                            if c_type == "reasoning_text":
                                reasoning_texts.append(c.get("text", ""))
                            elif c_type == "summary_text":
                                reasoning_texts.append(c.get("text", ""))
                    # summary 字段中的摘要也纳入
                    for s in item.get("summary", []):
                        if isinstance(s, dict) and s.get("type") == "summary_text":
                            text = s.get("text", "")
                            if text and text not in reasoning_texts:
                                reasoning_texts.append(text)
                    
                    if reasoning_texts:
                        thinking_block = {
                            "type": "thinking",
                            "thinking": "\n".join(reasoning_texts),
                            "signature": encrypted_content or "",  # Anthropic SDK: ThinkingBlock.signature
                        }
                        content_blocks.append(thinking_block)
            
            elif item_type == "message":
                content_list = item.get("content", [])
                for c in content_list:
                    if not isinstance(c, dict):
                        continue
                    c_type = c.get("type", "")
                    if c_type in ("output_text", "text"):
                        text_block = {
                            "type": "text",
                            "text": c.get("text", "")
                        }
                        # 保留 annotations → citations
                        if c.get("annotations"):
                            text_block["citations"] = c["annotations"]
                        content_blocks.append(text_block)
                    elif c_type == "refusal":
                        refusal_text = c.get("refusal", "")
                        if refusal_text:
                            content_blocks.append({
                                "type": "text",
                                "text": refusal_text
                            })
            
            elif item_type == "function_call":
                args_str = item.get("arguments", "{}")
                try:
                    args_obj = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, TypeError):
                    args_obj = {}
                
                content_blocks.append({
                    "type": "tool_use",
                    "id": item.get("call_id", item.get("id", f"toolu_{uuid.uuid4().hex[:24]}")),
                    "name": item.get("name", ""),
                    "input": args_obj
                })
        
        # 如果没有任何内容块，添加空文本
        if not content_blocks:
            content_blocks.append({"type": "text", "text": ""})
        
        # 映射停止原因
        status = response.get("status", "completed")
        if status == "incomplete":
            details = response.get("incomplete_details", {})
            reason = details.get("reason", "") if details else ""
            if reason == "max_output_tokens":
                stop_reason = "max_tokens"
            elif reason == "content_filter":
                stop_reason = "refusal"
                stop_details = {"reason": "content_policy"}
        elif status == "failed":
            stop_reason = "refusal"
            stop_details = {"reason": "content_policy"}
            error = response.get("error")
            if error and isinstance(error, dict):
                stop_details["message"] = error.get("message", "")
        
        # 转换 usage
        usage = response.get("usage", {})
        anthropic_usage = {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        
        # input_tokens_details.cached_tokens → cache_read_input_tokens
        input_tokens_details = usage.get("input_tokens_details")
        if isinstance(input_tokens_details, dict):
            cached_tokens = input_tokens_details.get("cached_tokens", 0)
            if cached_tokens:
                anthropic_usage["cache_read_input_tokens"] = cached_tokens
        
        # output_tokens_details.reasoning_tokens → 可用于推断 cache_creation
        output_tokens_details = usage.get("output_tokens_details")
        if isinstance(output_tokens_details, dict):
            reasoning_tokens = output_tokens_details.get("reasoning_tokens", 0)
            if reasoning_tokens:
                # Anthropic 没有直接暴露 reasoning_tokens，但保留为参考
                pass
        
        # Responses service_tier → Anthropic usage.service_tier
        responses_service_tier = response.get("service_tier")
        if responses_service_tier:
            responses_to_anthropic_tier = {
                "default": "standard",
                "auto": "standard",
                "flex": "standard",
                "scale": "standard",
                "priority": "priority",
            }
            anthropic_usage["service_tier"] = responses_to_anthropic_tier.get(
                responses_service_tier, responses_service_tier
            )
        
        result = {
            "id": response.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": response.get("model", ""),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": stop_details,
            "usage": anthropic_usage
        }
        
        # 保留 container 字段（Anthropic 响应可能包含此字段）
        if response.get("container"):
            result["container"] = response["container"]
        
        return result
    
    # ================================================================
    # 流式转换
    # ================================================================
    
    # 流式状态跟踪
    _stream_state = {
        "message_item_started": False,
        "content_part_started": False,
        "started_function_ids": set(),
        "output_index": 0,
        "reasoning_item_started": False,
        "reasoning_summary_started": False,
    }
    
    @classmethod
    def reset_stream_state(cls):
        """重置流式状态"""
        cls._stream_state = {
            "message_item_started": False,
            "content_part_started": False,
            "started_function_ids": set(),
            "output_index": 0,
            "reasoning_item_started": False,
            "reasoning_summary_started": False,
        }
    
    @classmethod
    def convert_stream_chunk(cls, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        转换 OpenAI Chat 流式块到 Responses 流式事件列表
        
        Responses 流式事件严格顺序:
        response.created -> response.in_progress -> 
        [response.output_item.added -> 
          [response.content_part.added -> 
            response.output_text.delta* -> response.output_text.done 
            | response.refusal.delta* -> response.refusal.done
          -> response.content_part.done]?
          [response.reasoning_summary_part.added -> 
            response.reasoning_summary_text.delta* -> response.reasoning_summary_text.done 
          -> response.reasoning_summary_part.done]?
          [response.function_call_arguments.delta* -> response.function_call_arguments.done]?
        -> response.output_item.done]* 
        -> response.completed | response.failed | response.incomplete
        
        完整事件类型列表:
        - response.created / response.in_progress / response.completed / response.failed / response.incomplete
        - response.output_item.added / response.output_item.done
        - response.content_part.added / response.content_part.done
        - response.output_text.delta / response.output_text.done
        - response.refusal.delta / response.refusal.done
        - response.reasoning_summary_part.added / response.reasoning_summary_part.done
        - response.reasoning_summary_text.delta / response.reasoning_summary_text.done
        - response.function_call_arguments.delta / response.function_call_arguments.done
        - response.reasoning_text.delta / response.reasoning_text.done
        - response.web_search_call.in_progress / .searching / .completed
        - response.file_search_call.in_progress / .searching / .completed
        - response.code_interpreter_call.in_progress / .completed
        - response.code_interpreter_call_code.delta / .done
        - response.computer_call.in_progress / .completed
        - response.ping
        
        注意：一个 OpenAI chunk 可能需要生成多个 Responses 事件
        
        Returns:
            List[Dict]: Responses 流式事件列表
        """
        events = []
        choices = chunk.get("choices", [])
        
        if not choices:
            return [{"type": "response.ping"}]
        
        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")
        content = delta.get("content", "")
        
        # 1. response.created + response.in_progress
        if delta.get("role") == "assistant":
            cls.reset_stream_state()
            events.append({
                "type": "response.created",
                "response": {
                    "id": chunk.get("id", ""),
                    "object": "response",
                    "status": "in_progress",
                    "model": chunk.get("model", ""),
                    "output": [],
                }
            })
            events.append({
                "type": "response.in_progress",
                "response": {
                    "id": chunk.get("id", ""),
                    "object": "response",
                    "status": "in_progress",
                    "model": chunk.get("model", ""),
                    "output": [],
                }
            })
            return events
        
        # 1.5 处理推理内容 (reasoning_content) -> reasoning 输出项
        reasoning_content = delta.get("reasoning_content")
        if reasoning_content is not None:
            output_idx = cls._stream_state["output_index"]
            
            # 如果还没开始 reasoning 项，先发 output_item.added
            if not cls._stream_state["reasoning_item_started"]:
                cls._stream_state["reasoning_item_started"] = True
                events.append({
                    "type": "response.output_item.added",
                    "output_index": output_idx,
                    "item": {
                        "type": "reasoning",
                        "id": f"rs_{uuid.uuid4().hex[:24]}",
                        "status": "in_progress",
                        "content": []
                    }
                })
            # 推理增量 - 使用 Responses API 官方事件类型 response.reasoning_text.delta
            events.append({
                "type": "response.reasoning_text.delta",
                "output_index": output_idx,
                "delta": reasoning_content
            })
            return events
        
        # 2. 文本内容
        if content:
            output_idx = cls._stream_state["output_index"]
            
            # 如果 reasoning 项已开始但未关闭，先关闭它
            if cls._stream_state["reasoning_item_started"]:
                events.append({
                    "type": "response.reasoning_text.done",
                    "output_index": output_idx,
                    "text": ""
                })
                events.append({
                    "type": "response.output_item.done",
                    "output_index": output_idx,
                    "item": {
                        "type": "reasoning",
                        "id": f"rs_{uuid.uuid4().hex[:24]}",
                        "status": "completed",
                        "content": []
                    }
                })
                cls._stream_state["reasoning_item_started"] = False
                cls._stream_state["output_index"] += 1
                output_idx = cls._stream_state["output_index"]
            
            # 如果还没开始消息项，先发 output_item.added + content_part.added
            if not cls._stream_state["message_item_started"]:
                cls._stream_state["message_item_started"] = True
                events.append({
                    "type": "response.output_item.added",
                    "output_index": output_idx,
                    "item": {
                        "type": "message",
                        "id": f"msg_{uuid.uuid4().hex[:24]}",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": []
                    }
                })
                events.append({
                    "type": "response.content_part.added",
                    "output_index": output_idx,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": "",
                        "annotations": []
                    }
                })
                cls._stream_state["content_part_started"] = True
            
            # 文本增量
            events.append({
                "type": "response.output_text.delta",
                "output_index": output_idx,
                "content_index": 0,
                "delta": content
            })
            return events
        
        # 3. 工具调用
        tool_calls = delta.get("tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                tc_func = tc.get("function", {})
                tc_id = tc.get("id", "")
                tc_idx = tc.get("index", 0)
                
                # 如果有 id，是新的 function_call 开始
                if tc_id and tc_id not in cls._stream_state["started_function_ids"]:
                    # 关闭之前的 reasoning 项（如果有）
                    if cls._stream_state["reasoning_item_started"]:
                        events.append({
                            "type": "response.reasoning_text.done",
                            "output_index": cls._stream_state["output_index"],
                            "text": ""
                        })
                        events.append({
                            "type": "response.output_item.done",
                            "output_index": cls._stream_state["output_index"],
                            "item": {
                                "type": "reasoning",
                                "id": f"rs_{uuid.uuid4().hex[:24]}",
                                "status": "completed",
                                "content": []
                            }
                        })
                        cls._stream_state["reasoning_item_started"] = False
                        cls._stream_state["output_index"] += 1
                    
                    # 关闭之前的消息项（如果有）
                    if cls._stream_state["message_item_started"]:
                        if cls._stream_state["content_part_started"]:
                            events.append({
                                "type": "response.output_text.done",
                                "output_index": cls._stream_state["output_index"],
                                "content_index": 0,
                                "text": ""
                            })
                            events.append({
                                "type": "response.content_part.done",
                                "output_index": cls._stream_state["output_index"],
                                "content_index": 0,
                                "part": {
                                    "type": "output_text",
                                    "text": "",
                                    "annotations": []
                                }
                            })
                            cls._stream_state["content_part_started"] = False
                        events.append({
                            "type": "response.output_item.done",
                            "output_index": cls._stream_state["output_index"],
                            "item": {
                                "type": "message",
                                "id": f"msg_{uuid.uuid4().hex[:24]}",
                                "status": "completed",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "", "annotations": []}]
                            }
                        })
                        cls._stream_state["message_item_started"] = False
                        cls._stream_state["output_index"] += 1
                    
                    # 新的 function_call
                    func_output_idx = cls._stream_state["output_index"]
                    cls._stream_state["started_function_ids"].add(tc_id)
                    cls._stream_state[f"_func_idx_{tc_id}"] = func_output_idx
                    
                    events.append({
                        "type": "response.output_item.added",
                        "output_index": func_output_idx,
                        "item": {
                            "type": "function_call",
                            "id": tc_id,
                            "call_id": tc_id,
                            "name": tc_func.get("name", ""),
                            "arguments": "",
                            "status": "in_progress"
                        }
                    })
                
                # function_call_arguments delta
                args_part = tc_func.get("arguments", "")
                if args_part and tc_id:
                    func_output_idx = cls._stream_state.get(f"_func_idx_{tc_id}", tc_idx)
                    events.append({
                        "type": "response.function_call_arguments.delta",
                        "output_index": func_output_idx,
                        "delta": args_part
                    })
            
            return events
        
        # 4. 消息结束
        if finish_reason:
            output_idx = cls._stream_state["output_index"]
            
            # 关闭 reasoning 项
            if cls._stream_state["reasoning_item_started"]:
                # 先发 response.reasoning_text.done 事件
                events.append({
                    "type": "response.reasoning_text.done",
                    "output_index": output_idx,
                    "text": ""  # 完整文本在流式场景中无法精确获取
                })
                events.append({
                    "type": "response.output_item.done",
                    "output_index": output_idx,
                    "item": {
                        "type": "reasoning",
                        "id": f"rs_{uuid.uuid4().hex[:24]}",
                        "status": "completed",
                        "content": []
                    }
                })
                cls._stream_state["reasoning_item_started"] = False
                cls._stream_state["output_index"] += 1
                output_idx = cls._stream_state["output_index"]
            
            # 关闭消息项
            if cls._stream_state["message_item_started"]:
                if cls._stream_state["content_part_started"]:
                    events.append({
                        "type": "response.output_text.done",
                        "output_index": output_idx,
                        "content_index": 0,
                        "text": ""
                    })
                    events.append({
                        "type": "response.content_part.done",
                        "output_index": output_idx,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": "",
                            "annotations": []
                        }
                    })
                events.append({
                    "type": "response.output_item.done",
                    "output_index": output_idx,
                    "item": {
                        "type": "message",
                        "id": f"msg_{uuid.uuid4().hex[:24]}",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "", "annotations": []}]
                    }
                })
            
            # 关闭 function_call 项
            for tc_id in cls._stream_state["started_function_ids"]:
                func_output_idx = cls._stream_state.get(f"_func_idx_{tc_id}", 1)
                events.append({
                    "type": "response.function_call_arguments.done",
                    "output_index": func_output_idx,
                    "arguments": ""
                })
                events.append({
                    "type": "response.output_item.done",
                    "output_index": func_output_idx,
                    "item": {
                        "type": "function_call",
                        "id": tc_id,
                        "call_id": tc_id,
                        "status": "completed"
                    }
                })
            
            # response.completed
            usage = chunk.get("usage", {})
            response_status = "completed"
            incomplete_details = None
            error_detail = None
            
            if finish_reason == "length":
                response_status = "incomplete"
                incomplete_details = {"reason": "max_output_tokens"}
            elif finish_reason == "content_filter":
                # content_filter 可以映射为 incomplete 或 failed
                # 对于安全性过滤，使用 failed 状态更准确
                response_status = "failed"
                error_detail = {"code": "content_filter", "message": "Content filtered by safety system"}
            
            completed_response = {
                "id": chunk.get("id", ""),
                "object": "response",
                "status": response_status,
                "model": chunk.get("model", ""),
                "output": [],
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                }
            }
            if incomplete_details:
                completed_response["incomplete_details"] = incomplete_details
            
            # 根据状态选择完成、失败或不完整事件类型
            if response_status == "failed":
                if error_detail:
                    completed_response["error"] = error_detail
                event_type = "response.failed"
            elif response_status == "incomplete":
                event_type = "response.incomplete"
            else:
                event_type = "response.completed"
            
            result = {
                "type": event_type,
                "response": completed_response
            }
            
            events.append(result)
            return events
        
        return [{"type": "response.ping"}]
    
    # ================================================================
    # Responses 事件 -> Chat 流式块 (用于 Responses 后端流式转换)
    # ================================================================

    @classmethod
    def _convert_responses_event_to_chat_chunk(cls, event_type: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        将 Responses SSE 事件转换为 OpenAI Chat 流式块
        
        用于 Responses 后端流式响应 → Chat/Anthropic 目标协议的转换场景。
        
        Args:
            event_type: Responses SSE 事件类型
            data: 事件数据
            
        Returns:
            Optional[Dict]: OpenAI Chat 格式的流式块，如果事件不需要转换则返回 None
        """
        if event_type == "response.output_text.delta":
            # 文本增量
            return {
                "id": data.get("response", {}).get("id", ""),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": data.get("response", {}).get("model", ""),
                "choices": [{
                    "index": 0,
                    "delta": {"content": data.get("delta", "")},
                    "finish_reason": None
                }]
            }
        
        elif event_type == "response.output_text.done":
            # 文本完成 - 不需要单独的 Chat 块
            return None
        
        elif event_type == "response.function_call_arguments.delta":
            # 函数调用参数增量
            tc_index = data.get("output_index", 0)
            return {
                "id": data.get("response", {}).get("id", ""),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": data.get("response", {}).get("model", ""),
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": tc_index,
                            "function": {"arguments": data.get("delta", "")}
                        }]
                    },
                    "finish_reason": None
                }]
            }
        
        elif event_type == "response.output_item.added":
            # 输出项添加
            item = data.get("item", {})
            item_type = item.get("type", "")
            
            if item_type == "message":
                # 消息项开始 - 发送 role delta
                return {
                    "id": data.get("response", {}).get("id", ""),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": data.get("response", {}).get("model", ""),
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None
                    }]
                }
            elif item_type == "function_call":
                # 函数调用开始
                return {
                    "id": data.get("response", {}).get("id", ""),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": data.get("response", {}).get("model", ""),
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": data.get("output_index", 0),
                                "id": item.get("id", item.get("call_id", "")),
                                "type": "function",
                                "function": {
                                    "name": item.get("name", ""),
                                    "arguments": ""
                                }
                            }]
                        },
                        "finish_reason": None
                    }]
                }
            elif item_type == "reasoning":
                # 推理项开始 - 不需要单独的 Chat 块
                # 推理内容通过 response.reasoning_text.delta 事件发送
                return {
                    "id": data.get("response", {}).get("id", ""),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": data.get("response", {}).get("model", ""),
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None
                    }]
                }
        
        elif event_type == "response.reasoning_text.delta":
            # 推理文本增量
            return {
                "id": data.get("response", {}).get("id", ""),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": data.get("response", {}).get("model", ""),
                "choices": [{
                    "index": 0,
                    "delta": {"reasoning_content": data.get("delta", "")},
                    "finish_reason": None
                }]
            }
        
        elif event_type == "response.reasoning_summary_text.delta":
            # 推理摘要文本增量 - 映射为 reasoning_content
            return {
                "id": data.get("response", {}).get("id", ""),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": data.get("response", {}).get("model", ""),
                "choices": [{
                    "index": 0,
                    "delta": {"reasoning_content": data.get("delta", "")},
                    "finish_reason": None
                }]
            }
        
        elif event_type in ("response.completed", "response.failed", "response.incomplete"):
            # 响应完成/失败/不完整
            resp_data = data.get("response", {})
            status = resp_data.get("status", "completed")
            finish_reason = "stop"
            if status == "incomplete":
                details = resp_data.get("incomplete_details", {})
                reason = details.get("reason", "") if details else ""
                if reason == "max_output_tokens":
                    finish_reason = "length"
                elif reason == "content_filter":
                    finish_reason = "content_filter"
            elif status == "failed":
                finish_reason = "stop"
            
            usage = resp_data.get("usage", {})
            chunk = {
                "id": resp_data.get("id", ""),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": resp_data.get("model", ""),
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason
                }],
                "usage": {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
            }
            return chunk
        
        elif event_type == "response.refusal.delta":
            # 拒绝内容增量 - 映射为 Chat content delta
            return {
                "id": data.get("response", {}).get("id", ""),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": data.get("response", {}).get("model", ""),
                "choices": [{
                    "index": 0,
                    "delta": {"content": data.get("delta", "")},
                    "finish_reason": None
                }]
            }
        
        return None
