"""
OpenAI Responses API 协议转换器

参考: https://platform.openai.com/docs/api-reference/responses
参考 SDK: openai-python ResponseCreateParams

OpenAI Responses API 请求参数:
- model (必填): 模型名称
- input (必填): string | ResponseInputParam[] (文本/图片/文件输入)
- instructions: 系统提示词
- tools: 工具定义 (function | web_search | file_search | computer | mcp | code_interpreter | image_generation | local_shell | custom)
- tool_choice: 工具选择策略 (auto | none | required | function | mcp | custom | apply_patch | shell)
- temperature: 温度 (0-2)
- top_p: nucleus 采样
- max_output_tokens: 最大输出 token 数
- stream: 是否流式
- previous_response_id: 上一个响应 ID (用于多轮对话)
- store: 是否存储响应
- metadata: 元数据
- reasoning: 推理配置 (o系列模型)
- text: 文本输出配置 (structured outputs)
- parallel_tool_calls: 是否允许并行工具调用
- truncation: 截断策略 ("auto" | "disabled")
- service_tier: 服务层级
- background: 是否后台运行
- max_tool_calls: 最大工具调用次数
- conversation: 对话参数
- context_management: 上下文管理
- include: 额外输出数据
- prompt: 提示模板
- safety_identifier: 安全标识符
- prompt_cache_key: 缓存键
- user: 用户标识符

OpenAI Responses 输入项类型:
- message: 消息 {"type": "message", "role", "content": [...]}
- function_call: 函数调用 {"type": "function_call", "call_id", "name", "arguments"}
- function_call_output: 函数调用结果 {"type": "function_call_output", "call_id", "output"}

OpenAI Responses 输出项类型:
- message: 消息 {"type": "message", "role", "content": [...]}
- function_call: 函数调用
- web_search_call: 网页搜索调用
- file_search_call: 文件搜索调用
- computer_call: 计算机调用
- reasoning: 推理项

OpenAI Responses 流式事件:
- response.created
- response.in_progress
- response.output_item.added
- response.content_part.added
- response.output_text.delta
- response.output_text.done
- response.function_call_arguments.delta
- response.function_call_arguments.done
- response.output_item.done
- response.completed
- response.failed
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
        """
        messages = []
        
        # 1. 处理 input（可以是字符串或数组）
        input_data = request.get("input", [])
        if isinstance(input_data, str):
            # 简单字符串输入 -> 单条 user 消息
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, list):
            for item in input_data:
                msg = cls._convert_input_item_to_message(item)
                if msg:
                    # 可能返回多条消息（例如 tool 消息）
                    if isinstance(msg, list):
                        messages.extend(msg)
                    else:
                        messages.append(msg)
        
        # 2. 处理 instructions（相当于 system/developer prompt）
        instructions = request.get("instructions")
        if instructions:
            messages.insert(0, {
                "role": "developer",
                "content": instructions if isinstance(instructions, str) else ""
            })
        
        # 3. 转换 tools
        tools = cls._convert_tools(request.get("tools", []))
        
        # 4. 构建 Chat 请求
        chat_request = {
            "model": request.get("model", "gpt-4o"),
            "messages": messages,
        }
        
        # 5. 映射可选参数
        if request.get("stream") is not None:
            chat_request["stream"] = request["stream"]
        if request.get("temperature") is not None:
            chat_request["temperature"] = request["temperature"]
        if request.get("top_p") is not None:
            chat_request["top_p"] = request["top_p"]
        if request.get("max_output_tokens"):
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
        
        # 7. 处理 store 参数
        if request.get("store") is not None:
            chat_request["store"] = request["store"]
        
        # 8. 处理 reasoning 参数 -> reasoning_effort
        reasoning = request.get("reasoning")
        if reasoning and isinstance(reasoning, dict):
            effort = reasoning.get("effort")
            if effort:
                chat_request["reasoning_effort"] = effort
            # 保留 reasoning 的 summary 配置
            summary = reasoning.get("summary")
            if summary is not None:
                if "extra_body" not in chat_request:
                    chat_request["extra_body"] = {}
                chat_request["extra_body"]["reasoning"] = reasoning
        
        # 9. 处理 text 参数 (structured outputs) -> response_format
        text_config = request.get("text")
        if text_config and isinstance(text_config, dict):
            format_config = text_config.get("format")
            if format_config and isinstance(format_config, dict):
                fmt_type = format_config.get("type")
                if fmt_type == "json_schema":
                    chat_request["response_format"] = {
                        "type": "json_schema",
                        "json_schema": format_config.get("json_schema", format_config)
                    }
                elif fmt_type == "json_object":
                    chat_request["response_format"] = {"type": "json_object"}
        
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
        if request.get("safety_identifier"):
            extra["safety_identifier"] = request["safety_identifier"]
        if request.get("prompt_cache_key"):
            extra["prompt_cache_key"] = request["prompt_cache_key"]
        if request.get("stream_options"):
            extra["stream_options"] = request["stream_options"]
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
            return {
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": str(item.get("output", ""))
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
        
        # 未知类型，尝试作为消息处理
        if "role" in item and "content" in item:
            role = cls.ROLE_MAP.get(item.get("role", ""), item.get("role", "user"))
            return {"role": role, "content": str(item.get("content", ""))}
        
        return None
    
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
                if image_url:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })
                else:
                    # base64 图片
                    data_url = item.get("image_url", item.get("data", ""))
                    if data_url:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        })
            
            elif item_type == "input_file":
                has_multimodal = True
                file_data = item.get("file_data", item.get("file_url", ""))
                filename = item.get("filename", "")
                if file_data:
                    content_parts.append({
                        "type": "file",
                        "file": {
                            "mime_type": item.get("mime_type", "application/octet-stream"),
                            "file_data": file_data
                        }
                    })
                elif filename:
                    text_parts.append(f"[File: {filename}]")
            
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
            return content_parts if content_parts else "\n".join(text_parts)
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
        
        Chat 工具类型: function, web_search (web_search_options)
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
            
            elif tool_type == "web_search":
                # Chat API 支持 web_search_options
                ws_opts = tool.get("user_location", {})
                search_context_size = tool.get("search_context_size", "medium")
                converted.append({
                    "type": "web_search",
                    "web_search_options": {
                        "search_context_size": search_context_size,
                        **({"user_location": ws_opts} if ws_opts else {})
                    }
                })
            
            elif tool_type == "web_search_preview":
                # 预览版网页搜索
                ws_opts = tool.get("user_location", {})
                search_context_size = tool.get("search_context_size", "medium")
                converted.append({
                    "type": "web_search",
                    "web_search_options": {
                        "search_context_size": search_context_size,
                        **({"user_location": ws_opts} if ws_opts else {})
                    }
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
    
    # ================================================================
    # 请求转换：Chat -> Responses
    # ================================================================
    
    @classmethod
    def from_openai_chat_request(cls, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI Chat 请求转换为 OpenAI Responses 格式
        """
        messages = request.get("messages", [])
        
        # 提取简单文本输入（用于 OpenRouter 等兼容 API）
        simple_texts = []
        has_complex = False
        instructions = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system" or role == "developer":
                instructions = content if isinstance(content, str) else ""
            elif role == "user" and isinstance(content, str):
                simple_texts.append(content)
            elif role == "assistant" and msg.get("tool_calls"):
                has_complex = True
            elif role == "tool":
                has_complex = True
            elif role not in ("user",) or not isinstance(content, str):
                has_complex = True
        
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
                    if content:
                        input_items.append({
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": content}]
                        })
                    for tc in msg["tool_calls"]:
                        input_items.append({
                            "type": "function_call",
                            "call_id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": tc.get("function", {}).get("arguments", "{}")
                        })
                elif role == "tool":
                    input_items.append({
                        "type": "function_call_output",
                        "call_id": msg.get("tool_call_id", ""),
                        "output": content if isinstance(content, str) else str(content)
                    })
                elif isinstance(content, str):
                    input_items.append({
                        "type": "message",
                        "role": role,
                        "content": [{"type": "input_text", "text": content}]
                    })
                elif isinstance(content, list):
                    # 多模态内容
                    content_items = cls._convert_chat_content_to_responses(content)
                    input_items.append({
                        "type": "message",
                        "role": role,
                        "content": content_items
                    })
                else:
                    input_items.append({
                        "type": "message",
                        "role": role,
                        "content": [{"type": "input_text", "text": str(content)}]
                    })
            input_data = input_items
        else:
            input_data = ""
        
        # 构建 Responses 请求
        responses_request = {
            "model": request.get("model", "gpt-4o"),
            "input": input_data,
        }
        
        if instructions:
            responses_request["instructions"] = instructions
        
        # 可选参数映射
        if request.get("temperature") is not None:
            responses_request["temperature"] = request["temperature"]
        if request.get("top_p") is not None:
            responses_request["top_p"] = request["top_p"]
        if request.get("max_completion_tokens"):
            responses_request["max_output_tokens"] = request["max_completion_tokens"]
        elif request.get("max_tokens"):
            responses_request["max_output_tokens"] = request["max_tokens"]
        if request.get("stream"):
            responses_request["stream"] = request["stream"]
        if request.get("tools"):
            responses_request["tools"] = cls._convert_chat_tools_to_responses(request["tools"])
        if request.get("tool_choice"):
            responses_request["tool_choice"] = cls._convert_chat_tool_choice_to_responses(request["tool_choice"])
        if request.get("parallel_tool_calls") is not None:
            responses_request["parallel_tool_calls"] = request["parallel_tool_calls"]
        if request.get("store") is not None:
            responses_request["store"] = request["store"]
        if request.get("metadata"):
            responses_request["metadata"] = request["metadata"]
        
        # reasoning_effort -> reasoning
        if request.get("reasoning_effort"):
            responses_request["reasoning"] = {"effort": request["reasoning_effort"]}
        
        # response_format -> text
        if request.get("response_format"):
            fmt = request["response_format"]
            fmt_type = fmt.get("type", "")
            if fmt_type == "json_schema":
                responses_request["text"] = {
                    "format": {
                        "type": "json_schema",
                        "json_schema": fmt.get("json_schema", {})
                    }
                }
            elif fmt_type == "json_object":
                responses_request["text"] = {"format": {"type": "json_object"}}
        
        # service_tier
        if request.get("service_tier"):
            responses_request["service_tier"] = request["service_tier"]
        
        # user -> safety_identifier (新字段)
        if request.get("user"):
            responses_request["user"] = request["user"]
        
        return responses_request
    
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
                result.append({
                    "type": "input_file",
                    "file_data": file_data.get("file_data", ""),
                    "filename": file_data.get("filename", "")
                })
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
            elif tool_type == "web_search":
                ws_opts = tool.get("web_search_options", {})
                rt = {"type": "web_search"}
                if ws_opts.get("search_context_size"):
                    rt["search_context_size"] = ws_opts["search_context_size"]
                if ws_opts.get("user_location"):
                    rt["user_location"] = ws_opts["user_location"]
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
        
        return tool_choice
    
    # ================================================================
    # 响应转换：Chat Response -> Responses Response
    # ================================================================
    
    @classmethod
    def from_openai_chat(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI Chat 响应转换为 Responses 格式
        
        Chat: {"id", "model", "choices": [{"message": {"content", "tool_calls"}, "finish_reason"}], "usage"}
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
            if content:
                msg_content = []
                msg_content.append({
                    "type": "output_text",
                    "text": content,
                    "annotations": message.get("annotations", [])
                })
                
                output.append({
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "status": "completed",
                    "role": "assistant",
                    "content": msg_content
                })
            
            # 处理 refusal
            refusal = message.get("refusal")
            if refusal:
                status = "incomplete"
                incomplete_details = {"reason": "content_filter"}
            
            # 检查是否不完整
            if finish_reason == "length":
                status = "incomplete"
                incomplete_details = {"reason": "max_output_tokens"}
            elif finish_reason == "content_filter":
                status = "incomplete"
                incomplete_details = {"reason": "content_filter"}
        
        # 转换 usage
        usage = response.get("usage", {})
        
        result = {
            "id": response.get("id", f"resp_{uuid.uuid4().hex[:24]}"),
            "object": "response",
            "status": status,
            "created_at": response.get("created", int(time.time())),
            "model": response.get("model", ""),
            "output": output,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            },
            "incomplete_details": incomplete_details,
            "error": None,
            "metadata": response.get("metadata"),
        }
        
        # 添加可选字段（如果 Chat 响应中存在）
        if response.get("service_tier"):
            result["service_tier"] = response["service_tier"]
        if response.get("system_fingerprint"):
            result["system_fingerprint"] = response["system_fingerprint"]
        
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
    # 流式转换
    # ================================================================
    
    @classmethod
    def convert_stream_chunk(cls, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换 OpenAI Chat 流式块到 Responses 流式格式
        
        Responses 流式事件（完整列表）:
        - response.created: 响应创建
        - response.in_progress: 响应进行中
        - response.output_item.added: 输出项添加
        - response.content_part.added: 内容部分添加
        - response.output_text.delta: 文本增量
        - response.output_text.done: 文本完成
        - response.output_item.done: 输出项完成
        - response.function_call_arguments.delta: 函数调用参数增量
        - response.function_call_arguments.done: 函数调用参数完成
        - response.completed: 响应完成
        - response.failed: 响应失败
        - response.ping: 心跳
        """
        choices = chunk.get("choices", [])
        
        if not choices:
            return {"type": "response.ping"}
        
        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")
        content = delta.get("content", "")
        
        if delta.get("role") == "assistant":
            return {
                "type": "response.created",
                "response": {
                    "id": chunk.get("id", ""),
                    "object": "response",
                    "status": "in_progress",
                    "model": chunk.get("model", ""),
                    "output": [],
                }
            }
        
        if content:
            return {
                "type": "response.output_text.delta",
                "content_index": 0,
                "output_index": 0,
                "delta": content
            }
        
        tool_calls = delta.get("tool_calls", [])
        if tool_calls:
            tc = tool_calls[0]
            tc_func = tc.get("function", {})
            
            # 如果有 id，是 function_call 开始
            if tc.get("id"):
                return {
                    "type": "response.output_item.added",
                    "output_index": tc.get("index", 0),
                    "item": {
                        "type": "function_call",
                        "id": tc["id"],
                        "call_id": tc["id"],
                        "name": tc_func.get("name", ""),
                        "arguments": "",
                        "status": "in_progress"
                    }
                }
            
            # 否则是 function_call_arguments delta
            args_part = tc_func.get("arguments", "")
            if args_part:
                return {
                    "type": "response.function_call_arguments.delta",
                    "output_index": tc.get("index", 0),
                    "delta": args_part
                }
        
        if finish_reason:
            usage = chunk.get("usage", {})
            response_status = "completed"
            incomplete_details = None
            
            if finish_reason == "length":
                response_status = "incomplete"
                incomplete_details = {"reason": "max_output_tokens"}
            elif finish_reason == "content_filter":
                response_status = "incomplete"
                incomplete_details = {"reason": "content_filter"}
            
            result = {
                "type": "response.completed",
                "response": {
                    "id": chunk.get("id", ""),
                    "object": "response",
                    "status": response_status,
                    "output": [],
                    "usage": {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)
                    }
                }
            }
            if incomplete_details:
                result["response"]["incomplete_details"] = incomplete_details
            
            return result
        
        return {"type": "response.ping"}
