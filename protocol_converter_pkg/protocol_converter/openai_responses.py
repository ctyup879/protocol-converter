"""
OpenAI Responses API 协议转换器

参考: https://platform.openai.com/docs/api-reference/responses
参考 SDK: openai-python ResponseCreateParams
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
    
    # ================================================================
    # 请求转换：Responses -> Chat
    # ================================================================
    
    @classmethod
    def to_openai_chat(cls, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI Responses 请求转换为 OpenAI Chat 格式
        
        Responses 请求字段:
        - model (必填)
        - input: string | ResponseInputParam[] (必填)
        - instructions: string
        - tools: ToolParam[]
        - tool_choice: ToolChoice
        - temperature, top_p
        - max_output_tokens
        - stream
        - previous_response_id
        - store, truncation, metadata 等
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
                "role": "system",
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
        if request.get("metadata"):
            chat_request["user"] = request["metadata"].get("user_id", "")
        if request.get("store") is not None:
            chat_request["store"] = request["store"]
        if request.get("truncation"):
            # Chat API 没有 truncation 参数，但可以映射到 max_tokens 限制
            pass
        
        return chat_request
    
    @classmethod
    def _convert_input_item_to_message(cls, item: Dict[str, Any]) -> Optional[Union[Dict, List[Dict]]]:
        """转换 Responses 输入项到 Chat 消息格式"""
        item_type = item.get("type", "")
        
        if item_type == "message":
            role = cls.ROLE_MAP.get(item.get("role", ""), item.get("role", "user"))
            content = item.get("content", [])
            text = cls._convert_content_to_text(content)
            return {"role": role, "content": text}
        
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
                "content": f"[Computer screenshot output]"
            }
        
        return None
    
    @classmethod
    def _convert_content_to_text(cls, content: Union[str, List[Dict]]) -> str:
        """转换内容为纯文本"""
        if isinstance(content, str):
            return content
        
        text_parts = []
        for item in content:
            item_type = item.get("type", "")
            if item_type in ("text", "input_text"):
                text_parts.append(item.get("text", ""))
            elif item_type == "input_image":
                url = item.get("image_url", "")
                text_parts.append(f"[Image: {url}]" if url else "[Image]")
            elif item_type == "input_file":
                text_parts.append(f"[File: {item.get('filename', '')}]")
        
        return "\n".join(text_parts)
    
    @classmethod
    def _convert_tools(cls, tools: List[Dict]) -> List[Dict]:
        """
        转换 Responses 工具定义到 Chat 格式
        
        Responses 工具类型: function, web_search, file_search, computer, mcp 等
        Chat 工具类型: function (主要)
        """
        converted = []
        for tool in tools:
            tool_type = tool.get("type", "function")
            
            if tool_type == "function":
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
            # web_search, file_search 等内置工具 Chat API 也支持
            elif tool_type == "web_search":
                converted.append({"type": "web_search", "web_search_options": tool.get("web_search_options", {})})
        
        return converted
    
    @classmethod
    def _convert_tool_choice(cls, tool_choice: Any) -> Any:
        """
        转换 Responses tool_choice 到 Chat 格式
        
        Responses: "auto" | "none" | "required" | {"type": "function", "name": "..."} | ...
        Chat: "auto" | "none" | "required" | {"type": "function", "function": {"name": "..."}}
        """
        if isinstance(tool_choice, str):
            return tool_choice  # auto, none, required 直接传递
        
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                return {
                    "type": "function",
                    "function": {"name": tool_choice.get("name", "")}
                }
        
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
            
            if role == "system":
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
                
                if role == "system":
                    continue
                
                if role == "assistant" and msg.get("tool_calls"):
                    # assistant + tool_calls -> function_call items
                    if content:
                        input_items.append({
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "text", "text": content}]
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
                        "content": [{"type": "text", "text": content}]
                    })
                else:
                    input_items.append({
                        "type": "message",
                        "role": role,
                        "content": content
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
        
        # 可选参数
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
            responses_request["tools"] = request["tools"]
        if request.get("tool_choice"):
            responses_request["tool_choice"] = request["tool_choice"]
        if request.get("store") is not None:
            responses_request["store"] = request["store"]
        if request.get("metadata"):
            responses_request["metadata"] = request["metadata"]
        
        return responses_request
    
    # ================================================================
    # 响应转换：Chat Response -> Responses Response
    # ================================================================
    
    @classmethod
    def from_openai_chat(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI Chat 响应转换为 Responses 格式
        
        Chat: {"id", "model", "choices": [{"message": {"content", "tool_calls"}, "finish_reason"}], "usage"}
        Responses: {"id", "object", "status", "created_at", "model", "output": [...], "usage"}
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
            
            # 检查是否不完整
            if finish_reason == "length":
                status = "incomplete"
                incomplete_details = {"reason": "max_output_tokens"}
            elif finish_reason == "content_filter":
                status = "incomplete"
                incomplete_details = {"reason": "content_filter"}
        
        # 转换 usage
        usage = response.get("usage", {})
        
        return {
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
            "error": None
        }
    
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
        
        Responses 流式事件:
        - response.created
        - response.in_progress
        - response.output_item.added
        - response.content_part.added
        - response.output_text.delta
        - response.output_text.done
        - response.output_item.done
        - response.completed
        """
        choices = chunk.get("choices", [])
        
        if not choices:
            return {"type": "response.completed", "response": {}}
        
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
                    "model": chunk.get("model", "")
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
            return {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": tc.get("function", {}).get("arguments", "")
            }
        
        if finish_reason:
            return {
                "type": "response.completed",
                "response": {
                    "id": chunk.get("id", ""),
                    "status": "completed"
                }
            }
        
        return {"type": "response.ping"}
