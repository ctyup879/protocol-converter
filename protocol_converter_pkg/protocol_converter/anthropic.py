"""
Anthropic Messages API 协议转换器

参考: https://docs.anthropic.com/en/api/messages
参考 SDK: anthropic-sdk-python
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union, Iterator


class AnthropicConverter:
    """Anthropic Messages API 协议转换器"""
    
    PROTOCOL_NAME = "anthropic"
    
    # Anthropic -> OpenAI 角色映射
    ROLE_MAP = {
        "user": "user",
        "assistant": "assistant",
    }
    
    # OpenAI -> Anthropic 停止原因映射
    STOP_REASON_MAP = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "refusal",
        "function_call": "end_turn",
    }
    
    # Anthropic -> OpenAI 停止原因映射
    REVERSE_STOP_REASON_MAP = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
        "pause_turn": "stop",
        "refusal": "content_filter",
    }
    
    # Anthropic tool_choice -> OpenAI tool_choice 映射
    TOOL_CHOICE_MAP = {
        "auto": "auto",
        "any": "required",
        "none": "none",
    }
    
    # ================================================================
    # 请求转换：Anthropic -> OpenAI Chat
    # ================================================================
    
    @classmethod
    def to_openai_chat(cls, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anthropic Messages 请求转换为 OpenAI Chat 格式
        
        Args:
            request: Anthropic Messages 格式请求
            {"model", "max_tokens", "messages", "system", "tools", 
             "tool_choice", "temperature", "top_p", "top_k", 
             "stop_sequences", "stream", "thinking", "metadata"}
            
        Returns:
            Dict: OpenAI Chat 格式请求
        """
        messages = []
        
        # 1. 处理 system prompt（Anthropic 顶级参数 -> OpenAI system 消息）
        system_prompt = request.get("system")
        if system_prompt:
            if isinstance(system_prompt, str):
                messages.append({"role": "system", "content": system_prompt})
            elif isinstance(system_prompt, list):
                text_parts = []
                for block in system_prompt:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                if text_parts:
                    messages.append({"role": "system", "content": "\n".join(text_parts)})
        
        # 2. 处理 messages
        anthropic_messages = request.get("messages", [])
        for msg in anthropic_messages:
            converted_list = cls._convert_message(msg)
            messages.extend(converted_list)
        
        # 3. 转换 tools
        tools = cls._convert_tools(request.get("tools", []))
        
        # 4. 转换 tool_choice
        tool_choice = cls._convert_tool_choice(request.get("tool_choice"))
        
        # 5. 构建 Chat 请求
        chat_request = {
            "model": request.get("model", "claude-sonnet-4-20250514"),
            "messages": messages,
        }
        
        # 6. 映射参数
        if request.get("stream") is not None:
            chat_request["stream"] = request["stream"]
        if request.get("temperature") is not None:
            chat_request["temperature"] = request["temperature"]
        if request.get("top_p") is not None:
            chat_request["top_p"] = request["top_p"]
        if request.get("max_tokens") is not None:
            # Anthropic max_tokens -> OpenAI max_tokens (注意: OpenAI 新版推荐 max_completion_tokens)
            chat_request["max_tokens"] = request["max_tokens"]
        if tools:
            chat_request["tools"] = tools
        if tool_choice is not None:
            chat_request["tool_choice"] = tool_choice
        if request.get("stop_sequences"):
            chat_request["stop"] = request["stop_sequences"]
        if request.get("metadata"):
            chat_request["user"] = request["metadata"].get("user_id", "")
        
        # 7. Anthropic 特有参数（OpenAI 不直接支持的）
        extra = {}
        if request.get("top_k") is not None:
            extra["top_k"] = request["top_k"]
        if request.get("thinking"):
            extra["thinking"] = request["thinking"]
        if extra:
            chat_request["extra_body"] = extra
        
        return chat_request
    
    @classmethod
    def _convert_message(cls, msg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        转换单条 Anthropic 消息，可能返回多条 OpenAI 消息
        
        Anthropic assistant 消息可能包含 tool_use 块，需要拆分：
        - text 块 -> content
        - tool_use 块 -> tool_calls
        Anthropic user 消息可能包含 tool_result 块，需要拆分为 tool 消息
        """
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role not in cls.ROLE_MAP and role != "user":
            return []
        
        openai_role = cls.ROLE_MAP.get(role, role)
        result = []
        
        if isinstance(content, str):
            # 简单字符串内容
            result.append({"role": openai_role, "content": content})
        
        elif isinstance(content, list):
            # 内容块数组 - 需要分类处理
            text_parts = []
            tool_calls = []
            tool_results = []
            
            for block in content:
                block_type = block.get("type", "")
                
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                
                elif block_type == "tool_use":
                    # assistant 消息中的工具调用
                    tool_calls.append({
                        "id": block.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {}), ensure_ascii=False)
                        }
                    })
                
                elif block_type == "tool_result":
                    # user 消息中的工具结果
                    tool_results.append(block)
                
                elif block_type == "thinking":
                    # thinking 块 - OpenAI 不支持，跳过或记录
                    pass
                
                elif block_type == "image":
                    # 图片块 - 转为 OpenAI image_url 格式
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        media_type = source.get("media_type", "image/png")
                        data = source.get("data", "")
                        text_parts.append(f"[Image: base64 {media_type}]")
                    elif source.get("type") == "url":
                        text_parts.append(f"[Image: {source.get('url', '')}]")
            
            # 构建 assistant 消息（可能带 tool_calls）
            if role == "assistant":
                assistant_msg = {"role": "assistant"}
                if text_parts:
                    assistant_msg["content"] = "\n".join(text_parts)
                else:
                    assistant_msg["content"] = None
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                result.append(assistant_msg)
            
            # 构建 user 消息
            elif role == "user":
                if text_parts and not tool_results:
                    result.append({"role": "user", "content": "\n".join(text_parts)})
                elif text_parts and tool_results:
                    result.append({"role": "user", "content": "\n".join(text_parts)})
                
                # tool_result -> role=tool 消息
                for tr in tool_results:
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                    }
                    tr_content = tr.get("content", "")
                    if isinstance(tr_content, str):
                        tool_msg["content"] = tr_content
                    elif isinstance(tr_content, list):
                        # 提取文本
                        text_list = []
                        for b in tr_content:
                            if b.get("type") == "text":
                                text_list.append(b.get("text", ""))
                        tool_msg["content"] = "\n".join(text_list)
                    else:
                        tool_msg["content"] = str(tr_content)
                    result.append(tool_msg)
        
        else:
            result.append({"role": openai_role, "content": str(content)})
        
        return result
    
    @classmethod
    def _convert_tools(cls, tools: List[Dict]) -> List[Dict]:
        """
        转换 Anthropic 工具定义到 OpenAI 格式
        
        Anthropic: {"name", "description", "input_schema"}
        OpenAI: {"type": "function", "function": {"name", "description", "parameters"}}
        """
        converted = []
        for tool in tools:
            if "name" in tool:
                func_def = {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                }
                # Anthropic 用 input_schema, OpenAI 用 parameters
                if "input_schema" in tool:
                    func_def["parameters"] = tool["input_schema"]
                converted.append({
                    "type": "function",
                    "function": func_def
                })
        return converted
    
    @classmethod
    def _convert_tool_choice(cls, tool_choice: Any) -> Optional[Any]:
        """
        转换 Anthropic tool_choice 到 OpenAI 格式
        
        Anthropic: "auto" | "any" | "none" | {"type": "tool", "name": "..."}
        OpenAI: "auto" | "none" | "required" | {"type": "function", "function": {"name": "..."}}
        """
        if tool_choice is None:
            return None
        
        if isinstance(tool_choice, str):
            return cls.TOOL_CHOICE_MAP.get(tool_choice, tool_choice)
        
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "tool":
                return {
                    "type": "function",
                    "function": {"name": tool_choice.get("name", "")}
                }
            if tool_choice.get("type") == "auto":
                return "auto"
            if tool_choice.get("type") == "any":
                return "required"
        
        return None
    
    # ================================================================
    # 响应转换：OpenAI Chat -> Anthropic
    # ================================================================
    
    @classmethod
    def from_openai_chat(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI Chat 响应转换为 Anthropic 格式
        
        OpenAI: {"id", "model", "choices": [{"message": {"content", "tool_calls"}, "finish_reason"}], "usage"}
        Anthropic: {"id", "type", "role", "content": [{"type": "text"}|{"type": "tool_use"}], 
                    "model", "stop_reason", "stop_sequence", "usage"}
        """
        choices = response.get("choices", [])
        content_blocks = []
        stop_reason = "end_turn"
        
        for choice in choices:
            message = choice.get("message", {})
            content = message.get("content")
            tool_calls = message.get("tool_calls", [])
            
            # 处理文本内容
            if content:
                content_blocks.append({
                    "type": "text",
                    "text": content
                })
            
            # 处理工具调用
            for tc in tool_calls:
                tc_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    args_obj = json.loads(args_str)
                except (json.JSONDecodeError, TypeError):
                    args_obj = {}
                
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc_id,
                    "name": func.get("name", ""),
                    "input": args_obj
                })
            
            # 获取停止原因
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                stop_reason = cls.STOP_REASON_MAP.get(finish_reason, "end_turn")
        
        # 如果没有任何内容块，添加空文本
        if not content_blocks:
            content_blocks.append({"type": "text", "text": ""})
        
        # 转换 usage
        usage = response.get("usage", {})
        
        return {
            "id": response.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": response.get("model", ""),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0
            }
        }
    
    @classmethod
    def _map_stop_reason(cls, openai_reason: str) -> str:
        """映射 OpenAI 停止原因到 Anthropic"""
        return cls.STOP_REASON_MAP.get(openai_reason, "end_turn")
    
    # ================================================================
    # 流式转换：OpenAI Chat Stream -> Anthropic SSE
    # ================================================================
    
    @classmethod
    def convert_stream_chunk(cls, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换 OpenAI Chat 流式块到 Anthropic SSE 格式
        
        Anthropic SSE 事件类型:
        - message_start: 包含完整 message 对象（id, type, role, model, usage）
        - content_block_start: 新内容块开始 {"type": "text"|"tool_use", ...}
        - content_block_delta: 内容增量 {"type": "text_delta"|"input_json_delta", ...}
        - content_block_stop: 内容块结束
        - message_delta: 消息级增量 (stop_reason, usage)
        - message_stop: 消息结束
        """
        choices = chunk.get("choices", [])
        
        if not choices:
            return {"type": "message_stop"}
        
        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")
        
        # 判断事件类型
        if delta.get("role") == "assistant":
            # 第一个块 - message_start
            return {
                "type": "message_start",
                "message": {
                    "id": chunk.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": chunk.get("model", ""),
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0}
                }
            }
        
        if delta.get("content") is not None and delta["content"] != "":
            # 文本增量
            return {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": delta["content"]
                }
            }
        
        tool_calls = delta.get("tool_calls", [])
        if tool_calls:
            tc = tool_calls[0]
            tc_idx = tc.get("index", 0)
            tc_func = tc.get("function", {})
            
            # 如果有 id 和 name，是 tool_use 开始
            if tc.get("id"):
                return {
                    "type": "content_block_start",
                    "index": tc_idx + 1,  # +1 因为 text 通常是 index 0
                    "content_block": {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc_func.get("name", ""),
                        "input": {}
                    }
                }
            
            # 否则是 input_json_delta
            args_part = tc_func.get("arguments", "")
            if args_part:
                return {
                    "type": "content_block_delta",
                    "index": tc_idx + 1,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": args_part
                    }
                }
        
        if finish_reason:
            # 消息结束
            usage = chunk.get("usage", {})
            return {
                "type": "message_delta",
                "delta": {
                    "stop_reason": cls._map_stop_reason(finish_reason),
                    "stop_sequence": None
                },
                "usage": {
                    "output_tokens": usage.get("completion_tokens", 0)
                }
            }
        
        return {"type": "ping"}
    
    # ================================================================
    # 响应转换：Anthropic -> OpenAI Responses
    # ================================================================
    
    @classmethod
    def to_openai_responses(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anthropic 响应转换为 OpenAI Responses 格式
        """
        output = []
        content = response.get("content", [])
        
        for block in content:
            block_type = block.get("type")
            
            if block_type == "text":
                output.append({
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": block.get("text", "")
                    }]
                })
            
            elif block_type == "tool_use":
                output.append({
                    "type": "function_call",
                    "id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "call_id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                    "status": "completed"
                })
        
        usage = response.get("usage", {})
        return {
            "id": response.get("id", f"resp_{uuid.uuid4().hex[:24]}"),
            "object": "response",
            "status": "completed",
            "created_at": time.time(),
            "model": response.get("model", ""),
            "output": output,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            }
        }


class AnthropicStreamingParser:
    """Anthropic 流式响应解析器"""
    
    @staticmethod
    def parse_sse_line(line: str) -> Optional[tuple]:
        """解析 SSE 行"""
        if not line or line.startswith(":"):
            return None
        
        if line.startswith("event:"):
            event_type = line[6:].strip()
            return (event_type, None)
        
        if line.startswith("data:"):
            data = line[5:].strip()
            if data == "[DONE]":
                return None
            try:
                return (None, json.loads(data))
            except json.JSONDecodeError:
                return (None, data)
        
        return None
    
    @staticmethod
    def parse_sse_chunk(chunk: bytes) -> Iterator[Dict[str, Any]]:
        """解析 SSE 块"""
        text = chunk.decode("utf-8")
        for line in text.split("\n"):
            result = AnthropicStreamingParser.parse_sse_line(line)
            if result:
                event_type, data = result
                if data:
                    yield {"event": event_type, "data": data}
