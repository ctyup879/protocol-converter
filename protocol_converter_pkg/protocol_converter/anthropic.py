"""
Anthropic Messages API 协议转换器

参考: https://docs.anthropic.com/en/api/messages
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
    
    # OpenAI -> Anthropic 角色映射
    REVERSE_ROLE_MAP = {
        "user": "user",
        "assistant": "assistant", 
        "system": "system",
        "tool": "user",  # tool 消息映射为 user
        "developer": "developer",
    }
    
    # Anthropic 停止原因
    STOP_REASONS = {
        "end_turn": "end_turn",
        "max_tokens": "max_tokens", 
        "stop_sequence": "stop_sequence",
        "tool_use": "tool_use",
        "pause_turn": "pause_turn",
        "refusal": "refusal",
    }
    
    @classmethod
    def to_openai_chat(cls, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anthropic Messages 请求转换为 OpenAI Chat 格式
        
        Args:
            request: Anthropic Messages 格式请求
            
        Returns:
            Dict: OpenAI Chat 格式请求
        """
        messages = []
        
        # 处理 system prompt
        system_prompt = request.get("system")
        if system_prompt:
            if isinstance(system_prompt, str):
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            elif isinstance(system_prompt, list):
                # 处理内容块数组
                text_content = []
                for block in system_prompt:
                    if block.get("type") == "text":
                        text_content.append(block.get("text", ""))
                if text_content:
                    messages.append({
                        "role": "system",
                        "content": "\n".join(text_content)
                    })
        
        # 处理 messages
        anthropic_messages = request.get("messages", [])
        for msg in anthropic_messages:
            converted = cls._convert_message(msg)
            if converted:
                messages.append(converted)
        
        # 转换 tools
        tools = cls._convert_tools(request.get("tools", []))
        
        # 构建 Chat 请求
        chat_request = {
            "model": cls._map_model(request.get("model", "claude-sonnet-4-20250514")),
            "messages": messages,
            "stream": request.get("stream", False),
        }
        
        # 可选参数
        if request.get("temperature") is not None:
            chat_request["temperature"] = request["temperature"]
        if request.get("top_p") is not None:
            chat_request["top_p"] = request["top_p"]
        if request.get("max_tokens"):
            chat_request["max_tokens"] = request["max_tokens"]
        if tools:
            chat_request["tools"] = tools
        if request.get("tool_choice"):
            chat_request["tool_choice"] = request["tool_choice"]
        if request.get("stop_sequences"):
            chat_request["stop"] = request["stop_sequences"]
        if request.get("thinking"):
            # Extended thinking - OpenAI 不直接支持，但可以放在 extra_body
            chat_request["extra_body"] = {
                "thinking": request["thinking"]
            }
        
        return chat_request
    
    @classmethod
    def _map_model(cls, anthropic_model: str) -> str:
        """保持 Anthropic 模型名称不变（协议网关场景下保持原始模型）"""
        # 在协议转换场景中，保持原始模型名称
        # 后端服务需要能够处理这些模型名称
        return anthropic_model
    
    @classmethod
    def _convert_message(cls, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """转换单条消息"""
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role not in cls.ROLE_MAP:
            return None
        
        # 处理 content
        if isinstance(content, list):
            converted_content = cls._convert_content_blocks(content)
        else:
            converted_content = content
        
        converted = {
            "role": cls.ROLE_MAP.get(role, role),
            "content": converted_content
        }
        
        # 处理 tool_use 消息 (Anthropic 格式)
        if role == "user" and isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result":
                    converted["tool_call_id"] = block.get("tool_use_id")
                    converted["name"] = block.get("tool_use_id", "").replace("toolu_", "")
                    break
        
        return converted
    
    @classmethod
    def _convert_content_blocks(cls, blocks: List[Dict]) -> Union[str, List[Dict]]:
        """转换 Anthropic 内容块到 OpenAI 格式"""
        text_parts = []
        tool_calls = []
        
        for block in blocks:
            block_type = block.get("type")
            
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            
            elif block_type == "tool_use":
                func = block.get("function", block)
                tool_calls.append({
                    "id": block.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", func.get("name", "")),
                        "arguments": json.dumps(block.get("input", func.get("input", {})))
                    }
                })
            
            elif block_type == "thinking":
                # Extended thinking - OpenAI 可能不支持，转换为注释
                text_parts.append(f"\n[Thinking: {block.get('thinking', '')[:100]}...]\n")
            
            elif block_type == "image":
                # 图片处理 - 转换为 URL 或 base64
                source = block.get("source", {})
                if source.get("type") == "base64":
                    text_parts.append(f"\n[Image: base64 encoded, {source.get('media_type', 'image/png')}]\n")
                elif source.get("type") == "url":
                    text_parts.append(f"\n[Image: {source.get('url', '')}]\n")
        
        if tool_calls:
            # 如果有工具调用，返回带有工具调用的文本内容
            tool_text = "\n".join([
                f"[Calling tool: {tc['function']['name']}]" 
                for tc in tool_calls
            ])
            return ("\n".join(text_parts) + "\n" + tool_text).strip()
        
        return "\n".join(text_parts)
    
    @classmethod
    def _convert_tools(cls, tools: List[Dict]) -> List[Dict]:
        """转换 Anthropic 工具定义到 OpenAI 格式"""
        converted = []
        
        for tool in tools:
            if "name" in tool:
                converted.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {"type": "object"})
                    }
                })
        
        return converted
    
    @classmethod
    def from_openai_chat(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI Chat 响应转换为 Anthropic 格式
        
        Args:
            response: OpenAI Chat 响应
            
        Returns:
            Dict: Anthropic Messages 格式响应
        """
        choices = response.get("choices", [])
        content_blocks = []
        stop_reason = None
        
        for choice in choices:
            message = choice.get("message", {})
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])
            
            # 处理文本内容
            if content:
                content_blocks.append({
                    "type": "text",
                    "text": content
                })
            
            # 处理工具调用
            for tc in tool_calls:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    "name": tc.get("function", {}).get("name", ""),
                    "input": json.loads(tc.get("function", {}).get("arguments", "{}"))
                })
            
            # 获取停止原因
            finish_reason = choice.get("finish_reason", "stop")
            stop_reason = cls._map_stop_reason(finish_reason)
        
        return {
            "id": response.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": response.get("model", ""),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                "output_tokens": response.get("usage", {}).get("completion_tokens", 0),
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0
            }
        }
    
    @classmethod
    def _map_stop_reason(cls, openai_reason: str) -> str:
        """映射 OpenAI 停止原因到 Anthropic"""
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "refusal",
            "function_call": "end_turn",
        }
        return mapping.get(openai_reason, "end_turn")
    
    @classmethod
    def convert_stream_chunk(cls, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换流式响应块到 Anthropic 格式
        
        Args:
            chunk: OpenAI Chat 流式响应块
            
        Returns:
            Dict: Anthropic SSE 流式格式
        """
        choices = chunk.get("choices", [])
        
        # 处理不同的事件类型
        if not choices:
            return {"type": "message_stop"}
        
        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")
        
        event_type = "content_block_delta"
        index = choice.get("index", 0)
        
        # 确定事件类型
        if delta.get("role"):
            event_type = "message_start"
        elif delta.get("content"):
            event_type = "content_block_delta"
        elif delta.get("tool_calls"):
            event_type = "content_block_delta"
        elif finish_reason:
            event_type = "message_delta"
        
        content_block = None
        delta_text = delta.get("content", "")
        
        if delta_text:
            content_block = {
                "type": "text",
                "text": delta_text
            }
        
        # 处理工具调用
        tool_calls = delta.get("tool_calls", [])
        if tool_calls:
            tc = tool_calls[0]
            content_block = {
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": tc.get("function", {}).get("name", ""),
                "input": json.loads(tc.get("function", {}).get("arguments", "{}"))
            }
        
        result = {
            "type": event_type,
            "index": index
        }
        
        if event_type == "message_start":
            result["message"] = {
                "id": chunk.get("id", ""),
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": chunk.get("model", "")
            }
        elif event_type == "content_block_delta":
            result["content_block"] = content_block
            result["delta"] = delta_text
        elif event_type == "message_delta":
            result["usage"] = {
                "output_tokens": chunk.get("usage", {}).get("completion_tokens", 0)
            }
            if finish_reason:
                result["delta"] = {
                    "stop_reason": cls._map_stop_reason(finish_reason)
                }
        
        return result
    
    @classmethod
    def to_openai_responses(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anthropic 响应转换为 OpenAI Responses 格式
        
        Args:
            response: Anthropic Messages 响应
            
        Returns:
            Dict: OpenAI Responses 格式
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
                    "arguments": json.dumps(block.get("input", {}))
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
        """
        解析 SSE 行
        
        Args:
            line: SSE 行
            
        Returns:
            tuple: (event_type, data) 或 None
        """
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
        """
        解析 SSE 块
        
        Args:
            chunk: SSE 数据块
            
        Yields:
            Dict: 解析后的数据
        """
        text = chunk.decode("utf-8")
        for line in text.split("\n"):
            result = AnthropicStreamingParser.parse_sse_line(line)
            if result:
                event_type, data = result
                if data:
                    yield {"event": event_type, "data": data}
