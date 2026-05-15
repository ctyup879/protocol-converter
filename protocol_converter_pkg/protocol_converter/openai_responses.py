"""
OpenAI Responses API 协议转换器

参考: https://platform.openai.com/docs/api-reference/responses
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union, Iterator


class OpenAIResponsesConverter:
    """OpenAI Responses API 协议转换器"""
    
    PROTOCOL_NAME = "openai_responses"
    
    # 角色映射
    ROLE_MAP = {
        "user": "user",
        "system": "system", 
        "developer": "developer",
        "assistant": "assistant",
    }
    
    # 停止原因映射
    STOP_REASON_MAP = {
        "stop": "stop",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "incomplete",
    }
    
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
        
        # 处理 input
        input_data = request.get("input", [])
        for item in input_data:
            msg = cls._convert_input_item_to_message(item)
            if msg:
                messages.append(msg)
        
        # 处理 instructions (相当于 system prompt)
        instructions = request.get("instructions")
        if instructions:
            messages.insert(0, {
                "role": "system",
                "content": instructions if isinstance(instructions, str) else ""
            })
        
        # 转换 tools
        tools = cls._convert_tools(request.get("tools", []))
        
        # 构建 Chat 请求
        chat_request = {
            "model": request.get("model", "gpt-4o"),
            "messages": messages,
            "stream": request.get("stream", False),
        }
        
        # 可选参数
        if request.get("temperature") is not None:
            chat_request["temperature"] = request["temperature"]
        if request.get("top_p") is not None:
            chat_request["top_p"] = request["top_p"]
        if request.get("max_output_tokens"):
            chat_request["max_tokens"] = request["max_output_tokens"]
        if tools:
            chat_request["tools"] = tools
        if request.get("tool_choice"):
            chat_request["tool_choice"] = request["tool_choice"]
        
        return chat_request
    
    @classmethod
    def _convert_input_item_to_message(cls, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """转换输入项到消息格式"""
        item_type = item.get("type", "")
        
        if item_type == "message":
            role = cls.ROLE_MAP.get(item.get("role", ""), item.get("role", "user"))
            content = cls._convert_content_to_text(item.get("content", []))
            return {
                "role": role,
                "content": content
            }
        
        elif item_type == "function_call_output":
            return {
                "role": "tool",
                "tool_call_id": item.get("call_id"),
                "content": str(item.get("output", ""))
            }
        
        elif item_type == "function_call":
            # function_call 是输入项目，需要转换为消息
            return None
        
        elif item_type == "computer_call_output":
            return {
                "role": "user",
                "content": f"[Computer screenshot: {item.get('output', {}).get('url', '')}]"
            }
        
        return None
    
    @classmethod
    def _convert_content_to_text(cls, content: Union[str, List[Dict]]) -> str:
        """转换内容为纯文本"""
        if isinstance(content, str):
            return content
        
        text_parts = []
        for item in content:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif item.get("type") == "input_text":
                text_parts.append(item.get("text", ""))
        
        return "\n".join(text_parts)
    
    @classmethod
    def _convert_tools(cls, tools: List[Dict]) -> List[Dict]:
        """转换工具定义"""
        converted = []
        
        for tool in tools:
            tool_type = tool.get("type", "function")
            
            if tool_type == "function":
                converted.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", tool.get("input_schema", {}))
                    }
                })
            elif tool_type == "computer":
                # computer 工具需要特殊处理
                converted.append({
                    "type": "function", 
                    "function": {
                        "name": "computer",
                        "description": tool.get("description", "Computer tool for GUI automation"),
                        "parameters": tool.get("parameters", {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string"},
                                "element_id": {"type": "string"},
                                "text": {"type": "string"}
                            }
                        })
                    }
                })
            elif tool_type == "file_search":
                converted.append({
                    "type": "file_search"
                })
            elif tool_type == "web_search":
                converted.append({
                    "type": "web_search"
                })
        
        return converted
    
    @classmethod
    def from_openai_chat(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI Chat 响应转换为 Responses 格式
        
        Args:
            response: OpenAI Chat 响应
            
        Returns:
            Dict: OpenAI Responses 格式响应
        """
        output = []
        
        choices = response.get("choices", [])
        for i, choice in enumerate(choices):
            message = choice.get("message", {})
            content = message.get("content", "")
            
            # 处理文本内容
            if content:
                output.append({
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": content,
                            "annotations": message.get("annotations", [])
                        }
                    ]
                })
            
            # 处理工具调用
            tool_calls = message.get("tool_calls", [])
            for tc in tool_calls:
                output.append({
                    "type": "function_call",
                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "call_id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", "{}")
                })
        
        # 转换 usage
        usage = response.get("usage", {})
        response_usage = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }
        
        return {
            "id": response.get("id", f"resp_{uuid.uuid4().hex[:24]}"),
            "object": "response",
            "status": "completed",
            "created_at": response.get("created", time.time()),
            "model": response.get("model", ""),
            "instructions": None,
            "output": output,
            "usage": response_usage,
            "incomplete_details": None,
            "error": None
        }
    
    @classmethod
    def convert_stream_chunk(cls, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换流式响应块到 Responses 格式
        
        Args:
            chunk: OpenAI Chat 流式响应块
            
        Returns:
            Dict: OpenAI Responses 流式格式
        """
        output_item = None
        choices = chunk.get("choices", [])
        
        for choice in choices:
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            
            if content:
                output_item = {
                    "type": "content_block",
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "index": choice.get("index", 0),
                    "content_block": {
                        "type": "output_text",
                        "text": content
                    }
                }
        
        return {
            "type": "response.output_item.added",
            "item": output_item,
            "output_index": 0,
            "content_index": 0
        }
    
    @classmethod
    def to_anthropic(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI Chat 响应转换为 Anthropic 格式
        
        Args:
            response: OpenAI Chat 响应
            
        Returns:
            Dict: Anthropic 格式响应
        """
        choices = response.get("choices", [])
        content_blocks = []
        
        for choice in choices:
            message = choice.get("message", {})
            content = message.get("content", "")
            
            if content:
                content_blocks.append({
                    "type": "text",
                    "text": content
                })
            
            # 处理工具调用
            tool_calls = message.get("tool_calls", [])
            for tc in tool_calls:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "input": json.loads(tc.get("function", {}).get("arguments", "{}"))
                })
        
        finish_reason = choices[0].get("finish_reason", "stop") if choices else "stop"
        
        return {
            "id": response.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": response.get("model", ""),
            "stop_reason": cls.STOP_REASON_MAP.get(finish_reason, finish_reason),
            "usage": {
                "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                "output_tokens": response.get("usage", {}).get("completion_tokens", 0),
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0
            }
        }
