"""
OpenAI Chat Completions 协议转换器

参考: https://platform.openai.com/docs/api-reference/chat
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union, Iterator
from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    """Chat 消息结构"""
    role: str
    content: Optional[Union[str, List[Dict]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    annotations: Optional[List[Dict]] = None
    audio: Optional[Dict] = None
    refusal: Optional[str] = None


@dataclass
class ToolCall:
    """工具调用结构"""
    id: str
    type: str = "function"
    function: "Function" = None


@dataclass 
class Function:
    """函数定义"""
    name: str
    arguments: str


@dataclass
class ChatCompletionRequest:
    """OpenAI Chat Completion 请求"""
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    max_tokens: Optional[int] = None
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Union[str, Dict]] = None
    response_format: Optional[Dict] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[int, float]] = None
    user: Optional[str] = None
    extra_headers: Optional[Dict[str, str]] = None
    extra_body: Optional[Dict] = None
    timeout: Optional[float] = None


@dataclass
class ChatCompletionResponse:
    """OpenAI Chat Completion 响应"""
    id: str
    model: str
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    choices: List[Dict[str, Any]] = field(default_factory=list)
    usage: Optional[Dict[str, Any]] = None
    service_tier: Optional[str] = None
    system_fingerprint: Optional[str] = None


class OpenAIChatConverter:
    """OpenAI Chat Completions 协议转换器"""
    
    # 协议标识
    PROTOCOL_NAME = "openai_chat"
    
    # Anthropic -> OpenAI 角色映射
    ROLE_MAP = {
        "user": "user",
        "assistant": "assistant",
        "system": "system",
    }
    
    # Anthropic 停止原因 -> OpenAI 停止原因
    STOP_REASON_MAP = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
        "pause_turn": "stop",
        "refusal": "content_filter",
    }
    
    @classmethod
    def to_openai_chat(cls, request: Dict[str, Any]) -> ChatCompletionRequest:
        """
        通用请求转换为 OpenAI Chat 格式
        
        Args:
            request: 可能是 OpenAI Chat、OpenAI Responses 或 Anthropic 请求
            
        Returns:
            ChatCompletionRequest: OpenAI Chat 格式请求
        """
        # 检查请求来源协议
        # Anthropic 特征：max_tokens + messages + (tools 或 system 或模型名以 claude- 开头)
        model = request.get("model", "")
        has_max_tokens = "max_tokens" in request
        has_messages = "messages" in request
        has_anthropic_marker = (
            "tools" in request or 
            "system" in request or 
            model.startswith("claude-")
        )
        
        if has_max_tokens and has_messages and has_anthropic_marker:
            return cls._from_anthropic(request)
        
        # 已经是 OpenAI Chat 格式
        return cls._normalize_request(request)
    
    @classmethod
    def _normalize_request(cls, request: Dict[str, Any]) -> ChatCompletionRequest:
        """标准化 OpenAI Chat 请求"""
        return ChatCompletionRequest(
            model=request.get("model", "gpt-4o"),
            messages=request.get("messages", []),
            stream=request.get("stream", False),
            temperature=request.get("temperature"),
            top_p=request.get("top_p"),
            n=request.get("n", 1),
            max_tokens=request.get("max_tokens"),
            tools=request.get("tools"),
            tool_choice=request.get("tool_choice"),
            response_format=request.get("response_format"),
            seed=request.get("seed"),
            stop=request.get("stop"),
            presence_penalty=request.get("presence_penalty"),
            frequency_penalty=request.get("frequency_penalty"),
            logit_bias=request.get("logit_bias"),
            user=request.get("user"),
        )
    
    @classmethod
    def _from_anthropic(cls, request: Dict[str, Any]) -> ChatCompletionRequest:
        """从 Anthropic 格式转换"""
        messages = cls._convert_anthropic_messages(request.get("messages", []))
        
        # 处理 system 提示
        system_prompt = request.get("system")
        if system_prompt:
            system_msg = {
                "role": "system",
                "content": system_prompt if isinstance(system_prompt, str) else system_prompt[0].get("text", "")
            }
            messages = [system_msg] + messages
        
        # 转换 tools
        tools = cls._convert_anthropic_tools(request.get("tools", []))
        
        # Anthropic 的 max_tokens 对应 OpenAI 的 max_tokens
        max_tokens = request.get("max_tokens", 1024)
        
        return ChatCompletionRequest(
            model=request.get("model", "claude-sonnet-4-6"),
            messages=messages,
            stream=request.get("stream", False),
            temperature=request.get("temperature"),
            top_p=request.get("top_p"),
            tools=tools if tools else None,
            max_tokens=max_tokens,
        )
    
    @classmethod
    def _convert_anthropic_messages(cls, messages: List[Dict]) -> List[Dict]:
        """转换 Anthropic 消息格式到 OpenAI 格式"""
        converted = []
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # Anthropic content 可以是字符串或内容块数组
            if isinstance(content, list):
                # 转换内容块
                openai_content = cls._convert_anthropic_content_blocks(content)
            else:
                openai_content = content
            
            converted_msg = {
                "role": cls.ROLE_MAP.get(role, role),
                "content": openai_content
            }
            
            # 处理工具结果
            if role == "user" and isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result":
                        # 工具结果需要放在下一个 user 消息中
                        pass
            
            converted.append(converted_msg)
        
        return converted
    
    @classmethod
    def _convert_anthropic_content_blocks(cls, blocks: List[Dict]) -> Union[str, List[Dict]]:
        """转换 Anthropic 内容块"""
        result_parts = []
        tool_calls = []
        
        for block in blocks:
            block_type = block.get("type")
            
            if block_type == "text":
                result_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}))
                    }
                })
        
        if tool_calls:
            # 如果有工具调用，返回带有 tool_calls 的内容
            return result_parts[0] if result_parts else "" + "\n[Tool calls: " + ", ".join(tc["function"]["name"] for tc in tool_calls) + "]"
        
        return "\n".join(result_parts) if result_parts else ""
    
    @classmethod
    def _convert_anthropic_tools(cls, tools: List[Dict]) -> List[Dict]:
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
    def from_openai_chat(cls, response: Dict[str, Any], target_protocol: str = None) -> Dict[str, Any]:
        """
        OpenAI Chat 响应转换（保持原格式）
        
        Args:
            response: OpenAI Chat 响应
            target_protocol: 目标协议，如果指定则转换为目标协议格式
            
        Returns:
            Dict: 转换后的响应
        """
        if target_protocol == "anthropic":
            return cls._to_anthropic(response)
        elif target_protocol == "openai_responses":
            return cls._to_openai_responses(response)
        
        return response
    
    @classmethod
    def _to_anthropic(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """转换为 Anthropic 格式"""
        choices = response.get("choices", [])
        content_blocks = []
        stop_reason = None
        
        for choice in choices:
            message = choice.get("message", {})
            msg_content = message.get("content", "")
            
            if msg_content:
                content_blocks.append({
                    "type": "text",
                    "text": msg_content
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
            
            stop_reason = cls.STOP_REASON_MAP.get(
                choice.get("finish_reason", ""),
                choice.get("finish_reason", "end_turn")
            )
        
        return {
            "id": response.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": response.get("model", ""),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": cls._convert_usage_to_anthropic(response.get("usage"))
        }
    
    @classmethod
    def _to_openai_responses(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """转换为 OpenAI Responses 格式"""
        choices = response.get("choices", [])
        output_items = []
        
        for choice in choices:
            message = choice.get("message", {})
            msg_content = message.get("content", "")
            
            content = []
            if msg_content:
                content.append({
                    "type": "output_text",
                    "text": msg_content
                })
            
            # 处理工具调用
            tool_calls = message.get("tool_calls", [])
            for tc in tool_calls:
                output_items.append({
                    "type": "function_call",
                    "id": tc.get("id", ""),
                    "call_id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", "{}")
                })
            
            if content:
                output_items.append({
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "role": "assistant",
                    "content": content
                })
        
        return {
            "id": response.get("id", f"resp_{uuid.uuid4().hex[:24]}"),
            "object": "response",
            "status": "completed",
            "created_at": response.get("created", int(time.time())),
            "model": response.get("model", ""),
            "output": output_items,
            "usage": cls._convert_usage_to_responses(response.get("usage"))
        }
    
    @classmethod
    def _convert_usage_to_anthropic(cls, usage: Optional[Dict]) -> Optional[Dict]:
        """转换 usage 到 Anthropic 格式"""
        if not usage:
            return None
        return {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0
        }
    
    @classmethod
    def _convert_usage_to_responses(cls, usage: Optional[Dict]) -> Optional[Dict]:
        """转换 usage 到 OpenAI Responses 格式"""
        if not usage:
            return None
        return {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }
    
    @classmethod
    def convert_stream_chunk(cls, chunk: Dict[str, Any], target_protocol: str = None) -> Dict[str, Any]:
        """
        转换流式响应块
        
        Args:
            chunk: OpenAI 流式响应块
            target_protocol: 目标协议
            
        Returns:
            Dict: 转换后的块
        """
        if target_protocol == "anthropic":
            return cls._convert_stream_chunk_to_anthropic(chunk)
        elif target_protocol == "openai_responses":
            return cls._convert_stream_chunk_to_responses(chunk)
        
        return chunk
    
    @classmethod
    def _convert_stream_chunk_to_anthropic(cls, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """转换流式块到 Anthropic 格式"""
        choices = chunk.get("choices", [])
        content = ""
        tool_use = None
        index = 0
        finish_reason = None
        
        for choice in choices:
            delta = choice.get("delta", {})
            content += delta.get("content", "")
            
            tool_calls = delta.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    if tool_use is None:
                        tool_use = {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "input": {}
                        }
                    if tc.get("function", {}).get("arguments"):
                        tool_use["input"] = json.loads(
                            tool_use.get("input", "{}") + tc.get("function", {}).get("arguments", "")
                        )
            
            finish_reason = cls.STOP_REASON_MAP.get(
                choice.get("finish_reason", ""),
                choice.get("finish_reason")
            )
            index = choice.get("index", 0)
        
        event_type = chunk.get("type", "content_block_delta")
        event_map = {
            "content_block_delta": "content_block_delta",
            "message_start": "message_start", 
            "message_delta": "message_delta",
            "message_stop": "message_stop"
        }
        
        return {
            "type": event_map.get(event_type, event_type),
            "index": index,
            "content_block": {
                "type": "text" if content else "tool_use",
                "text": content if content else None,
            } if not tool_use else tool_use,
            "message": {
                "id": chunk.get("id", ""),
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": chunk.get("model", "")
            },
            "usage": cls._convert_usage_to_anthropic(chunk.get("usage")),
            "delta": content,
            "stop_reason": finish_reason
        }
    
    @classmethod
    def _convert_stream_chunk_to_responses(cls, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """转换流式块到 OpenAI Responses 格式"""
        choices = chunk.get("choices", [])
        content = ""
        index = 0
        
        for choice in choices:
            delta = choice.get("delta", {})
            content += delta.get("content", "")
            index = choice.get("index", 0)
        
        item = {
            "type": "response.output_item",
            "output_index": index,
            "content_index": 0
        }
        
        if content:
            item["type"] = "response.content_part.done"
            item["part"] = {
                "type": "output_text",
                "text": content
            }
        
        return {
            "type": "response.output_item.added" if content else "response.done",
            "item": item,
            "usage": cls._convert_usage_to_responses(chunk.get("usage"))
        }
