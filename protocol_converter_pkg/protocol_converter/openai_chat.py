"""
OpenAI Chat Completions 协议转换器

参考: https://platform.openai.com/docs/api-reference/chat
参考 SDK: openai-python CompletionCreateParams
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union, Iterator
from dataclasses import dataclass, field


@dataclass
class ChatCompletionRequest:
    """OpenAI Chat Completion 请求（核心字段）"""
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Union[str, Dict]] = None
    response_format: Optional[Dict] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    user: Optional[str] = None
    store: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    parallel_tool_calls: Optional[bool] = None
    web_search_options: Optional[Dict] = None
    service_tier: Optional[str] = None


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
    
    PROTOCOL_NAME = "openai_chat"
    
    # Anthropic -> OpenAI 停止原因映射
    STOP_REASON_MAP = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
        "pause_turn": "stop",
        "refusal": "content_filter",
    }
    
    # OpenAI -> Anthropic 停止原因映射
    REVERSE_STOP_REASON_MAP = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "refusal",
        "function_call": "end_turn",
    }
    
    @classmethod
    def to_openai_chat(cls, request: Dict[str, Any]) -> ChatCompletionRequest:
        """
        通用请求转换为 OpenAI Chat 格式
        
        自动检测输入协议并转换
        """
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
            max_completion_tokens=request.get("max_completion_tokens"),
            tools=request.get("tools"),
            tool_choice=request.get("tool_choice"),
            response_format=request.get("response_format"),
            seed=request.get("seed"),
            stop=request.get("stop"),
            presence_penalty=request.get("presence_penalty"),
            frequency_penalty=request.get("frequency_penalty"),
            logit_bias=request.get("logit_bias"),
            logprobs=request.get("logprobs"),
            top_logprobs=request.get("top_logprobs"),
            user=request.get("user"),
            store=request.get("store"),
            reasoning_effort=request.get("reasoning_effort"),
            parallel_tool_calls=request.get("parallel_tool_calls"),
            web_search_options=request.get("web_search_options"),
            service_tier=request.get("service_tier"),
        )
    
    @classmethod
    def _from_anthropic(cls, request: Dict[str, Any]) -> ChatCompletionRequest:
        """从 Anthropic 格式转换"""
        from .anthropic import AnthropicConverter
        
        chat_dict = AnthropicConverter.to_openai_chat(request)
        
        return ChatCompletionRequest(
            model=chat_dict.get("model", "claude-sonnet-4-20250514"),
            messages=chat_dict.get("messages", []),
            stream=chat_dict.get("stream", False),
            temperature=chat_dict.get("temperature"),
            top_p=chat_dict.get("top_p"),
            max_tokens=chat_dict.get("max_tokens"),
            tools=chat_dict.get("tools"),
            tool_choice=chat_dict.get("tool_choice"),
            stop=chat_dict.get("stop"),
            user=chat_dict.get("user"),
        )
    
    # ================================================================
    # 响应转换
    # ================================================================
    
    @classmethod
    def from_openai_chat(cls, response: Dict[str, Any], target_protocol: str = None) -> Dict[str, Any]:
        """OpenAI Chat 响应转换"""
        if target_protocol == "anthropic":
            return cls._to_anthropic(response)
        elif target_protocol == "openai_responses":
            return cls._to_openai_responses(response)
        return response
    
    @classmethod
    def _to_anthropic(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """转换为 Anthropic 格式"""
        from .anthropic import AnthropicConverter
        return AnthropicConverter.from_openai_chat(response)
    
    @classmethod
    def _to_openai_responses(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """转换为 OpenAI Responses 格式"""
        from .openai_responses import OpenAIResponsesConverter
        return OpenAIResponsesConverter.from_openai_chat(response)
    
    # ================================================================
    # 流式转换
    # ================================================================
    
    @classmethod
    def convert_stream_chunk(cls, chunk: Dict[str, Any], target_protocol: str = None) -> Dict[str, Any]:
        """转换流式响应块"""
        if target_protocol == "anthropic":
            from .anthropic import AnthropicConverter
            return AnthropicConverter.convert_stream_chunk(chunk)
        elif target_protocol == "openai_responses":
            from .openai_responses import OpenAIResponsesConverter
            return OpenAIResponsesConverter.convert_stream_chunk(chunk)
        return chunk
    
    # ================================================================
    # Usage 转换
    # ================================================================
    
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
