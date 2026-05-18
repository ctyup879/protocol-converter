"""
OpenAI Chat Completions 协议转换器

参考: https://platform.openai.com/docs/api-reference/chat
参考 SDK: openai-python CompletionCreateParams

OpenAI Chat Completions API 请求参数:
- model (必填): 模型名称
- messages (必填): 消息列表
- temperature: 温度 (0-2)
- top_p: nucleus 采样
- n: 生成选择数量
- stream: 是否流式
- stream_options: 流式选项 {"include_usage": true}
- max_tokens: 最大 token 数 (已弃用，推荐 max_completion_tokens)
- max_completion_tokens: 最大完成 token 数
- tools: 工具定义
- tool_choice: 工具选择策略
- response_format: 响应格式 (text | json_object | json_schema)
- seed: 随机种子
- stop: 停止序列
- presence_penalty: 存在惩罚 (-2.0 ~ 2.0)
- frequency_penalty: 频率惩罚 (-2.0 ~ 2.0)
- logit_bias: token 偏置
- logprobs: 是否返回 logprobs
- top_logprobs: 返回的 top logprobs 数量 (0-20)
- user: 用户标识符 (被 safety_identifier/prompt_cache_key 替代)
- store: 是否存储
- reasoning_effort: 推理努力 (none, minimal, low, medium, high, xhigh)
  - gpt-5.1 默认 none，支持 none/low/medium/high
  - gpt-5-pro 默认且仅支持 high
  - xhigh 支持 gpt-5.1-codex-max 之后的模型
- parallel_tool_calls: 是否允许并行工具调用
- web_search_options: 网页搜索选项 (search_context_size + user_location)
- service_tier: 服务层级 (auto, default, flex, scale, priority)
- metadata: 元数据 (最多16个键值对)
- modalities: 输出类型 (text, audio)
- audio: 音频输出参数
- prediction: 预测内容
- verbosity: 输出详细程度 (low, medium, high)
- safety_identifier: 安全标识符
- prompt_cache_key: 缓存键
- prompt_cache_retention: 缓存保留策略 ("in_memory" | "24h")
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
    metadata: Optional[Dict[str, str]] = None
    modalities: Optional[List[str]] = None
    audio: Optional[Dict] = None
    prediction: Optional[Dict] = None
    stream_options: Optional[Dict] = None
    verbosity: Optional[str] = None
    safety_identifier: Optional[str] = None
    prompt_cache_key: Optional[str] = None
    prompt_cache_retention: Optional[str] = None
    extra_body: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，过滤 None 值"""
        result: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result


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
    metadata: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，过滤 None 值"""
        result: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result


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
        "function_call": "tool_use",  # legacy function_call is semantically tool_use
    }
    
    @classmethod
    def to_openai_chat(cls, request: Dict[str, Any]) -> ChatCompletionRequest:
        """
        通用请求转换为 OpenAI Chat 格式
        
        自动检测输入协议并转换
        
        Raises:
            TypeError: 如果 request 不是字典
        """
        if not isinstance(request, dict):
            raise TypeError(f"request must be a dict, got {type(request).__name__}")
        
        from .protocol_detector import ProtocolDetector, Protocol

        if ProtocolDetector.detect(request) == Protocol.ANTHROPIC:
            return cls._from_anthropic(request)
        
        return cls._normalize_request(request)
    
    @classmethod
    def _normalize_request(cls, request: Dict[str, Any]) -> ChatCompletionRequest:
        """标准化 OpenAI Chat 请求"""
        return ChatCompletionRequest(
            model=request.get("model") or "gpt-4o",
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
            metadata=request.get("metadata"),
            modalities=request.get("modalities"),
            audio=request.get("audio"),
            prediction=request.get("prediction"),
            stream_options=request.get("stream_options"),
            verbosity=request.get("verbosity"),
            safety_identifier=request.get("safety_identifier"),
            prompt_cache_key=request.get("prompt_cache_key"),
            prompt_cache_retention=request.get("prompt_cache_retention"),
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
            n=chat_dict.get("n", 1),
            max_completion_tokens=chat_dict.get("max_completion_tokens"),
            max_tokens=chat_dict.get("max_tokens"),
            tools=chat_dict.get("tools"),
            tool_choice=chat_dict.get("tool_choice"),
            response_format=chat_dict.get("response_format"),
            seed=chat_dict.get("seed"),
            stop=chat_dict.get("stop"),
            presence_penalty=chat_dict.get("presence_penalty"),
            frequency_penalty=chat_dict.get("frequency_penalty"),
            logit_bias=chat_dict.get("logit_bias"),
            logprobs=chat_dict.get("logprobs"),
            top_logprobs=chat_dict.get("top_logprobs"),
            user=chat_dict.get("user"),
            store=chat_dict.get("store"),
            reasoning_effort=chat_dict.get("reasoning_effort"),
            parallel_tool_calls=chat_dict.get("parallel_tool_calls"),
            web_search_options=chat_dict.get("web_search_options"),
            service_tier=chat_dict.get("service_tier"),
            metadata=chat_dict.get("metadata"),
            modalities=chat_dict.get("modalities"),
            audio=chat_dict.get("audio"),
            prediction=chat_dict.get("prediction"),
            stream_options=chat_dict.get("stream_options"),
            verbosity=chat_dict.get("verbosity"),
            safety_identifier=chat_dict.get("safety_identifier"),
            prompt_cache_key=chat_dict.get("prompt_cache_key"),
            prompt_cache_retention=chat_dict.get("prompt_cache_retention"),
            extra_body=chat_dict.get("extra_body"),
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
        cache_read = 0
        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            cache_read = prompt_details.get("cached_tokens", 0)
        return {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cache_read,
        }

    @classmethod
    def _convert_usage_to_responses(cls, usage: Optional[Dict]) -> Optional[Dict]:
        """转换 usage 到 OpenAI Responses 格式"""
        if not usage:
            return None
        result = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens"):
            result["input_tokens_details"] = {"cached_tokens": prompt_details["cached_tokens"]}
        completion_details = usage.get("completion_tokens_details")
        if isinstance(completion_details, dict) and completion_details.get("reasoning_tokens"):
            result["output_tokens_details"] = {"reasoning_tokens": completion_details["reasoning_tokens"]}
        return result
