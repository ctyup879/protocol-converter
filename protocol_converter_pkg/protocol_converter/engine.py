"""
协议转换引擎 - 统一处理各种协议的转换
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union, Iterator, Callable
from dataclasses import dataclass, field
from enum import Enum

from .protocol_detector import ProtocolDetector, Protocol
from .openai_chat import OpenAIChatConverter
from .openai_responses import OpenAIResponsesConverter
from .anthropic import AnthropicConverter


class ConversionMode(Enum):
    """转换模式"""
    TO_OPENAI_CHAT = "to_openai_chat"
    TO_OPENAI_RESPONSES = "to_openai_responses"
    TO_ANTHROPIC = "to_anthropic"
    AUTO_DETECT = "auto_detect"


@dataclass
class ConverterConfig:
    """转换器配置"""
    # 目标后端（OpenAI Chat API 端点）
    backend_url: str = "https://api.openai.com/v1/chat/completions"
    
    # API 密钥
    api_key: Optional[str] = None
    
    # 默认模型
    default_model: str = "gpt-4o"
    
    # 超时时间（秒）
    timeout: float = 60.0
    
    # 是否启用流式响应
    stream: bool = False
    
    # 请求头
    extra_headers: Dict[str, str] = field(default_factory=dict)
    
    # 其他请求参数
    extra_body: Dict[str, Any] = field(default_factory=dict)
    
    # 模型映射表：Anthropic/OpenAI 模型名 -> 目标后端模型名
    model_mapping: Dict[str, str] = field(default_factory=dict)
    
    def get_model(self, model: str) -> str:
        """获取映射后的模型名"""
        return self.model_mapping.get(model, model)


@dataclass
class StreamChunk:
    """流式响应块"""
    event: str
    data: Dict[str, Any]
    
    def to_sse(self) -> str:
        """转换为 SSE 格式"""
        if self.event:
            return f"event: {self.event}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"
        return f"data: {json.dumps(self.data, ensure_ascii=False)}\n\n"
    
    def to_anthropic_sse(self) -> str:
        """转换为 Anthropic SSE 格式"""
        event_map = {
            "content_block_delta": "content_block_delta",
            "message_start": "message_start",
            "message_delta": "message_delta",
            "message_stop": "message_stop",
        }
        event = event_map.get(self.event, self.event)
        return f"event: {event}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"
    
    def to_openai_chunk(self) -> str:
        """保持为 OpenAI SSE 格式"""
        return f"data: {json.dumps(self.data, ensure_ascii=False)}\n\n"


class ProtocolConverterEngine:
    """
    协议转换引擎
    
    支持:
    - 自动检测请求协议类型
    - 转换为 OpenAI Chat 格式发送到后端
    - 将响应转换回用户请求的协议格式
    - 支持流式和非流式响应
    """
    
    def __init__(self, config: Optional[ConverterConfig] = None):
        """
        初始化转换引擎
        
        Args:
            config: 转换器配置
        """
        self.config = config or ConverterConfig()
        self.detector = ProtocolDetector()
        self.openai_chat = OpenAIChatConverter()
        self.openai_responses = OpenAIResponsesConverter()
        self.anthropic = AnthropicConverter()
    
    def detect_protocol(self, request: Dict[str, Any]) -> Protocol:
        """
        检测请求使用的协议
        
        Args:
            request: 请求字典
            
        Returns:
            Protocol: 检测到的协议
        """
        return self.detector.detect(request)
    
    def convert_request(
        self, 
        request: Dict[str, Any], 
        target_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        转换请求到目标格式
        
        Args:
            request: 输入请求
            target_format: 目标格式，None 则自动检测后转为 OpenAI Chat
            
        Returns:
            Dict: 转换后的请求
        """
        # 检测源协议
        source_protocol = self.detect_protocol(request)
        result = None
        
        if target_format is None or target_format == "openai_chat":
            # 统一转换为 OpenAI Chat 格式
            if source_protocol == Protocol.ANTHROPIC:
                result = self.anthropic.to_openai_chat(request)
            elif source_protocol == Protocol.OPENAI_RESPONSES:
                result = self.openai_responses.to_openai_chat(request)
            else:
                # 已经是 OpenAI Chat 或未知
                result = request.copy() if isinstance(request, dict) else request
        
        elif target_format == "anthropic":
            # 转换为 Anthropic 格式（再转为 Chat 发送）
            result = self.anthropic.to_openai_chat(request)
            result["_source_format"] = "anthropic"
        
        elif target_format == "openai_responses":
            # 转换为 Responses 格式（再转为 Chat 发送）
            result = self.openai_responses.to_openai_chat(request)
            result["_source_format"] = "openai_responses"
        
        else:
            result = request
        
        # 应用模型映射
        if isinstance(result, dict) and "model" in result:
            original_model = result["model"]
            mapped_model = self.config.get_model(original_model)
            if mapped_model != original_model:
                result["model"] = mapped_model
        
        return result
    
    def convert_response(
        self, 
        response: Dict[str, Any],
        target_protocol: Protocol
    ) -> Dict[str, Any]:
        """
        转换响应到目标协议格式
        
        Args:
            response: OpenAI Chat 格式响应
            target_protocol: 目标协议
            
        Returns:
            Dict: 目标协议格式的响应
        """
        if target_protocol == Protocol.ANTHROPIC:
            return self.anthropic.from_openai_chat(response)
        elif target_protocol == Protocol.OPENAI_RESPONSES:
            return self.openai_responses.from_openai_chat(response)
        elif target_protocol == Protocol.OPENAI_CHAT:
            return response
        
        return response
    
    def convert_stream_chunk(
        self,
        chunk: Dict[str, Any],
        target_protocol: Protocol
    ) -> StreamChunk:
        """
        转换流式响应块
        
        Args:
            chunk: OpenAI Chat 流式块
            target_protocol: 目标协议
            
        Returns:
            StreamChunk: 转换后的块
        """
        if target_protocol == Protocol.ANTHROPIC:
            data = self.anthropic.convert_stream_chunk(chunk)
            # 确定事件类型
            event = data.get("type", "content_block_delta")
            return StreamChunk(event=event, data=data)
        
        elif target_protocol == Protocol.OPENAI_RESPONSES:
            data = self.openai_responses.convert_stream_chunk(chunk)
            return StreamChunk(event="response.output_item.added", data=data)
        
        else:
            # OpenAI Chat - 提取事件类型
            event = "content_block_delta"
            if chunk.get("choices") and chunk["choices"][0].get("index", 0) == 0:
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("role"):
                    event = "message_start"
                elif delta.get("content") is None and delta.get("tool_calls"):
                    event = "content_block_delta"
                elif chunk["choices"][0].get("finish_reason"):
                    event = "message_delta"
            
            return StreamChunk(event=event, data=chunk)
    
    async def convert_and_forward(
        self,
        request: Dict[str, Any],
        http_client: Callable,
    ) -> Union[Dict[str, Any], Iterator[StreamChunk]]:
        """
        转换请求并转发到后端
        
        Args:
            request: 输入请求
            http_client: HTTP 客户端函数（异步）
            
        Returns:
            Union[Dict, Iterator]: 转换后的响应或流式迭代器
        """
        # 检测源协议
        source_protocol = self.detect_protocol(request)
        
        # 转换请求
        chat_request = self.convert_request(request)
        
        # 添加认证信息
        headers = {
            "Content-Type": "application/json",
            **self.config.extra_headers
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        # 合并 extra_body
        if self.config.extra_body:
            chat_request["extra_body"] = {
                **chat_request.get("extra_body", {}),
                **self.config.extra_body
            }
        
        # 发送请求
        is_stream = chat_request.get("stream", False)
        
        if is_stream:
            return self._handle_stream_response(
                chat_request, 
                headers, 
                source_protocol,
                http_client
            )
        else:
            response = await http_client(
                url=self.config.backend_url,
                method="POST",
                headers=headers,
                json=chat_request,
                timeout=self.config.timeout
            )
            
            # 转换响应
            return self.convert_response(response, source_protocol)
    
    async def _handle_stream_response(
        self,
        chat_request: Dict[str, Any],
        headers: Dict[str, str],
        source_protocol: Protocol,
        http_client: Callable,
    ) -> Iterator[StreamChunk]:
        """
        处理流式响应
        """
        # 确保 stream 为 True
        chat_request["stream"] = True
        
        # 添加 stream_options
        if "stream_options" not in chat_request:
            chat_request["stream_options"] = {"include_usage": True}
        
        # 发起流式请求
        response = await http_client(
            url=self.config.backend_url,
            method="POST",
            headers=headers,
            json=chat_request,
            timeout=self.config.timeout,
            stream=True
        )
        
        async for line in response.content:
            line = line.decode("utf-8").strip()
            if not line or line.startswith(":"):
                continue
            
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    yield StreamChunk(event="message_stop", data={}).to_sse()
                    break
                
                try:
                    chunk = json.loads(data_str)
                    stream_chunk = self.convert_stream_chunk(chunk, source_protocol)
                    
                    # 根据目标协议选择输出格式
                    if source_protocol == Protocol.ANTHROPIC:
                        yield stream_chunk.to_anthropic_sse()
                    elif source_protocol == Protocol.OPENAI_RESPONSES:
                        yield stream_chunk.to_sse()
                    else:
                        yield stream_chunk.to_openai_chunk()
                except json.JSONDecodeError:
                    continue


def create_http_client() -> Callable:
    """
    创建 HTTP 客户端（使用 httpx）
    """
    try:
        import httpx
        import asyncio
        
        async def client(
            url: str,
            method: str = "GET",
            headers: Optional[Dict] = None,
            json: Optional[Dict] = None,
            timeout: float = 60.0,
            stream: bool = False
        ):
            async with httpx.AsyncClient(timeout=timeout) as client:
                if method == "POST":
                    response = await client.post(url, json=json, headers=headers)
                else:
                    response = await client.get(url, headers=headers)
                return response.json()
        
        return client
    except ImportError:
        # 如果没有 httpx，返回一个简单的占位符
        async def placeholder_client(*args, **kwargs):
            raise ImportError("请安装 httpx: pip install httpx")
        return placeholder_client


# 示例用法
if __name__ == "__main__":
    # 示例：检测协议
    engine = ProtocolConverterEngine()
    
    # OpenAI Chat 请求
    openai_request = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
    
    # Anthropic 请求
    anthropic_request = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
    
    # OpenAI Responses 请求
    responses_request = {
        "model": "gpt-4o",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "text", "text": "Hello!"}]}
        ]
    }
    
    print("OpenAI Chat:", engine.detect_protocol(openai_request))
    print("Anthropic:", engine.detect_protocol(anthropic_request))
    print("OpenAI Responses:", engine.detect_protocol(responses_request))
    
    # 转换 Anthropic 请求到 OpenAI Chat
    converted = engine.convert_request(anthropic_request)
    print("\n转换后的 Anthropic 请求:")
    print(json.dumps(converted, indent=2, ensure_ascii=False))
