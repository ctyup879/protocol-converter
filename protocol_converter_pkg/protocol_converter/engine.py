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
    # 目标后端类型:
    #   "openai"          - OpenAI Chat Completions 兼容 (/v1/chat/completions)
    #   "openai_responses" - OpenAI Responses 兼容 (/v1/responses)
    #   "anthropic"       - Anthropic Messages 兼容 (/v1/messages)
    backend_type: str = "openai"
    
    # 目标后端 URL
    # OpenAI Chat:      https://api.openai.com/v1/chat/completions
    # OpenAI Responses: https://api.openai.com/v1/responses
    # Anthropic:        https://api.anthropic.com/v1/messages
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
    
    # 模型映射表：模型名 -> 目标后端模型名
    model_mapping: Dict[str, str] = field(default_factory=dict)
    
    # Anthropic 特有配置
    # 推理地理区域 (Anthropic inference_geo 参数)
    inference_geo: Optional[str] = None
    
    # Anthropic API 版本
    anthropic_version: str = "2023-06-01"
    
    # OpenAI 特有配置
    # 提示缓存键 (替代 user 字段)
    prompt_cache_key: Optional[str] = None
    # 提示缓存保留策略 ("in_memory" | "24h")
    prompt_cache_retention: Optional[str] = None
    
    def get_model(self, model: str) -> str:
        """获取映射后的模型名"""
        return self.model_mapping.get(model, model)
    
    def get_auth_headers(self) -> Dict[str, str]:
        """获取认证请求头"""
        if self.backend_type == "anthropic":
            return {
                "x-api-key": self.api_key or "",
                "anthropic-version": self.anthropic_version
            }
        else:
            # openai 和 openai_responses 均使用 Bearer 认证
            return {
                "Authorization": f"Bearer {self.api_key or ''}"
            }


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
    - 根据后端类型转换为对应格式发送
    - 将响应转换回用户请求的协议格式
    - 支持流式和非流式响应
    - 支持三种后端: openai (Chat), openai_responses (Responses), anthropic
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
        self._pending_events: List[Dict[str, Any]] = []
    
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
            target_format: 目标格式，None 则根据后端类型自动选择
            
        Returns:
            Dict: 转换后的请求
        """
        # 检测源协议
        source_protocol = self.detect_protocol(request)
        
        # 确定目标格式
        if target_format is None:
            target_format = self._get_target_format()
        
        result = None
        
        # 根据源协议和目标格式进行转换
        if target_format == "openai_chat":
            if source_protocol == Protocol.ANTHROPIC:
                result = self.anthropic.to_openai_chat(request)
            elif source_protocol == Protocol.OPENAI_RESPONSES:
                result = self.openai_responses.to_openai_chat(request)
            else:
                # 已经是 OpenAI Chat 或未知
                result = request.copy() if isinstance(request, dict) else request
        
        elif target_format == "anthropic":
            # 转换为 Anthropic 格式
            if source_protocol == Protocol.OPENAI_CHAT:
                result = self._chat_to_anthropic_request(request)
            elif source_protocol == Protocol.OPENAI_RESPONSES:
                # 先转为 Chat，再转为 Anthropic
                chat_req = self.openai_responses.to_openai_chat(request)
                result = self._chat_to_anthropic_request(chat_req)
            else:
                result = request.copy() if isinstance(request, dict) else request
        
        elif target_format == "openai_responses":
            # 转换为 OpenAI Responses 格式
            if source_protocol == Protocol.OPENAI_CHAT:
                result = self.openai_responses.from_openai_chat_request(request)
            elif source_protocol == Protocol.ANTHROPIC:
                # 先转为 Chat，再转为 Responses
                chat_req = self.anthropic.to_openai_chat(request)
                result = self.openai_responses.from_openai_chat_request(chat_req)
            else:
                result = request.copy() if isinstance(request, dict) else request
        
        else:
            result = request
        
        # 应用模型映射
        if isinstance(result, dict) and "model" in result:
            original_model = result["model"]
            mapped_model = self.config.get_model(original_model)
            if mapped_model != original_model:
                result["model"] = mapped_model
        
        # 对 Chat 后端，将 developer 角色降级为 system（部分后端不支持 developer）
        if isinstance(result, dict) and target_format == "openai_chat":
            messages = result.get("messages", [])
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "developer":
                    msg["role"] = "system"
        
        # 合并 extra_body（将 extra_body 中的字段合并到请求中）
        if isinstance(result, dict) and "extra_body" in result:
            extra = result.pop("extra_body")
            if isinstance(extra, dict):
                for key, value in extra.items():
                    if key not in result:
                        result[key] = value
        
        return result
    
    def _get_target_format(self) -> str:
        """根据后端类型获取目标格式"""
        backend_type = self.config.backend_type
        if backend_type == "anthropic":
            return "anthropic"
        elif backend_type == "openai_responses":
            return "openai_responses"
        else:
            # openai 和默认都使用 openai_chat
            return "openai_chat"
    
    def _chat_to_anthropic_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """将 OpenAI Chat 请求转换为 Anthropic 格式"""
        messages = request.get("messages", [])
        anthropic_messages = []
        system_prompt = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system" or role == "developer":
                # system / developer 消息提取为顶级 system 参数
                if isinstance(content, str):
                    system_prompt = content
                elif isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    system_prompt = "\n".join(text_parts) if text_parts else None
                continue
            
            if role == "tool":
                # tool 消息转换为 user 消息中的 tool_result 块
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": content if isinstance(content, str) else str(content)
                    }]
                })
                continue
            
            if role == "assistant":
                assistant_content = []
                if content:
                    if isinstance(content, str):
                        assistant_content.append({"type": "text", "text": content})
                    elif isinstance(content, list):
                        assistant_content.extend([
                            {"type": "text", "text": b.get("text", "")}
                            for b in content if isinstance(b, dict) and b.get("type") == "text"
                        ])
                
                # 转换 tool_calls
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    func = tc.get("function", {})
                    args_str = func.get("arguments", "{}")
                    try:
                        args_obj = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except (json.JSONDecodeError, TypeError):
                        args_obj = {}
                    
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                        "name": func.get("name", ""),
                        "input": args_obj
                    })
                
                if assistant_content:
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": assistant_content
                    })
                continue
            
            # user 消息
            if isinstance(content, str):
                anthropic_messages.append({
                    "role": "user",
                    "content": content
                })
            elif isinstance(content, list):
                # 多模态内容
                anthropic_content = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type", "")
                    if block_type == "text":
                        anthropic_content.append({"type": "text", "text": block.get("text", "")})
                    elif block_type == "image_url":
                        url = block.get("image_url", {})
                        if isinstance(url, dict):
                            url_str = url.get("url", "")
                            if url_str.startswith("data:"):
                                # data:image/png;base64,...
                                parts = url_str.split(";", 1)
                                media_type = parts[0].replace("data:", "") if len(parts) > 1 else "image/png"
                                data = parts[1].replace("base64,", "") if len(parts) > 1 else ""
                                anthropic_content.append({
                                    "type": "image",
                                    "source": {"type": "base64", "media_type": media_type, "data": data}
                                })
                            else:
                                anthropic_content.append({
                                    "type": "image",
                                    "source": {"type": "url", "url": url_str}
                                })
                    else:
                        text = block.get("text", "")
                        if text:
                            anthropic_content.append({"type": "text", "text": text})
                
                if anthropic_content:
                    anthropic_messages.append({
                        "role": "user",
                        "content": anthropic_content
                    })
        
        # 构建 Anthropic 请求
        anthropic_request = {
            "model": request.get("model", "claude-sonnet-4-20250514"),
            "max_tokens": request.get("max_completion_tokens") or request.get("max_tokens", 4096),
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
        if request.get("stop"):
            anthropic_request["stop_sequences"] = request["stop"] if isinstance(request["stop"], list) else [request["stop"]]
        if request.get("metadata"):
            anthropic_request["metadata"] = request["metadata"]
        
        # 转换 tools
        chat_tools = request.get("tools", [])
        if chat_tools:
            anthropic_tools = []
            for tool in chat_tools:
                if isinstance(tool, dict) and tool.get("type") == "function":
                    func = tool.get("function", {})
                    at = {"name": func.get("name", ""), "input_schema": func.get("parameters", {"type": "object", "properties": {}})}
                    if func.get("description"):
                        at["description"] = func["description"]
                    anthropic_tools.append(at)
            if anthropic_tools:
                anthropic_request["tools"] = anthropic_tools
        
        # 转换 tool_choice
        chat_tool_choice = request.get("tool_choice")
        if chat_tool_choice:
            if isinstance(chat_tool_choice, str):
                tc_map = {"auto": "auto", "none": "none", "required": "any"}
                anthropic_request["tool_choice"] = tc_map.get(chat_tool_choice, chat_tool_choice)
            elif isinstance(chat_tool_choice, dict):
                if chat_tool_choice.get("type") == "function":
                    func = chat_tool_choice.get("function", {})
                    anthropic_request["tool_choice"] = {"type": "tool", "name": func.get("name", "")}
        
        # reasoning_effort -> thinking
        reasoning_effort = request.get("reasoning_effort")
        if reasoning_effort:
            effort_budget_map = {
                "none": 0,
                "minimal": 0,
                "low": 1024,
                "medium": 10000,
                "high": 32000,
                "xhigh": 64000,
            }
            budget = effort_budget_map.get(reasoning_effort, 10000)
            if budget > 0:
                anthropic_request["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget,
                }
            else:
                anthropic_request["thinking"] = {"type": "disabled"}
        
        # service_tier 映射
        service_tier = request.get("service_tier")
        if service_tier:
            tier_map = {"default": "standard_only", "auto": "auto"}
            anthropic_request["service_tier"] = tier_map.get(service_tier, service_tier)
        
        # inference_geo (Anthropic 特有)
        inference_geo = request.get("inference_geo") or self.config.inference_geo
        if inference_geo:
            anthropic_request["inference_geo"] = inference_geo
        
        return anthropic_request
    
    def convert_response(
        self, 
        response: Dict[str, Any],
        target_protocol: Protocol
    ) -> Dict[str, Any]:
        """
        转换响应到目标协议格式
        
        Args:
            response: 后端响应（格式取决于后端类型）
            target_protocol: 目标协议
            
        Returns:
            Dict: 目标协议格式的响应
        """
        # 先判断响应格式（基于后端类型）
        backend_format = self._get_target_format()
        
        # 如果后端是 anthropic 格式，需要先转换为目标协议
        if backend_format == "anthropic":
            if target_protocol == Protocol.ANTHROPIC:
                return response
            elif target_protocol == Protocol.OPENAI_CHAT:
                return self._anthropic_to_chat_response(response)
            elif target_protocol == Protocol.OPENAI_RESPONSES:
                chat_resp = self._anthropic_to_chat_response(response)
                return self.openai_responses.from_openai_chat(chat_resp)
        
        # 如果后端是 openai_responses 格式
        elif backend_format == "openai_responses":
            if target_protocol == Protocol.OPENAI_RESPONSES:
                return response
            elif target_protocol == Protocol.ANTHROPIC:
                # Responses -> Chat -> Anthropic
                chat_resp = self._responses_to_chat_response(response)
                return self.anthropic.from_openai_chat(chat_resp)
            elif target_protocol == Protocol.OPENAI_CHAT:
                return self._responses_to_chat_response(response)
        
        # 默认：后端是 openai_chat 格式
        if target_protocol == Protocol.ANTHROPIC:
            return self.anthropic.from_openai_chat(response)
        elif target_protocol == Protocol.OPENAI_RESPONSES:
            return self.openai_responses.from_openai_chat(response)
        elif target_protocol == Protocol.OPENAI_CHAT:
            return response
        
        return response
    
    def _anthropic_to_chat_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """将 Anthropic 响应转换为 OpenAI Chat 响应"""
        content = response.get("content", [])
        message_content = None
        tool_calls = []
        reasoning_content = None
        
        for block in content:
            if block.get("type") == "text":
                message_content = block.get("text", "")
            elif block.get("type") == "thinking":
                # thinking 块 -> reasoning_content (OpenAI o系列格式)
                reasoning_content = block.get("thinking", "")
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False)
                    }
                })
        
        # 映射停止原因
        stop_reason = response.get("stop_reason", "end_turn")
        reverse_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
            "pause_turn": "stop",
            "refusal": "content_filter",
        }
        finish_reason = reverse_map.get(stop_reason, "stop")
        
        usage = response.get("usage", {})
        
        # 构建 message
        message = {
            "role": "assistant",
            "content": message_content,
        }
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        if tool_calls:
            message["tool_calls"] = tool_calls
        
        return {
            "id": response.get("id", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response.get("model", ""),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            }
        }
    
    def _responses_to_chat_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """将 OpenAI Responses 响应转换为 OpenAI Chat 响应"""
        return self.openai_responses.to_chat_response(response)
    
    def convert_stream_chunk(
        self,
        chunk: Dict[str, Any],
        target_protocol: Protocol
    ) -> StreamChunk:
        """
        转换流式响应块
        
        Args:
            chunk: 后端流式块
            target_protocol: 目标协议
            
        Returns:
            StreamChunk: 转换后的块
        """
        if target_protocol == Protocol.ANTHROPIC:
            # Anthropic 流式转换可能返回多个事件
            data_list = self.anthropic.convert_stream_chunk(chunk)
            if isinstance(data_list, list) and len(data_list) > 0:
                # 返回第一个事件，后续事件需要通过 convert_stream_chunk_multi 获取
                data = data_list[0]
                event = data.get("type", "content_block_delta")
                # 存储额外事件供后续处理
                self._pending_events = data_list[1:] if len(data_list) > 1 else []
                return StreamChunk(event=event, data=data)
            elif isinstance(data_list, dict):
                event = data_list.get("type", "content_block_delta")
                return StreamChunk(event=event, data=data_list)
            return StreamChunk(event="ping", data={"type": "ping"})
        
        elif target_protocol == Protocol.OPENAI_RESPONSES:
            data = self.openai_responses.convert_stream_chunk(chunk)
            event = data.get("type", "response.output_text.delta")
            return StreamChunk(event=event, data=data)
        
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
    
    def convert_stream_chunk_multi(
        self,
        chunk: Dict[str, Any],
        target_protocol: Protocol
    ) -> List[StreamChunk]:
        """
        转换流式响应块，返回所有事件（支持一对多映射）
        
        Args:
            chunk: 后端流式块
            target_protocol: 目标协议
            
        Returns:
            List[StreamChunk]: 转换后的所有事件块
        """
        if target_protocol == Protocol.ANTHROPIC:
            data_list = self.anthropic.convert_stream_chunk(chunk)
            if isinstance(data_list, list):
                return [
                    StreamChunk(event=d.get("type", "content_block_delta"), data=d)
                    for d in data_list
                ]
            elif isinstance(data_list, dict):
                event = data_list.get("type", "content_block_delta")
                return [StreamChunk(event=event, data=data_list)]
            return [StreamChunk(event="ping", data={"type": "ping"})]
        
        elif target_protocol == Protocol.OPENAI_RESPONSES:
            data_list = self.openai_responses.convert_stream_chunk(chunk)
            if isinstance(data_list, list):
                return [
                    StreamChunk(event=d.get("type", "response.output_text.delta"), data=d)
                    for d in data_list
                ]
            data = data_list
            event = data.get("type", "response.output_text.delta")
            return [StreamChunk(event=event, data=data)]
        
        else:
            # OpenAI Chat - 1:1 映射
            return [self.convert_stream_chunk(chunk, target_protocol)]
    
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
        backend_request = self.convert_request(request)
        
        # 添加认证信息（使用 get_auth_headers 方法）
        headers = {
            "Content-Type": "application/json",
            **self.config.get_auth_headers(),
            **self.config.extra_headers
        }
        
        # 合并 extra_body 到请求体中
        if isinstance(backend_request, dict) and "extra_body" in backend_request:
            extra = backend_request.pop("extra_body")
            if isinstance(extra, dict):
                for key, value in extra.items():
                    if key not in backend_request:
                        backend_request[key] = value
        
        # 合并全局 extra_body
        if self.config.extra_body:
            for key, value in self.config.extra_body.items():
                if key not in backend_request:
                    backend_request[key] = value
        
        # 发送请求
        is_stream = backend_request.get("stream", False) or self.config.stream
        
        if is_stream:
            return self._handle_stream_response(
                backend_request, 
                headers, 
                source_protocol,
                http_client
            )
        else:
            response = await http_client(
                url=self.config.backend_url,
                method="POST",
                headers=headers,
                json=backend_request,
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
        
        # 添加 stream_options（仅对 OpenAI Chat 后端）
        backend_format = self._get_target_format()
        if backend_format == "openai_chat" and "stream_options" not in chat_request:
            chat_request["stream_options"] = {"include_usage": True}
        
        # 对于 Anthropic 后端，重置流式状态
        if backend_format == "anthropic":
            self.anthropic.reset_stream_state()
        
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
                    # 根据目标协议发送结束事件
                    if source_protocol == Protocol.ANTHROPIC:
                        yield StreamChunk(event="message_stop", data={"type": "message_stop"}).to_anthropic_sse()
                    elif source_protocol == Protocol.OPENAI_RESPONSES:
                        yield StreamChunk(event="response.completed", data={"type": "response.completed"}).to_sse()
                    else:
                        yield "data: [DONE]\n\n"
                    break
                
                try:
                    chunk = json.loads(data_str)
                    # 使用多事件转换，以支持 Anthropic 的一对多映射
                    stream_chunks = self.convert_stream_chunk_multi(chunk, source_protocol)
                    
                    # 根据目标协议选择输出格式
                    for sc in stream_chunks:
                        if source_protocol == Protocol.ANTHROPIC:
                            yield sc.to_anthropic_sse()
                        elif source_protocol == Protocol.OPENAI_RESPONSES:
                            yield sc.to_sse()
                        else:
                            yield sc.to_openai_chunk()
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
