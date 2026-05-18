"""
协议转换引擎 - 统一处理各种协议的转换
"""

import json
import copy
import time
import uuid
from typing import Any, Dict, List, Optional, Union, Iterator, Callable
from dataclasses import dataclass, field
from enum import Enum

from .protocol_detector import ProtocolDetector, Protocol
from .openai_chat import OpenAIChatConverter
from .openai_responses import OpenAIResponsesConverter
from .anthropic import AnthropicConverter
from .exceptions import ProtocolConversionError, UnsupportedProtocolError, InvalidRequestError


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
    
    # 流式响应中是否将模型名反向替换为原始请求中的模型名
    # True: 用户看到原始模型名；False: 用户看到后端的模型名（默认）
    reverse_model_mapping_in_stream: bool = False
    
    # Anthropic 特有配置
    # 推理地理区域 (Anthropic inference_geo 参数)
    inference_geo: Optional[str] = None
    
    # Anthropic API 版本
    anthropic_version: str = "2023-06-01"
    
    # OpenAI 特有配置
    # 提示缓存键 (替代 user 字段)
    prompt_cache_key: Optional[str] = None
    # 提示缓存保留策略 ("in-memory" | "24h")
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

    def _is_same_protocol(self, source_protocol: Protocol, target_format: str) -> bool:
        """判断源协议和目标格式是否为同一协议"""
        return (
            (source_protocol == Protocol.OPENAI_CHAT and target_format == "openai_chat")
            or (source_protocol == Protocol.ANTHROPIC and target_format == "anthropic")
            or (source_protocol == Protocol.OPENAI_RESPONSES and target_format == "openai_responses")
        )

    def _needs_request_modification(self, request: Dict[str, Any], target_format: str) -> bool:
        """
        判断同协议请求是否需要修改。
        即使同协议，若存在以下情况也需要做最小修改：
        - 模型映射（请求模型与后端模型不一致）
        - developer 角色需降级为 system
        - 请求自身包含 extra_body 需要合并到顶层
        """
        if not isinstance(request, dict):
            return False

        # 模型映射
        model = request.get("model")
        if model and self.config.get_model(model) != model:
            return True

        # developer 降级
        if target_format == "openai_chat":
            for msg in request.get("messages", []):
                if isinstance(msg, dict) and msg.get("role") == "developer":
                    return True

        # 请求自身的 extra_body 需要合并
        if "extra_body" in request:
            return True

        return False
    
    def convert_request(
        self, 
        request: Dict[str, Any], 
        target_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        转换请求到目标格式
        
        同协议且无需任何修改时直接透传（浅拷贝），
        避免不必要的 deepcopy 和字段变更。
        
        Args:
            request: 输入请求
            target_format: 目标格式，None 则根据后端类型自动选择
            
        Returns:
            Dict: 转换后的请求
            
        Raises:
            InvalidRequestError: 如果 request 不是字典
            UnsupportedProtocolError: 如果检测到的协议类型为 UNKNOWN
        """
        if not isinstance(request, dict):
            raise InvalidRequestError(
                f"request must be a dict, got {type(request).__name__}",
                field="request"
            )
        # 检测源协议
        source_protocol = self.detect_protocol(request)
        
        # 确定目标格式
        if target_format is None:
            target_format = self._get_target_format()
        
        # 同协议快速路径：尽可能透传，只做必要的最小修改
        if self._is_same_protocol(source_protocol, target_format):
            needs_mod = self._needs_request_modification(request, target_format)
            if not needs_mod:
                # 完全透传：返回浅拷贝，避免调用方被后续修改污染
                if isinstance(request, dict):
                    return dict(request)
                return request
            # 需要修改：深拷贝后只做最小变更（避免修改原始请求的嵌套对象）
            result = copy.deepcopy(request) if isinstance(request, dict) else request
            # 应用模型映射
            if isinstance(result, dict) and "model" in result:
                original_model = result["model"]
                mapped_model = self.config.get_model(original_model)
                if mapped_model != original_model:
                    result["model"] = mapped_model
            # developer 降级
            if isinstance(result, dict) and target_format == "openai_chat":
                messages = result.get("messages", [])
                for msg in messages:
                    if isinstance(msg, dict) and msg.get("role") == "developer":
                        msg["role"] = "system"
            # 合并 extra_body
            if isinstance(result, dict) and "extra_body" in result:
                extra = result.pop("extra_body")
                if isinstance(extra, dict):
                    for key, value in extra.items():
                        if key not in result:
                            result[key] = value
            return result
        
        result = None
        
        # 根据源协议和目标格式进行转换
        if target_format == "openai_chat":
            if source_protocol == Protocol.ANTHROPIC:
                result = self.anthropic.to_openai_chat(request)
            elif source_protocol == Protocol.OPENAI_RESPONSES:
                result = self.openai_responses.to_openai_chat(request)
            else:
                result = copy.deepcopy(request) if isinstance(request, dict) else request
        
        elif target_format == "anthropic":
            if source_protocol == Protocol.OPENAI_CHAT:
                result = self._chat_to_anthropic_request(request)
            elif source_protocol == Protocol.OPENAI_RESPONSES:
                result = self.openai_responses.to_anthropic_request(request)
            else:
                result = copy.deepcopy(request) if isinstance(request, dict) else request
        
        elif target_format == "openai_responses":
            if source_protocol == Protocol.OPENAI_CHAT:
                result = self.openai_responses.from_openai_chat_request(request)
            elif source_protocol == Protocol.ANTHROPIC:
                # 直接转换路径，避免经 Chat 中转丢失 reasoning/output_format 等数据
                result = self._anthropic_to_responses_request(request)
            else:
                result = copy.deepcopy(request) if isinstance(request, dict) else request
        
        else:
            result = request
        
        # 应用模型映射
        if isinstance(result, dict) and "model" in result:
            original_model = result["model"]
            mapped_model = self.config.get_model(original_model)
            if mapped_model != original_model:
                result["model"] = mapped_model
        
        # 对 Chat 后端，将 developer 角色降级为 system
        if isinstance(result, dict) and target_format == "openai_chat":
            messages = result.get("messages", [])
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "developer":
                    msg["role"] = "system"
        
        # 合并 extra_body
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
        system_parts = []
        system_blocks = []  # 保留含 cache_control 的 system TextBlockParam
        has_system_cache_control = False
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system" or role == "developer":
                # system / developer 消息提取为顶级 system 参数
                if isinstance(content, str):
                    if content:
                        system_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text = block.get("text", "")
                                if block.get("cache_control"):
                                    # 保留含 cache_control 的块为 Anthropic TextBlockParam 格式
                                    has_system_cache_control = True
                                    tb = {"type": "text", "text": text}
                                    if isinstance(block["cache_control"], dict):
                                        tb["cache_control"] = block["cache_control"]
                                    system_blocks.append(tb)
                                else:
                                    system_parts.append(text)
                            else:
                                # 其他类型降级为文本
                                text = block.get("text", "")
                                if text:
                                    system_parts.append(text)
                        else:
                            # 非字典块降级为字符串
                            text = str(block) if block is not None else ""
                            if text:
                                system_parts.append(text)
                continue
            
            if role == "tool":
                # tool 消息转换为 user 消息中的 tool_result 块
                is_error = False
                if isinstance(content, str):
                    tool_content = content
                    # 检测 [Error] 前缀标记，反向映射为 Anthropic is_error 字段
                    if tool_content.startswith("[Error] "):
                        is_error = True
                        tool_content = tool_content[8:]  # 去掉 "[Error] " 前缀
                elif isinstance(content, list):
                    # Chat tool 消息的 content 可能是列表（多模态工具结果）
                    # 转换为 Anthropic tool_result 的 content 块列表
                    content_blocks = []
                    for b in content:
                        if isinstance(b, dict):
                            b_type = b.get("type", "")
                            if b_type == "text":
                                content_blocks.append({"type": "text", "text": b.get("text", "")})
                            else:
                                # 其他类型降级为文本
                                text = b.get("text", "")
                                if text:
                                    content_blocks.append({"type": "text", "text": text})
                        else:
                            content_blocks.append({"type": "text", "text": str(b)})
                    if content_blocks:
                        tool_content = content_blocks
                    elif isinstance(content, list) and not content:
                        tool_content = ""
                    else:
                        tool_content = str(content)
                    # 检测列表中第一项文本是否为 [Error] 标记
                    if (isinstance(tool_content, list) and len(tool_content) > 0
                            and isinstance(tool_content[0], dict)
                            and tool_content[0].get("text", "").startswith("[Error] ")):
                        is_error = True
                        tool_content[0]["text"] = tool_content[0]["text"][8:]
                else:
                    tool_content = str(content) if content is not None else ""
                
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": tool_content
                }
                if is_error:
                    tool_result_block["is_error"] = True
                anthropic_messages.append({
                    "role": "user",
                    "content": [tool_result_block]
                })
                continue
            
            if role == "assistant":
                assistant_content = []
                if content is not None:
                    if isinstance(content, str):
                        assistant_content.append({"type": "text", "text": content})
                    elif isinstance(content, list):
                        for b in content:
                            if not isinstance(b, dict):
                                continue
                            b_type = b.get("type", "")
                            if b_type == "text":
                                text_block = {"type": "text", "text": b.get("text", "")}
                                # 保留 annotations -> citations 映射
                                if b.get("annotations"):
                                    text_block["citations"] = b["annotations"]
                                assistant_content.append(text_block)
                            elif b_type == "image_url":
                                # Chat image_url -> Anthropic image 块
                                url = b.get("image_url", {})
                                if isinstance(url, dict):
                                    url_str = url.get("url", "")
                                    if url_str.startswith("data:"):
                                        parts = url_str.split(";", 1)
                                        media_type = parts[0].replace("data:", "") if len(parts) > 1 else "image/png"
                                        data = parts[1].replace("base64,", "") if len(parts) > 1 else ""
                                        assistant_content.append({
                                            "type": "image",
                                            "source": {"type": "base64", "media_type": media_type, "data": data}
                                        })
                                    else:
                                        assistant_content.append({
                                            "type": "image",
                                            "source": {"type": "url", "url": url_str}
                                        })
                            elif b_type == "file":
                                file_data = b.get("file", {})
                                file_url = file_data.get("file_data", "")
                                mime_type = file_data.get("mime_type", "application/octet-stream")
                                if file_url:
                                    if file_url.startswith("data:"):
                                        parts = file_url.split(";", 1)
                                        media_type = parts[0].replace("data:", "") if len(parts) > 1 else mime_type
                                        data = parts[1].replace("base64,", "") if len(parts) > 1 else ""
                                        assistant_content.append({
                                            "type": "document",
                                            "source": {"type": "base64", "media_type": media_type, "data": data}
                                        })
                                    elif file_url.startswith("http"):
                                        assistant_content.append({
                                            "type": "document",
                                            "source": {"type": "url", "url": file_url}
                                        })
                                    else:
                                        assistant_content.append({
                                            "type": "document",
                                            "source": {"type": "base64", "media_type": mime_type, "data": file_url}
                                        })
                            else:
                                # 未知类型降级为文本
                                text = b.get("text", "")
                                if text:
                                    assistant_content.append({"type": "text", "text": text})
                
                # 处理 reasoning_content -> thinking/redacted_thinking 块
                # Anthropic SDK: ThinkingBlock 必须包含 thinking + signature 字段
                reasoning_content = msg.get("reasoning_content")
                if reasoning_content:
                    # 检测 [redacted_thinking: ...] 格式，恢复为 redacted_thinking 块
                    if (isinstance(reasoning_content, str) 
                            and reasoning_content.startswith("[redacted_thinking: ") 
                            and reasoning_content.endswith("]")):
                        redacted_data = reasoning_content[20:-1]  # 去掉 "[redacted_thinking: " 和 "]"
                        assistant_content.insert(0, {
                            "type": "redacted_thinking",
                            "data": redacted_data,
                        })
                    else:
                        assistant_content.insert(0, {
                            "type": "thinking",
                            "thinking": reasoning_content,
                            "signature": "",  # Chat API 不提供 signature，设为空字符串
                        })
                
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
                else:
                    # 无内容块但角色为 assistant，保留空消息以维持消息序列
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": []
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
                    elif block_type == "file":
                        # Chat file 内容块 -> Anthropic document 块
                        file_data = block.get("file", {})
                        file_url = file_data.get("file_data", "")
                        mime_type = file_data.get("mime_type", "application/octet-stream")
                        if file_url:
                            if file_url.startswith("data:"):
                                # data:application/pdf;base64,...
                                parts = file_url.split(";", 1)
                                media_type = parts[0].replace("data:", "") if len(parts) > 1 else mime_type
                                data = parts[1].replace("base64,", "") if len(parts) > 1 else ""
                                anthropic_content.append({
                                    "type": "document",
                                    "source": {"type": "base64", "media_type": media_type, "data": data}
                                })
                            elif file_url.startswith("http"):
                                anthropic_content.append({
                                    "type": "document",
                                    "source": {"type": "url", "url": file_url}
                                })
                            else:
                                # 可能是纯 base64 数据
                                anthropic_content.append({
                                    "type": "document",
                                    "source": {"type": "base64", "media_type": mime_type, "data": file_url}
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
        
        # 确保 anthropic_messages 不为空（Anthropic API 要求至少一条消息）
        if not anthropic_messages:
            anthropic_messages.append({"role": "user", "content": ""})
        
        # 构建 Anthropic 请求
        # 注意：max_tokens=0 是有效值（仅预热缓存），必须使用 is not None 判断保留 0 值
        max_completion = request.get("max_completion_tokens")
        max_tok = request.get("max_tokens")
        anthropic_request = {
            "model": request.get("model") or "claude-sonnet-4-20250514",
            "max_tokens": max_completion if max_completion is not None else (max_tok if max_tok is not None else 4096),
            "messages": anthropic_messages,
        }
        
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        if has_system_cache_control:
            # 含 cache_control 的 system 块必须使用 TextBlockParam[] 格式
            all_blocks = []
            if system_prompt:
                all_blocks.append({"type": "text", "text": system_prompt})
            all_blocks.extend(system_blocks)
            if all_blocks:
                anthropic_request["system"] = all_blocks
        elif system_prompt:
            anthropic_request["system"] = system_prompt
        if request.get("stream") is not None:
            anthropic_request["stream"] = request["stream"]
        if request.get("temperature") is not None:
            anthropic_request["temperature"] = request["temperature"]
        if request.get("top_p") is not None:
            anthropic_request["top_p"] = request["top_p"]
        if request.get("stop") is not None:
            stop_val = request["stop"]
            if isinstance(stop_val, list) and stop_val:
                valid_stops = [s for s in stop_val if s is not None and s != ""]
                if valid_stops:
                    anthropic_request["stop_sequences"] = valid_stops
            elif isinstance(stop_val, str) and stop_val:
                anthropic_request["stop_sequences"] = [stop_val]
        
        # metadata 合并（请求中的 metadata 可能已有其他字段）
        req_metadata = request.get("metadata")
        if req_metadata and isinstance(req_metadata, dict):
            if "metadata" not in anthropic_request:
                anthropic_request["metadata"] = {}
            anthropic_request["metadata"].update(req_metadata)

        # user 字段 -> Anthropic metadata.user_id（优先级高于 req_metadata）
        user_id = request.get("user")
        if user_id:
            if "metadata" not in anthropic_request:
                anthropic_request["metadata"] = {}
            anthropic_request["metadata"]["user_id"] = user_id
        
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
                    # Anthropic SDK ToolParam: strict 字段 (Ref: anthropic-sdk-python ToolParam)
                    if func.get("strict") is not None:
                        at["strict"] = func["strict"]
                    anthropic_tools.append(at)
            if anthropic_tools:
                anthropic_request["tools"] = anthropic_tools
        
        # 转换 tool_choice
        chat_tool_choice = request.get("tool_choice")
        # Chat parallel_tool_calls: False → Anthropic tool_choice.disable_parallel_tool_use: True
        parallel_tool_calls = request.get("parallel_tool_calls")
        disable_parallel = parallel_tool_calls is False
        
        if chat_tool_choice:
            if isinstance(chat_tool_choice, str):
                tc_map = {"auto": "auto", "none": "none", "required": "any"}
                tc_value = tc_map.get(chat_tool_choice, chat_tool_choice)
                if disable_parallel and tc_value in ("auto", "any"):
                    anthropic_request["tool_choice"] = {"type": tc_value, "disable_parallel_tool_use": True}
                else:
                    anthropic_request["tool_choice"] = tc_value
            elif isinstance(chat_tool_choice, dict):
                if chat_tool_choice.get("type") == "function":
                    func = chat_tool_choice.get("function", {})
                    tc_dict = {"type": "tool", "name": func.get("name", "")}
                    if disable_parallel:
                        tc_dict["disable_parallel_tool_use"] = True
                    anthropic_request["tool_choice"] = tc_dict
        elif disable_parallel:
            # parallel_tool_calls=False but no explicit tool_choice
            # Anthropic requires disable_parallel_tool_use inside tool_choice,
            # so set tool_choice to {"type":"auto","disable_parallel_tool_use":True}
            anthropic_request["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        
        # reasoning_effort -> thinking
        reasoning_effort = request.get("reasoning_effort")
        if reasoning_effort:
            effort_budget_map = {
                "none": 0,
                "minimal": 1024,
                "low": 1024,
                "medium": 10000,
                "high": 32000,
                "xhigh": 64000,
            }
            budget = effort_budget_map.get(reasoning_effort, 10000)
            if budget > 0:
                thinking_config = {
                    "type": "enabled",
                    "budget_tokens": budget,
                }
                # 如果 extra_body 中有原始 thinking 配置（含 display），保留
                if isinstance(request.get("extra_body"), dict):
                    original_thinking = request["extra_body"].get("thinking")
                    if isinstance(original_thinking, dict) and original_thinking.get("display"):
                        thinking_config["display"] = original_thinking["display"]
                    # 如果 extra_body 中有 reasoning 配置（从 Responses 转换过来的），提取 summary → display
                    original_reasoning = request["extra_body"].get("reasoning")
                    if isinstance(original_reasoning, dict) and original_reasoning.get("summary") is not None:
                        summary = original_reasoning["summary"]
                        if summary in ("concise", "detailed"):
                            thinking_config["display"] = "summarized"
                        elif summary == "omitted":
                            thinking_config["display"] = "omitted"
                anthropic_request["thinking"] = thinking_config
            else:
                anthropic_request["thinking"] = {"type": "disabled"}
        # 如果请求中直接包含 thinking 参数（从 Anthropic 转换过来的），优先使用
        elif isinstance(request.get("extra_body"), dict) and isinstance(request["extra_body"].get("thinking"), dict):
            original_thinking = request["extra_body"]["thinking"]
            # enabled 或 adaptive 时恢复（adaptive 无 budget_tokens）
            if original_thinking.get("type") == "enabled":
                thinking_config = {
                    "type": "enabled",
                    "budget_tokens": original_thinking.get("budget_tokens", 10000),
                }
                if original_thinking.get("display"):
                    thinking_config["display"] = original_thinking["display"]
                anthropic_request["thinking"] = thinking_config
            elif original_thinking.get("type") == "adaptive":
                # adaptive 类型直接恢复，不添加 budget_tokens
                thinking_config = {"type": "adaptive"}
                if original_thinking.get("display"):
                    thinking_config["display"] = original_thinking["display"]
                anthropic_request["thinking"] = thinking_config
        
        # service_tier 映射
        service_tier = request.get("service_tier")
        if service_tier:
            tier_map = {
                "default": "standard_only",
                "auto": "auto",
                "flex": "standard_only",
                "scale": "standard_only",
                "priority": "standard_only",
            }
            anthropic_request["service_tier"] = tier_map.get(service_tier, service_tier)
        # 从 extra_body 中恢复原始 Anthropic service_tier（解决往返转换信息丢失）
        if isinstance(request.get("extra_body"), dict):
            original_st = request["extra_body"].get("original_service_tier")
            if original_st and original_st in ("auto", "standard_only"):
                anthropic_request["service_tier"] = original_st
        
        # inference_geo (Anthropic 特有)
        inference_geo = request.get("inference_geo") or self.config.inference_geo
        if inference_geo:
            anthropic_request["inference_geo"] = inference_geo
        
        # Anthropic 特有参数恢复（从 Anthropic→Chat 转换时通过 extra_body 合并到顶层）
        # 这些参数在 Chat API 中不存在，但在 Anthropic→Chat→Anthropic 往返转换中需要恢复
        if request.get("top_k") is not None:
            anthropic_request["top_k"] = request["top_k"]
        if request.get("container") is not None:
            anthropic_request["container"] = request["container"]
        if request.get("cache_control") is not None:
            anthropic_request["cache_control"] = request["cache_control"]
        if request.get("output_config") is not None:
            anthropic_request["output_config"] = request["output_config"]
        
        # 恢复 Anthropic 服务器工具（从 extra_body.anthropic_server_tools 合并到顶层的）
        anthropic_server_tools = request.get("anthropic_server_tools")
        if isinstance(anthropic_server_tools, list) and anthropic_server_tools:
            existing_tools = anthropic_request.get("tools", [])
            existing_ids = {t.get("type") for t in existing_tools if isinstance(t, dict)}
            for st in anthropic_server_tools:
                if isinstance(st, dict) and st.get("type") not in existing_ids:
                    existing_tools.append(st)
            anthropic_request["tools"] = existing_tools
        
        # Chat API 不支持的参数 -> extra_body
        extra = {}
        if request.get("parallel_tool_calls") is not None:
            extra["parallel_tool_calls"] = request["parallel_tool_calls"]
        if request.get("frequency_penalty") is not None:
            extra["frequency_penalty"] = request["frequency_penalty"]
        if request.get("presence_penalty") is not None:
            extra["presence_penalty"] = request["presence_penalty"]
        if request.get("seed") is not None:
            extra["seed"] = request["seed"]
        if request.get("n") is not None and request["n"] != 1:
            extra["n"] = request["n"]
        # response_format → Anthropic output_format (结构化输出映射)
        # Chat: {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}
        # Anthropic: {"type": "json_schema", "name": "...", "schema": {...}}
        response_format = request.get("response_format")
        if response_format and isinstance(response_format, dict):
            fmt_type = response_format.get("type", "")
            if fmt_type == "json_schema":
                json_schema = response_format.get("json_schema") or {}
                output_format = {"type": "json_schema"}
                if json_schema.get("name"):
                    output_format["name"] = json_schema["name"]
                if json_schema.get("schema"):
                    output_format["schema"] = json_schema["schema"]
                if json_schema.get("strict") is not None:
                    output_format["strict"] = json_schema["strict"]
                anthropic_request["output_format"] = output_format
            elif fmt_type == "json_object":
                anthropic_request["output_format"] = {"type": "json_object"}
            # 保留原始 response_format 在 extra_body 中以兼容后端
            extra["response_format"] = response_format
        if request.get("logprobs") is not None:
            extra["logprobs"] = request["logprobs"]
        if request.get("top_logprobs") is not None:
            extra["top_logprobs"] = request["top_logprobs"]
        if request.get("logit_bias") is not None:
            extra["logit_bias"] = request["logit_bias"]
        if request.get("web_search_options") is not None:
            extra["web_search_options"] = request["web_search_options"]
        # Chat API 新增参数（Anthropic 不直接支持）
        if request.get("verbosity") is not None:
            extra["verbosity"] = request["verbosity"]
        if request.get("modalities") is not None:
            extra["modalities"] = request["modalities"]
        if request.get("audio") is not None:
            extra["audio"] = request["audio"]
        if request.get("prediction") is not None:
            extra["prediction"] = request["prediction"]
        if request.get("safety_identifier") is not None:
            extra["safety_identifier"] = request["safety_identifier"]
        if request.get("prompt_cache_key") is not None:
            extra["prompt_cache_key"] = request["prompt_cache_key"]
        if request.get("prompt_cache_retention") is not None:
            extra["prompt_cache_retention"] = request["prompt_cache_retention"]
        if request.get("store") is not None:
            extra["store"] = request["store"]
        if extra:
            anthropic_request["extra_body"] = extra
        
        return anthropic_request
    
    def _anthropic_to_responses_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """将 Anthropic 请求直接转换为 OpenAI Responses 格式（避免经 Chat 中转丢失数据）"""
        # 先转换为 Chat 中间格式以复用已有的 Anthropic→Chat 转换逻辑
        chat_request = self.anthropic.to_openai_chat(request)
        # 再从 Chat 转换为 Responses 格式
        result = self.openai_responses.from_openai_chat_request(chat_request)
        
        # 恢复 Chat 中转丢失的 Anthropic 特有数据
        # 1. thinking 参数中的 display 和 adaptive 类型
        thinking = request.get("thinking")
        if thinking and isinstance(thinking, dict):
            thinking_type = thinking.get("type")
            if thinking_type in ("enabled", "adaptive"):
                effort = chat_request.get("reasoning_effort", "medium")
                reasoning_config = {"effort": effort}
                # reasoning.summary 映射
                display = thinking.get("display")
                if display == "summarized":
                    reasoning_config["summary"] = "concise"
                elif display == "omitted":
                    reasoning_config["summary"] = "omitted"
                # adaptive 类型标记
                if thinking_type == "adaptive":
                    reasoning_config["effort"] = "medium"  # adaptive 无确切 effort，默认 medium
                # 保留 from_openai_chat_request 生成的 generate_summary 等字段
                existing_reasoning = result.get("reasoning", {})
                if isinstance(existing_reasoning, dict) and existing_reasoning.get("generate_summary") is not None:
                    reasoning_config["generate_summary"] = existing_reasoning["generate_summary"]
                result["reasoning"] = reasoning_config
            elif thinking_type == "disabled":
                result["reasoning"] = {"effort": "none"}
        
        # 2. output_format → text.format（Anthropic 结构化输出）
        output_format = request.get("output_format")
        if output_format and isinstance(output_format, dict):
            fmt_type = output_format.get("type", "")
            text_format = {}
            if fmt_type == "json_schema":
                text_format = {"type": "json_schema"}
                if output_format.get("name"):
                    text_format["name"] = output_format["name"]
                if output_format.get("schema"):
                    text_format["schema"] = output_format["schema"]
                if output_format.get("strict") is not None:
                    text_format["strict"] = output_format["strict"]
            elif fmt_type == "json_object":
                text_format = {"type": "json_object"}
            if text_format:
                result["text"] = {"format": text_format}
        
        # 3. parallel_tool_calls → disable_parallel_tool_use 反向映射
        # Chat 中转时 parallel_tool_calls 已被设置，但如果 Anthropic 原始请求
        # 中 tool_choice.disable_parallel_tool_use 为 True，需要确保 Responses 也禁用并行
        tool_choice_raw = request.get("tool_choice")
        if isinstance(tool_choice_raw, dict) and tool_choice_raw.get("disable_parallel_tool_use"):
            result["parallel_tool_calls"] = False
        
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
        target_protocol: Protocol,
        original_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        转换响应到目标协议格式
        
        Args:
            response: 后端响应（格式取决于后端类型）
            target_protocol: 目标协议
            original_model: 用户请求中的原始模型名（用于反向替换响应中的 model）
            
        Returns:
            Dict: 目标协议格式的响应
            
        Raises:
            InvalidRequestError: 如果 response 不是字典
            UnsupportedProtocolError: 如果 target_protocol 为 UNKNOWN
        """
        if not isinstance(response, dict):
            raise InvalidRequestError(
                f"response must be a dict, got {type(response).__name__}",
                field="response"
            )
        if target_protocol == Protocol.UNKNOWN:
            raise UnsupportedProtocolError(
                "Cannot convert response to UNKNOWN protocol",
                target_protocol="unknown"
            )
        # 先判断响应格式（基于后端类型）
        backend_format = self._get_target_format()
        
        result = response
        
        # 如果后端是 anthropic 格式，需要先转换为目标协议
        if backend_format == "anthropic":
            if target_protocol == Protocol.ANTHROPIC:
                result = response
            elif target_protocol == Protocol.OPENAI_CHAT:
                result = self._anthropic_to_chat_response(response)
            elif target_protocol == Protocol.OPENAI_RESPONSES:
                # 直接转换，保留 redacted_thinking (data字段) → encrypted_content
                result = self.anthropic.to_openai_responses(response)
        
        # 如果后端是 openai_responses 格式
        elif backend_format == "openai_responses":
            if target_protocol == Protocol.OPENAI_RESPONSES:
                result = response
            elif target_protocol == Protocol.ANTHROPIC:
                # Responses -> Anthropic: 直接转换保留 summary 和 encrypted_content
                result = self.openai_responses.to_anthropic(response)
            elif target_protocol == Protocol.OPENAI_CHAT:
                result = self._responses_to_chat_response(response)
        
        # 默认：后端是 openai_chat 格式
        else:
            if target_protocol == Protocol.ANTHROPIC:
                result = self.anthropic.from_openai_chat(response)
            elif target_protocol == Protocol.OPENAI_RESPONSES:
                result = self.openai_responses.from_openai_chat(response)
            elif target_protocol == Protocol.OPENAI_CHAT:
                result = response
        
        # 反向模型映射：若配置启用且模型不一致，将响应中的 model 替换为原始模型名
        if (self.config.reverse_model_mapping_in_stream
                and original_model is not None
                and isinstance(result, dict)
                and "model" in result):
            mapped_model = self.config.get_model(original_model)
            if mapped_model != original_model and result.get("model") == mapped_model:
                result = dict(result)
                result["model"] = original_model
        
        return result
    
    def _anthropic_to_chat_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """将 Anthropic 响应转换为 OpenAI Chat 响应"""
        content = response.get("content", [])
        message_content = None
        message_annotations = None
        tool_calls = []
        reasoning_content = None
        
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text_val = block.get("text", "")
                # 合并多个 text 块的内容，避免后覆盖前
                if message_content is None:
                    message_content = text_val
                else:
                    message_content = f"{message_content}\n{text_val}"
                # 保留 citations -> Chat annotations（多个 text 块的 citations 合并）
                if block.get("citations"):
                    if message_annotations is None:
                        message_annotations = []
                    message_annotations.extend(block["citations"])
            elif block.get("type") == "thinking":
                # thinking 块 -> reasoning_content (OpenAI o系列格式)
                thinking_text = block.get("thinking", "")
                if thinking_text:
                    reasoning_content = thinking_text
            elif block.get("type") == "redacted_thinking":
                # 脱敏思考块 - 保留 data 字段用于往返转换
                # Chat API 无等价字段，但可保留在 reasoning_content 中
                data = block.get("data") or block.get("signature", "")
                if data:
                    # 标记为脱敏思考内容，后续转换时可恢复
                    reasoning_content = f"[redacted_thinking: {data}]"
            elif block.get("type") == "tool_use":
                tc = {
                    "id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False)
                    }
                }
                if block.get("caller"):
                    tc["caller"] = block["caller"]
                tool_calls.append(tc)
            elif block.get("type") == "server_tool_use":
                tc = {
                    "id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False)
                    }
                }
                if block.get("caller"):
                    tc["caller"] = block["caller"]
                tool_calls.append(tc)
        
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
        
        # 构建 Chat usage (含 prompt_tokens_details 和 completion_tokens_details)
        chat_usage = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        }
        
        # 缓存 token 信息
        cached_tokens = usage.get("cache_read_input_tokens", 0)
        cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)
        if cached_tokens or cache_creation_tokens:
            prompt_tokens_details = {}
            if cached_tokens:
                prompt_tokens_details["cached_tokens"] = cached_tokens
            chat_usage["prompt_tokens_details"] = prompt_tokens_details
        
        # Anthropic usage 中的 service_tier -> Chat service_tier
        # Anthropic response: "standard"|"priority"|"batch"
        # Chat response: "auto"|"default"|"flex"|"scale"|"priority"
        anthropic_service_tier = usage.get("service_tier")
        if anthropic_service_tier:
            from .anthropic import AnthropicConverter
            chat_usage["service_tier"] = AnthropicConverter.RESPONSE_SERVICE_TIER_MAP.get(
                anthropic_service_tier, anthropic_service_tier
            )
        if usage.get("inference_geo"):
            chat_usage["inference_geo"] = usage["inference_geo"]
        
        # Anthropic server_tool_use -> Chat 无等价字段，保留在 usage 中
        if usage.get("server_tool_use"):
            chat_usage["server_tool_use"] = usage["server_tool_use"]
        
        # Anthropic cache_creation 详情 (breakdown by TTL)
        if usage.get("cache_creation"):
            chat_usage["cache_creation"] = usage["cache_creation"]
        
        # 保留 Anthropic 特有的 usage 字段（用于往返转换场景）
        if usage.get("cache_creation_input_tokens"):
            chat_usage["cache_creation_input_tokens"] = usage["cache_creation_input_tokens"]
        if usage.get("cache_read_input_tokens"):
            chat_usage["cache_read_input_tokens"] = usage["cache_read_input_tokens"]
        
        # 构建 message
        message = {
            "role": "assistant",
            "content": message_content,
        }
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        if tool_calls:
            message["tool_calls"] = tool_calls
        # 保留 annotations（从 Anthropic citations 转换）
        if message_annotations is not None:
            message["annotations"] = message_annotations
        
        result = {
            "id": response.get("id", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response.get("model", ""),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }],
            "usage": chat_usage
        }
        
        # 保留 Anthropic container 字段（在顶层响应中）
        if response.get("container"):
            result["container"] = response["container"]
        
        return result
    
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
        
        # 记录模型映射关系，供流式响应反向替换使用
        original_model = request.get("model") if isinstance(request, dict) else None
        mapped_model = backend_request.get("model") if isinstance(backend_request, dict) else None
        
        if is_stream:
            return self._handle_stream_response(
                backend_request, 
                headers, 
                source_protocol,
                http_client,
                original_model=original_model,
                mapped_model=mapped_model,
            )
        else:
            response = await http_client(
                url=self.config.backend_url,
                method="POST",
                headers=headers,
                json=backend_request,
                timeout=self.config.timeout
            )
            
            # 转换响应（传入 original_model 支持反向模型映射）
            return self.convert_response(response, source_protocol, original_model=original_model)
    
    async def _handle_stream_response(
        self,
        chat_request: Dict[str, Any],
        headers: Dict[str, str],
        source_protocol: Protocol,
        http_client: Callable,
        original_model: Optional[str] = None,
        mapped_model: Optional[str] = None,
    ) -> Iterator[StreamChunk]:
        """
        处理流式响应
        
        Args:
            original_model: 用户请求中的原始模型名
            mapped_model: 经 convert_request 映射后的模型名（发给后端的）
        """
        # 是否有模型映射（模型不一致且配置启用时，反向替换响应中的 model）
        has_model_mapping = (
            self.config.reverse_model_mapping_in_stream
            and original_model is not None
            and mapped_model is not None
            and original_model != mapped_model
        )
        
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
        
        # Anthropic 后端使用独立的 event: 和 data: 行
        # 需要缓存 event type 直到收到对应的 data
        pending_event_type = None
        
        # 使用 aiter_lines() 正确解析 SSE 行（避免原始字节块包含多行导致解析错误）
        line_iter = response.aiter_lines() if hasattr(response, "aiter_lines") else response.content
        async for line in line_iter:
            if hasattr(response, "aiter_lines"):
                line = line.strip()
            else:
                line = line.decode("utf-8").strip()
            if not line or line.startswith(":"):
                continue
            
            # 处理 event: 行（Anthropic SSE 格式）
            if line.startswith("event:"):
                pending_event_type = line[6:].strip()
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
                
                # 保存当前 event type 并清空，供后续各分支使用
                current_event_type = pending_event_type
                pending_event_type = None
                
                try:
                    chunk = json.loads(data_str)
                    
                    # Anthropic 后端返回 Anthropic SSE 格式数据
                    if backend_format == "anthropic":
                        event_type = current_event_type or chunk.get("type", "")
                        
                        if source_protocol == Protocol.ANTHROPIC:
                            # Anthropic→Anthropic: 直接转发原始 SSE
                            # 若存在模型映射，将响应中的 model 替换回原始模型名
                            if has_model_mapping and "model" in chunk:
                                chunk["model"] = original_model
                            if event_type:
                                yield f"event: {event_type}\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            else:
                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            continue
                        else:
                            # Anthropic→Chat/Responses: 将 Anthropic SSE 事件转换为 Chat 格式
                            chat_chunks = self.anthropic.convert_anthropic_event_to_chat(event_type, chunk)
                            
                            # 将 Chat 格式块转换为目标协议
                            for chat_chunk in chat_chunks:
                                stream_chunks = self.convert_stream_chunk_multi(chat_chunk, source_protocol)
                                for sc in stream_chunks:
                                    if source_protocol == Protocol.ANTHROPIC:
                                        yield sc.to_anthropic_sse()
                                    elif source_protocol == Protocol.OPENAI_RESPONSES:
                                        yield sc.to_sse()
                                    else:
                                        yield sc.to_openai_chunk()
                            continue
                    
                    # Responses 后端使用命名 SSE 事件
                    if backend_format == "openai_responses":
                        # Responses→Responses: 直接转发原始 SSE（保留 event 行）
                        if source_protocol == Protocol.OPENAI_RESPONSES:
                            # 若存在模型映射，将响应中的 model 替换回原始模型名
                            if has_model_mapping and "model" in chunk:
                                chunk["model"] = original_model
                            if current_event_type is not None:
                                yield f"event: {current_event_type}\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            else:
                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            continue
                        # Responses→Chat/Anthropic: Responses 后端返回 Responses 格式事件
                        event_type = current_event_type
                        if event_type and event_type not in ("response.created", "response.in_progress", "response.ping"):
                            if event_type in ("response.completed", "response.failed", "response.incomplete"):
                                # 完成/失败/不完整事件 - 转换为 Chat 格式流式块并发送
                                chat_chunk = self.openai_responses._convert_responses_event_to_chat_chunk(event_type, chunk)
                                if chat_chunk:
                                    stream_chunks = self.convert_stream_chunk_multi(chat_chunk, source_protocol)
                                    for sc in stream_chunks:
                                        if source_protocol == Protocol.ANTHROPIC:
                                            yield sc.to_anthropic_sse()
                                        elif source_protocol == Protocol.OPENAI_RESPONSES:
                                            yield sc.to_sse()
                                        else:
                                            yield sc.to_openai_chunk()
                            else:
                                # 对于其他输出事件，尝试转换为 Chat 流式块
                                chat_chunk = self.openai_responses._convert_responses_event_to_chat_chunk(event_type, chunk)
                                if chat_chunk:
                                    stream_chunks = self.convert_stream_chunk_multi(chat_chunk, source_protocol)
                                    for sc in stream_chunks:
                                        if source_protocol == Protocol.ANTHROPIC:
                                            yield sc.to_anthropic_sse()
                                        elif source_protocol == Protocol.OPENAI_RESPONSES:
                                            yield sc.to_sse()
                                        else:
                                            yield sc.to_openai_chunk()
                            continue
                    
                    # OpenAI Chat 后端
                    if backend_format == "openai_chat" and source_protocol == Protocol.OPENAI_CHAT:
                        # Chat→Chat: 直接转发原始 SSE，避免 json.loads/dumps 的字段顺序变化与性能损耗
                        # 若存在模型映射，将响应中的 model 替换回原始模型名
                        if has_model_mapping and "model" in chunk:
                            chunk["model"] = original_model
                        if current_event_type is not None:
                            yield f"event: {current_event_type}\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        else:
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        continue
                    
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
    
    支持:
    - 非流式请求: 返回 JSON 响应
    - 流式请求 (stream=True): 返回 httpx.Response 对象（含 aiter_lines 方法）
    """
    try:
        import httpx
        import asyncio
        
        # 使用模块级客户端以支持连接池和 keep-alive
        _client = httpx.AsyncClient(timeout=None)
        
        async def client(
            url: str,
            method: str = "GET",
            headers: Optional[Dict] = None,
            json: Optional[Dict] = None,
            timeout: float = 60.0,
            stream: bool = False
        ):
            if stream:
                # 流式请求: 返回 httpx.Response 对象
                # 调用方需使用 async with 或手动关闭
                request = _client.build_request(
                    method, url, json=json, headers=headers, timeout=timeout
                )
                response = await _client.send(request, stream=True)
                return response
            else:
                # 非流式请求: 返回 JSON 响应
                if method == "POST":
                    response = await _client.post(url, json=json, headers=headers, timeout=timeout)
                else:
                    response = await _client.get(url, headers=headers, timeout=timeout)
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
