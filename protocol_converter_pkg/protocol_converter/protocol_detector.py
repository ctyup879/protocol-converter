"""
协议检测器 - 自动检测请求使用的协议类型
"""

from enum import Enum
from typing import Any, Dict


class Protocol(Enum):
    """支持的协议类型"""
    OPENAI_CHAT = "openai_chat"
    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC = "anthropic"
    UNKNOWN = "unknown"


class ProtocolDetector:
    """检测请求使用的协议类型"""
    
    # OpenAI Chat Completions API 特征
    OPENAI_CHAT_KEYS = {"messages", "model", "stream"}
    OPENAI_CHAT_MESSAGE_ROLES = {"system", "user", "assistant", "tool", "function", "developer"}
    
    # OpenAI Responses API 特征
    OPENAI_RESPONSES_KEYS = {"model", "input", "max_output_tokens"}
    OPENAI_RESPONSES_INPUT_TYPES = {"message", "text", "image"}
    
    # Anthropic Messages API 特征
    ANTHROPIC_KEYS = {"messages", "model", "max_tokens"}
    ANTHROPIC_ROLES = {"user", "assistant"}
    ANTHROPIC_CONTENT_TYPES = {"text", "tool_use", "tool_result"}
    
    @classmethod
    def detect(cls, request: Dict[str, Any]) -> Protocol:
        """
        检测请求使用的协议类型
        
        Args:
            request: 请求字典
            
        Returns:
            Protocol: 检测到的协议类型
        """
        if not request or not isinstance(request, dict):
            return Protocol.UNKNOWN
        
        keys = set(request.keys())
        
        # 检测 Anthropic Messages API
        if cls._is_anthropic(request, keys):
            return Protocol.ANTHROPIC
        
        # 检测 OpenAI Responses API
        if cls._is_openai_responses(request, keys):
            return Protocol.OPENAI_RESPONSES
        
        # 检测 OpenAI Chat Completions API
        if cls._is_openai_chat(request, keys):
            return Protocol.OPENAI_CHAT
        
        return Protocol.UNKNOWN
    
    @classmethod
    def _is_anthropic(cls, request: Dict[str, Any], keys: set) -> bool:
        """检查是否为 Anthropic Messages API"""
        # Anthropic 必须有 max_tokens（不是 max_tokens）
        if "max_tokens" not in keys:
            return False
        
        # Anthropic messages 是必填的
        if "messages" not in keys:
            return False
        
        messages = request.get("messages", [])
        if not messages or not isinstance(messages, list):
            return False
        
        # 检查消息格式
        for msg in messages[:3]:  # 只检查前几条消息
            if not isinstance(msg, dict):
                return False
            if "role" not in msg:
                return False
            role = msg.get("role")
            if role not in cls.ANTHROPIC_ROLES:
                return False
        
        # Anthropic 的关键特征：检查模型名称（claude- 开头）
        model = request.get("model", "")
        if model.startswith("claude-"):
            return True
        
        # Anthropic 特有参数：stop_sequences, thinking, cache_control
        anthropic_specific_params = {"stop_sequences", "thinking", "cache_control"}
        if any(p in keys for p in anthropic_specific_params):
            return True
        
        # Anthropic 有 system 作为顶级参数
        if "system" in keys:
            return True
        
        return False
    
    @classmethod
    def _is_openai_responses(cls, request: Dict[str, Any], keys: set) -> bool:
        """检查是否为 OpenAI Responses API"""
        # Responses API 使用 input 而不是 messages
        if "input" in keys and "model" in keys:
            input_data = request.get("input", [])
            if isinstance(input_data, list) and len(input_data) > 0:
                first_item = input_data[0]
                if isinstance(first_item, dict) and first_item.get("type") in cls.OPENAI_RESPONSES_INPUT_TYPES:
                    return True
        
        # 检查是否有 response 输出相关的结构
        if "instructions" in keys and "tools" in keys:
            return True
        
        return False
    
    @classmethod
    def _is_openai_chat(cls, request: Dict[str, Any], keys: set) -> bool:
        """检查是否为 OpenAI Chat Completions API"""
        # Chat API 使用 messages
        if "messages" not in keys:
            return False
        
        messages = request.get("messages", [])
        if not messages or not isinstance(messages, list):
            return False
        
        # 检查消息格式
        for msg in messages[:3]:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg:
                return False
            role = msg.get("role")
            # Chat API 有更多角色类型
            if role not in cls.OPENAI_CHAT_MESSAGE_ROLES:
                return False
        
        # 如果有 max_tokens 且模型是 claude-，这可能是 Anthropic（已在上面检测）
        # OpenAI Chat 通常使用 gpt- 系列模型
        model = request.get("model", "")
        if model.startswith("claude-"):
            return False
        
        return True
