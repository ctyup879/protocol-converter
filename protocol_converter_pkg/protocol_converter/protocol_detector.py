"""
协议检测器 - 自动检测请求使用的协议类型

检测依据:
- Anthropic Messages API: 有 max_tokens + messages + (claude- 模型 或 system 顶级参数 或 stop_sequences/thinking/cache_control)
- OpenAI Responses API: 有 input 参数 或 instructions/max_output_tokens/previous_response_id/reasoning/truncation/prompt/text 等特有参数
- OpenAI Chat Completions API: 有 messages 参数
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
    OPENAI_RESPONSES_INPUT_TYPES = {"message", "text", "image", "function_call", "function_call_output",
                                     "input_text", "input_image", "input_file", "input_audio",
                                     "computer_call_output", "reasoning",
                                     "local_shell_call_output", "mcp_approval_response"}
    # Responses API 特有参数（Chat API 不存在这些参数）
    OPENAI_RESPONSES_SPECIFIC_KEYS = {
        "previous_response_id", "reasoning", "text", "truncation",
        "background", "max_tool_calls", "context_management",
        "conversation", "include", "prompt",
    }
    
    # Anthropic Messages API 特征
    ANTHROPIC_KEYS = {"messages", "model", "max_tokens"}
    ANTHROPIC_ROLES = {"user", "assistant"}
    ANTHROPIC_CONTENT_TYPES = {"text", "tool_use", "tool_result", "thinking", "redacted_thinking", 
                               "image", "document", "search_result", "server_tool_use",
                               "web_search_tool_result", "web_fetch_tool_result",
                               "code_execution_tool_result", "bash_code_execution_tool_result",
                               "text_editor_code_execution_tool_result", "tool_search_tool_result",
                               "container_upload"}
    # Anthropic 特有参数
    ANTHROPIC_SPECIFIC_KEYS = {
        "stop_sequences", "thinking", "cache_control", "top_k",
        "container", "output_config", "output_format", "inference_geo",
    }
    # Anthropic 特有的服务器工具类型
    ANTHROPIC_SERVER_TOOL_TYPES = {
        "bash_20250124", "text_editor_20250124", "text_editor_20250429", "text_editor_20250728",
        "code_execution_20250522", "code_execution_20250825", "code_execution_20260120",
        "web_search_20250305", "web_search_20260209",
        "web_fetch_20250910", "web_fetch_20260209", "web_fetch_20260309",
        "memory_20250818",
        "tool_search_bm25_20251119", "tool_search_regex_20251119",
    }
    
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
        # Anthropic 必须有 max_tokens（注意：这是 Anthropic 的必填参数）
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
        
        # Anthropic 特有参数
        if any(p in keys for p in cls.ANTHROPIC_SPECIFIC_KEYS):
            return True
        
        # Anthropic 有 system 作为顶级参数
        if "system" in keys:
            return True
        
        # Anthropic 的 tool_choice 支持 "any"（Chat API 不支持）
        tool_choice = request.get("tool_choice")
        if tool_choice == "any":
            return True
        if isinstance(tool_choice, dict) and tool_choice.get("type") in ("tool", "any"):
            return True
        
        # 检查 tools 列表中是否包含 Anthropic 服务器工具类型
        tools = request.get("tools", [])
        if isinstance(tools, list):
            for tool in tools[:5]:
                if isinstance(tool, dict):
                    tool_type = tool.get("type", "")
                    if tool_type in cls.ANTHROPIC_SERVER_TOOL_TYPES:
                        return True
        
        # 检查消息内容是否包含 Anthropic 特有的内容类型
        for msg in messages[:5]:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content[:3]:
                    if isinstance(block, dict):
                        block_type = block.get("type", "")
                        if block_type in cls.ANTHROPIC_CONTENT_TYPES:
                            return True
        
        return False
    
    @classmethod
    def _is_openai_responses(cls, request: Dict[str, Any], keys: set) -> bool:
        """检查是否为 OpenAI Responses API"""
        # Responses API 使用 input 而不是 messages
        if "input" in keys and "model" in keys:
            input_data = request.get("input")
            # input 可以是字符串
            if isinstance(input_data, str):
                return True
            # input 也可以是数组
            if isinstance(input_data, list) and len(input_data) > 0:
                first_item = input_data[0]
                if isinstance(first_item, dict) and first_item.get("type") in cls.OPENAI_RESPONSES_INPUT_TYPES:
                    return True
        
        # Responses API 特有参数检测
        # 有 instructions 且无 messages -> 可能是 Responses
        if "instructions" in keys and "messages" not in keys:
            return True
        # max_output_tokens 是 Responses 特有参数名
        if "max_output_tokens" in keys and "messages" not in keys:
            return True
        # previous_response_id 是 Responses API 特有
        if "previous_response_id" in keys:
            return True
        # reasoning 参数是 Responses 特有（Chat 用 reasoning_effort）
        if "reasoning" in keys and "messages" not in keys:
            return True
        # text 配置是 Responses 特有（Chat 用 response_format）
        if "text" in keys and "messages" not in keys:
            return True
        # truncation 是 Responses 特有
        if "truncation" in keys:
            return True
        # background 是 Responses 特有
        if "background" in keys:
            return True
        # max_tool_calls 是 Responses 特有
        if "max_tool_calls" in keys:
            return True
        # 其他 Responses 特有参数
        if any(p in keys for p in cls.OPENAI_RESPONSES_SPECIFIC_KEYS):
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
