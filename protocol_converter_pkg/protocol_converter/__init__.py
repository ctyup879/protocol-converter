"""
Protocol Converter - 支持 OpenAI Chat、OpenAI Responses 和 Anthropic 协议的相互转换
"""

from .engine import ProtocolConverterEngine, ConverterConfig
from .protocol_detector import ProtocolDetector, Protocol
from .openai_chat import OpenAIChatConverter
from .openai_responses import OpenAIResponsesConverter
from .anthropic import AnthropicConverter

__version__ = "1.0.0"
__all__ = [
    "ProtocolConverterEngine",
    "ConverterConfig",
    "ProtocolDetector", 
    "Protocol",
    "OpenAIChatConverter",
    "OpenAIResponsesConverter",
    "AnthropicConverter",
]
