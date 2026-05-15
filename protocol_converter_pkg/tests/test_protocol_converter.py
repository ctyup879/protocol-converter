"""
测试用例 - 协议转换器
"""

import pytest
import json
from protocol_converter import (
    ProtocolDetector,
    Protocol,
    OpenAIChatConverter,
    OpenAIResponsesConverter,
    AnthropicConverter,
    ProtocolConverterEngine,
    ConverterConfig,
)


class TestProtocolDetector:
    """协议检测器测试"""
    
    def test_detect_openai_chat(self):
        """测试 OpenAI Chat 请求检测"""
        request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        assert ProtocolDetector.detect(request) == Protocol.OPENAI_CHAT
    
    def test_detect_anthropic(self):
        """测试 Anthropic 请求检测"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        assert ProtocolDetector.detect(request) == Protocol.ANTHROPIC
    
    def test_detect_openai_responses(self):
        """测试 OpenAI Responses 请求检测"""
        request = {
            "model": "gpt-4o",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ]
        }
        assert ProtocolDetector.detect(request) == Protocol.OPENAI_RESPONSES
    
    def test_detect_with_thinking(self):
        """测试带 thinking 参数的 Anthropic 请求检测"""
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        assert ProtocolDetector.detect(request) == Protocol.ANTHROPIC
    
    def test_detect_empty_request(self):
        """测试空请求检测"""
        assert ProtocolDetector.detect({}) == Protocol.UNKNOWN
        assert ProtocolDetector.detect(None) == Protocol.UNKNOWN


class TestOpenAIChatConverter:
    """OpenAI Chat 转换器测试"""
    
    def test_from_anthropic(self):
        """测试从 Anthropic 转换"""
        anthropic_request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": "You are Claude.",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        
        result = OpenAIChatConverter.to_openai_chat(anthropic_request)
        
        assert result.model == "claude-sonnet-4-20250514"
        assert len(result.messages) == 2
        assert result.messages[0]["role"] == "system"
        assert result.messages[0]["content"] == "You are Claude."
        assert result.messages[1]["role"] == "user"
    
    def test_from_anthropic_with_tools(self):
        """测试带工具的 Anthropic 请求转换"""
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "What's the weather?"}
            ],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}}
                    }
                }
            ]
        }
        
        result = OpenAIChatConverter.to_openai_chat(request)
        
        assert result.tools is not None
        assert len(result.tools) == 1
        assert result.tools[0]["function"]["name"] == "get_weather"
    
    def test_to_anthropic(self):
        """测试转换为 Anthropic 格式"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {"content": "Hello!"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        
        result = OpenAIChatConverter.from_openai_chat(chat_response, "anthropic")
        
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert len(result["content"]) == 1
        assert result["content"][0]["text"] == "Hello!"
    
    def test_to_anthropic_with_tool_calls(self):
        """测试带工具调用的响应转换"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Beijing"}'
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ]
        }
        
        result = OpenAIChatConverter.from_openai_chat(chat_response, "anthropic")
        
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["name"] == "get_weather"


class TestAnthropicConverter:
    """Anthropic 转换器测试"""
    
    def test_to_openai_chat(self):
        """测试转换为 OpenAI Chat"""
        anthropic_request = {
            "model": "claude-opus-4-6",
            "max_tokens": 1024,
            "system": "You are Claude.",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 0.7
        }
        
        result = AnthropicConverter.to_openai_chat(anthropic_request)
        
        # 保持原始模型名称（协议网关场景）
        assert result["model"] == "claude-opus-4-6"
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["temperature"] == 0.7
    
    def test_convert_content_blocks(self):
        """测试内容块转换 - 通过 _convert_message 间接测试"""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": " World"}
            ]
        }
        
        result = AnthropicConverter._convert_message(msg)
        
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Hello" in result[0]["content"]
    
    def test_convert_tool_use_blocks(self):
        """测试工具调用块转换 - assistant 消息中的 tool_use"""
        msg = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"city": "Beijing"}
                }
            ]
        }
        
        result = AnthropicConverter._convert_message(msg)
        
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0].get("tool_calls") is not None
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    
    def test_from_openai_chat(self):
        """测试从 OpenAI Chat 转换"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {"content": "Hello!"},
                    "finish_reason": "stop"
                }
            ]
        }
        
        result = AnthropicConverter.from_openai_chat(chat_response)
        
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["content"][0]["text"] == "Hello!"
        assert result["stop_reason"] == "end_turn"
    
    def test_map_stop_reason(self):
        """测试停止原因映射"""
        assert AnthropicConverter._map_stop_reason("stop") == "end_turn"
        assert AnthropicConverter._map_stop_reason("length") == "max_tokens"
        assert AnthropicConverter._map_stop_reason("tool_calls") == "tool_use"


class TestOpenAIResponsesConverter:
    """OpenAI Responses 转换器测试"""
    
    def test_to_openai_chat(self):
        """测试转换为 OpenAI Chat"""
        responses_request = {
            "model": "gpt-4o",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello!"}]
                }
            ],
            "instructions": "You are helpful.",
            "max_output_tokens": 1024
        }
        
        result = OpenAIResponsesConverter.to_openai_chat(responses_request)
        
        assert result["model"] == "gpt-4o"
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"
        # max_output_tokens 映射为 max_completion_tokens (OpenAI 新版推荐)
        assert result.get("max_completion_tokens") == 1024 or result.get("max_tokens") == 1024
    
    def test_from_openai_chat(self):
        """测试从 OpenAI Chat 转换"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {"content": "Hello!"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        
        result = OpenAIResponsesConverter.from_openai_chat(chat_response)
        
        assert result["object"] == "response"
        assert result["status"] == "completed"
        assert len(result["output"]) == 1
        assert result["output"][0]["content"][0]["text"] == "Hello!"


class TestProtocolConverterEngine:
    """协议转换引擎测试"""
    
    def test_engine_initialization(self):
        """测试引擎初始化"""
        config = ConverterConfig(
            backend_url="https://api.openai.com/v1/chat/completions",
            api_key="test-key"
        )
        engine = ProtocolConverterEngine(config)
        
        assert engine.config.backend_url == "https://api.openai.com/v1/chat/completions"
        assert engine.config.api_key == "test-key"
    
    def test_detect_protocol(self):
        """测试协议检测"""
        engine = ProtocolConverterEngine()
        
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}]
        }
        
        assert engine.detect_protocol(request) == Protocol.ANTHROPIC
    
    def test_convert_request_anthropic(self):
        """测试转换 Anthropic 请求"""
        engine = ProtocolConverterEngine()
        
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}]
        }
        
        result = engine.convert_request(request)
        
        assert result["model"] == "claude-opus-4-6"
        assert "messages" in result
    
    def test_convert_response(self):
        """测试响应转换"""
        engine = ProtocolConverterEngine()
        
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {"content": "Hello!"},
                    "finish_reason": "stop"
                }
            ]
        }
        
        result = engine.convert_response(chat_response, Protocol.ANTHROPIC)
        
        assert result["type"] == "message"
        assert result["content"][0]["text"] == "Hello!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
