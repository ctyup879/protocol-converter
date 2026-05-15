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
        # instructions 映射为 developer 角色（Responses API 的 instructions 对应 Chat API 的 developer 角色）
        assert result["messages"][0]["role"] == "developer"
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
    
    def test_auth_headers_openai(self):
        """测试 OpenAI 后端认证头"""
        config = ConverterConfig(backend_type="openai", api_key="sk-test")
        assert config.get_auth_headers() == {"Authorization": "Bearer sk-test"}
    
    def test_auth_headers_anthropic(self):
        """测试 Anthropic 后端认证头"""
        config = ConverterConfig(backend_type="anthropic", api_key="sk-test")
        headers = config.get_auth_headers()
        assert headers["x-api-key"] == "sk-test"
        assert headers["anthropic-version"] == "2023-06-01"
    
    def test_convert_request_to_anthropic_backend(self):
        """测试转换请求到 Anthropic 后端"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        
        chat_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"}
            ]
        }
        
        result = engine.convert_request(chat_request)
        
        assert result["model"] == "gpt-4o"
        assert "max_tokens" in result
        assert result["system"] == "You are helpful."
    
    def test_convert_request_to_responses_backend(self):
        """测试转换请求到 Responses 后端"""
        config = ConverterConfig(backend_type="openai_responses")
        engine = ProtocolConverterEngine(config)
        
        chat_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        
        result = engine.convert_request(chat_request)
        
        assert result["model"] == "gpt-4o"
        assert "input" in result
    
    def test_model_mapping(self):
        """测试模型映射"""
        config = ConverterConfig(model_mapping={"gpt-4o": "gpt-4o-mini"})
        engine = ProtocolConverterEngine(config)
        
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}]
        }
        
        result = engine.convert_request(request)
        assert result["model"] == "gpt-4o-mini"


class TestProtocolDetectorExtended:
    """协议检测器扩展测试"""
    
    def test_detect_responses_with_reasoning(self):
        """测试带 reasoning 参数的 Responses 请求检测"""
        request = {
            "model": "o3",
            "input": "Solve this math problem",
            "reasoning": {"effort": "high"}
        }
        assert ProtocolDetector.detect(request) == Protocol.OPENAI_RESPONSES
    
    def test_detect_responses_with_truncation(self):
        """测试带 truncation 参数的 Responses 请求检测"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "truncation": "auto"
        }
        assert ProtocolDetector.detect(request) == Protocol.OPENAI_RESPONSES
    
    def test_detect_responses_with_previous_response_id(self):
        """测试带 previous_response_id 的 Responses 请求检测"""
        request = {
            "model": "gpt-4o",
            "input": "Continue",
            "previous_response_id": "resp_123"
        }
        assert ProtocolDetector.detect(request) == Protocol.OPENAI_RESPONSES
    
    def test_detect_anthropic_with_system(self):
        """测试带 system 顶级参数的 Anthropic 请求检测"""
        request = {
            "model": "my-model",
            "max_tokens": 1024,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}]
        }
        assert ProtocolDetector.detect(request) == Protocol.ANTHROPIC
    
    def test_detect_anthropic_with_tool_choice_any(self):
        """测试带 tool_choice=any 的 Anthropic 请求检测"""
        request = {
            "model": "my-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": "any"
        }
        assert ProtocolDetector.detect(request) == Protocol.ANTHROPIC


class TestAnthropicConverterExtended:
    """Anthropic 转换器扩展测试"""
    
    def test_image_conversion(self):
        """测试图片内容正确转换为 OpenAI image_url 格式"""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgo="
                    }
                }
            ]
        }
        
        result = AnthropicConverter._convert_message(msg)
        
        assert len(result) == 1
        assert result[0]["role"] == "user"
        content = result[0]["content"]
        assert isinstance(content, list)
        # 文本块
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "What's in this image?"
        # 图片块 -> image_url
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    
    def test_image_url_source(self):
        """测试图片 URL 源转换"""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/image.png"
                    }
                }
            ]
        }
        
        result = AnthropicConverter._convert_message(msg)
        
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"] == "https://example.com/image.png"
    
    def test_tool_result_conversion(self):
        """测试工具结果正确转换"""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": "The weather is sunny."
                }
            ]
        }
        
        result = AnthropicConverter._convert_message(msg)
        
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "toolu_123"
        assert result[0]["content"] == "The weather is sunny."
    
    def test_thinking_conversion_to_reasoning_effort(self):
        """测试 thinking 参数转换为 reasoning_effort"""
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 16000,
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "messages": [{"role": "user", "content": "Think hard"}]
        }
        
        result = AnthropicConverter.to_openai_chat(request)
        
        assert "reasoning_effort" in result
        assert result["reasoning_effort"] in ("low", "medium", "high")
    
    def test_thinking_disabled(self):
        """测试 thinking 禁用"""
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 1024,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        result = AnthropicConverter.to_openai_chat(request)
        
        assert result.get("reasoning_effort") == "none"
    
    def test_max_tokens_to_max_completion_tokens(self):
        """测试 max_tokens 映射为 max_completion_tokens"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        result = AnthropicConverter.to_openai_chat(request)
        
        assert result.get("max_completion_tokens") == 2048
    
    def test_response_stop_details(self):
        """测试响应中包含 stop_details"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "message": {"content": "I can't help with that.", "refusal": "Content policy violation"},
                "finish_reason": "stop"
            }]
        }
        
        result = AnthropicConverter.from_openai_chat(chat_response)
        
        assert result["stop_reason"] == "refusal"
        assert result["stop_details"] is not None
        assert result["stop_details"]["reason"] == "content_policy"
    
    def test_system_list_blocks(self):
        """测试 system 参数为列表时的转换"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "You are helpful."},
                {"type": "text", "text": "Be concise."}
            ],
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        result = AnthropicConverter.to_openai_chat(request)
        
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert "helpful" in result["messages"][0]["content"]
        assert "concise" in result["messages"][0]["content"]
    
    def test_service_tier_mapping(self):
        """测试 service_tier 映射"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "service_tier": "standard_only",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        result = AnthropicConverter.to_openai_chat(request)
        
        assert result.get("service_tier") == "default"


class TestOpenAIResponsesConverterExtended:
    """OpenAI Responses 转换器扩展测试"""
    
    def test_reasoning_to_reasoning_effort(self):
        """测试 reasoning 参数映射"""
        request = {
            "model": "o3",
            "input": "Think about this",
            "reasoning": {"effort": "high"}
        }
        
        result = OpenAIResponsesConverter.to_openai_chat(request)
        
        assert result.get("reasoning_effort") == "high"
    
    def test_text_config_to_response_format(self):
        """测试 text 配置映射为 response_format"""
        request = {
            "model": "gpt-4o",
            "input": "Generate JSON",
            "text": {
                "format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "my_schema",
                        "schema": {"type": "object", "properties": {"name": {"type": "string"}}}
                    }
                }
            }
        }
        
        result = OpenAIResponsesConverter.to_openai_chat(request)
        
        assert "response_format" in result
        assert result["response_format"]["type"] == "json_schema"
    
    def test_parallel_tool_calls_mapping(self):
        """测试 parallel_tool_calls 映射"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "parallel_tool_calls": False
        }
        
        result = OpenAIResponsesConverter.to_openai_chat(request)
        
        assert result.get("parallel_tool_calls") is False
    
    def test_function_call_input_conversion(self):
        """测试 function_call 输入项转换"""
        request = {
            "model": "gpt-4o",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "text", "text": "Call the function"}]},
                {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": '{"city": "NYC"}'},
                {"type": "function_call_output", "call_id": "call_1", "output": "Sunny, 72F"}
            ]
        }
        
        result = OpenAIResponsesConverter.to_openai_chat(request)
        
        messages = result["messages"]
        # 应该有 user 消息 + assistant 消息(tool_calls) + tool 消息
        roles = [m["role"] for m in messages]
        assert "user" in roles
        assert "assistant" in roles
        assert "tool" in roles
    
    def test_from_openai_chat_request_with_tools(self):
        """测试 Chat 请求转换为 Responses 格式（带工具）"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }
            }]
        }
        
        result = OpenAIResponsesConverter.from_openai_chat_request(chat_request)
        
        assert result["model"] == "gpt-4o"
        assert isinstance(result["input"], str)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["name"] == "get_weather"
    
    def test_from_openai_chat_request_reasoning_effort(self):
        """测试 reasoning_effort 映射为 reasoning"""
        chat_request = {
            "model": "o3",
            "messages": [{"role": "user", "content": "Solve this"}],
            "reasoning_effort": "high"
        }
        
        result = OpenAIResponsesConverter.from_openai_chat_request(chat_request)
        
        assert "reasoning" in result
        assert result["reasoning"]["effort"] == "high"
    
    def test_response_incomplete_details(self):
        """测试响应中包含 incomplete_details"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "message": {"content": "Hello..."},
                "finish_reason": "length"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        
        result = OpenAIResponsesConverter.from_openai_chat(chat_response)
        
        assert result["status"] == "incomplete"
        assert result["incomplete_details"]["reason"] == "max_output_tokens"
    
    def test_web_search_tool_conversion(self):
        """测试 web_search 工具转换"""
        request = {
            "model": "gpt-4o",
            "input": "Search the web",
            "tools": [{
                "type": "web_search",
                "search_context_size": "high",
                "user_location": {"type": "approximate", "city": "San Francisco"}
            }]
        }
        
        result = OpenAIResponsesConverter.to_openai_chat(request)
        
        tools = result.get("tools", [])
        web_search_tools = [t for t in tools if t.get("type") == "web_search"]
        assert len(web_search_tools) == 1
        assert web_search_tools[0]["web_search_options"]["search_context_size"] == "high"
    
    def test_response_with_metadata(self):
        """测试响应包含 metadata"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "message": {"content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "service_tier": "auto"
        }
        
        result = OpenAIResponsesConverter.from_openai_chat(chat_response)
        
        assert result["service_tier"] == "auto"


class TestEngineBackendConversions:
    """引擎后端格式转换测试"""
    
    def test_chat_to_anthropic_request(self):
        """测试 Chat 请求转换为 Anthropic 格式"""
        engine = ProtocolConverterEngine(ConverterConfig(backend_type="anthropic"))
        
        chat_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ],
            "max_completion_tokens": 1024,
            "temperature": 0.7
        }
        
        result = engine.convert_request(chat_request)
        
        assert result["model"] == "gpt-4o"
        assert result["system"] == "You are helpful."
        assert result["max_tokens"] == 1024
        assert result["temperature"] == 0.7
        # 检查消息格式
        assert all(msg["role"] in ("user", "assistant") for msg in result["messages"])
    
    def test_chat_to_anthropic_with_tool_calls(self):
        """测试带 tool_calls 的 Chat 请求转换为 Anthropic 格式"""
        engine = ProtocolConverterEngine(ConverterConfig(backend_type="anthropic"))
        
        chat_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'}
                    }]
                },
                {"role": "tool", "tool_call_id": "call_123", "content": "Sunny, 72F"}
            ],
            "tools": [{
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}
            }]
        }
        
        result = engine.convert_request(chat_request)
        
        assert "tools" in result
        assert result["tools"][0]["name"] == "get_weather"
        # 检查 assistant 消息中有 tool_use 块
        assistant_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        # 检查 tool_result 在 user 消息中
        user_msgs = [m for m in result["messages"] if m["role"] == "user"]
        tool_result_found = any(
            isinstance(m.get("content"), list) and any(b.get("type") == "tool_result" for b in m["content"])
            for m in user_msgs
        )
        assert tool_result_found
    
    def test_anthropic_backend_response_conversion(self):
        """测试 Anthropic 后端响应转换为 Chat 格式"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello from Claude!"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        
        result = engine.convert_response(anthropic_response, Protocol.OPENAI_CHAT)
        
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hello from Claude!"
        assert result["choices"][0]["finish_reason"] == "stop"
    
    def test_responses_backend_response_conversion(self):
        """测试 Responses 后端响应转换为 Chat 格式"""
        config = ConverterConfig(backend_type="openai_responses")
        engine = ProtocolConverterEngine(config)
        
        responses_response = {
            "id": "resp_123",
            "object": "response",
            "status": "completed",
            "model": "gpt-4o",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello!"}]
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        
        result = engine.convert_response(responses_response, Protocol.OPENAI_CHAT)
        
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
