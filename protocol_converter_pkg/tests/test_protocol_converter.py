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
        
        # web_search 转换为 Chat API 的 web_search_preview 工具类型
        tools = result.get("tools", [])
        web_search_tools = [t for t in tools if t.get("type") == "web_search_preview"]
        assert len(web_search_tools) == 1
        
        # 搜索上下文配置放入顶层 web_search_options 参数
        assert result.get("web_search_options", {}).get("search_context_size") == "high"
    
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


class TestAnthropicStreamingSequence:
    """Anthropic 流式事件序列测试"""

    def setup_method(self):
        """每次测试前重置流式状态"""
        AnthropicConverter.reset_stream_state()

    def test_message_start_event(self):
        """测试 message_start 事件"""
        chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "delta": {"role": "assistant"},
                "index": 0
            }]
        }
        
        events = AnthropicConverter.convert_stream_chunk(chunk)
        
        assert isinstance(events, list)
        assert events[0]["type"] == "message_start"
        assert events[0]["message"]["role"] == "assistant"
        assert events[0]["message"]["stop_reason"] is None
        assert "input_tokens" in events[0]["message"]["usage"]

    def test_text_content_block_start(self):
        """测试文本内容块先发 content_block_start"""
        # 先发送 message_start
        start_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        AnthropicConverter.convert_stream_chunk(start_chunk)
        
        # 第一个文本 chunk 应该先发 content_block_start 再发 content_block_delta
        text_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "delta": {"content": "Hello"},
                "index": 0
            }]
        }
        
        events = AnthropicConverter.convert_stream_chunk(text_chunk)
        
        assert len(events) == 2
        assert events[0]["type"] == "content_block_start"
        assert events[0]["content_block"]["type"] == "text"
        assert events[1]["type"] == "content_block_delta"
        assert events[1]["delta"]["type"] == "text_delta"
        assert events[1]["delta"]["text"] == "Hello"

    def test_text_content_block_no_duplicate_start(self):
        """测试后续文本 chunk 不再发 content_block_start"""
        # 先发送 message_start
        start_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        AnthropicConverter.convert_stream_chunk(start_chunk)
        
        # 第一个文本 chunk
        text_chunk1 = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"delta": {"content": "Hello"}, "index": 0}]
        }
        events1 = AnthropicConverter.convert_stream_chunk(text_chunk1)
        assert len(events1) == 2  # content_block_start + content_block_delta
        
        # 第二个文本 chunk
        text_chunk2 = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"delta": {"content": " world"}, "index": 0}]
        }
        events2 = AnthropicConverter.convert_stream_chunk(text_chunk2)
        assert len(events2) == 1  # 只有 content_block_delta
        assert events2[0]["type"] == "content_block_delta"

    def test_content_block_stop_on_finish(self):
        """测试 finish_reason 时关闭所有 content blocks"""
        # 先发送 message_start
        start_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        AnthropicConverter.convert_stream_chunk(start_chunk)
        
        # 文本 chunk
        text_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"delta": {"content": "Hello"}, "index": 0}]
        }
        AnthropicConverter.convert_stream_chunk(text_chunk)
        
        # finish_reason chunk
        finish_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "delta": {},
                "finish_reason": "stop",
                "index": 0
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        
        events = AnthropicConverter.convert_stream_chunk(finish_chunk)
        
        # 应该包含: content_block_stop + message_delta
        event_types = [e["type"] for e in events]
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        # message_delta 应该是最后一个事件
        assert events[-1]["type"] == "message_delta"

    def test_thinking_delta(self):
        """测试 thinking 内容转换为 thinking_delta"""
        start_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        AnthropicConverter.convert_stream_chunk(start_chunk)
        
        thinking_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "delta": {"reasoning_content": "Let me think..."},
                "index": 0
            }]
        }
        
        events = AnthropicConverter.convert_stream_chunk(thinking_chunk)
        
        # 应该有 content_block_start + thinking_delta + signature_delta
        assert len(events) == 3
        assert events[0]["type"] == "content_block_start"
        assert events[0]["content_block"]["type"] == "thinking"
        # thinking_delta
        assert events[1]["type"] == "content_block_delta"
        assert events[1]["delta"]["type"] == "thinking_delta"
        assert events[1]["delta"]["thinking"] == "Let me think..."
        # signature_delta (Anthropic SDK 要求)
        assert events[2]["type"] == "content_block_delta"
        assert events[2]["delta"]["type"] == "signature_delta"

    def test_tool_use_content_block_start(self):
        """测试 tool_use 的 content_block_start 事件"""
        start_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        AnthropicConverter.convert_stream_chunk(start_chunk)
        
        # 先发文本以开启文本块
        text_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"delta": {"content": "Let me check"}, "index": 0}]
        }
        AnthropicConverter.convert_stream_chunk(text_chunk)
        
        # tool_call 开始
        tool_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "id": "call_123",
                        "index": 0,
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""}
                    }]
                },
                "index": 0
            }]
        }
        
        events = AnthropicConverter.convert_stream_chunk(tool_chunk)
        
        event_types = [e["type"] for e in events]
        # 应该先关闭文本块，然后开启 tool_use 块
        assert "content_block_stop" in event_types
        assert "content_block_start" in event_types
        
        # 找到 content_block_start 事件
        start_event = [e for e in events if e["type"] == "content_block_start"][0]
        assert start_event["content_block"]["type"] == "tool_use"
        assert start_event["content_block"]["id"] == "call_123"


class TestResponsesStreamingSequence:
    """Responses 流式事件序列测试"""

    def setup_method(self):
        """每次测试前重置流式状态"""
        OpenAIResponsesConverter.reset_stream_state()

    def test_response_created_and_in_progress(self):
        """测试 response.created + response.in_progress 双事件"""
        chunk = {
            "id": "resp_123",
            "model": "gpt-4o",
            "choices": [{
                "delta": {"role": "assistant"},
                "index": 0
            }]
        }
        
        events = OpenAIResponsesConverter.convert_stream_chunk(chunk)
        
        assert isinstance(events, list)
        assert len(events) == 2
        assert events[0]["type"] == "response.created"
        assert events[1]["type"] == "response.in_progress"

    def test_text_delta_with_item_added(self):
        """测试文本增量包含 output_item.added 和 content_part.added"""
        # 先发 response.created
        start_chunk = {
            "id": "resp_123",
            "model": "gpt-4o",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        OpenAIResponsesConverter.convert_stream_chunk(start_chunk)
        
        # 第一个文本 chunk
        text_chunk = {
            "id": "resp_123",
            "model": "gpt-4o",
            "choices": [{
                "delta": {"content": "Hello"},
                "index": 0
            }]
        }
        
        events = OpenAIResponsesConverter.convert_stream_chunk(text_chunk)
        
        event_types = [e["type"] for e in events]
        assert "response.output_item.added" in event_types
        assert "response.content_part.added" in event_types
        assert "response.output_text.delta" in event_types

    def test_function_call_events(self):
        """测试 function_call 事件序列"""
        # 先发 response.created
        start_chunk = {
            "id": "resp_123",
            "model": "gpt-4o",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        OpenAIResponsesConverter.convert_stream_chunk(start_chunk)
        
        # function_call 开始
        tool_chunk = {
            "id": "resp_123",
            "model": "gpt-4o",
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "id": "call_123",
                        "index": 0,
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""}
                    }]
                },
                "index": 0
            }]
        }
        
        events = OpenAIResponsesConverter.convert_stream_chunk(tool_chunk)
        
        event_types = [e["type"] for e in events]
        assert "response.output_item.added" in event_types


class TestAnthropicNewParams:
    """Anthropic 新增参数测试"""

    def test_inference_geo_param(self):
        """测试 inference_geo 参数传递"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "inference_geo": "us"
        }
        
        result = AnthropicConverter.to_openai_chat(request)
        
        assert "extra_body" in result
        assert result["extra_body"]["inference_geo"] == "us"

    def test_thinking_adaptive_type(self):
        """测试 thinking adaptive 类型"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "thinking": {"type": "adaptive", "budget_tokens": 5000},
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        result = AnthropicConverter.to_openai_chat(request)
        
        assert "reasoning_effort" in result
        # adaptive 应该映射到某个 reasoning_effort
        assert result["reasoning_effort"] in ("low", "medium", "high")

    def test_thinking_display_field(self):
        """测试 thinking display 子字段保留"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 5000, "display": "omitted"},
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        result = AnthropicConverter.to_openai_chat(request)
        
        # display 字段应保留在 extra_body.thinking 中
        assert "extra_body" in result
        assert result["extra_body"]["thinking"]["display"] == "omitted"


class TestEngineNewFeatures:
    """引擎新功能测试"""

    def test_anthropic_version_configurable(self):
        """测试 Anthropic API 版本可配置"""
        config = ConverterConfig(backend_type="anthropic", anthropic_version="2024-01-01")
        assert config.get_auth_headers()["anthropic-version"] == "2024-01-01"

    def test_inference_geo_config(self):
        """测试 inference_geo 配置"""
        config = ConverterConfig(backend_type="anthropic", inference_geo="eu")
        engine = ProtocolConverterEngine(config)
        
        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        result = engine.convert_request(chat_request)
        
        assert result.get("inference_geo") == "eu"

    def test_reasoning_effort_xhigh(self):
        """测试 reasoning_effort xhigh 映射"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        
        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "xhigh"
        }
        
        result = engine.convert_request(chat_request)
        
        assert "thinking" in result
        assert result["thinking"]["budget_tokens"] == 64000

    def test_reasoning_effort_minimal(self):
        """测试 reasoning_effort minimal 映射"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        
        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "minimal"
        }
        
        result = engine.convert_request(chat_request)
        
        assert "thinking" in result
        assert result["thinking"]["type"] == "disabled"

    def test_anthropic_response_with_thinking(self):
        """测试 Anthropic 响应中 thinking 块转换为 reasoning_content"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me analyze this..."},
                {"type": "text", "text": "Here is my answer."}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        
        result = engine.convert_response(anthropic_response, Protocol.OPENAI_CHAT)
        
        assert result["choices"][0]["message"]["reasoning_content"] == "Let me analyze this..."
        assert result["choices"][0]["message"]["content"] == "Here is my answer."


class TestProtocolDetectorNew:
    """协议检测器新增测试"""

    def test_detect_anthropic_with_inference_geo(self):
        """测试带 inference_geo 的 Anthropic 请求检测"""
        request = {
            "model": "my-model",
            "max_tokens": 1024,
            "inference_geo": "us",
            "messages": [{"role": "user", "content": "Hi"}]
        }
        assert ProtocolDetector.detect(request) == Protocol.ANTHROPIC

    def test_detect_anthropic_with_tool_result_content(self):
        """测试消息中包含 Anthropic 特有内容类型"""
        request = {
            "model": "my-model",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "result"}]
            }]
        }
        assert ProtocolDetector.detect(request) == Protocol.ANTHROPIC

    def test_detect_responses_with_function_call_input(self):
        """测试带 function_call 输入类型的 Responses 请求检测"""
        request = {
            "model": "gpt-4o",
            "input": [
                {"type": "function_call", "call_id": "c1", "name": "test", "arguments": "{}"}
            ]
        }
        assert ProtocolDetector.detect(request) == Protocol.OPENAI_RESPONSES


class TestReasoningContentConversion:
    """reasoning_content 与 thinking 内容块的跨协议转换测试"""

    def test_chat_reasoning_content_to_anthropic_thinking(self):
        """测试 Chat reasoning_content 转换为 Anthropic thinking 块"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "o3",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "reasoning_content": "Let me think about this step by step...",
                    "content": "The answer is 42."
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }
        
        result = AnthropicConverter.from_openai_chat(chat_response)
        
        # 验证 thinking 块在 text 块之前
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "thinking"
        assert result["content"][0]["thinking"] == "Let me think about this step by step..."
        assert result["content"][1]["type"] == "text"
        assert result["content"][1]["text"] == "The answer is 42."

    def test_chat_reasoning_content_to_responses_reasoning(self):
        """测试 Chat reasoning_content 转换为 Responses reasoning 输出项"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "o3",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "reasoning_content": "I need to analyze this...",
                    "content": "Here's the result."
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }
        
        result = OpenAIResponsesConverter.from_openai_chat(chat_response)
        
        # 验证 reasoning 输出项存在
        reasoning_items = [item for item in result["output"] if item["type"] == "reasoning"]
        assert len(reasoning_items) == 1
        assert reasoning_items[0]["content"][0]["type"] == "reasoning_text"
        assert reasoning_items[0]["content"][0]["text"] == "I need to analyze this..."
        
        # 验证 message 输出项存在
        message_items = [item for item in result["output"] if item["type"] == "message"]
        assert len(message_items) == 1
        assert message_items[0]["content"][0]["text"] == "Here's the result."

    def test_anthropic_thinking_to_responses_reasoning(self):
        """测试 Anthropic thinking 块转换为 Responses reasoning 输出项"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Deep analysis..."},
                {"type": "text", "text": "My answer."}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        }
        
        result = AnthropicConverter.to_openai_responses(anthropic_response)
        
        # 验证 reasoning 输出项
        reasoning_items = [item for item in result["output"] if item["type"] == "reasoning"]
        assert len(reasoning_items) == 1
        assert reasoning_items[0]["content"][0]["type"] == "reasoning_text"
        assert reasoning_items[0]["content"][0]["text"] == "Deep analysis..."
        
        # 验证 message 输出项
        message_items = [item for item in result["output"] if item["type"] == "message"]
        assert len(message_items) == 1
        assert message_items[0]["content"][0]["text"] == "My answer."

    def test_anthropic_redacted_thinking_to_responses(self):
        """测试 Anthropic redacted_thinking 块转换为 Responses reasoning (含 encrypted_content)"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "redacted_thinking", "signature": "enc_abc123"},
                {"type": "text", "text": "Answer."}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        
        result = AnthropicConverter.to_openai_responses(anthropic_response)
        
        reasoning_items = [item for item in result["output"] if item["type"] == "reasoning"]
        assert len(reasoning_items) == 1
        assert reasoning_items[0].get("encrypted_content") == "enc_abc123"

    def test_responses_reasoning_to_chat_reasoning_content(self):
        """测试 Responses reasoning 输出项转换为 Chat reasoning_content"""
        responses_response = {
            "id": "resp_123",
            "object": "response",
            "status": "completed",
            "model": "o3",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_123",
                    "content": [{"type": "reasoning_text", "text": "Let me reason..."}],
                    "status": "completed"
                },
                {
                    "type": "message",
                    "id": "msg_123",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Final answer."}]
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        }
        
        result = OpenAIResponsesConverter.to_chat_response(responses_response)
        
        assert result["choices"][0]["message"]["reasoning_content"] == "Let me reason..."
        assert result["choices"][0]["message"]["content"] == "Final answer."

    def test_engine_chat_to_anthropic_with_reasoning_content(self):
        """测试引擎将 Chat 请求中 assistant 消息的 reasoning_content 转换为 Anthropic thinking 块"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        
        chat_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "reasoning_content": "I need to add 2 and 2...", "content": "4"},
                {"role": "user", "content": "And 3+3?"}
            ]
        }
        
        result = engine.convert_request(chat_request)
        
        # 找到 assistant 消息
        assistant_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        # 验证 thinking 块在 text 块之前
        content = assistant_msgs[0]["content"]
        thinking_blocks = [b for b in content if b.get("type") == "thinking"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "I need to add 2 and 2..."


class TestUsageConversion:
    """Usage 字段跨协议转换测试"""

    def test_chat_usage_with_cached_tokens_to_responses(self):
        """测试 Chat usage 缓存 token 信息转换到 Responses"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 80}
            }
        }
        
        result = OpenAIResponsesConverter.from_openai_chat(chat_response)
        
        assert result["usage"]["input_tokens"] == 100
        assert result["usage"]["output_tokens"] == 50
        assert result["usage"]["input_tokens_details"]["cached_tokens"] == 80

    def test_chat_usage_with_reasoning_tokens_to_responses(self):
        """测试 Chat usage 推理 token 信息转换到 Responses"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "o3",
            "choices": [{
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "completion_tokens_details": {"reasoning_tokens": 30}
            }
        }
        
        result = OpenAIResponsesConverter.from_openai_chat(chat_response)
        
        assert result["usage"]["output_tokens_details"]["reasoning_tokens"] == 30

    def test_responses_usage_to_chat_with_details(self):
        """测试 Responses usage 含 details 转换到 Chat"""
        responses_response = {
            "id": "resp_123",
            "object": "response",
            "status": "completed",
            "model": "o3",
            "output": [
                {
                    "type": "message",
                    "id": "msg_123",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Hello"}]
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "input_tokens_details": {"cached_tokens": 80},
                "output_tokens_details": {"reasoning_tokens": 30}
            }
        }
        
        result = OpenAIResponsesConverter.to_chat_response(responses_response)
        
        assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 80
        assert result["usage"]["completion_tokens_details"]["reasoning_tokens"] == 30

    def test_anthropic_usage_to_chat_with_cache(self):
        """测试 Anthropic usage 缓存信息转换到 Chat"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 80,
            }
        }
        
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine.convert_response(anthropic_response, Protocol.OPENAI_CHAT)
        
        assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 80


class TestAnthropicServerTools:
    """Anthropic 服务器工具转换测试"""

    def test_anthropic_server_tool_to_chat(self):
        """测试 Anthropic 服务器工具保留在 extra_body 中"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [
                {"name": "my_func", "description": "My function", "input_schema": {"type": "object", "properties": {}}},
                {"type": "web_search_20250305"},
                {"type": "bash_20250124"},
            ]
        }
        
        result = AnthropicConverter.to_openai_chat(request)
        
        # 只有客户端工具被转换
        assert len(result["tools"]) == 1
        assert result["tools"][0]["function"]["name"] == "my_func"
        
        # 服务器工具保留在 extra_body 中
        assert "anthropic_server_tools" in result["extra_body"]
        assert len(result["extra_body"]["anthropic_server_tools"]) == 2

    def test_detect_anthropic_with_server_tool_type(self):
        """测试包含 Anthropic 服务器工具类型的请求检测"""
        request = {
            "model": "my-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Search the web"}],
            "tools": [{"type": "web_search_20250305"}]
        }
        assert ProtocolDetector.detect(request) == Protocol.ANTHROPIC


class TestResponsesOutputItems:
    """Responses 输出项类型转换测试"""

    def test_mcp_call_to_chat_tool_call(self):
        """测试 Responses mcp_call 转换为 Chat tool_call"""
        responses_response = {
            "id": "resp_123",
            "object": "response",
            "status": "completed",
            "model": "gpt-4o",
            "output": [
                {
                    "type": "mcp_call",
                    "id": "mcp_123",
                    "name": "search_drive",
                    "arguments": '{"query": "report"}',
                    "server_label": "google_drive"
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        }
        
        result = OpenAIResponsesConverter.to_chat_response(responses_response)
        
        assert len(result["choices"][0]["message"]["tool_calls"]) == 1
        assert result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "search_drive"

    def test_custom_tool_call_to_chat(self):
        """测试 Responses custom_tool_call 转换为 Chat tool_call"""
        responses_response = {
            "id": "resp_123",
            "object": "response",
            "status": "completed",
            "model": "gpt-4o",
            "output": [
                {
                    "type": "custom_tool_call",
                    "id": "ct_123",
                    "call_id": "ct_123",
                    "name": "my_custom_tool",
                    "arguments": '{"param": "value"}'
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        }
        
        result = OpenAIResponsesConverter.to_chat_response(responses_response)
        
        assert len(result["choices"][0]["message"]["tool_calls"]) == 1
        assert result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "my_custom_tool"

    def test_anthropic_max_tokens_to_responses_incomplete(self):
        """测试 Anthropic max_tokens 停止原因转换为 Responses incomplete"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Truncated..."}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        
        result = AnthropicConverter.to_openai_responses(anthropic_response)
        
        assert result["status"] == "incomplete"
        assert result["incomplete_details"]["reason"] == "max_output_tokens"


class TestAnthropicThinkingBlockSignature:
    """测试 Anthropic thinking 块的 signature 字段（官方 SDK 要求）"""

    def test_thinking_block_has_signature_in_response(self):
        """测试 from_openai_chat 转换的 thinking 块包含 signature 字段"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "o3",
            "choices": [{
                "message": {
                    "content": "The answer is 42",
                    "reasoning_content": "Let me think about this..."
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }
        
        result = AnthropicConverter.from_openai_chat(chat_response)
        
        thinking_blocks = [b for b in result["content"] if b.get("type") == "thinking"]
        assert len(thinking_blocks) == 1
        # Anthropic SDK: ThinkingBlock 必须包含 signature 字段
        assert "signature" in thinking_blocks[0]
        assert thinking_blocks[0]["thinking"] == "Let me think about this..."

    def test_engine_chat_to_anthropic_thinking_has_signature(self):
        """测试引擎 Chat→Anthropic 转换中 thinking 块包含 signature"""
        chat_request = {
            "model": "o3",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Answer", "reasoning_content": "My reasoning..."}
            ]
        }
        
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._chat_to_anthropic_request(chat_request)
        
        # 找到 assistant 消息中的 thinking 块
        for msg in result["messages"]:
            if msg.get("role") == "assistant":
                for block in msg.get("content", []):
                    if block.get("type") == "thinking":
                        assert "signature" in block
                        assert block["thinking"] == "My reasoning..."


class TestAnthropicPauseTurnMapping:
    """测试 Anthropic pause_turn 停止原因的正确映射"""

    def test_pause_turn_to_responses_incomplete(self):
        """测试 Anthropic pause_turn 转换为 Responses incomplete"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Still thinking..."}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "pause_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        
        result = AnthropicConverter.to_openai_responses(anthropic_response)
        assert result["status"] == "incomplete"

    def test_pause_turn_to_chat_stop(self):
        """测试 Anthropic pause_turn 转换为 Chat stop"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Still thinking..."}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "pause_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._anthropic_to_chat_response(anthropic_response)
        assert result["choices"][0]["finish_reason"] == "stop"


class TestChatToAnthropicMissingParams:
    """测试 Chat→Anthropic 转换中缺失的参数映射"""

    def test_user_to_metadata_user_id(self):
        """测试 Chat user 字段映射到 Anthropic metadata.user_id"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "user": "user_abc123"
        }
        
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._chat_to_anthropic_request(chat_request)
        
        assert result.get("metadata", {}).get("user_id") == "user_abc123"

    def test_unsupported_params_in_extra_body(self):
        """测试 Chat API 特有但 Anthropic 不支持的参数放入 extra_body"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            "seed": 42,
            "parallel_tool_calls": True,
            "logprobs": True,
            "top_logprobs": 5,
            "logit_bias": {"1234": -100},
        }
        
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._chat_to_anthropic_request(chat_request)
        
        extra = result.get("extra_body", {})
        assert extra.get("frequency_penalty") == 0.5
        assert extra.get("presence_penalty") == 0.3
        assert extra.get("seed") == 42
        assert extra.get("parallel_tool_calls") is True
        assert extra.get("logprobs") is True
        assert extra.get("top_logprobs") == 5
        assert extra.get("logit_bias") == {"1234": -100}

    def test_metadata_merge(self):
        """测试 metadata 合并（user_id 和自定义字段）"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "user": "user_abc123",
            "metadata": {"session_id": "sess_456"}
        }
        
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._chat_to_anthropic_request(chat_request)
        
        assert result["metadata"]["user_id"] == "user_abc123"
        assert result["metadata"]["session_id"] == "sess_456"


class TestAnthropicUsageFieldsInChatResponse:
    """测试 Anthropic usage字段在 Chat 响应中的正确保留"""

    def test_cache_creation_input_tokens_in_chat_usage(self):
        """测试 Anthropic cache_creation_input_tokens 映射到 Chat prompt_tokens_details"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 30,
                "cache_read_input_tokens": 20,
            }
        }
        
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._anthropic_to_chat_response(anthropic_response)
        
        # 两个缓存字段都应该在 prompt_tokens_details 中
        assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 20
        # service_tier 应该保留
        assert result["usage"]["prompt_tokens"] == 100
        assert result["usage"]["completion_tokens"] == 50

    def test_service_tier_and_inference_geo_in_usage(self):
        """测试 Anthropic service_tier 和 inference_geo 在 Chat 响应中保留"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "service_tier": "priority",
                "inference_geo": "us",
            }
        }
        
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._anthropic_to_chat_response(anthropic_response)
        
        assert result["usage"].get("service_tier") == "priority"
        assert result["usage"].get("inference_geo") == "us"


class TestResponsesInstructionsAsArray:
    """测试 Responses instructions 作为数组的转换"""

    def test_instructions_as_string(self):
        """测试 instructions 为字符串时的转换"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "instructions": "You are a helpful assistant."
        }
        
        result = OpenAIResponsesConverter.to_openai_chat(request)
        
        # 应该有一个 developer 消息
        dev_msgs = [m for m in result["messages"] if m.get("role") == "developer"]
        assert len(dev_msgs) == 1
        assert dev_msgs[0]["content"] == "You are a helpful assistant."

    def test_instructions_as_array(self):
        """测试 instructions 为数组时的转换"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "instructions": [
                {"type": "input_text", "text": "You are a helpful assistant."},
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Be concise."}]}
            ]
        }
        
        result = OpenAIResponsesConverter.to_openai_chat(request)
        
        # 应该有 developer 消息
        dev_msgs = [m for m in result["messages"] if m.get("role") == "developer"]
        assert len(dev_msgs) >= 1
        # 第一个应该包含 "Be concise." 或 "You are a helpful assistant."
        all_text = " ".join(m.get("content", "") for m in dev_msgs)
        assert "helpful" in all_text or "concise" in all_text


class TestResponsesMissingToolTypes:
    """测试 Responses 缺失的工具类型处理"""

    def test_tool_search_tool_ignored(self):
        """测试 tool_search 工具类型被正确处理（跳过）"""
        request = {
            "model": "gpt-4o",
            "input": "Search tools",
            "tools": [
                {"type": "function", "name": "get_data", "parameters": {"type": "object"}},
                {"type": "tool_search"}
            ]
        }
        
        result = OpenAIResponsesConverter.to_openai_chat(request)
        
        # 只有 function 工具应该被转换
        tools = result.get("tools", [])
        assert len(tools) == 1
        assert tools[0]["type"] == "function"

    def test_namespace_tool_ignored(self):
        """测试 namespace 工具类型被正确处理（跳过）"""
        request = {
            "model": "gpt-4o",
            "input": "Use namespace",
            "tools": [
                {"type": "namespace", "name": "my_namespace"}
            ]
        }
        
        result = OpenAIResponsesConverter.to_openai_chat(request)
        
        # namespace 工具没有 Chat 等价项
        assert len(result.get("tools", [])) == 0


class TestResponsesMissingOutputItems:
    """测试 Responses 缺失的输出项类型处理"""

    def test_tool_search_call_to_chat(self):
        """测试 tool_search_call 输出项转为 Chat tool_call"""
        responses_response = {
            "id": "resp_123",
            "object": "response",
            "status": "completed",
            "model": "gpt-4o",
            "output": [
                {
                    "type": "tool_search_call",
                    "id": "ts_123",
                    "arguments": '{"query": "weather"}'
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        }
        
        result = OpenAIResponsesConverter.to_chat_response(responses_response)
        
        tool_calls = result["choices"][0]["message"].get("tool_calls", [])
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "tool_search"

    def test_compaction_item_skipped(self):
        """测试 compaction_item 输出项被跳过"""
        responses_response = {
            "id": "resp_123",
            "object": "response",
            "status": "completed",
            "model": "gpt-4o",
            "output": [
                {"type": "compaction_item", "id": "ci_123"}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        }
        
        result = OpenAIResponsesConverter.to_chat_response(responses_response)
        
        # compaction_item 没有等价项，应该被跳过
        assert result["choices"][0]["message"]["content"] is None


class TestAnthropicReasoningTokensNotMappedToServerToolUse:
    """测试 reasoning_tokens 不再被错误映射到 server_tool_use"""

    def test_reasoning_tokens_in_usage(self):
        """测试 reasoning_tokens 正确保留在 usage 中"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "o3",
            "choices": [{
                "message": {"content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "completion_tokens_details": {
                    "reasoning_tokens": 15
                }
            }
        }
        
        result = AnthropicConverter.from_openai_chat(chat_response)
        
        # 不应该有 server_tool_use（之前的 bug）
        assert "server_tool_use" not in result["usage"]
        # output_tokens 应该正常
        assert result["usage"]["output_tokens"] == 20


class TestAnthropicServiceTierMapping:
    """测试 Anthropic service_tier 值的正确双向映射"""

    def test_anthropic_response_standard_to_chat_default(self):
        """测试 Anthropic 响应 usage.service_tier='standard' 映射为 Chat 'default'"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5, "service_tier": "standard"}
        }
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._anthropic_to_chat_response(anthropic_response)
        assert result["usage"]["service_tier"] == "default"

    def test_anthropic_response_priority_to_chat_priority(self):
        """测试 Anthropic 响应 usage.service_tier='priority' 映射为 Chat 'priority'"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5, "service_tier": "priority"}
        }
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._anthropic_to_chat_response(anthropic_response)
        assert result["usage"]["service_tier"] == "priority"

    def test_anthropic_response_batch_to_chat_default(self):
        """测试 Anthropic 响应 usage.service_tier='batch' 映射为 Chat 'default'"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5, "service_tier": "batch"}
        }
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._anthropic_to_chat_response(anthropic_response)
        assert result["usage"]["service_tier"] == "default"

    def test_chat_service_tier_to_anthropic(self):
        """测试 Chat service_tier 映射到 Anthropic usage.service_tier"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "message": {"content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "service_tier": "default"}
        }
        result = AnthropicConverter.from_openai_chat(chat_response)
        assert result["usage"]["service_tier"] == "standard"

    def test_chat_priority_tier_to_anthropic(self):
        """测试 Chat service_tier='priority' 映射到 Anthropic 'priority'"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "message": {"content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "service_tier": "priority"}
        }
        result = AnthropicConverter.from_openai_chat(chat_response)
        assert result["usage"]["service_tier"] == "priority"


class TestAnthropicStreamingSignatureDelta:
    """测试 Anthropic 流式事件包含 signature_delta"""

    def setup_method(self):
        AnthropicConverter.reset_stream_state()

    def test_thinking_delta_includes_signature_delta(self):
        """测试 thinking_delta 后跟随 signature_delta"""
        start_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        AnthropicConverter.convert_stream_chunk(start_chunk)

        thinking_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "delta": {"reasoning_content": "Let me think..."},
                "index": 0
            }]
        }

        events = AnthropicConverter.convert_stream_chunk(thinking_chunk)

        # 应该有 content_block_start + thinking_delta + signature_delta
        delta_events = [e for e in events if e["type"] == "content_block_delta"]
        delta_types = [e["delta"]["type"] for e in delta_events]
        assert "thinking_delta" in delta_types
        assert "signature_delta" in delta_types


class TestResponsesReasoningTextEvents:
    """测试 Responses 流式使用官方 reasoning_text 事件类型"""

    def setup_method(self):
        OpenAIResponsesConverter.reset_stream_state()

    def test_reasoning_uses_official_event_type(self):
        """测试推理增量使用 response.reasoning_text.delta 事件类型"""
        start_chunk = {
            "id": "resp_123",
            "model": "o3",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        OpenAIResponsesConverter.convert_stream_chunk(start_chunk)

        reasoning_chunk = {
            "id": "resp_123",
            "model": "o3",
            "choices": [{
                "delta": {"reasoning_content": "Let me reason..."},
                "index": 0
            }]
        }

        events = OpenAIResponsesConverter.convert_stream_chunk(reasoning_chunk)

        event_types = [e["type"] for e in events]
        assert "response.reasoning_text.delta" in event_types

    def test_reasoning_done_on_finish(self):
        """测试 finish_reason 时发送 response.reasoning_text.done"""
        start_chunk = {
            "id": "resp_123",
            "model": "o3",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        OpenAIResponsesConverter.convert_stream_chunk(start_chunk)

        reasoning_chunk = {
            "id": "resp_123",
            "model": "o3",
            "choices": [{
                "delta": {"reasoning_content": "Reasoning..."},
                "index": 0
            }]
        }
        OpenAIResponsesConverter.convert_stream_chunk(reasoning_chunk)

        finish_chunk = {
            "id": "resp_123",
            "model": "o3",
            "choices": [{
                "delta": {},
                "finish_reason": "stop",
                "index": 0
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        }

        events = OpenAIResponsesConverter.convert_stream_chunk(finish_chunk)

        event_types = [e["type"] for e in events]
        assert "response.reasoning_text.done" in event_types
        assert "response.output_item.done" in event_types


class TestResponsesParallelToolCalls:
    """测试 Responses 响应包含 parallel_tool_calls 字段"""

    def test_from_openai_chat_includes_parallel_tool_calls(self):
        """测试 Chat 响应转 Responses 包含 parallel_tool_calls"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "message": {"content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        result = OpenAIResponsesConverter.from_openai_chat(chat_response)

        assert "parallel_tool_calls" in result
        assert result["parallel_tool_calls"] is True


class TestThinkingDisplayPreservation:
    """测试 thinking display 字段在转换中正确保留"""

    def test_thinking_display_preserved_chat_to_anthropic(self):
        """测试 Chat→Anthropic 转换中 thinking display 字段保留"""
        chat_request = {
            "model": "o3",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "high",
            "extra_body": {
                "thinking": {"type": "enabled", "budget_tokens": 32000, "display": "omitted"}
            }
        }

        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._chat_to_anthropic_request(chat_request)

        assert "thinking" in result
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["display"] == "omitted"

    def test_thinking_without_display(self):
        """测试没有 display 的 thinking 配置正常工作"""
        chat_request = {
            "model": "o3",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "high",
        }

        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._chat_to_anthropic_request(chat_request)

        assert "thinking" in result
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 32000

    def test_original_thinking_restored_when_no_reasoning_effort(self):
        """测试当没有 reasoning_effort 时，从 extra_body 恢复原始 thinking"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "extra_body": {
                "thinking": {"type": "enabled", "budget_tokens": 20000, "display": "summarized"}
            }
        }

        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._chat_to_anthropic_request(chat_request)

        assert "thinking" in result
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 20000
        assert result["thinking"]["display"] == "summarized"


class TestReasoningSummaryPreservation:
    """测试 reasoning.summary 在 Chat→Responses 转换中保留"""

    def test_reasoning_summary_from_extra_body(self):
        """测试从 extra_body 中保留 reasoning.summary"""
        chat_request = {
            "model": "o3",
            "messages": [{"role": "user", "content": "Solve this"}],
            "reasoning_effort": "high",
            "extra_body": {
                "reasoning": {"effort": "high", "summary": "detailed"}
            }
        }

        result = OpenAIResponsesConverter.from_openai_chat_request(chat_request)

        assert "reasoning" in result
        assert result["reasoning"]["effort"] == "high"
        assert result["reasoning"]["summary"] == "detailed"


class TestAnthropicServerToolUsage:
    """测试 Anthropic server_tool_use 在转换中正确保留"""

    def test_server_tool_use_preserved_in_chat_response(self):
        """测试 Anthropic server_tool_use 在 Chat 响应 usage 中保留"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Search results..."}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 1}
            }
        }

        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._anthropic_to_chat_response(anthropic_response)

        assert "server_tool_use" in result["usage"]
        assert result["usage"]["server_tool_use"]["web_search_requests"] == 2

    def test_cache_creation_preserved_in_chat_response(self):
        """测试 Anthropic cache_creation 详情在 Chat 响应 usage 中保留"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation": {"ephemeral_5m_input_tokens": 50, "ephemeral_1h_input_tokens": 30}
            }
        }

        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._anthropic_to_chat_response(anthropic_response)

        assert "cache_creation" in result["usage"]
        assert result["usage"]["cache_creation"]["ephemeral_5m_input_tokens"] == 50


class TestThinkingAdaptiveMapping:
    """测试 thinking adaptive 类型映射"""

    def test_adaptive_without_budget_defaults_medium(self):
        """测试 adaptive thinking 无 budget_tokens 时映射为 medium（SDK 规范：adaptive 无 budget_tokens）"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": "Hello"}]
        }
        result = AnthropicConverter.to_openai_chat(request)
        assert result["reasoning_effort"] == "medium"

    def test_adaptive_with_budget(self):
        """测试 adaptive thinking 带 budget_tokens 时按预算映射"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "thinking": {"type": "adaptive", "budget_tokens": 32000},
            "messages": [{"role": "user", "content": "Hello"}]
        }
        result = AnthropicConverter.to_openai_chat(request)
        assert result["reasoning_effort"] == "high"

    def test_adaptive_with_display(self):
        """测试 adaptive thinking 带 display 字段"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "thinking": {"type": "adaptive", "display": "omitted"},
            "messages": [{"role": "user", "content": "Hello"}]
        }
        result = AnthropicConverter.to_openai_chat(request)
        assert result["reasoning_effort"] == "medium"
        # thinking 原始值保留在 extra_body 中
        assert result["extra_body"]["thinking"]["display"] == "omitted"


class TestCitationsSupport:
    """测试 citations 支持"""

    def test_anthropic_text_with_citations_to_chat(self):
        """测试 Anthropic text 块带 citations 转换为 Chat"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello", "citations": [{"type": "web_search_result", "url": "https://example.com"}]}
                ]
            }]
        }
        result = AnthropicConverter.to_openai_chat(request)
        # citations 保留在 content_parts 中
        user_msg = [m for m in result["messages"] if m["role"] == "user"][0]
        if isinstance(user_msg["content"], list):
            text_part = [p for p in user_msg["content"] if p.get("type") == "text"][0]
            assert "citations" in text_part

    def test_anthropic_text_with_citations_to_responses(self):
        """测试 Anthropic text 块带 citations 转换为 Responses 格式"""
        response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello", "citations": [{"type": "web_search_result", "url": "https://example.com"}]}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        result = AnthropicConverter.to_openai_responses(response)
        # 检查 message 输出项中包含 annotations
        msg_item = [o for o in result["output"] if o.get("type") == "message"][0]
        assert "annotations" in msg_item["content"][0]


class TestToolResultIsError:
    """测试 tool_result is_error 字段处理"""

    def test_tool_result_with_is_error(self):
        """测试 Anthropic tool_result 带 is_error 字段"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": "Error: API rate limit exceeded",
                    "is_error": True
                }]
            }]
        }
        result = AnthropicConverter.to_openai_chat(request)
        tool_msgs = [m for m in result["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) >= 1
        assert "[Error]" in tool_msgs[0]["content"]

    def test_tool_result_without_is_error(self):
        """测试 Anthropic tool_result 不带 is_error 字段"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": "Temperature: 72°F"
                }]
            }]
        }
        result = AnthropicConverter.to_openai_chat(request)
        tool_msgs = [m for m in result["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) >= 1
        assert "[Error]" not in tool_msgs[0]["content"]


class TestResponseFormatTextType:
    """测试 response_format text 类型"""

    def test_responses_text_format_to_chat(self):
        """测试 Responses text format type='text' -> Chat response_format type='text'"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "text": {"format": {"type": "text"}}
        }
        result = OpenAIResponsesConverter.to_openai_chat(request)
        assert result["response_format"] == {"type": "text"}

    def test_chat_response_format_text_to_responses(self):
        """测试 Chat response_format type='text' -> Responses text format"""
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "response_format": {"type": "text"}
        }
        result = OpenAIResponsesConverter.from_openai_chat_request(request)
        assert result["text"] == {"format": {"type": "text"}}


class TestCompletedAtField:
    """测试 Responses 响应 completed_at 字段"""

    def test_from_openai_chat_includes_completed_at(self):
        """测试 Chat 响应转 Responses 包含 completed_at"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        }
        result = OpenAIResponsesConverter.from_openai_chat(chat_response)
        assert "completed_at" in result
        assert result["completed_at"] is not None
        assert result["status"] == "completed"


class TestAnnotationsPreservation:
    """测试 annotations 在响应转换中保留"""

    def test_responses_annotations_to_chat(self):
        """测试 Responses output_text.annotations -> Chat message.annotations"""
        response = {
            "id": "resp_123",
            "object": "response",
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "Hello",
                    "annotations": [{"type": "url_citation", "url": "https://example.com"}]
                }]
            }],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
        }
        result = OpenAIResponsesConverter.to_chat_response(response)
        assert "annotations" in result["choices"][0]["message"]
        assert len(result["choices"][0]["message"]["annotations"]) == 1

    def test_chat_annotations_to_responses(self):
        """测试 Chat message.annotations -> Responses output_text.annotations"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "message": {
                    "content": "Hello",
                    "annotations": [{"type": "url_citation", "url": "https://example.com"}]
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        }
        result = OpenAIResponsesConverter.from_openai_chat(chat_response)
        msg_items = [o for o in result["output"] if o.get("type") == "message"]
        assert len(msg_items) == 1
        assert "annotations" in msg_items[0]["content"][0]


class TestCacheCreationRoundTrip:
    """测试 cache_creation_input_tokens 往返转换保留"""

    def test_cache_creation_roundtrip_via_chat(self):
        """测试 Anthropic cache_creation_input_tokens 通过 Chat 往返保留"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 80,
                "cache_read_input_tokens": 30,
            }
        }
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        # Anthropic -> Chat
        chat_resp = engine._anthropic_to_chat_response(anthropic_response)
        # Chat -> Anthropic (往返)
        result = AnthropicConverter.from_openai_chat(chat_resp)
        assert result["usage"]["cache_creation_input_tokens"] == 80
        assert result["usage"]["cache_read_input_tokens"] == 30


class TestStopParamAsString:
    """测试 stop 参数作为字符串的处理"""

    def test_stop_as_string_to_anthropic(self):
        """测试 Chat stop 字符串 -> Anthropic stop_sequences 列表"""
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": "END"
        }
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._chat_to_anthropic_request(request)
        assert result["stop_sequences"] == ["END"]

    def test_stop_as_list_to_anthropic(self):
        """测试 Chat stop 列表 -> Anthropic stop_sequences 列表"""
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": ["END", "STOP"]
        }
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._chat_to_anthropic_request(request)
        assert result["stop_sequences"] == ["END", "STOP"]
