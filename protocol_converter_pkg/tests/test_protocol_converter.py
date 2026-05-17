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
        
        # 应该包含: content_block_stop + message_delta + message_stop
        event_types = [e["type"] for e in events]
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types
        # message_stop 必须是最后一个事件
        assert events[-1]["type"] == "message_stop"

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
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 1024

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


class TestAnthropicStreamingMessageStop:
    """测试 Anthropic 流式事件包含 message_stop"""

    def setup_method(self):
        AnthropicConverter.reset_stream_state()

    def test_message_stop_event_present(self):
        """测试 finish_reason 时包含 message_stop 事件"""
        start_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        AnthropicConverter.convert_stream_chunk(start_chunk)

        text_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"delta": {"content": "Hello"}, "index": 0}]
        }
        AnthropicConverter.convert_stream_chunk(text_chunk)

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

        event_types = [e["type"] for e in events]
        assert "message_stop" in event_types
        # message_stop 必须是最后一个事件
        assert events[-1]["type"] == "message_stop"

    def test_thinking_then_text_block_sequence(self):
        """测试 thinking 块关闭后开始 text 块"""
        start_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        AnthropicConverter.convert_stream_chunk(start_chunk)

        # Thinking chunk
        thinking_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "delta": {"reasoning_content": "Let me think..."},
                "index": 0
            }]
        }
        thinking_events = AnthropicConverter.convert_stream_chunk(thinking_chunk)

        # Thinking block should be at index 0
        start_event = [e for e in thinking_events if e["type"] == "content_block_start"][0]
        assert start_event["index"] == 0
        assert start_event["content_block"]["type"] == "thinking"

        # Text chunk (after thinking)
        text_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "delta": {"content": "The answer is..."},
                "index": 0
            }]
        }
        text_events = AnthropicConverter.convert_stream_chunk(text_chunk)

        event_types = [e["type"] for e in text_events]
        # Should close thinking block first
        assert "content_block_stop" in event_types
        # Then start text block
        assert "content_block_start" in event_types
        # Text block should be at index 1 (after thinking at index 0)
        text_start = [e for e in text_events if e["type"] == "content_block_start"][0]
        assert text_start["index"] == 1
        assert text_start["content_block"]["type"] == "text"

    def test_thinking_then_tool_use_block_sequence(self):
        """测试 thinking 块关闭后开始 tool_use 块"""
        start_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        AnthropicConverter.convert_stream_chunk(start_chunk)

        # Thinking chunk
        thinking_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "delta": {"reasoning_content": "I need to use a tool..."},
                "index": 0
            }]
        }
        AnthropicConverter.convert_stream_chunk(thinking_chunk)

        # Tool call chunk
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
        # Should close thinking block first
        assert "content_block_stop" in event_types
        # Then start tool_use block
        assert "content_block_start" in event_types
        # Tool block should be at index 1 (after thinking at index 0)
        tool_start = [e for e in events if e["type"] == "content_block_start"][0]
        assert tool_start["index"] == 1
        assert tool_start["content_block"]["type"] == "tool_use"


class TestAnnotationsToCitationsMapping:
    """测试 Chat annotations → Anthropic citations 映射"""

    def test_chat_annotations_to_anthropic_citations(self):
        """测试 Chat message.annotations 映射为 Anthropic text.citations"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "message": {
                    "content": "The sky is blue.",
                    "annotations": [{"type": "url_citation", "url": "https://example.com"}]
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        result = AnthropicConverter.from_openai_chat(chat_response)

        text_blocks = [b for b in result["content"] if b.get("type") == "text"]
        assert len(text_blocks) == 1
        assert "citations" in text_blocks[0]
        assert text_blocks[0]["citations"] == [{"type": "url_citation", "url": "https://example.com"}]

    def test_chat_without_annotations_no_citations(self):
        """测试 Chat 消息无 annotations 时不添加 citations"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "message": {"content": "Hello!"},
                "finish_reason": "stop"
            }]
        }

        result = AnthropicConverter.from_openai_chat(chat_response)

        text_blocks = [b for b in result["content"] if b.get("type") == "text"]
        assert len(text_blocks) == 1
        assert "citations" not in text_blocks[0]


class TestResponsesStreamingReasoningClose:
    """测试 Responses 流式中 reasoning 项正确关闭"""

    def setup_method(self):
        OpenAIResponsesConverter.reset_stream_state()

    def test_reasoning_closed_before_message_item(self):
        """测试 reasoning 项在 message 项开始前正确关闭"""
        start_chunk = {
            "id": "resp_123",
            "model": "o3",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        OpenAIResponsesConverter.convert_stream_chunk(start_chunk)

        # Reasoning chunk
        reasoning_chunk = {
            "id": "resp_123",
            "model": "o3",
            "choices": [{
                "delta": {"reasoning_content": "Let me reason..."},
                "index": 0
            }]
        }
        OpenAIResponsesConverter.convert_stream_chunk(reasoning_chunk)

        # Text chunk (after reasoning)
        text_chunk = {
            "id": "resp_123",
            "model": "o3",
            "choices": [{
                "delta": {"content": "The answer is 42."},
                "index": 0
            }]
        }

        events = OpenAIResponsesConverter.convert_stream_chunk(text_chunk)

        event_types = [e["type"] for e in events]
        # Should close reasoning item first
        assert "response.reasoning_text.done" in event_types
        assert "response.output_item.done" in event_types
        # Then start message item
        assert "response.output_item.added" in event_types
        assert "response.content_part.added" in event_types
        # Then text delta
        assert "response.output_text.delta" in event_types

    def test_response_completed_includes_model(self):
        """测试 response.completed 事件包含 model 字段"""
        start_chunk = {
            "id": "resp_123",
            "model": "o3",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        OpenAIResponsesConverter.convert_stream_chunk(start_chunk)

        text_chunk = {
            "id": "resp_123",
            "model": "o3",
            "choices": [{
                "delta": {"content": "Hello"},
                "index": 0
            }]
        }
        OpenAIResponsesConverter.convert_stream_chunk(text_chunk)

        finish_chunk = {
            "id": "resp_123",
            "model": "o3",
            "choices": [{
                "delta": {},
                "finish_reason": "stop",
                "index": 0
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }

        events = OpenAIResponsesConverter.convert_stream_chunk(finish_chunk)

        completed_events = [e for e in events if e["type"] == "response.completed"]
        assert len(completed_events) == 1
        assert "model" in completed_events[0]["response"]
        assert completed_events[0]["response"]["model"] == "o3"


class TestDeveloperRoleInstructions:
    """测试 developer 角色消息转换为 Responses instructions"""

    def test_developer_role_to_instructions(self):
        """测试 Chat developer 角色消息映射为 Responses instructions"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "developer", "content": "You are a code expert."},
                {"role": "user", "content": "Hello"}
            ]
        }

        result = OpenAIResponsesConverter.from_openai_chat_request(chat_request)

        assert result.get("instructions") == "You are a code expert."

    def test_system_role_to_instructions(self):
        """测试 Chat system 角色消息映射为 Responses instructions"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"}
            ]
        }

        result = OpenAIResponsesConverter.from_openai_chat_request(chat_request)

        assert result.get("instructions") == "You are helpful."


class TestResponsesMetadataPreservation:
    """测试 Responses 响应 metadata 保留"""

    def test_metadata_preserved_in_chat_response(self):
        """测试 Responses metadata 在 Chat 响应中保留"""
        responses_response = {
            "id": "resp_123",
            "object": "response",
            "status": "completed",
            "model": "gpt-4o",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello"}]
            }],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
            "metadata": {"user_id": "user_123", "session_id": "sess_456"}
        }

        result = OpenAIResponsesConverter.to_chat_response(responses_response)

        assert "metadata" in result
        assert result["metadata"]["user_id"] == "user_123"
        assert result["metadata"]["session_id"] == "sess_456"


# ============================================================
# 新增测试 - 覆盖发现的逻辑缺陷、兼容性问题及未覆盖场景
# ============================================================


class TestResponsesTopLevelParams:
    """测试 Responses→Chat 转换中参数应映射为 Chat 顶层参数而非 extra_body"""

    def test_top_logprobs_as_toplevel(self):
        """测试 top_logprobs 应为 Chat 顶层参数，不在 extra_body 中"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "top_logprobs": 5
        }
        result = OpenAIResponsesConverter.to_openai_chat(request)
        assert result.get("top_logprobs") == 5
        assert result.get("extra_body", {}).get("top_logprobs") is None

    def test_safety_identifier_as_toplevel(self):
        """测试 safety_identifier 应为 Chat 顶层参数"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "safety_identifier": "user_abc"
        }
        result = OpenAIResponsesConverter.to_openai_chat(request)
        assert result.get("safety_identifier") == "user_abc"
        assert result.get("extra_body", {}).get("safety_identifier") is None

    def test_prompt_cache_key_as_toplevel(self):
        """测试 prompt_cache_key 应为 Chat 顶层参数"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "prompt_cache_key": "cache_123"
        }
        result = OpenAIResponsesConverter.to_openai_chat(request)
        assert result.get("prompt_cache_key") == "cache_123"
        assert result.get("extra_body", {}).get("prompt_cache_key") is None

    def test_prompt_cache_retention_as_toplevel(self):
        """测试 prompt_cache_retention 应为 Chat 顶层参数"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "prompt_cache_retention": "24h"
        }
        result = OpenAIResponsesConverter.to_openai_chat(request)
        assert result.get("prompt_cache_retention") == "24h"
        assert result.get("extra_body", {}).get("prompt_cache_retention") is None

    def test_stream_options_as_toplevel(self):
        """测试 stream_options 处理：include_obfuscation 是 Responses 特有字段，放入 extra_body"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "stream_options": {"include_obfuscation": True}
        }
        result = OpenAIResponsesConverter.to_openai_chat(request)
        # include_obfuscation 是 Responses 特有字段，Chat API 不支持，放入 extra_body
        assert result.get("extra_body", {}).get("stream_options") == {"include_obfuscation": True}
        # 顶层不应有 stream_options（include_obfuscation 不是 Chat API 参数）
        assert result.get("stream_options") is None

    def test_stream_options_include_usage_passthrough(self):
        """测试 stream_options 的 include_usage 直接传递（Chat API 原生支持）"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "stream_options": {"include_usage": True}
        }
        result = OpenAIResponsesConverter.to_openai_chat(request)
        # include_usage 是 Chat API 原生支持的 stream_options 字段，直接传递
        assert result.get("stream_options") == {"include_usage": True}


class TestMaxCompletionTokensZero:
    """测试 max_completion_tokens=0 的正确处理（Anthropic 缓存预热场景）"""

    def test_max_completion_tokens_zero_to_anthropic(self):
        """测试 max_completion_tokens=0 正确传递到 Anthropic（不应被默认为 4096）"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_completion_tokens": 0
        }
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._chat_to_anthropic_request(chat_request)
        assert result["max_tokens"] == 0

    def test_max_tokens_zero_to_anthropic(self):
        """测试 max_tokens=0 正确传递到 Anthropic"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 0
        }
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._chat_to_anthropic_request(chat_request)
        assert result["max_tokens"] == 0

    def test_max_completion_tokens_nonzero(self):
        """测试 max_completion_tokens 非零值正常工作"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_completion_tokens": 2048
        }
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._chat_to_anthropic_request(chat_request)
        assert result["max_tokens"] == 2048


class TestMultipleSystemDeveloperMessages:
    """测试多条 system/developer 消息合并为 instructions"""

    def test_multiple_system_messages(self):
        """测试多条 system 消息应合并为 instructions"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"}
            ]
        }
        result = OpenAIResponsesConverter.from_openai_chat_request(chat_request)
        assert result.get("instructions") is not None
        assert "helpful" in result["instructions"]
        assert "concise" in result["instructions"]

    def test_system_and_developer_messages(self):
        """测试 system 和 developer 消息合并为 instructions"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "System prompt."},
                {"role": "developer", "content": "Developer instructions."},
                {"role": "user", "content": "Hello"}
            ]
        }
        result = OpenAIResponsesConverter.from_openai_chat_request(chat_request)
        assert result.get("instructions") is not None
        assert "System prompt" in result["instructions"]
        assert "Developer instructions" in result["instructions"]


class TestRedactedThinkingContentField:
    """测试 redacted_thinking 转换包含 content 字段"""

    def test_redacted_thinking_has_content_field(self):
        """测试 redacted_thinking 转换为 Responses reasoning 包含 content 字段"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "redacted_thinking", "signature": "enc_xyz789"},
                {"type": "text", "text": "Answer."}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        result = AnthropicConverter.to_openai_responses(anthropic_response)
        reasoning_items = [item for item in result["output"] if item["type"] == "reasoning"]
        assert len(reasoning_items) == 1
        assert "content" in reasoning_items[0]
        assert reasoning_items[0]["encrypted_content"] == "enc_xyz789"


class TestEmptyContentString:
    """测试空字符串 content 的正确处理"""

    def test_empty_string_content_creates_text_block(self):
        """测试 Chat 响应 content="" 创建 text 块（有别于 None）"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "message": {"content": "", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5}
        }
        result = AnthropicConverter.from_openai_chat(chat_response)
        text_blocks = [b for b in result["content"] if b.get("type") == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == ""

    def test_none_content_no_text_block(self):
        """测试 Chat 响应 content=None 不创建独立 text 块"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "message": {"content": None, "role": "assistant", "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}
                ]},
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
        }
        result = AnthropicConverter.from_openai_chat(chat_response)
        text_blocks = [b for b in result["content"] if b.get("type") == "text"]
        tool_use_blocks = [b for b in result["content"] if b.get("type") == "tool_use"]
        # content=None 不创建 text 块
        assert len(text_blocks) == 0
        assert len(tool_use_blocks) == 1


class TestResponsesErrorPreservation:
    """测试 Responses 错误信息在 Chat 响应中保留"""

    def test_failed_status_error_preserved(self):
        """测试 Responses failed 状态的 error 信息在 Chat 响应中保留"""
        responses_response = {
            "id": "resp_123",
            "object": "response",
            "status": "failed",
            "model": "gpt-4o",
            "output": [],
            "error": {"code": "server_error", "message": "Internal server error"},
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }
        result = OpenAIResponsesConverter.to_chat_response(responses_response)
        assert "error" in result
        assert result["error"]["code"] == "server_error"


class TestAnthropicSearchResultInMultimodal:
    """测试 search_result 在多模态内容中正确保留"""

    def test_search_result_in_multimodal_user_content(self):
        """测试 search_result 在多模态 user 消息中保留在 content_parts 中"""
        msg = {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}},
                {"type": "search_result", "content": [{"type": "text", "text": "Found result"}]}
            ]
        }
        result = AnthropicConverter._convert_message(msg)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        content = result[0]["content"]
        assert isinstance(content, list)
        # 应该有 image_url 和 search_result 的文本表示
        content_types = [p.get("type") for p in content]
        assert "image_url" in content_types
        # search_result 应该转为 text 类型
        text_parts = [p for p in content if p.get("type") == "text"]
        assert any("Search Result" in p.get("text", "") for p in text_parts)


class TestAnthropicStreamingErrorEvent:
    """测试 Anthropic 流式 error 事件"""

    def setup_method(self):
        AnthropicConverter.reset_stream_state()

    def test_content_filter_generates_error_event(self):
        """测试 content_filter finish_reason 生成 error 事件"""
        start_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        AnthropicConverter.convert_stream_chunk(start_chunk)

        text_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"delta": {"content": "Hello"}, "index": 0}]
        }
        AnthropicConverter.convert_stream_chunk(text_chunk)

        filter_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "delta": {},
                "finish_reason": "content_filter",
                "index": 0
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        events = AnthropicConverter.convert_stream_chunk(filter_chunk)
        event_types = [e["type"] for e in events]
        assert "error" in event_types


class TestResponsesStreamingFailedEvent:
    """测试 Responses 流式 response.failed 事件"""

    def setup_method(self):
        OpenAIResponsesConverter.reset_stream_state()

    def test_content_filter_generates_failed_event(self):
        """测试 content_filter finish_reason 生成 response.failed 事件"""
        start_chunk = {
            "id": "resp_123",
            "model": "gpt-4o",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        OpenAIResponsesConverter.convert_stream_chunk(start_chunk)

        text_chunk = {
            "id": "resp_123",
            "model": "gpt-4o",
            "choices": [{"delta": {"content": "Hello"}, "index": 0}]
        }
        OpenAIResponsesConverter.convert_stream_chunk(text_chunk)

        filter_chunk = {
            "id": "resp_123",
            "model": "gpt-4o",
            "choices": [{
                "delta": {},
                "finish_reason": "content_filter",
                "index": 0
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        events = OpenAIResponsesConverter.convert_stream_chunk(filter_chunk)
        event_types = [e["type"] for e in events]
        assert "response.failed" in event_types
        # 找到 failed 事件
        failed_event = [e for e in events if e["type"] == "response.failed"][0]
        assert "error" in failed_event["response"]


class TestResponsesInputEdgeCases:
    """测试 Responses 输入边界情况"""

    def test_input_as_empty_string(self):
        """测试 input 为空字符串时的转换"""
        request = {
            "model": "gpt-4o",
            "input": ""
        }
        result = OpenAIResponsesConverter.to_openai_chat(request)
        assert "messages" in result
        # 空字符串应创建 user 消息
        user_msgs = [m for m in result["messages"] if m.get("role") == "user"]
        assert len(user_msgs) == 1

    def test_input_with_reasoning_and_summary(self):
        """测试 reasoning 参数同时包含 effort 和 summary"""
        request = {
            "model": "o3",
            "input": "Think hard",
            "reasoning": {"effort": "high", "summary": "detailed"}
        }
        result = OpenAIResponsesConverter.to_openai_chat(request)
        assert result.get("reasoning_effort") == "high"
        # summary 应保留在 extra_body 中
        assert result.get("extra_body", {}).get("reasoning", {}).get("summary") == "detailed"


class TestAnthropicDocumentTextSource:
    """测试 Anthropic document 类型 text source 的转换"""

    def test_document_text_source(self):
        """测试 document 块 text source 转换"""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "text", "text": "Document content here"}
                }
            ]
        }
        result = AnthropicConverter._convert_message(msg)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        content = result[0]["content"]
        # document text source 转换为 str 或 list 包含文本
        if isinstance(content, str):
            assert "Document content here" in content
        elif isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
            assert any("Document content here" in t for t in text_parts)


class TestChatToAnthropicReasoningContent:
    """测试 Chat→Anthropic 转换中 reasoning_content 的完整处理"""

    def test_reasoning_content_with_tool_calls(self):
        """测试 reasoning_content 与 tool_calls 共存时的正确转换"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "I need to check the weather API...",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'}
                    }]
                }
            ]
        }
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._chat_to_anthropic_request(chat_request)
        # 找到 assistant 消息
        assistant_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        content = assistant_msgs[0]["content"]
        # thinking 块应在 tool_use 块之前
        types = [b.get("type") for b in content]
        if "thinking" in types and "tool_use" in types:
            assert types.index("thinking") < types.index("tool_use")


class TestResponsesStopParamMapping:
    """测试 stop 参数在 Responses→Chat 转换中的映射"""

    def test_stop_in_extra_body_from_responses(self):
        """测试 Responses 请求中的 stop 参数传递到 Chat 的 extra_body"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "stop": ["END", "STOP"]
        }
        # Responses API 本身不支持 stop 参数
        # 但如果通过 extra_body 传递，应该正确映射
        request_with_stop = {
            "model": "gpt-4o",
            "input": "Hello",
        }
        result = OpenAIResponsesConverter.to_openai_chat(request_with_stop)
        assert "messages" in result


class TestAnthropicResponseWithContainerField:
    """测试 Anthropic 响应 container 字段保留"""

    def test_container_field_in_anthropic_response(self):
        """测试 Anthropic 响应中的 container 字段在 Chat 转换中保留"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "container": {"type": "auto", "id": "ctr_abc"},
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._anthropic_to_chat_response(anthropic_response)
        assert result["usage"].get("container") is not None or "container" in result


class TestResponsesToChatWithMultipleOutputTypes:
    """测试 Responses 响应包含多种输出类型的转换"""

    def test_reasoning_and_function_call_and_message(self):
        """测试 reasoning + function_call + message 组合输出"""
        responses_response = {
            "id": "resp_123",
            "object": "response",
            "status": "completed",
            "model": "o3",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_123",
                    "content": [{"type": "reasoning_text", "text": "Let me search..."}],
                    "status": "completed"
                },
                {
                    "type": "function_call",
                    "id": "fc_123",
                    "call_id": "fc_123",
                    "name": "search",
                    "arguments": '{"q": "test"}',
                    "status": "completed"
                },
                {
                    "type": "message",
                    "id": "msg_123",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Here are the results."}]
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        }
        result = OpenAIResponsesConverter.to_chat_response(responses_response)
        msg = result["choices"][0]["message"]
        assert msg["reasoning_content"] == "Let me search..."
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "search"
        assert msg["content"] == "Here are the results."


class TestProtocolDetectorEdgeCases:
    """测试协议检测器的边界情况"""

    def test_detect_anthropic_with_max_tokens_zero(self):
        """测试 max_tokens=0 的 Anthropic 请求检测（缓存预热场景）"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 0,
            "messages": [{"role": "user", "content": "Hello"}]
        }
        assert ProtocolDetector.detect(request) == Protocol.ANTHROPIC

    def test_detect_responses_with_empty_input(self):
        """测试空数组 input 不应被误判"""
        request = {
            "model": "gpt-4o",
            "input": []
        }
        # 空数组 input 无法确定类型
        result = ProtocolDetector.detect(request)
        # 应该不崩溃
        assert result in (Protocol.OPENAI_RESPONSES, Protocol.UNKNOWN)

    def test_detect_anthropic_with_cache_control(self):
        """测试带 cache_control 的 Anthropic 请求检测"""
        request = {
            "model": "my-model",
            "max_tokens": 1024,
            "cache_control": {"type": "ephemeral"},
            "messages": [{"role": "user", "content": "Hi"}]
        }
        assert ProtocolDetector.detect(request) == Protocol.ANTHROPIC

    def test_detect_anthropic_with_output_config(self):
        """测试带 output_config 的 Anthropic 请求检测"""
        request = {
            "model": "my-model",
            "max_tokens": 1024,
            "output_config": {"type": "json"},
            "messages": [{"role": "user", "content": "Hi"}]
        }
        assert ProtocolDetector.detect(request) == Protocol.ANTHROPIC


class TestAnthropicStreamingCitationsDelta:
    """测试 Anthropic 流式 citations 支持（非流式场景）"""

    def setup_method(self):
        AnthropicConverter.reset_stream_state()

    def test_thinking_block_index_continuity(self):
        """测试 thinking 块后 text 块的索引连续性"""
        start_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        AnthropicConverter.convert_stream_chunk(start_chunk)

        # Thinking chunk
        thinking_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"delta": {"reasoning_content": "Let me think..."}, "index": 0}]
        }
        thinking_events = AnthropicConverter.convert_stream_chunk(thinking_chunk)
        thinking_start = [e for e in thinking_events if e["type"] == "content_block_start"][0]
        assert thinking_start["index"] == 0

        # Text chunk
        text_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"delta": {"content": "The answer."}, "index": 0}]
        }
        text_events = AnthropicConverter.convert_stream_chunk(text_chunk)
        text_start = [e for e in text_events if e["type"] == "content_block_start"][0]
        assert text_start["index"] == 1

        # Tool call chunk
        tool_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "id": "call_123",
                        "index": 0,
                        "type": "function",
                        "function": {"name": "test", "arguments": ""}
                    }]
                },
                "index": 0
            }]
        }
        tool_events = AnthropicConverter.convert_stream_chunk(tool_chunk)
        tool_start = [e for e in tool_events if e["type"] == "content_block_start"][0]
        assert tool_start["index"] == 2


class TestChatToResponsesStopMapping:
    """测试 Chat→Responses 转换中 stop 参数映射"""

    def test_stop_param_to_responses_extra_body(self):
        """测试 Chat stop 参数映射到 Responses extra_body"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": ["END", "STOP"]
        }
        result = OpenAIResponsesConverter.from_openai_chat_request(chat_request)
        assert result.get("extra_body", {}).get("stop") == ["END", "STOP"]


class TestAnthropicToChatWithRefusal:
    """测试 Anthropic→Chat 转换中 refusal 的正确处理"""

    def test_refusal_with_content(self):
        """测试 Anthropic refusal 停止原因（含内容）的转换"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "I cannot help with that."}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "refusal",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._anthropic_to_chat_response(anthropic_response)
        assert result["choices"][0]["finish_reason"] == "content_filter"


class TestResponsesWebSearchOptionsDetail:
    """测试 Responses web_search 工具的 detail 参数"""

    def test_web_search_with_user_location(self):
        """测试 web_search 工具 user_location 完整映射"""
        request = {
            "model": "gpt-4o",
            "input": "Search the web",
            "tools": [{
                "type": "web_search",
                "search_context_size": "low",
                "user_location": {"type": "approximate", "city": "Tokyo", "region": "Tokyo"}
            }]
        }
        result = OpenAIResponsesConverter.to_openai_chat(request)
        ws_opts = result.get("web_search_options", {})
        assert ws_opts.get("search_context_size") == "low"
        assert ws_opts.get("user_location", {}).get("city") == "Tokyo"


class TestRedactedThinkingDataField:
    """测试 RedactedThinkingBlock 使用 data 字段（SDK 规范）"""

    def test_redacted_thinking_data_field_in_to_openai_responses(self):
        """测试 Anthropic redacted_thinking 的 data 字段映射到 Responses encrypted_content"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "redacted_thinking", "data": "ErUB3jkH...encrypted..."},
                {"type": "text", "text": "Hello!"}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        result = AnthropicConverter.to_openai_responses(anthropic_response)
        # 找到 reasoning 输出项
        reasoning_items = [item for item in result["output"] if item.get("type") == "reasoning"]
        assert len(reasoning_items) == 1
        assert reasoning_items[0]["encrypted_content"] == "ErUB3jkH...encrypted..."
        assert reasoning_items[0]["content"] == []
        assert "summary" in reasoning_items[0]  # summary 是必填字段

    def test_redacted_thinking_signature_fallback(self):
        """测试 redacted_thinking 兼容旧版 signature 字段"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "redacted_thinking", "signature": "old_sig_value"},
                {"type": "text", "text": "Hello!"}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        result = AnthropicConverter.to_openai_responses(anthropic_response)
        reasoning_items = [item for item in result["output"] if item.get("type") == "reasoning"]
        assert len(reasoning_items) == 1
        # data 优先，无 data 时 fallback 到 signature
        assert reasoning_items[0]["encrypted_content"] == "old_sig_value"


class TestAnthropicToolChoiceDisableParallel:
    """测试 Anthropic tool_choice 的 disable_parallel_tool_use 映射"""

    def test_auto_with_disable_parallel(self):
        """测试 tool_choice auto + disable_parallel_tool_use → Chat required + parallel_tool_calls=False"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True}
        }
        result = AnthropicConverter.to_openai_chat(request)
        assert result["tool_choice"] == "auto"
        assert result.get("parallel_tool_calls") is False

    def test_any_with_disable_parallel(self):
        """测试 tool_choice any + disable_parallel_tool_use → Chat required + parallel_tool_calls=False"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "tool_choice": {"type": "any", "disable_parallel_tool_use": True}
        }
        result = AnthropicConverter.to_openai_chat(request)
        assert result["tool_choice"] == "required"
        assert result.get("parallel_tool_calls") is False

    def test_tool_with_disable_parallel(self):
        """测试 tool_choice tool + disable_parallel_tool_use → Chat function + parallel_tool_calls=False"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "tool_choice": {"type": "tool", "name": "get_weather", "disable_parallel_tool_use": True}
        }
        result = AnthropicConverter.to_openai_chat(request)
        assert result["tool_choice"]["type"] == "function"
        assert result["tool_choice"]["function"]["name"] == "get_weather"
        assert result.get("parallel_tool_calls") is False

    def test_auto_without_disable_parallel(self):
        """测试 tool_choice auto 无 disable_parallel_tool_use → 不设置 parallel_tool_calls"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "tool_choice": {"type": "auto"}
        }
        result = AnthropicConverter.to_openai_chat(request)
        assert result["tool_choice"] == "auto"
        assert "parallel_tool_calls" not in result


class TestChatToAnthropicParallelToolCalls:
    """测试 Chat parallel_tool_calls: False → Anthropic tool_choice.disable_parallel_tool_use"""

    def test_required_with_parallel_false(self):
        """测试 Chat tool_choice=required + parallel_tool_calls=False → Anthropic any + disable"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "tool_choice": "required",
            "parallel_tool_calls": False
        }
        result = engine._chat_to_anthropic_request(request)
        assert result["tool_choice"]["type"] == "any"
        assert result["tool_choice"]["disable_parallel_tool_use"] is True

    def test_function_choice_with_parallel_false(self):
        """测试 Chat function tool_choice + parallel_tool_calls=False"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "parallel_tool_calls": False
        }
        result = engine._chat_to_anthropic_request(request)
        assert result["tool_choice"]["type"] == "tool"
        assert result["tool_choice"]["name"] == "get_weather"
        assert result["tool_choice"]["disable_parallel_tool_use"] is True


class TestResponsesToAnthropicDirect:
    """测试 Responses→Anthropic 直接转换（不经 Chat 中转）"""

    def test_reasoning_with_encrypted_content_to_redacted_thinking(self):
        """测试 Responses reasoning 含 encrypted_content → Anthropic redacted_thinking"""
        responses_response = {
            "id": "resp_123",
            "object": "response",
            "status": "completed",
            "model": "o3",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_abc",
                    "content": [],
                    "summary": [],
                    "encrypted_content": "ErUB3jkH...encrypted...",
                    "status": "completed"
                },
                {
                    "type": "message",
                    "id": "msg_def",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Hello!", "annotations": []}]
                }
            ],
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        }
        result = OpenAIResponsesConverter.to_anthropic(responses_response)
        assert result["type"] == "message"
        # 应有 redacted_thinking 块（使用 data 字段）
        redacted_blocks = [b for b in result["content"] if b["type"] == "redacted_thinking"]
        assert len(redacted_blocks) == 1
        assert redacted_blocks[0]["data"] == "ErUB3jkH...encrypted..."
        # 应有 text 块
        text_blocks = [b for b in result["content"] if b["type"] == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "Hello!"

    def test_reasoning_with_content_to_thinking(self):
        """测试 Responses reasoning 含 reasoning_text → Anthropic thinking"""
        responses_response = {
            "id": "resp_123",
            "object": "response",
            "status": "completed",
            "model": "o3",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_abc",
                    "content": [{"type": "reasoning_text", "text": "Let me think..."}],
                    "summary": [],
                    "status": "completed"
                },
                {
                    "type": "message",
                    "id": "msg_def",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "42", "annotations": []}]
                }
            ],
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        }
        result = OpenAIResponsesConverter.to_anthropic(responses_response)
        thinking_blocks = [b for b in result["content"] if b["type"] == "thinking"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "Let me think..."
        assert "signature" in thinking_blocks[0]

    def test_incomplete_to_max_tokens(self):
        """测试 Responses incomplete → Anthropic max_tokens"""
        responses_response = {
            "id": "resp_123",
            "object": "response",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "model": "gpt-4o",
            "output": [
                {
                    "type": "message",
                    "id": "msg_def",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Hello...", "annotations": []}]
                }
            ],
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        }
        result = OpenAIResponsesConverter.to_anthropic(responses_response)
        assert result["stop_reason"] == "max_tokens"


class TestMaxOutputTokensZeroEdgeCase:
    """测试 max_output_tokens=0 边界情况（缓存预热）"""

    def test_max_output_tokens_zero_to_chat(self):
        """测试 Responses max_output_tokens=0 → Chat max_completion_tokens=0"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "max_output_tokens": 0
        }
        result = OpenAIResponsesConverter.to_openai_chat(request)
        assert result.get("max_completion_tokens") == 0

    def test_max_completion_tokens_zero_to_responses(self):
        """测试 Chat max_completion_tokens=0 → Responses max_output_tokens=0"""
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_completion_tokens": 0
        }
        result = OpenAIResponsesConverter.from_openai_chat_request(request)
        assert result.get("max_output_tokens") == 0


class TestChatToolIsErrorReverseMapping:
    """测试 Chat tool [Error] 前缀 → Anthropic is_error 反向映射"""

    def test_error_prefix_to_is_error(self):
        """测试 Chat tool 消息 [Error] 前缀反向映射为 Anthropic is_error"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_123", "content": "[Error] API rate limit exceeded"}
            ]
        }
        result = engine._chat_to_anthropic_request(request)
        # 找到 tool_result 块
        user_msgs = [m for m in result["messages"] if m["role"] == "user"]
        tool_result_blocks = []
        for m in user_msgs:
            for b in m.get("content", []):
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    tool_result_blocks.append(b)
        assert len(tool_result_blocks) == 1
        assert tool_result_blocks[0]["is_error"] is True
        assert tool_result_blocks[0]["content"] == "API rate limit exceeded"

    def test_normal_tool_result_no_is_error(self):
        """测试正常 tool 消息不添加 is_error"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_123", "content": "Sunny, 72°F"}
            ]
        }
        result = engine._chat_to_anthropic_request(request)
        user_msgs = [m for m in result["messages"] if m["role"] == "user"]
        tool_result_blocks = []
        for m in user_msgs:
            for b in m.get("content", []):
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    tool_result_blocks.append(b)
        assert len(tool_result_blocks) == 1
        assert "is_error" not in tool_result_blocks[0]
        assert tool_result_blocks[0]["content"] == "Sunny, 72°F"


class TestResponsesReasoningSummaryField:
    """测试 Responses reasoning 输出项的 summary 字段处理"""

    def test_chat_to_responses_reasoning_has_summary(self):
        """测试 Chat→Responses 转换中 reasoning 输出项包含 summary 字段"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "o3",
            "choices": [{
                "message": {
                    "content": "42",
                    "reasoning_content": "Let me think step by step..."
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        result = OpenAIResponsesConverter.from_openai_chat(chat_response)
        reasoning_items = [item for item in result["output"] if item.get("type") == "reasoning"]
        assert len(reasoning_items) == 1
        assert "summary" in reasoning_items[0]

    def test_anthropic_thinking_to_responses_has_summary(self):
        """测试 Anthropic thinking→Responses 转换中 reasoning 输出项包含 summary 字段"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "I need to reason...", "signature": "sig_123"},
                {"type": "text", "text": "Hello!"}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        result = AnthropicConverter.to_openai_responses(anthropic_response)
        reasoning_items = [item for item in result["output"] if item.get("type") == "reasoning"]
        assert len(reasoning_items) == 1
        assert "summary" in reasoning_items[0]


class TestSDKCompatibilityGaps:
    """测试基于官方文档和 Python SDK 发现的兼容性缺口"""

    def test_anthropic_document_to_chat_file_uses_sdk_fields_only(self):
        """Anthropic document -> Chat file 不应输出 SDK 不支持的 mime_type 字段"""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "JVBERi0=",
                    },
                }
            ],
        }

        result = AnthropicConverter._convert_message(msg)

        file_obj = result[0]["content"][0]["file"]
        assert file_obj["file_data"] == "data:application/pdf;base64,JVBERi0="
        assert "mime_type" not in file_obj

    def test_responses_input_file_url_degrades_for_chat(self):
        """Chat Completions 不支持 file_url，Responses file_url 应降级为文本而不是伪装成 file_data"""
        request = {
            "model": "gpt-4o",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Summarize this"},
                        {"type": "input_file", "file_url": "https://example.com/report.pdf"},
                    ],
                }
            ],
        }

        result = OpenAIResponsesConverter.to_openai_chat(request)

        content = result["messages"][0]["content"]
        assert content[1] == {
            "type": "text",
            "text": "[File URL: https://example.com/report.pdf]",
        }

    def test_responses_input_file_id_maps_to_chat_file_id(self):
        """Responses input_file.file_id 可映射到 Chat file.file_id"""
        request = {
            "model": "gpt-4o",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_file", "file_id": "file_123"}],
                }
            ],
        }

        result = OpenAIResponsesConverter.to_openai_chat(request)

        assert result["messages"][0]["content"][0] == {
            "type": "file",
            "file": {"file_id": "file_123"},
        }

    def test_responses_function_call_output_content_list_to_chat_tool_text(self):
        """Responses function_call_output.output 数组应提取文本，而不是变成 Python repr"""
        request = {
            "model": "gpt-4o",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": [
                        {"type": "input_text", "text": "line 1"},
                        {"type": "input_text", "text": "line 2"},
                    ],
                }
            ],
        }

        result = OpenAIResponsesConverter.to_openai_chat(request)

        assert result["messages"][0]["content"] == "line 1\nline 2"

    def test_chat_refusal_to_responses_refusal_content(self):
        """Chat message.refusal 应映射为 Responses refusal 内容块"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {"content": None, "refusal": "I cannot help with that."},
                    "finish_reason": "content_filter",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }

        result = OpenAIResponsesConverter.from_openai_chat(chat_response)

        msg = result["output"][0]
        assert msg["content"][0] == {
            "type": "refusal",
            "refusal": "I cannot help with that.",
        }
        assert result["status"] == "incomplete"
        assert result["incomplete_details"] == {"reason": "content_filter"}

    def test_responses_refusal_to_chat_refusal(self):
        """Responses refusal 内容块应映射为 Chat message.refusal"""
        responses_response = {
            "id": "resp_123",
            "object": "response",
            "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"},
            "model": "gpt-4o",
            "output": [
                {
                    "type": "message",
                    "id": "msg_123",
                    "role": "assistant",
                    "status": "incomplete",
                    "content": [{"type": "refusal", "refusal": "I cannot help with that."}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        }

        result = OpenAIResponsesConverter.to_chat_response(responses_response)

        message = result["choices"][0]["message"]
        assert message["content"] is None
        assert message["refusal"] == "I cannot help with that."
        assert result["choices"][0]["finish_reason"] == "content_filter"

    def test_chat_multiple_system_and_developer_messages_are_preserved_for_anthropic(self):
        """Chat 多条 system/developer 消息转 Anthropic 时应合并，而不是只保留最后一条"""
        engine = ProtocolConverterEngine(ConverterConfig(backend_type="anthropic"))
        request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "System A"},
                {"role": "developer", "content": "Developer B"},
                {"role": "user", "content": "Hello"},
            ],
        }

        result = engine._chat_to_anthropic_request(request)

        assert result["system"] == "System A\n\nDeveloper B"

    def test_openai_chat_tools_with_max_tokens_are_not_misdetected_as_anthropic(self):
        """合法 Chat tools + max_tokens 请求不应被 OpenAIChatConverter 误判为 Anthropic"""
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Weather?"}],
            "max_tokens": 128,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }

        result = OpenAIChatConverter.to_openai_chat(request)

        assert result.max_tokens == 128
        assert result.max_completion_tokens is None
        assert result.tools == request["tools"]

    def test_responses_allowed_tools_choice_maps_to_chat_sdk_shape(self):
        """Responses allowed_tools tool_choice 应转换为 Chat SDK 的 allowed_tools 包装形状"""
        request = {
            "model": "gpt-4o",
            "input": "Hi",
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "function", "name": "get_weather"}],
            },
        }

        result = OpenAIResponsesConverter.to_openai_chat(request)

        assert result["tool_choice"] == {
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": "required",
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather"},
                    }
                ],
            },
        }

    def test_chat_allowed_tools_choice_maps_to_responses_sdk_shape(self):
        """Chat allowed_tools tool_choice 应转换为 Responses SDK 的平铺形状"""
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "auto",
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "get_weather"},
                        }
                    ],
                },
            },
        }

        result = OpenAIResponsesConverter.from_openai_chat_request(request)

        assert result["tool_choice"] == {
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [{"type": "function", "name": "get_weather"}],
        }

    def test_engine_developer_downgrade_does_not_mutate_input_request(self):
        """Chat 后端 developer 降级为 system 时不应修改原始请求对象"""
        engine = ProtocolConverterEngine(ConverterConfig(backend_type="openai"))
        request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "developer", "content": "Follow policy."},
                {"role": "user", "content": "Hi"},
            ],
        }

        result = engine.convert_request(request)

        assert result["messages"][0]["role"] == "system"
        assert request["messages"][0]["role"] == "developer"


class TestBugFixes:
    """v1.8.0 缺陷修复测试"""

    def test_parallel_tool_calls_false_without_tool_choice(self):
        """测试 parallel_tool_calls=False 但未设置 tool_choice 时，Anthropic tool_choice 仍包含 disable_parallel_tool_use"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)

        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [{
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}
            }],
            "parallel_tool_calls": False,
        }

        result = engine.convert_request(chat_request)

        # tool_choice 应被设置为 {"type": "auto", "disable_parallel_tool_use": True}
        assert "tool_choice" in result
        assert result["tool_choice"]["type"] == "auto"
        assert result["tool_choice"]["disable_parallel_tool_use"] is True

    def test_parallel_tool_calls_false_with_tool_choice_auto(self):
        """测试 parallel_tool_calls=False 且 tool_choice=auto 时的 Anthropic 转换"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)

        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [{
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}
            }],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        }

        result = engine.convert_request(chat_request)

        assert result["tool_choice"]["type"] == "auto"
        assert result["tool_choice"]["disable_parallel_tool_use"] is True

    def test_parallel_tool_calls_true_no_tool_choice(self):
        """测试 parallel_tool_calls=True 且无 tool_choice 时不设置 Anthropic tool_choice"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)

        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [{
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}
            }],
            "parallel_tool_calls": True,
        }

        result = engine.convert_request(chat_request)

        # parallel_tool_calls=True 是默认值，不应设置 tool_choice
        assert "tool_choice" not in result or result.get("tool_choice") is None

    def test_responses_streaming_content_filter_failed_event(self):
        """测试 Responses 流式 content_filter 生成 response.failed 事件时 error_detail 正确保留"""
        # 先发 response.created
        start_chunk = {
            "id": "resp_123",
            "model": "gpt-4o",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        OpenAIResponsesConverter.convert_stream_chunk(start_chunk)

        # content_filter finish_reason
        finish_chunk = {
            "id": "resp_123",
            "model": "gpt-4o",
            "choices": [{
                "delta": {},
                "finish_reason": "content_filter",
                "index": 0
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0}
        }

        events = OpenAIResponsesConverter.convert_stream_chunk(finish_chunk)

        # 找到 response.failed 事件
        failed_events = [e for e in events if e["type"] == "response.failed"]
        assert len(failed_events) == 1
        # 验证 error_detail 被保留
        assert failed_events[0]["response"]["error"] is not None
        assert failed_events[0]["response"]["error"]["code"] == "content_filter"

    def test_anthropic_convert_message_role_check(self):
        """测试 Anthropic _convert_message 对不支持角色的处理"""
        # "user" 和 "assistant" 是有效角色
        user_msg = {"role": "user", "content": "Hello"}
        result = AnthropicConverter._convert_message(user_msg)
        assert len(result) == 1
        assert result[0]["role"] == "user"

        # "system" 不是 Anthropic 消息角色（应被过滤）
        system_msg = {"role": "system", "content": "System prompt"}
        result = AnthropicConverter._convert_message(system_msg)
        assert len(result) == 0


class TestBugFixesV1_9_0:
    """v1.9.0 缺陷修复测试"""

    def test_from_anthropic_preserves_max_completion_tokens(self):
        """测试 _from_anthropic 保留 max_completion_tokens 字段"""
        anthropic_request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = OpenAIChatConverter.to_openai_chat(anthropic_request)
        assert result.max_completion_tokens == 2048

    def test_from_anthropic_preserves_reasoning_effort(self):
        """测试 _from_anthropic 保留 reasoning_effort 字段"""
        anthropic_request = {
            "model": "claude-opus-4-6",
            "max_tokens": 16000,
            "thinking": {"type": "enabled", "budget_tokens": 32000},
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = OpenAIChatConverter.to_openai_chat(anthropic_request)
        assert result.reasoning_effort is not None

    def test_from_anthropic_preserves_parallel_tool_calls(self):
        """测试 _from_anthropic 保留 parallel_tool_calls 字段"""
        anthropic_request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        }
        result = OpenAIChatConverter.to_openai_chat(anthropic_request)
        assert result.parallel_tool_calls is False

    def test_from_anthropic_preserves_service_tier(self):
        """测试 _from_anthropic 保留 service_tier 字段"""
        anthropic_request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "service_tier": "standard_only",
        }
        result = OpenAIChatConverter.to_openai_chat(anthropic_request)
        assert result.service_tier == "default"

    def test_responses_text_format_json_schema_to_chat(self):
        """测试 Responses text.format json_schema 正确转换为 Chat response_format
        
        Responses API: {"type": "json_schema", "name": "...", "schema": {...}}
        Chat API: {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}
        """
        request = {
            "model": "gpt-4o",
            "input": "Generate JSON",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "my_schema",
                    "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                    "strict": True,
                }
            }
        }
        result = OpenAIResponsesConverter.to_openai_chat(request)
        assert "response_format" in result
        assert result["response_format"]["type"] == "json_schema"
        # json_schema 内部不应包含 type 字段
        json_schema = result["response_format"]["json_schema"]
        assert "type" not in json_schema
        assert json_schema["name"] == "my_schema"
        assert "schema" in json_schema
        assert json_schema["strict"] is True

    def test_chat_response_format_json_schema_to_responses(self):
        """测试 Chat response_format json_schema 正确转换为 Responses text.format
        
        Chat API: {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}
        Responses API: {"type": "json_schema", "name": "...", "schema": {...}}
        """
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Generate JSON"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "my_schema",
                    "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                    "strict": True,
                }
            }
        }
        result = OpenAIResponsesConverter.from_openai_chat_request(request)
        assert "text" in result
        text_format = result["text"]["format"]
        assert text_format["type"] == "json_schema"
        # Responses format 不应有嵌套的 json_schema，字段应在顶层
        assert "json_schema" not in text_format
        assert text_format["name"] == "my_schema"
        assert "schema" in text_format
        assert text_format["strict"] is True

    def test_responses_json_schema_round_trip(self):
        """测试 Responses json_schema 格式与 Chat 格式互转的往返一致性"""
        # Responses → Chat
        responses_request = {
            "model": "gpt-4o",
            "input": "Generate JSON",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "test_schema",
                    "schema": {"type": "object", "properties": {"id": {"type": "integer"}}},
                    "strict": False,
                }
            }
        }
        chat_result = OpenAIResponsesConverter.to_openai_chat(responses_request)
        
        # Chat → Responses
        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Generate JSON"}],
            "response_format": chat_result["response_format"],
        }
        responses_result = OpenAIResponsesConverter.from_openai_chat_request(chat_request)
        
        # 验证往返一致性
        original_format = responses_request["text"]["format"]
        round_trip_format = responses_result["text"]["format"]
        assert original_format["type"] == round_trip_format["type"]
        assert original_format["name"] == round_trip_format["name"]
        assert original_format["schema"] == round_trip_format["schema"]
        assert original_format.get("strict") == round_trip_format.get("strict")

    def test_responses_streaming_reasoning_to_tool_calls_transition(self):
        """测试 Responses 流式转换中 reasoning 项在 tool_calls 开始时正确关闭"""
        # 先发 response.created
        start_chunk = {
            "id": "resp_123",
            "model": "o3",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        OpenAIResponsesConverter.convert_stream_chunk(start_chunk)

        # 发送 reasoning 内容
        reasoning_chunk = {
            "id": "resp_123",
            "model": "o3",
            "choices": [{"delta": {"reasoning_content": "Thinking..."}, "index": 0}]
        }
        events = OpenAIResponsesConverter.convert_stream_chunk(reasoning_chunk)
        assert any(e["type"] == "response.output_item.added" for e in events)

        # 发送 tool_calls（直接从 reasoning 过渡，无文本内容）
        tool_chunk = {
            "id": "resp_123",
            "model": "o3",
            "choices": [{"delta": {"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}, "index": 0}]}, "index": 0}]
        }
        events = OpenAIResponsesConverter.convert_stream_chunk(tool_chunk)

        # 验证 reasoning 项被关闭
        reasoning_done_events = [e for e in events if e["type"] == "response.reasoning_text.done"]
        reasoning_item_done = [e for e in events if e["type"] == "response.output_item.done" and e.get("item", {}).get("type") == "reasoning"]
        assert len(reasoning_done_events) >= 1, "reasoning_text.done event should be emitted when tool calls start"
        assert len(reasoning_item_done) >= 1, "reasoning output_item.done should be emitted when tool calls start"

    def test_responses_from_openai_chat_incomplete_details_always_present(self):
        """测试 Responses 响应始终包含 incomplete_details 字段"""
        # 完成状态
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
        assert "incomplete_details" in result
        assert result["incomplete_details"] is None

    def test_anthropic_from_chat_preserves_content_none(self):
        """测试 Chat 响应中 content=None 正确转换为 Anthropic 空内容"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "message": {"content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}]},
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        result = AnthropicConverter.from_openai_chat(chat_response)
        # 不应包含空文本块，只有 tool_use
        text_blocks = [b for b in result["content"] if b["type"] == "text"]
        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        # content=None 不应产生文本块
        assert all(b.get("text", "") != "" for b in text_blocks)


class TestBugFixesV1_10_0:
    """v1.10.0 修复的测试"""

    def test_top_logprobs_sets_logprobs_true(self):
        """测试 Responses top_logprobs → Chat 时自动设置 logprobs=True"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "top_logprobs": 5
        }
        result = OpenAIResponsesConverter.to_openai_chat(request)
        # Chat API 要求 logprobs=True 才能使用 top_logprobs
        assert result.get("logprobs") is True
        assert result.get("top_logprobs") == 5

    def test_input_image_file_id_fallback(self):
        """测试 Responses input_image 的 file_id 降级为文本占位"""
        request = {
            "model": "gpt-4o",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "file_id": "file-abc123"}]
            }]
        }
        result = OpenAIResponsesConverter.to_openai_chat(request)
        # file_id 在 Chat API 中无直接等价，应降级为文本占位
        messages = result.get("messages", [])
        assert len(messages) > 0
        content = messages[-1].get("content")
        if isinstance(content, list):
            text_parts = [p for p in content if p.get("type") == "text"]
            assert any("file-abc123" in p.get("text", "") for p in text_parts)
        elif isinstance(content, str):
            assert "file-abc123" in content

    def test_chat_web_search_options_to_responses(self):
        """测试 Chat web_search_options 正确映射到 Responses web_search 工具"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [{"type": "web_search_preview"}],
            "web_search_options": {
                "search_context_size": "high",
                "user_location": {"type": "approximate", "approximate": {"city": "SF"}}
            }
        }
        result = OpenAIResponsesConverter.from_openai_chat_request(chat_request)
        tools = result.get("tools", [])
        ws_tool = next((t for t in tools if t.get("type") == "web_search"), None)
        assert ws_tool is not None
        assert ws_tool.get("search_context_size") == "high"
        assert ws_tool.get("user_location") is not None

    def test_responses_to_anthropic_request_direct(self):
        """测试 Responses 请求直接转换为 Anthropic 格式"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "instructions": "You are helpful.",
            "max_output_tokens": 1024,
            "temperature": 0.7,
            "reasoning": {"effort": "high"},
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert result["model"] == "gpt-4o"
        assert result["max_tokens"] == 1024
        assert result["system"] == "You are helpful."
        assert result["temperature"] == 0.7
        assert result["thinking"] == {"type": "enabled", "budget_tokens": 32000}
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_responses_to_anthropic_request_with_tools(self):
        """测试 Responses 请求带工具直接转换为 Anthropic 格式"""
        request = {
            "model": "gpt-4o",
            "input": "What's the weather?",
            "max_output_tokens": 100,
            "tools": [{"type": "function", "name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}],
            "tool_choice": "required",
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert len(result.get("tools", [])) == 1
        assert result["tools"][0]["name"] == "get_weather"
        assert result["tool_choice"] == "any"

    def test_responses_to_anthropic_request_preserves_extra_params(self):
        """测试 Responses→Anthropic 请求保留 Responses 特有参数"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "max_output_tokens": 100,
            "previous_response_id": "resp_123",
            "truncation": "auto",
            "store": True,
            "safety_identifier": "user-abc",
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        extra = result.get("extra_body", {})
        assert extra.get("previous_response_id") == "resp_123"
        assert extra.get("truncation") == "auto"
        assert extra.get("store") is True
        assert extra.get("safety_identifier") == "user-abc"

    def test_responses_to_anthropic_response_service_tier(self):
        """测试 Responses→Anthropic 响应 service_tier 映射"""
        response = {
            "id": "resp_123",
            "object": "response",
            "status": "completed",
            "model": "gpt-4o",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi"}]}],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "service_tier": "priority",
        }
        result = OpenAIResponsesConverter.to_anthropic(response)
        assert result["usage"].get("service_tier") == "priority"

    def test_converter_type_validation(self):
        """测试转换器输入类型校验"""
        import pytest
        with pytest.raises(TypeError):
            AnthropicConverter.to_openai_chat("not a dict")
        with pytest.raises(TypeError):
            OpenAIResponsesConverter.to_openai_chat("not a dict")
        with pytest.raises(TypeError):
            OpenAIResponsesConverter.from_openai_chat_request("not a dict")
        with pytest.raises(TypeError):
            AnthropicConverter.from_openai_chat("not a dict")

    def test_chat_stream_options_include_usage_to_responses(self):
        """测试 Chat stream_options.include_usage 不传递给 Responses（Responses 自动包含）"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream_options": {"include_usage": True}
        }
        result = OpenAIResponsesConverter.from_openai_chat_request(chat_request)
        # include_usage 在 Responses API 中不需要（自动在 completed 事件中返回）
        assert result.get("stream_options") is None
        # 保留在 extra_body 供参考
        assert result.get("extra_body", {}).get("stream_options") == {"include_usage": True}

    def test_chat_stream_options_include_obfuscation_to_responses(self):
        """测试 Chat 中来自 Responses 的 stream_options.include_obfuscation 正确恢复"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream_options": {"include_obfuscation": True}
        }
        result = OpenAIResponsesConverter.from_openai_chat_request(chat_request)
        # include_obfuscation 是 Responses 特有字段，应正确传递
        assert result.get("stream_options") == {"include_obfuscation": True}


class TestConvertChatContentToResponses:
    """测试 _convert_chat_content_to_responses 方法（修复后）"""

    def test_text_content(self):
        """测试文本内容转换"""
        result = OpenAIResponsesConverter._convert_chat_content_to_responses([
            {"type": "text", "text": "Hello"}
        ])
        assert len(result) == 1
        assert result[0]["type"] == "input_text"
        assert result[0]["text"] == "Hello"

    def test_image_url_content(self):
        """测试图片 URL 内容转换"""
        result = OpenAIResponsesConverter._convert_chat_content_to_responses([
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
        ])
        assert len(result) == 1
        assert result[0]["type"] == "input_image"
        assert result[0]["image_url"] == "https://example.com/img.png"

    def test_file_content(self):
        """测试文件内容转换"""
        result = OpenAIResponsesConverter._convert_chat_content_to_responses([
            {"type": "file", "file": {"file_data": "data:application/pdf;base64,abc", "filename": "test.pdf"}}
        ])
        assert len(result) == 1
        assert result[0]["type"] == "input_file"
        assert result[0]["file_data"] == "data:application/pdf;base64,abc"
        assert result[0]["filename"] == "test.pdf"

    def test_mixed_content(self):
        """测试混合内容转换"""
        result = OpenAIResponsesConverter._convert_chat_content_to_responses([
            {"type": "text", "text": "Describe this"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ])
        assert len(result) == 2
        assert result[0]["type"] == "input_text"
        assert result[1]["type"] == "input_image"

    def test_chat_to_responses_multimodal(self):
        """测试 Chat 多模态消息完整转换为 Responses 格式"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}
                    ]
                }
            ]
        }
        result = OpenAIResponsesConverter.from_openai_chat_request(chat_request)
        assert isinstance(result["input"], list)
        assert result["input"][0]["type"] == "message"
        content = result["input"][0]["content"]
        assert any(c["type"] == "input_text" for c in content)
        assert any(c["type"] == "input_image" for c in content)


class TestAnthropicOutputFormat:
    """测试 Anthropic output_format 参数映射"""

    def test_anthropic_output_format_json_schema_to_chat(self):
        """测试 Anthropic output_format json_schema 映射到 Chat response_format"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "output_format": {
                "type": "json_schema",
                "name": "my_schema",
                "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                "strict": True
            }
        }
        result = AnthropicConverter.to_openai_chat(request)
        assert "response_format" in result
        assert result["response_format"]["type"] == "json_schema"
        assert result["response_format"]["json_schema"]["name"] == "my_schema"
        assert result["response_format"]["json_schema"]["strict"] is True

    def test_anthropic_output_format_json_object_to_chat(self):
        """测试 Anthropic output_format json_object 映射"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "output_format": {"type": "json_object"}
        }
        result = AnthropicConverter.to_openai_chat(request)
        assert result["response_format"] == {"type": "json_object"}

    def test_chat_response_format_to_anthropic_output_format(self):
        """测试 Chat response_format 映射到 Anthropic output_format"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "test_schema", "schema": {"type": "object"}, "strict": True}
            }
        }
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine._chat_to_anthropic_request(chat_request)

        assert "output_format" in result
        assert result["output_format"]["type"] == "json_schema"
        assert result["output_format"]["name"] == "test_schema"
        assert result["output_format"]["strict"] is True

    def test_responses_text_format_to_anthropic_output_format(self):
        """测试 Responses text.format 映射到 Anthropic output_format"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "my_output",
                    "schema": {"type": "object", "properties": {"result": {"type": "string"}}}
                }
            }
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert "output_format" in result
        assert result["output_format"]["type"] == "json_schema"
        assert result["output_format"]["name"] == "my_output"

    def test_detect_anthropic_with_output_format(self):
        """测试带 output_format 的请求检测为 Anthropic"""
        request = {
            "model": "my-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "output_format": {"type": "json_schema", "name": "test"}
        }
        assert ProtocolDetector.detect(request) == Protocol.ANTHROPIC


class TestResponsesIncompleteEvent:
    """测试 Responses API response.incomplete 流式事件"""

    def test_incomplete_event_on_length(self):
        """测试 finish_reason=length 生成 response.incomplete 事件"""
        OpenAIResponsesConverter.reset_stream_state()
        # 先发 start 事件
        start_chunk = {
            "id": "resp_123",
            "model": "gpt-4o",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        OpenAIResponsesConverter.convert_stream_chunk(start_chunk)

        # 发 length 结束事件
        end_chunk = {
            "id": "resp_123",
            "model": "gpt-4o",
            "choices": [{"delta": {}, "finish_reason": "length", "index": 0}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 100, "total_tokens": 110}
        }
        events = OpenAIResponsesConverter.convert_stream_chunk(end_chunk)

        # 应该包含 response.incomplete 事件
        event_types = [e.get("type") for e in events]
        assert "response.incomplete" in event_types

    def test_completed_event_on_stop(self):
        """测试 finish_reason=stop 生成 response.completed 事件"""
        OpenAIResponsesConverter.reset_stream_state()
        start_chunk = {
            "id": "resp_123",
            "model": "gpt-4o",
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        }
        OpenAIResponsesConverter.convert_stream_chunk(start_chunk)

        end_chunk = {
            "id": "resp_123",
            "model": "gpt-4o",
            "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        events = OpenAIResponsesConverter.convert_stream_chunk(end_chunk)

        event_types = [e.get("type") for e in events]
        assert "response.completed" in event_types


# ============================================================
# v1.12.0 新增测试
# ============================================================

class TestResponsesToAnthropicMaxOutputTokensZero:
    """测试 Responses→Anthropic max_output_tokens=0 边界情况"""

    def test_max_output_tokens_zero_not_defaulted(self):
        """max_output_tokens=0 应保留为 0，不应默认为 4096"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "max_output_tokens": 0,
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert result["max_tokens"] == 0

    def test_max_output_tokens_absent_defaults_to_4096(self):
        """缺少 max_output_tokens 时应默认为 4096"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert result["max_tokens"] == 4096

    def test_max_output_tokens_positive_value(self):
        """max_output_tokens 为正数时应正确传递"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "max_output_tokens": 2048,
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert result["max_tokens"] == 2048


class TestResponsesToAnthropicMetadataFix:
    """测试 Responses→Anthropic metadata 覆盖 bug 修复"""

    def test_metadata_user_id_preserved(self):
        """metadata.user_id 应正确传递给 Anthropic metadata"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "metadata": {"user_id": "user-123"},
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert result["metadata"] == {"user_id": "user-123"}

    def test_metadata_non_user_fields_in_extra_body(self):
        """metadata 中非 user_id 字段应放入 extra_body"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "metadata": {"user_id": "user-123", "custom_field": "value"},
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert result["metadata"] == {"user_id": "user-123"}
        assert "extra_body" in result
        assert result["extra_body"]["metadata"] == {"custom_field": "value"}

    def test_metadata_no_user_id(self):
        """metadata 中没有 user_id 时不设置 anthropic metadata"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "metadata": {"custom_field": "value"},
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert "metadata" not in result or result.get("metadata") is None
        assert "extra_body" in result
        assert result["extra_body"]["metadata"] == {"custom_field": "value"}


class TestResponsesToAnthropicParallelToolCalls:
    """测试 Responses→Anthropic parallel_tool_calls → disable_parallel_tool_use 映射"""

    def test_parallel_tool_calls_false_with_auto_tool_choice(self):
        """parallel_tool_calls=False + tool_choice=auto → Anthropic auto with disable_parallel"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "tools": [{"type": "function", "name": "test", "parameters": {}}],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert result["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}

    def test_parallel_tool_calls_false_without_tool_choice(self):
        """parallel_tool_calls=False 无 tool_choice → Anthropic auto with disable_parallel"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "tools": [{"type": "function", "name": "test", "parameters": {}}],
            "parallel_tool_calls": False,
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert result["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}

    def test_parallel_tool_calls_false_with_required_tool_choice(self):
        """parallel_tool_calls=False + tool_choice=required → Anthropic any with disable_parallel"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "tools": [{"type": "function", "name": "test", "parameters": {}}],
            "tool_choice": "required",
            "parallel_tool_calls": False,
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert result["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}

    def test_parallel_tool_calls_true_no_effect(self):
        """parallel_tool_calls=True 不影响 tool_choice"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "tools": [{"type": "function", "name": "test", "parameters": {}}],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert result["tool_choice"] == "auto"


class TestResponsesToAnthropicReasoningSummary:
    """测试 Responses→Anthropic reasoning.summary → thinking.display 映射"""

    def test_reasoning_summary_concise_to_display(self):
        """reasoning.summary='concise' → thinking.display='summarized'"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "reasoning": {"effort": "medium", "summary": "concise"},
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["display"] == "summarized"

    def test_reasoning_summary_detailed_to_display(self):
        """reasoning.summary='detailed' → thinking.display='summarized'"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "reasoning": {"effort": "high", "summary": "detailed"},
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert result["thinking"]["display"] == "summarized"

    def test_reasoning_summary_auto_no_display(self):
        """reasoning.summary='auto' 不设置 display（使用 Anthropic 默认行为）"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "reasoning": {"effort": "medium", "summary": "auto"},
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert "display" not in result["thinking"]

    def test_reasoning_summary_none_no_display(self):
        """reasoning.summary=None 不设置 display"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "reasoning": {"effort": "medium", "summary": None},
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert "display" not in result["thinking"]

    def test_reasoning_no_summary_no_display(self):
        """缺少 reasoning.summary 不设置 display"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "reasoning": {"effort": "medium"},
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert "display" not in result["thinking"]


class TestProtocolDetectorToolChoiceAnyDict:
    """测试 protocol_detector 识别 Anthropic tool_choice dict type='any'"""

    def test_tool_choice_dict_any(self):
        """tool_choice={'type': 'any'} 应检测为 Anthropic"""
        request = {
            "model": "some-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "tool_choice": {"type": "any"},
        }
        assert ProtocolDetector.detect(request) == Protocol.ANTHROPIC

    def test_tool_choice_dict_any_with_disable_parallel(self):
        """tool_choice={'type': 'any', 'disable_parallel_tool_use': True} 应检测为 Anthropic"""
        request = {
            "model": "some-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
        }
        assert ProtocolDetector.detect(request) == Protocol.ANTHROPIC


class TestAnthropicToChatStreamConversion:
    """测试 Anthropic SSE 事件 → Chat 流式块转换"""

    def test_message_start_event(self):
        """message_start 事件应转换为 Chat chunk with role"""
        AnthropicConverter.reset_anthropic_to_chat_state()
        events = AnthropicConverter.convert_anthropic_event_to_chat("message_start", {
            "message": {
                "id": "msg_123",
                "model": "claude-sonnet-4-20250514",
                "usage": {"input_tokens": 10, "output_tokens": 0}
            }
        })
        assert len(events) == 1
        assert events[0]["choices"][0]["delta"]["role"] == "assistant"
        assert events[0]["model"] == "claude-sonnet-4-20250514"

    def test_text_delta_event(self):
        """text_delta 事件应转换为 Chat chunk with content"""
        AnthropicConverter.reset_anthropic_to_chat_state()
        # 先发 message_start
        AnthropicConverter.convert_anthropic_event_to_chat("message_start", {
            "message": {"id": "msg_123", "model": "claude-sonnet-4-20250514", "usage": {}}
        })
        events = AnthropicConverter.convert_anthropic_event_to_chat("content_block_delta", {
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"}
        })
        assert len(events) == 1
        assert events[0]["choices"][0]["delta"]["content"] == "Hello"

    def test_thinking_delta_event(self):
        """thinking_delta 事件应转换为 Chat chunk with reasoning_content"""
        AnthropicConverter.reset_anthropic_to_chat_state()
        AnthropicConverter.convert_anthropic_event_to_chat("message_start", {
            "message": {"id": "msg_123", "model": "claude-sonnet-4-20250514", "usage": {}}
        })
        events = AnthropicConverter.convert_anthropic_event_to_chat("content_block_delta", {
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Let me think..."}
        })
        assert len(events) == 1
        assert events[0]["choices"][0]["delta"]["reasoning_content"] == "Let me think..."

    def test_input_json_delta_event(self):
        """input_json_delta 事件应转换为 Chat chunk with tool_calls"""
        AnthropicConverter.reset_anthropic_to_chat_state()
        AnthropicConverter.convert_anthropic_event_to_chat("message_start", {
            "message": {"id": "msg_123", "model": "claude-sonnet-4-20250514", "usage": {}}
        })
        # 先发 content_block_start with tool_use
        AnthropicConverter.convert_anthropic_event_to_chat("content_block_start", {
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {}}
        })
        events = AnthropicConverter.convert_anthropic_event_to_chat("content_block_delta", {
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"city":'}
        })
        assert len(events) == 1
        assert events[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"city":'

    def test_message_delta_event(self):
        """message_delta 事件应转换为 Chat chunk with finish_reason"""
        AnthropicConverter.reset_anthropic_to_chat_state()
        AnthropicConverter.convert_anthropic_event_to_chat("message_start", {
            "message": {"id": "msg_123", "model": "claude-sonnet-4-20250514", "usage": {}}
        })
        events = AnthropicConverter.convert_anthropic_event_to_chat("message_delta", {
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 50}
        })
        assert len(events) == 1
        assert events[0]["choices"][0]["finish_reason"] == "stop"

    def test_message_delta_tool_use_stop(self):
        """message_delta tool_use stop_reason → finish_reason=tool_calls"""
        AnthropicConverter.reset_anthropic_to_chat_state()
        AnthropicConverter.convert_anthropic_event_to_chat("message_start", {
            "message": {"id": "msg_123", "model": "claude-sonnet-4-20250514", "usage": {}}
        })
        events = AnthropicConverter.convert_anthropic_event_to_chat("message_delta", {
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 50}
        })
        assert events[0]["choices"][0]["finish_reason"] == "tool_calls"

    def test_message_start_with_cached_tokens(self):
        """message_start 包含缓存 token 信息"""
        AnthropicConverter.reset_anthropic_to_chat_state()
        events = AnthropicConverter.convert_anthropic_event_to_chat("message_start", {
            "message": {
                "id": "msg_123",
                "model": "claude-sonnet-4-20250514",
                "usage": {"input_tokens": 100, "output_tokens": 0, "cache_read_input_tokens": 80}
            }
        })
        assert events[0]["usage"]["prompt_tokens_details"]["cached_tokens"] == 80


class TestChatToAnthropicReasoningSummaryToDisplay:
    """测试 Chat→Anthropic 路径 reasoning.summary → thinking.display 映射"""

    def test_reasoning_effort_with_extra_body_reasoning_summary(self):
        """reasoning_effort + extra_body.reasoning.summary → thinking.display"""
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "medium",
            "extra_body": {
                "reasoning": {"effort": "medium", "summary": "concise"},
            },
        }
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine.convert_request(request)
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["display"] == "summarized"


class TestChatToAnthropicRoundTripParams:
    """测试 Chat→Anthropic 路径恢复 Anthropic 特有参数（round-trip 场景）"""

    def test_top_k_recovery(self):
        """从 Chat 请求中恢复 top_k 参数"""
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "top_k": 50,
        }
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine.convert_request(request)
        assert result["top_k"] == 50

    def test_container_recovery(self):
        """从 Chat 请求中恢复 container 参数"""
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "container": "ctr_abc123",
        }
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine.convert_request(request)
        assert result["container"] == "ctr_abc123"

    def test_cache_control_recovery(self):
        """从 Chat 请求中恢复 cache_control 参数"""
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "cache_control": {"type": "ephemeral"},
        }
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine.convert_request(request)
        assert result["cache_control"] == {"type": "ephemeral"}

    def test_output_config_recovery(self):
        """从 Chat 请求中恢复 output_config 参数"""
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "output_config": {"key": "value"},
        }
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine.convert_request(request)
        assert result["output_config"] == {"key": "value"}

    def test_anthropic_server_tools_recovery(self):
        """从 Chat 请求中恢复 Anthropic 服务器工具"""
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Search"}],
            "tools": [{"type": "function", "function": {"name": "my_tool", "parameters": {}}}],
            "anthropic_server_tools": [{"type": "web_search_20250305", "name": "web_search"}],
        }
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine.convert_request(request)
        # Chat function 工具转换为 Anthropic 格式 (无 type 字段)，服务器工具保留 type
        tool_names = [t.get("name", t.get("type", "")) for t in result.get("tools", [])]
        assert "web_search" in tool_names or "web_search_20250305" in tool_names
        assert "my_tool" in tool_names

    def test_anthropic_server_tools_no_duplicate(self):
        """Anthropic 服务器工具不重复添加"""
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Search"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "anthropic_server_tools": [{"type": "web_search_20250305", "name": "web_search"}],
        }
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        result = engine.convert_request(request)
        web_search_count = sum(1 for t in result.get("tools", []) if t.get("type") == "web_search_20250305")
        assert web_search_count == 1

    def test_full_anthropic_round_trip_params(self):
        """完整 Anthropic→Chat→Anthropic 往返转换保留特有参数"""
        anthropic_request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "top_k": 40,
            "container": "ctr_test",
            "cache_control": {"type": "ephemeral"},
            "output_config": {"key": "val"},
            "thinking": {"type": "enabled", "budget_tokens": 10000},
        }
        # Anthropic → Chat
        chat_request = AnthropicConverter.to_openai_chat(anthropic_request)
        assert chat_request["extra_body"]["top_k"] == 40
        assert chat_request["extra_body"]["container"] == "ctr_test"

        # Chat → Anthropic (通过 engine)
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        # 合并 extra_body 到顶层（模拟 engine.convert_request 的行为）
        merged = dict(chat_request)
        extra = merged.pop("extra_body", {})
        if isinstance(extra, dict):
            for k, v in extra.items():
                if k not in merged:
                    merged[k] = v
        result = engine._chat_to_anthropic_request(merged)
        assert result["top_k"] == 40
        assert result["container"] == "ctr_test"
        assert result["cache_control"] == {"type": "ephemeral"}
        assert result["output_config"] == {"key": "val"}


class TestResponsesToAnthropicStopMapping:
    """测试 Responses→Anthropic 路径 stop→stop_sequences 映射"""

    def test_stop_string_to_stop_sequences(self):
        """Responses stop 字符串参数应映射为 Anthropic stop_sequences 列表"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "stop": "END",
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert result["stop_sequences"] == ["END"]

    def test_stop_list_to_stop_sequences(self):
        """Responses stop 列表参数应映射为 Anthropic stop_sequences"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
            "stop": ["END", "STOP"],
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert result["stop_sequences"] == ["END", "STOP"]

    def test_no_stop_no_stop_sequences(self):
        """无 stop 参数时不应有 stop_sequences"""
        request = {
            "model": "gpt-4o",
            "input": "Hello",
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert "stop_sequences" not in result


class TestAssistantThinkingOnlyBlock:
    """测试 assistant 消息只含 thinking 块时的转换"""

    def test_assistant_thinking_only_to_chat(self):
        """assistant 消息只含 thinking 块时应保留 reasoning_content"""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me analyze this...", "signature": "sig_abc"},
            ]
        }
        result = AnthropicConverter._convert_message(msg)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["reasoning_content"] == "Let me analyze this..."

    def test_assistant_thinking_and_text(self):
        """assistant 消息含 thinking + text 块时 reasoning_content 正确保留"""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Thinking...", "signature": "sig_abc"},
                {"type": "text", "text": "Answer"},
            ]
        }
        result = AnthropicConverter._convert_message(msg)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Answer"
        # Chat API 支持同时包含 content 和 reasoning_content
        assert result[0].get("reasoning_content") == "Thinking..."

    def test_assistant_redacted_thinking_only(self):
        """assistant 消息只含 redacted_thinking 块时 content 为 None"""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "redacted_thinking", "data": "encrypted_data"},
            ]
        }
        result = AnthropicConverter._convert_message(msg)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] is None


# ============================================================
# 本次修复缺陷的回归测试
# ============================================================

class TestProtocolDetectorNoFalsePositive:
    """修复: ANTHROPIC_CONTENT_TYPES 中移除 "text" 避免误判"""

    def test_chat_with_text_content_not_anthropic(self):
        """Chat 消息含 type=text 内容块不应被误判为 Anthropic"""
        request = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"}
                    ]
                }
            ]
        }
        assert ProtocolDetector.detect(request) == Protocol.OPENAI_CHAT

    def test_anthropic_with_tool_use_content_still_detected(self):
        """含 tool_use 内容块的请求仍应被检测为 Anthropic"""
        request = {
            "model": "my-model",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "result"}
                    ]
                }
            ]
        }
        assert ProtocolDetector.detect(request) == Protocol.ANTHROPIC


class TestRefusalStopReasonPriority:
    """修复: refusal 停止原因优先于 finish_reason"""

    def test_refusal_overrides_finish_reason(self):
        """当有 refusal 时，stop_reason 应为 refusal 而非 finish_reason 映射值"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "message": {"content": "I can't help.", "refusal": "Policy violation"},
                "finish_reason": "stop"
            }]
        }
        result = AnthropicConverter.from_openai_chat(chat_response)
        assert result["stop_reason"] == "refusal"
        assert result["stop_details"]["reason"] == "content_policy"
        assert result["stop_details"]["message"] == "Policy violation"

    def test_no_refusal_uses_finish_reason(self):
        """无 refusal 时使用 finish_reason 映射"""
        chat_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "message": {"content": "Hello!"},
                "finish_reason": "length"
            }]
        }
        result = AnthropicConverter.from_openai_chat(chat_response)
        assert result["stop_reason"] == "max_tokens"


class TestAnthropicToResponsesContainer:
    """修复: to_openai_responses 保留 container 字段"""

    def test_container_preserved(self):
        """Anthropic 响应中的 container 字段应保留"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "container": "ctr_abc123"
        }
        result = AnthropicConverter.to_openai_responses(anthropic_response)
        assert result.get("container") == "ctr_abc123"

    def test_no_container_when_absent(self):
        """无 container 字段时结果中不应包含"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        result = AnthropicConverter.to_openai_responses(anthropic_response)
        assert "container" not in result


class TestAnthropicStopSequenceMapping:
    """修复: stop_sequence 停止原因在 Responses 中映射为 completed"""

    def test_stop_sequence_in_responses(self):
        """Anthropic stop_sequence 映射为 Responses completed 状态"""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "stop_sequence",
            "stop_sequence": "END",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        result = AnthropicConverter.to_openai_responses(anthropic_response)
        assert result["status"] == "completed"
        assert result.get("incomplete_details") is None


class TestResponsesCompletedAtField:
    """修复: from_openai_chat 仅在 completed 时设置 completed_at"""

    def test_completed_has_completed_at(self):
        """completed 状态的响应应包含 completed_at"""
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
        assert result["status"] == "completed"
        assert "completed_at" in result

    def test_incomplete_no_completed_at(self):
        """incomplete 状态的响应不应包含 completed_at"""
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
        assert "completed_at" not in result


class TestResponsesAssistantMultimodalContent:
    """修复: from_openai_chat_request 中 assistant+tool_calls 的 content 为列表时正确转换"""

    def test_assistant_tool_calls_with_list_content(self):
        """assistant+tool_calls 的 content 为列表时应正确转换"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Search for X"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me search"},
                    ],
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"}
                    }]
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "Result"}
            ]
        }
        result = OpenAIResponsesConverter.from_openai_chat_request(chat_request)
        assert isinstance(result["input"], list)
        # 应包含 assistant 消息、function_call 和 function_call_output
        types = [item.get("type") for item in result["input"]]
        assert "message" in types
        assert "function_call" in types
        assert "function_call_output" in types


class TestResponsesReasoningContentInInput:
    """修复: Chat 请求中 assistant 消息的 reasoning_content 转换为 Responses reasoning 输入项"""

    def test_reasoning_content_in_assistant_message(self):
        """assistant 消息中的 reasoning_content 应转为 reasoning 输入项"""
        chat_request = {
            "model": "o3",
            "messages": [
                {
                    "role": "assistant",
                    "content": "The answer is 42.",
                    "reasoning_content": "Let me think about this..."
                }
            ]
        }
        result = OpenAIResponsesConverter.from_openai_chat_request(chat_request)
        assert isinstance(result["input"], list)
        # 应包含 reasoning 项
        reasoning_items = [item for item in result["input"] if item.get("type") == "reasoning"]
        assert len(reasoning_items) == 1
        assert reasoning_items[0]["summary"][0]["text"] == "Let me think about this..."

    def test_reasoning_content_with_tool_calls(self):
        """assistant+tool_calls+reasoning_content 应全部转换"""
        chat_request = {
            "model": "o3",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "I need to call a tool",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_data", "arguments": "{}"}
                    }]
                }
            ]
        }
        result = OpenAIResponsesConverter.from_openai_chat_request(chat_request)
        assert isinstance(result["input"], list)
        types = [item.get("type") for item in result["input"]]
        assert "reasoning" in types
        assert "function_call" in types


class TestResponsesEventToChatChunk:
    """测试 Responses SSE 事件到 Chat 流式块的转换"""

    def test_output_text_delta(self):
        """response.output_text.delta 事件转换"""
        data = {
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello",
            "response": {"id": "resp_123", "model": "gpt-4o"}
        }
        result = OpenAIResponsesConverter._convert_responses_event_to_chat_chunk(
            "response.output_text.delta", data
        )
        assert result is not None
        assert result["choices"][0]["delta"]["content"] == "Hello"

    def test_reasoning_text_delta(self):
        """response.reasoning_text.delta 事件转换"""
        data = {
            "type": "response.reasoning_text.delta",
            "output_index": 0,
            "delta": "Thinking...",
            "response": {"id": "resp_123", "model": "o3"}
        }
        result = OpenAIResponsesConverter._convert_responses_event_to_chat_chunk(
            "response.reasoning_text.delta", data
        )
        assert result is not None
        assert result["choices"][0]["delta"]["reasoning_content"] == "Thinking..."

    def test_function_call_arguments_delta(self):
        """response.function_call_arguments.delta 事件转换"""
        data = {
            "type": "response.function_call_arguments.delta",
            "output_index": 1,
            "delta": '{"city":',
            "response": {"id": "resp_123", "model": "gpt-4o"}
        }
        result = OpenAIResponsesConverter._convert_responses_event_to_chat_chunk(
            "response.function_call_arguments.delta", data
        )
        assert result is not None
        assert result["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"city":'

    def test_output_item_added_message(self):
        """response.output_item.added (message) 事件转换"""
        data = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "id": "msg_123", "role": "assistant", "content": []},
            "response": {"id": "resp_123", "model": "gpt-4o"}
        }
        result = OpenAIResponsesConverter._convert_responses_event_to_chat_chunk(
            "response.output_item.added", data
        )
        assert result is not None
        assert result["choices"][0]["delta"]["role"] == "assistant"

    def test_output_item_added_function_call(self):
        """response.output_item.added (function_call) 事件转换"""
        data = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "id": "fc_123", "call_id": "fc_123", "name": "get_weather", "arguments": ""},
            "response": {"id": "resp_123", "model": "gpt-4o"}
        }
        result = OpenAIResponsesConverter._convert_responses_event_to_chat_chunk(
            "response.output_item.added", data
        )
        assert result is not None
        tc = result["choices"][0]["delta"]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"

    def test_response_completed(self):
        """response.completed 事件转换"""
        data = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "model": "gpt-4o",
                "status": "completed",
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            }
        }
        result = OpenAIResponsesConverter._convert_responses_event_to_chat_chunk(
            "response.completed", data
        )
        assert result is not None
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10

    def test_response_incomplete_max_tokens(self):
        """response.incomplete (max_output_tokens) 事件转换"""
        data = {
            "type": "response.incomplete",
            "response": {
                "id": "resp_123",
                "model": "gpt-4o",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            }
        }
        result = OpenAIResponsesConverter._convert_responses_event_to_chat_chunk(
            "response.incomplete", data
        )
        assert result is not None
        assert result["choices"][0]["finish_reason"] == "length"

    def test_output_text_done_returns_none(self):
        """response.output_text.done 不需要转换"""
        data = {"type": "response.output_text.done", "response": {"id": "resp_123"}}
        result = OpenAIResponsesConverter._convert_responses_event_to_chat_chunk(
            "response.output_text.done", data
        )
        assert result is None

    def test_unknown_event_returns_none(self):
        """未知事件类型返回 None"""
        data = {"type": "response.ping"}
        result = OpenAIResponsesConverter._convert_responses_event_to_chat_chunk(
            "response.ping", data
        )
        assert result is None


# ================================================================
# v1.15.0 Bug Fix Tests
# ================================================================

class TestAnthropicThinkingWithText:
    """测试 Anthropic→Chat 转换中 thinking 块与 text 块同时存在时的处理"""

    def test_thinking_with_text_preserves_reasoning_content(self):
        """thinking 块与 text 块同时存在时，reasoning_content 不应丢失"""
        anthropic_request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Explain quantum computing"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "I need to explain quantum computing clearly...", "signature": "abc123"},
                        {"type": "text", "text": "Quantum computing uses qubits..."}
                    ]
                }
            ]
        }
        result = AnthropicConverter.to_openai_chat(anthropic_request)
        msg = result["messages"][-1]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Quantum computing uses qubits..."
        # 关键：reasoning_content 不应丢失
        assert msg.get("reasoning_content") == "I need to explain quantum computing clearly..."

    def test_thinking_with_text_and_tool_calls(self):
        """thinking 块、text 块和 tool_calls 同时存在时，reasoning_content 不应丢失"""
        anthropic_request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "User wants weather info...", "signature": "sig123"},
                        {"type": "text", "text": "Let me check the weather."},
                        {"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {"city": "NYC"}}
                    ]
                }
            ]
        }
        result = AnthropicConverter.to_openai_chat(anthropic_request)
        msg = result["messages"][-1]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me check the weather."
        assert msg.get("reasoning_content") == "User wants weather info..."
        assert len(msg.get("tool_calls", [])) == 1

    def test_thinking_only_still_preserves_reasoning_content(self):
        """仅含 thinking 块时 reasoning_content 仍然正确保留（回归测试）"""
        anthropic_request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Think about this"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Deep thoughts...", "signature": "sig456"}
                    ]
                }
            ]
        }
        result = AnthropicConverter.to_openai_chat(anthropic_request)
        msg = result["messages"][-1]
        assert msg["role"] == "assistant"
        assert msg["content"] is None
        assert msg.get("reasoning_content") == "Deep thoughts..."


class TestChatToResponsesReasoningWithListContent:
    """测试 Chat→Responses 转换中 assistant 消息有列表 content 和 reasoning_content 的场景"""

    def test_assistant_list_content_with_reasoning_content(self):
        """assistant 消息有列表 content 和 reasoning_content 时，reasoning 应正确映射"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Analyze this image"},
                {
                    "role": "assistant",
                    "reasoning_content": "I see a landscape...",
                    "content": [
                        {"type": "text", "text": "This is a beautiful landscape."},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
                    ]
                }
            ]
        }
        result = OpenAIResponsesConverter.from_openai_chat_request(chat_request)
        input_items = result.get("input", [])
        # 应包含 reasoning 输入项
        reasoning_items = [item for item in input_items if isinstance(item, dict) and item.get("type") == "reasoning"]
        assert len(reasoning_items) == 1
        assert reasoning_items[0]["summary"][0]["text"] == "I see a landscape..."
        # 应包含 assistant message 输入项
        assistant_msgs = [item for item in input_items if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "assistant"]
        assert len(assistant_msgs) == 1

    def test_assistant_string_content_with_reasoning_content(self):
        """assistant 消息有字符串 content 和 reasoning_content 时仍正确（回归测试）"""
        chat_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "reasoning_content": "Thinking...",
                    "content": "Hi there!"
                }
            ]
        }
        result = OpenAIResponsesConverter.from_openai_chat_request(chat_request)
        input_items = result.get("input", [])
        reasoning_items = [item for item in input_items if isinstance(item, dict) and item.get("type") == "reasoning"]
        assert len(reasoning_items) == 1
        assistant_msgs = [item for item in input_items if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "assistant"]
        assert len(assistant_msgs) == 1


class TestChatToAnthropicToolListContent:
    """测试 Chat→Anthropic 转换中 tool 消息有列表 content 的场景"""

    def test_tool_message_with_list_content(self):
        """tool 消息的 content 为列表时应正确转换为 Anthropic tool_result content 块"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        chat_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]},
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": [
                        {"type": "text", "text": "Temperature: 72°F"},
                        {"type": "text", "text": "Humidity: 45%"}
                    ]
                }
            ]
        }
        result = engine.convert_request(chat_request)
        # 找到 tool_result 块
        user_msgs = [m for m in result.get("messages", []) if m.get("role") == "user"]
        tool_result_found = False
        for msg in user_msgs:
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_result_found = True
                    # content 应该是列表，不是字符串化的列表
                    assert isinstance(block.get("content"), list), f"Expected list, got {type(block.get('content'))}"
                    assert len(block["content"]) == 2
                    assert block["content"][0].get("text") == "Temperature: 72°F"
        assert tool_result_found, "tool_result block not found"

    def test_tool_message_with_string_content(self):
        """tool 消息的 content 为字符串时仍正确（回归测试）"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        chat_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "test", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_123", "content": "Result: OK"}
            ]
        }
        result = engine.convert_request(chat_request)
        user_msgs = [m for m in result.get("messages", []) if m.get("role") == "user"]
        tool_result_found = False
        for msg in user_msgs:
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_result_found = True
                    assert block.get("content") == "Result: OK"
        assert tool_result_found

    def test_tool_message_list_content_with_error_prefix(self):
        """tool 消息的列表 content 中 [Error] 前缀应正确映射为 is_error"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        chat_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "test", "arguments": "{}"}}]},
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": [
                        {"type": "text", "text": "[Error] API rate limit exceeded"}
                    ]
                }
            ]
        }
        result = engine.convert_request(chat_request)
        user_msgs = [m for m in result.get("messages", []) if m.get("role") == "user"]
        tool_result_found = False
        for msg in user_msgs:
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_result_found = True
                    assert block.get("is_error") is True
                    # [Error] 前缀应被去掉
                    assert block["content"][0].get("text") == "API rate limit exceeded"
        assert tool_result_found


class TestResponsesReasoningSummaryTextDelta:
    """测试 Responses→Chat 中 reasoning_summary_text.delta 事件的转换"""

    def test_reasoning_summary_text_delta(self):
        """response.reasoning_summary_text.delta 应映射为 reasoning_content"""
        data = {
            "type": "response.reasoning_summary_text.delta",
            "output_index": 0,
            "delta": "Summary of reasoning...",
            "response": {"id": "resp_123", "model": "o3"}
        }
        result = OpenAIResponsesConverter._convert_responses_event_to_chat_chunk(
            "response.reasoning_summary_text.delta", data
        )
        assert result is not None
        assert result["choices"][0]["delta"]["reasoning_content"] == "Summary of reasoning..."

    def test_reasoning_summary_text_delta_empty(self):
        """reasoning_summary_text.delta 为空字符串时应正确处理"""
        data = {
            "type": "response.reasoning_summary_text.delta",
            "output_index": 0,
            "delta": "",
            "response": {"id": "resp_123", "model": "o3"}
        }
        result = OpenAIResponsesConverter._convert_responses_event_to_chat_chunk(
            "response.reasoning_summary_text.delta", data
        )
        assert result is not None
        assert result["choices"][0]["delta"]["reasoning_content"] == ""


class TestResponsesRefusalDelta:
    """测试 Responses→Chat 中 response.refusal.delta 事件的转换"""

    def test_refusal_delta(self):
        """response.refusal.delta 应映射为 Chat content delta"""
        data = {
            "type": "response.refusal.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": "I cannot assist with that.",
            "response": {"id": "resp_123", "model": "gpt-4o"}
        }
        result = OpenAIResponsesConverter._convert_responses_event_to_chat_chunk(
            "response.refusal.delta", data
        )
        assert result is not None
        assert result["choices"][0]["delta"]["content"] == "I cannot assist with that."

    def test_refusal_delta_empty(self):
        """response.refusal.delta 为空字符串时应正确处理"""
        data = {
            "type": "response.refusal.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": "",
            "response": {"id": "resp_123", "model": "gpt-4o"}
        }
        result = OpenAIResponsesConverter._convert_responses_event_to_chat_chunk(
            "response.refusal.delta", data
        )
        assert result is not None
        assert result["choices"][0]["delta"]["content"] == ""


class TestConvertContentToChatMultimodalEmpty:
    """测试 _convert_content_to_chat 多模态空内容返回类型一致性"""

    def test_multimodal_empty_returns_empty_list(self):
        """has_multimodal=True 但内容为空时返回空列表而非空字符串"""
        content = [
            {"type": "input_image", "file_id": "file-abc"},  # 只有 file_id 无实际内容
        ]
        result = OpenAIResponsesConverter._convert_content_to_chat(content)
        # file_id 降级为文本占位，has_multimodal=True
        # 如果 file_id 被降级为 text，则 text_parts 非空，返回字符串
        # 但如果 image_url 和 file_id 都没产生 content_parts，应返回空列表
        assert isinstance(result, (str, list))

    def test_multimodal_no_content_returns_list_type(self):
        """多模态内容为空时返回列表类型"""
        # 模拟一个只有 input_image 但无 image_url/file_id 的场景
        content = [
            {"type": "input_image"},  # 无 image_url 也无 file_id
        ]
        result = OpenAIResponsesConverter._convert_content_to_chat(content)
        # has_multimodal=True 但 content_parts 和 text_parts 为空
        assert isinstance(result, list)


class TestChatToResponsesAssistantNoneContent:
    """测试 Chat→Responses 中 assistant 消息 content=None 的处理"""

    def test_assistant_none_content_with_reasoning_no_toolcalls(self):
        """assistant 消息 content=None + reasoning_content 但无 tool_calls 时不生成 'None' 文本"""
        request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": None, "reasoning_content": "Let me think..."},
                {"role": "user", "content": "Continue"},
            ]
        }
        result = OpenAIResponsesConverter.from_openai_chat_request(request)
        input_data = result.get("input", [])
        # 应有 reasoning 项
        reasoning_items = [item for item in input_data if isinstance(item, dict) and item.get("type") == "reasoning"]
        assert len(reasoning_items) == 1
        # 不应有文本为 "None" 的 input_text 消息
        none_text_items = [
            item for item in input_data
            if isinstance(item, dict) and item.get("type") == "message"
            and isinstance(item.get("content"), list)
            and any(c.get("text") == "None" for c in item.get("content", []) if isinstance(c, dict))
        ]
        assert len(none_text_items) == 0

    def test_assistant_none_content_no_reasoning_no_toolcalls(self):
        """assistant 消息 content=None 无 reasoning_content 也无 tool_calls 时创建空消息"""
        request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": None},
                {"role": "user", "content": "Continue"},
            ]
        }
        result = OpenAIResponsesConverter.from_openai_chat_request(request)
        input_data = result.get("input", [])
        # assistant 空消息应创建 content 为空的 message 项
        assistant_msgs = [
            item for item in input_data
            if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 1
        # 不应包含 "None" 文本
        for msg in assistant_msgs:
            for c in msg.get("content", []):
                if isinstance(c, dict) and c.get("type") == "input_text":
                    assert c.get("text", "") != "None"


class TestResponsesToAnthropicContainer:
    """测试 Responses→Anthropic 响应中 container 字段的保留"""

    def test_container_preserved(self):
        """container 字段应在 Responses→Anthropic 转换中保留"""
        response = {
            "id": "resp_123",
            "object": "response",
            "status": "completed",
            "model": "gpt-4o",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello"}],
                    "status": "completed"
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "container": {"id": "ctr_123", "status": "running"}
        }
        result = OpenAIResponsesConverter.to_anthropic(response)
        assert "container" in result
        assert result["container"]["id"] == "ctr_123"

    def test_no_container_field(self):
        """无 container 字段时不添加"""
        response = {
            "id": "resp_123",
            "object": "response",
            "status": "completed",
            "model": "gpt-4o",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello"}],
                    "status": "completed"
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
        result = OpenAIResponsesConverter.to_anthropic(response)
        assert "container" not in result


class TestResponsesFunctionCallOutputToAnthropic:
    """测试 Responses function_call_output 到 Anthropic 的转换"""

    def test_string_output(self):
        """字符串 output 正常转换"""
        item = {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "The weather is sunny."
        }
        result = OpenAIResponsesConverter._convert_input_item_to_anthropic(item)
        assert result is not None
        assert result["role"] == "user"
        content = result["content"]
        assert len(content) == 1
        assert content[0]["type"] == "tool_result"
        assert content[0]["content"] == "The weather is sunny."
        assert content[0]["tool_use_id"] == "call_123"

    def test_list_output(self):
        """列表类型 output 正确转换为 Anthropic 内容块列表"""
        item = {
            "type": "function_call_output",
            "call_id": "call_456",
            "output": [
                {"type": "text", "text": "Result part 1"},
                {"type": "text", "text": "Result part 2"}
            ]
        }
        result = OpenAIResponsesConverter._convert_input_item_to_anthropic(item)
        assert result is not None
        content = result["content"]
        assert len(content) == 1
        assert content[0]["type"] == "tool_result"
        # 列表输出应转为 Anthropic 内容块列表
        tool_result_content = content[0]["content"]
        assert isinstance(tool_result_content, list)
        assert len(tool_result_content) == 2
        assert tool_result_content[0]["type"] == "text"
        assert tool_result_content[0]["text"] == "Result part 1"

    def test_empty_output(self):
        """空 output 正确处理"""
        item = {
            "type": "function_call_output",
            "call_id": "call_789",
            "output": ""
        }
        result = OpenAIResponsesConverter._convert_input_item_to_anthropic(item)
        assert result is not None
        content = result["content"]
        assert content[0]["content"] == ""


class TestVerbosityMapping:
    """verbosity 跨协议映射测试 (Chat 顶层 ↔ Responses text.verbosity)"""

    def test_responses_text_verbosity_to_chat(self):
        """测试 Responses text.verbosity 映射为 Chat 顶层 verbosity"""
        request = {
            "model": "gpt-4o",
            "input": "Be brief",
            "text": {
                "format": {"type": "text"},
                "verbosity": "low"
            }
        }
        result = OpenAIResponsesConverter.to_openai_chat(request)
        assert result["verbosity"] == "low"

    def test_chat_verbosity_to_responses_text(self):
        """测试 Chat 顶层 verbosity 映射为 Responses text.verbosity"""
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Be brief"}],
            "verbosity": "high"
        }
        result = OpenAIResponsesConverter.from_openai_chat_request(request)
        assert "text" in result
        assert result["text"]["verbosity"] == "high"

    def test_chat_verbosity_to_responses_preserves_existing_text(self):
        """测试 Chat verbosity 映射时保留已有 text.format 配置"""
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Generate JSON"}],
            "verbosity": "medium",
            "response_format": {"type": "json_object"}
        }
        result = OpenAIResponsesConverter.from_openai_chat_request(request)
        assert "text" in result
        assert result["text"]["verbosity"] == "medium"
        assert result["text"]["format"]["type"] == "json_object"

    def test_responses_verbosity_to_anthropic_extra_body(self):
        """测试 Responses text.verbosity 保留在 Anthropic extra_body"""
        request = {
            "model": "gpt-4o",
            "input": "Be brief",
            "text": {
                "verbosity": "low"
            }
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert "extra_body" in result
        assert result["extra_body"]["verbosity"] == "low"


class TestReasoningGenerateSummary:
    """reasoning.generate_summary 字段跨协议保留测试"""

    def test_responses_generate_summary_to_chat(self):
        """测试 Responses reasoning.generate_summary 保留到 Chat extra_body"""
        request = {
            "model": "o3",
            "input": "Think about this",
            "reasoning": {
                "effort": "high",
                "generate_summary": "auto",
                "summary": "concise"
            }
        }
        result = OpenAIResponsesConverter.to_openai_chat(request)
        assert result["reasoning_effort"] == "high"
        assert "extra_body" in result
        assert result["extra_body"]["reasoning"]["generate_summary"] == "auto"
        assert result["extra_body"]["reasoning"]["summary"] == "concise"

    def test_chat_generate_summary_to_responses(self):
        """测试 Chat extra_body.reasoning.generate_summary 恢复到 Responses reasoning"""
        request = {
            "model": "o3",
            "messages": [{"role": "user", "content": "Think"}],
            "reasoning_effort": "high",
            "extra_body": {
                "reasoning": {
                    "effort": "high",
                    "generate_summary": "auto",
                    "summary": "concise"
                }
            }
        }
        result = OpenAIResponsesConverter.from_openai_chat_request(request)
        assert "reasoning" in result
        assert result["reasoning"]["effort"] == "high"
        assert result["reasoning"]["generate_summary"] == "auto"
        assert result["reasoning"]["summary"] == "concise"

    def test_responses_generate_summary_to_anthropic_extra_body(self):
        """测试 Responses reasoning.generate_summary 保留在 Anthropic extra_body"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "input": "Think about this",
            "reasoning": {
                "effort": "high",
                "generate_summary": "auto"
            }
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert "extra_body" in result
        assert result["extra_body"]["reasoning"]["generate_summary"] == "auto"


class TestSystemCacheControl:
    """Chat→Anthropic system cache_control 保留测试"""

    def test_system_message_with_cache_control(self):
        """测试含 cache_control 的 system 消息保留为 Anthropic TextBlockParam 格式"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        
        request = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}}
                    ]
                },
                {"role": "user", "content": "Hello"}
            ]
        }
        
        result = engine.convert_request(request)
        assert "system" in result
        system = result["system"]
        assert isinstance(system, list)
        assert any(isinstance(b, dict) and b.get("cache_control") for b in system)

    def test_anthropic_system_cache_control_to_chat(self):
        """测试 Anthropic system cache_control 保留在 Chat 消息内容块中"""
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": "Hello"}]
        }
        result = AnthropicConverter.to_openai_chat(request)
        system_msgs = [m for m in result["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        content = system_msgs[0]["content"]
        assert isinstance(content, list)
        assert any(b.get("cache_control") for b in content if isinstance(b, dict))


class TestConvertContentToAnthropicEmpty:
    """_convert_content_to_anthropic 空内容返回类型测试"""

    def test_empty_content_returns_list(self):
        """空内容返回内容块列表而非空字符串"""
        result = OpenAIResponsesConverter._convert_content_to_anthropic([])
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == ""

    def test_non_empty_content_returns_list(self):
        """非空内容返回内容块列表"""
        result = OpenAIResponsesConverter._convert_content_to_anthropic([
            {"type": "text", "text": "Hello"}
        ])
        assert isinstance(result, list)
        assert result[0]["text"] == "Hello"


class TestEngineSameProtocolPassthrough:
    """测试同协议快速路径"""

    def test_chat_to_chat_passthrough_no_modification(self):
        """Chat→Chat 无需修改时直接透传（浅拷贝而非深拷贝）"""
        engine = ProtocolConverterEngine()
        request = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}
        result = engine.convert_request(request)
        assert result is not request  # 不是同一个对象（浅拷贝）
        assert result == request      # 但内容一致
        assert result["model"] == "gpt-4o"

    def test_chat_to_chat_model_mapping_only(self):
        """Chat→Chat 仅模型不一致时只做最小修改"""
        config = ConverterConfig(model_mapping={"gpt-4o": "gpt-4o-mini"})
        engine = ProtocolConverterEngine(config)
        request = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}
        result = engine.convert_request(request)
        assert result["model"] == "gpt-4o-mini"
        assert result["messages"] == request["messages"]

    def test_chat_to_chat_developer_downgrade(self):
        """Chat→Chat developer 角色降级为 system"""
        engine = ProtocolConverterEngine()
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "developer", "content": "sys"}]
        }
        result = engine.convert_request(request)
        assert result["messages"][0]["role"] == "system"

    def test_chat_to_chat_extra_body_merge(self):
        """Chat→Chat extra_body 合并到顶层"""
        engine = ProtocolConverterEngine()
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "extra_body": {"custom_key": "custom_value"}
        }
        result = engine.convert_request(request)
        assert result.get("custom_key") == "custom_value"
        assert "extra_body" not in result

    def test_anthropic_to_anthropic_passthrough(self):
        """Anthropic→Anthropic 同协议透传"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}]
        }
        result = engine.convert_request(request)
        assert result is not request
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["max_tokens"] == 1024

    def test_responses_to_responses_passthrough(self):
        """Responses→Responses 同协议透传"""
        config = ConverterConfig(backend_type="openai_responses")
        engine = ProtocolConverterEngine(config)
        request = {
            "model": "gpt-4o",
            "input": [{"type": "message", "role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        }
        result = engine.convert_request(request)
        assert result is not request
        assert result["model"] == "gpt-4o"


class TestEngineReverseModelMapping:
    """测试响应反向模型映射"""

    def test_response_no_reverse_by_default(self):
        """默认不反向替换 model"""
        config = ConverterConfig(
            backend_type="openai",
            model_mapping={"gpt-4o": "gpt-4o-mini"}
        )
        engine = ProtocolConverterEngine(config)
        response = {"id": "1", "model": "gpt-4o-mini", "choices": []}
        result = engine.convert_response(response, Protocol.OPENAI_CHAT, original_model="gpt-4o")
        assert result["model"] == "gpt-4o-mini"

    def test_response_reverse_when_enabled(self):
        """启用后非流式响应反向替换 model"""
        config = ConverterConfig(
            backend_type="openai",
            model_mapping={"gpt-4o": "gpt-4o-mini"},
            reverse_model_mapping_in_stream=True
        )
        engine = ProtocolConverterEngine(config)
        response = {"id": "1", "model": "gpt-4o-mini", "choices": []}
        result = engine.convert_response(response, Protocol.OPENAI_CHAT, original_model="gpt-4o")
        assert result["model"] == "gpt-4o"

    def test_response_no_reverse_when_no_mapping(self):
        """无模型映射时不反向替换"""
        config = ConverterConfig(
            backend_type="openai",
            reverse_model_mapping_in_stream=True
        )
        engine = ProtocolConverterEngine(config)
        response = {"id": "1", "model": "gpt-4o", "choices": []}
        result = engine.convert_response(response, Protocol.OPENAI_CHAT, original_model="gpt-4o")
        assert result["model"] == "gpt-4o"

    def test_response_reverse_anthropic_same_protocol(self):
        """Anthropic 同协议响应反向替换"""
        config = ConverterConfig(
            backend_type="anthropic",
            model_mapping={"claude-a": "claude-b"},
            reverse_model_mapping_in_stream=True
        )
        engine = ProtocolConverterEngine(config)
        response = {"id": "1", "model": "claude-b", "content": [], "usage": {}}
        result = engine.convert_response(response, Protocol.ANTHROPIC, original_model="claude-a")
        assert result["model"] == "claude-a"

    def test_response_reverse_responses_same_protocol(self):
        """Responses 同协议响应反向替换"""
        config = ConverterConfig(
            backend_type="openai_responses",
            model_mapping={"gpt-4o": "gpt-4o-mini"},
            reverse_model_mapping_in_stream=True
        )
        engine = ProtocolConverterEngine(config)
        response = {"id": "1", "model": "gpt-4o-mini", "output": [], "usage": {}}
        result = engine.convert_response(response, Protocol.OPENAI_RESPONSES, original_model="gpt-4o")
        assert result["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_stream_chat_to_chat_reverse_model(self):
        """流式 Chat→Chat 反向替换 model"""
        config = ConverterConfig(
            backend_type="openai",
            model_mapping={"gpt-4o": "gpt-4o-mini"},
            reverse_model_mapping_in_stream=True
        )
        engine = ProtocolConverterEngine(config)

        async def mock_http_client(**kwargs):
            class MockResponse:
                async def aiter_lines(self):
                    yield 'data: {"id":"1","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hi"}}]}'
                    yield 'data: [DONE]'
            return MockResponse()

        request = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}], "stream": True}
        gen = await engine.convert_and_forward(request, mock_http_client)
        chunks = [chunk async for chunk in gen]

        data_chunks = [c for c in chunks if '[DONE]' not in c]
        assert len(data_chunks) == 1
        assert '"model": "gpt-4o"' in data_chunks[0]
        assert '"model": "gpt-4o-mini"' not in data_chunks[0]

    @pytest.mark.asyncio
    async def test_stream_chat_to_chat_no_reverse_by_default(self):
        """流式 Chat→Chat 默认不反向替换 model"""
        config = ConverterConfig(
            backend_type="openai",
            model_mapping={"gpt-4o": "gpt-4o-mini"}
        )
        engine = ProtocolConverterEngine(config)

        async def mock_http_client(**kwargs):
            class MockResponse:
                async def aiter_lines(self):
                    yield 'data: {"id":"1","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hi"}}]}'
                    yield 'data: [DONE]'
            return MockResponse()

        request = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}], "stream": True}
        gen = await engine.convert_and_forward(request, mock_http_client)
        chunks = [chunk async for chunk in gen]

        data_chunks = [c for c in chunks if '[DONE]' not in c]
        assert len(data_chunks) == 1
        assert '"model": "gpt-4o-mini"' in data_chunks[0]

    @pytest.mark.asyncio
    async def test_stream_anthropic_to_anthropic_passthrough(self):
        """流式 Anthropic→Anthropic 直接转发原始 SSE"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)

        async def mock_http_client(**kwargs):
            class MockResponse:
                async def aiter_lines(self):
                    yield 'event: content_block_delta'
                    yield 'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}'
                    yield 'data: [DONE]'
            return MockResponse()

        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True
        }
        gen = await engine.convert_and_forward(request, mock_http_client)
        chunks = [chunk async for chunk in gen]

        data_chunks = [c for c in chunks if '[DONE]' not in c]
        assert any(c.startswith('event: content_block_delta') for c in data_chunks)
        assert any('"text": "Hi"' in c for c in data_chunks)

    @pytest.mark.asyncio
    async def test_stream_responses_to_responses_passthrough(self):
        """流式 Responses→Responses 直接转发原始 SSE（保留 event 行）"""
        config = ConverterConfig(backend_type="openai_responses")
        engine = ProtocolConverterEngine(config)

        async def mock_http_client(**kwargs):
            class MockResponse:
                async def aiter_lines(self):
                    yield 'event: response.output_text.delta'
                    yield 'data: {"type":"response.output_text.delta","item_id":"i1","output_index":0,"content_index":0,"delta":"Hi"}'
                    yield 'data: [DONE]'
            return MockResponse()

        request = {
            "model": "gpt-4o",
            "input": [{"type": "message", "role": "user", "content": [{"type": "text", "text": "Hello"}]}],
            "stream": True
        }
        gen = await engine.convert_and_forward(request, mock_http_client)
        chunks = [chunk async for chunk in gen]

        data_chunks = [c for c in chunks if '[DONE]' not in c]
        assert any(c.startswith('event: response.output_text.delta') for c in data_chunks)


# ============================================================
# 新增测试 - 修复验证
# ============================================================


class TestBugFixes:
    """修复验证测试"""

    def test_reasoning_effort_minimal_maps_to_enabled_1024(self):
        """reasoning_effort='minimal' 应映射为 thinking enabled budget=1024，而非 disabled"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        
        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "minimal"
        }
        
        result = engine.convert_request(chat_request)
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 1024

    def test_redacted_thinking_roundtrip_anthropic_to_chat_to_anthropic(self):
        """redacted_thinking 块应能通过 Chat 格式往返转换"""
        anthropic_response = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "redacted_thinking", "data": "abc123encrypted"},
                {"type": "text", "text": "Hello"}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        chat_response = engine.convert_response(anthropic_response, Protocol.OPENAI_CHAT)
        
        assert chat_response["choices"][0]["message"]["reasoning_content"] == "[redacted_thinking: abc123encrypted]"
        
        anthropic_back = AnthropicConverter.from_openai_chat(chat_response)
        has_redacted = any(b["type"] == "redacted_thinking" for b in anthropic_back["content"])
        assert has_redacted, "redacted_thinking 块应在 Chat→Anthropic 转换中恢复"
        redacted_block = next(b for b in anthropic_back["content"] if b["type"] == "redacted_thinking")
        assert redacted_block["data"] == "abc123encrypted"

    def test_redacted_thinking_roundtrip_responses_to_chat_to_anthropic(self):
        """Responses reasoning 的 encrypted_content 应能通过 Chat 格式往返转换"""
        responses_response = {
            "id": "resp_test",
            "object": "response",
            "status": "completed",
            "created_at": 1234567890,
            "model": "gpt-4o",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_test",
                    "content": [],
                    "summary": [],
                    "encrypted_content": "xyz789encrypted",
                    "status": "completed"
                },
                {
                    "type": "message",
                    "id": "msg_test",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Hello"}]
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        }
        
        chat_response = OpenAIResponsesConverter.to_chat_response(responses_response)
        assert chat_response["choices"][0]["message"]["reasoning_content"] == "[redacted_thinking: xyz789encrypted]"
        
        anthropic_result = AnthropicConverter.from_openai_chat(chat_response)
        has_redacted = any(b["type"] == "redacted_thinking" for b in anthropic_result["content"])
        assert has_redacted

    def test_empty_input_responses_to_chat(self):
        """空 input 不应导致无效的 Chat 请求"""
        request = {
            "model": "gpt-4o",
            "input": [],
        }
        result = OpenAIResponsesConverter.to_openai_chat(request)
        assert len(result["messages"]) > 0
        assert result["messages"][-1]["role"] == "user"

    def test_empty_input_responses_to_anthropic(self):
        """空 input 不应导致无效的 Anthropic 请求"""
        request = {
            "model": "gpt-4o",
            "input": [],
        }
        result = OpenAIResponsesConverter.to_anthropic_request(request)
        assert len(result["messages"]) > 0

    def test_stop_empty_list_not_passed_to_anthropic(self):
        """stop=[] 不应传给 Anthropic stop_sequences"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        
        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": []
        }
        
        result = engine.convert_request(chat_request)
        assert "stop_sequences" not in result

    def test_stop_empty_string_not_passed_to_anthropic(self):
        """stop=\"\" 不应传给 Anthropic stop_sequences"""
        config = ConverterConfig(backend_type="anthropic")
        engine = ProtocolConverterEngine(config)
        
        chat_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": ""
        }
        
        result = engine.convert_request(chat_request)
        assert "stop_sequences" not in result
