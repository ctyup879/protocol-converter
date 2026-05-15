"""
使用示例 - 协议转换器

本模块展示如何使用协议转换器进行各种协议之间的转换
"""

import json
import asyncio
from protocol_converter import (
    ProtocolConverterEngine,
    ConverterConfig,
    ProtocolDetector,
    Protocol,
    OpenAIChatConverter,
    OpenAIResponsesConverter,
    AnthropicConverter,
)


def example_detect_protocol():
    """示例：检测请求协议"""
    print("=" * 60)
    print("示例：协议检测")
    print("=" * 60)
    
    detector = ProtocolDetector()
    
    # OpenAI Chat 请求
    openai_request = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
    }
    
    # Anthropic 请求
    anthropic_request = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "system": "You are Claude.",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
    
    # OpenAI Responses 请求
    responses_request = {
        "model": "gpt-4o",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello!"}
                ]
            }
        ]
    }
    
    print(f"OpenAI Chat: {detector.detect(openai_request)}")
    print(f"Anthropic: {detector.detect(anthropic_request)}")
    print(f"OpenAI Responses: {detector.detect(responses_request)}")
    print()


def example_convert_to_openai_chat():
    """示例：转换为 OpenAI Chat 格式"""
    print("=" * 60)
    print("示例：转换为 OpenAI Chat 格式")
    print("=" * 60)
    
    # Anthropic 请求
    anthropic_request = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "system": "你是一个有帮助的助手。",
        "messages": [
            {"role": "user", "content": "你好，介绍一下自己"}
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "获取天气信息",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
                    },
                    "required": ["city"]
                }
            }
        ],
        "temperature": 0.7
    }
    
    converted = OpenAIChatConverter.to_openai_chat(anthropic_request)
    print(json.dumps(converted, indent=2, ensure_ascii=False))
    print()


def example_convert_anthropic_to_openai():
    """示例：Anthropic 请求转 OpenAI Chat"""
    print("=" * 60)
    print("示例：Anthropic 转 OpenAI Chat")
    print("=" * 60)
    
    # Anthropic 请求
    anthropic_request = {
        "model": "claude-opus-4-6",
        "max_tokens": 1024,
        "system": "You are Claude.",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ]
    }
    
    # 使用 Anthropic 转换器
    chat_request = AnthropicConverter.to_openai_chat(anthropic_request)
    print("转换后的请求:")
    print(json.dumps(chat_request, indent=2, ensure_ascii=False))
    print()


def example_convert_response():
    """示例：响应转换"""
    print("=" * 60)
    print("示例：响应转换")
    print("=" * 60)
    
    # OpenAI Chat 响应
    chat_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The capital of France is Paris."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30
        }
    }
    
    # 转换为 Anthropic 格式
    anthropic_response = AnthropicConverter.from_openai_chat(chat_response)
    print("Anthropic 格式响应:")
    print(json.dumps(anthropic_response, indent=2, ensure_ascii=False))
    print()
    
    # 转换为 OpenAI Responses 格式
    responses_response = OpenAIResponsesConverter.from_openai_chat(chat_response)
    print("OpenAI Responses 格式响应:")
    print(json.dumps(responses_response, indent=2, ensure_ascii=False))
    print()


def example_engine_usage():
    """示例：使用引擎进行转换"""
    print("=" * 60)
    print("示例：使用转换引擎")
    print("=" * 60)
    
    # 创建引擎配置
    config = ConverterConfig(
        backend_url="https://api.openai.com/v1/chat/completions",
        api_key="your-api-key",
        default_model="gpt-4o",
        timeout=60.0,
    )
    
    # 创建引擎
    engine = ProtocolConverterEngine(config)
    
    # 模拟 Anthropic 请求
    anthropic_request = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
    
    # 检测协议
    protocol = engine.detect_protocol(anthropic_request)
    print(f"检测到的协议: {protocol}")
    
    # 转换请求
    converted = engine.convert_request(anthropic_request)
    print("\n转换后的请求:")
    print(json.dumps(converted, indent=2, ensure_ascii=False))
    print()


def example_with_openai_responses():
    """示例：OpenAI Responses 协议转换"""
    print("=" * 60)
    print("示例：OpenAI Responses 协议转换")
    print("=" * 60)
    
    # OpenAI Responses 请求
    responses_request = {
        "model": "gpt-4o",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello!"}
                ]
            }
        ],
        "instructions": "You are a helpful assistant.",
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        ],
        "max_output_tokens": 1024
    }
    
    # 转换为 OpenAI Chat 格式
    chat_request = OpenAIResponsesConverter.to_openai_chat(responses_request)
    print("转换为 OpenAI Chat 格式:")
    print(json.dumps(chat_request, indent=2, ensure_ascii=False))
    print()


def example_stream_conversion():
    """示例：流式响应转换"""
    print("=" * 60)
    print("示例：流式响应转换")
    print("=" * 60)
    
    # OpenAI Chat 流式响应块
    chunk = {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1677652288,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": "Hello"
                },
                "finish_reason": None
            }
        ]
    }
    
    # 转换为 Anthropic 格式
    anthropic_chunk = AnthropicConverter.convert_stream_chunk(chunk)
    print("Anthropic 流式块:")
    print(json.dumps(anthropic_chunk, indent=2, ensure_ascii=False))
    print()
    
    # 转换为 OpenAI Responses 格式
    responses_chunk = OpenAIResponsesConverter.convert_stream_chunk(chunk)
    print("OpenAI Responses 流式块:")
    print(json.dumps(responses_chunk, indent=2, ensure_ascii=False))
    print()


def example_tool_calls():
    """示例：工具调用转换"""
    print("=" * 60)
    print("示例：工具调用转换")
    print("=" * 60)
    
    # Anthropic 请求（带工具）
    anthropic_request = {
        "model": "claude-opus-4-6",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Beijing?"
            }
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                }
            }
        ]
    }
    
    # 转换请求
    chat_request = AnthropicConverter.to_openai_chat(anthropic_request)
    print("转换后的请求（带工具）:")
    print(json.dumps(chat_request, indent=2, ensure_ascii=False))
    print()


def run_all_examples():
    """运行所有示例"""
    example_detect_protocol()
    example_convert_to_openai_chat()
    example_convert_anthropic_to_openai()
    example_convert_response()
    example_engine_usage()
    example_with_openai_responses()
    example_stream_conversion()
    example_tool_calls()


if __name__ == "__main__":
    run_all_examples()
