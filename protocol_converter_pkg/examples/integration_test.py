"""
集成测试 - 使用 MiniMax API 进行端到端测试
"""

import asyncio
import json
from protocol_converter import (
    ProtocolConverterEngine,
    ConverterConfig,
    Protocol,
    OpenAIChatConverter,
    OpenAIResponsesConverter,
    AnthropicConverter,
)


# MiniMax API 配置
CONFIG = ConverterConfig(
    backend_url="https://api.minimaxi.com/v1/chat/completions",
    api_key="REDACTED_MINIMAX_API_KEY",
    default_model="MiniMax-M2.7",
    timeout=60.0,
    # Anthropic 模型映射到 MiniMax 模型
    model_mapping={
        "claude-sonnet-4-20250514": "MiniMax-M2.7",
        "claude-opus-4-6": "MiniMax-M2.7",
        "claude-3-5-sonnet-latest": "MiniMax-M2.7",
        "claude-3-opus-latest": "MiniMax-M2.7",
        # OpenAI 模型也可以映射
        "gpt-4o": "MiniMax-M2.7",
        "gpt-4o-mini": "MiniMax-M2.7",
    }
)

# 创建引擎
engine = ProtocolConverterEngine(CONFIG)


async def test_openai_chat_request():
    """测试 OpenAI Chat 请求"""
    print("\n" + "=" * 60)
    print("测试 1: OpenAI Chat 请求")
    print("=" * 60)
    
    request = {
        "model": "MiniMax-M2.7",
        "messages": [
            {"role": "user", "content": "Hello, who are you?"}
        ]
    }
    
    # 检测协议
    protocol = engine.detect_protocol(request)
    print(f"检测到的协议: {protocol}")
    
    # 转换请求
    converted = engine.convert_request(request)
    print(f"转换后的请求:")
    print(json.dumps(converted, indent=2, ensure_ascii=False))
    
    return converted


async def test_anthropic_request():
    """测试 Anthropic 请求"""
    print("\n" + "=" * 60)
    print("测试 2: Anthropic 请求")
    print("=" * 60)
    
    request = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": "Hello, who are you?"}
        ]
    }
    
    # 检测协议
    protocol = engine.detect_protocol(request)
    print(f"检测到的协议: {protocol}")
    
    # 转换请求
    converted = engine.convert_request(request)
    print(f"转换后的请求:")
    print(json.dumps(converted, indent=2, ensure_ascii=False))
    
    return converted


async def test_openai_responses_request():
    """测试 OpenAI Responses 请求"""
    print("\n" + "=" * 60)
    print("测试 3: OpenAI Responses 请求")
    print("=" * 60)
    
    request = {
        "model": "gpt-4o",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello, who are you?"}
                ]
            }
        ],
        "instructions": "You are a helpful assistant."
    }
    
    # 检测协议
    protocol = engine.detect_protocol(request)
    print(f"检测到的协议: {protocol}")
    
    # 转换请求
    converted = engine.convert_request(request)
    print(f"转换后的请求:")
    print(json.dumps(converted, indent=2, ensure_ascii=False))
    
    return converted


async def test_with_minimax():
    """使用 MiniMax API 进行实际请求测试"""
    print("\n" + "=" * 60)
    print("测试 4: 使用 MiniMax API 进行实际请求")
    print("=" * 60)
    
    try:
        import httpx
        
        # 测试 Anthropic 格式请求 -> 转换为 OpenAI Chat -> 发送到 MiniMax
        anthropic_request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Say hello in one sentence."}
            ]
        }
        
        print("\n原始 Anthropic 请求:")
        print(json.dumps(anthropic_request, indent=2, ensure_ascii=False))
        
        # 转换为 OpenAI Chat
        converted = engine.convert_request(anthropic_request)
        print("\n转换后的 OpenAI Chat 请求:")
        print(json.dumps(converted, indent=2, ensure_ascii=False))
        
        # 发送到 MiniMax
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CONFIG.api_key}"
        }
        
        async with httpx.AsyncClient(timeout=CONFIG.timeout) as client:
            response = await client.post(
                CONFIG.backend_url,
                headers=headers,
                json=converted
            )
            
            print(f"\n状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("\n响应内容:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                
                # 转换回 Anthropic 格式
                anthropic_response = AnthropicConverter.from_openai_chat(result)
                print("\n转换后的 Anthropic 响应:")
                print(json.dumps(anthropic_response, indent=2, ensure_ascii=False))
                return True
            else:
                print(f"\n错误: {response.text}")
                return False
                
    except Exception as e:
        print(f"\n请求失败: {e}")
        return False


async def test_stream_with_minimax():
    """测试流式请求"""
    print("\n" + "=" * 60)
    print("测试 5: 流式请求测试")
    print("=" * 60)
    
    try:
        import httpx
        
        anthropic_request = {
            "model": "MiniMax-M2.7",
            "max_tokens": 200,
            "stream": True,
            "messages": [
                {"role": "user", "content": "Count from 1 to 5."}
            ]
        }
        
        # 转换为 OpenAI Chat
        converted = engine.convert_request(anthropic_request)
        converted["stream"] = True
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CONFIG.api_key}"
        }
        
        async with httpx.AsyncClient(timeout=CONFIG.timeout) as client:
            async with client.stream(
                "POST",
                CONFIG.backend_url,
                headers=headers,
                json=converted
            ) as response:
                print(f"\n状态码: {response.status_code}")
                
                if response.status_code == 200:
                    full_content = ""
                    print("\n流式响应:")
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if delta:
                                    print(delta, end="", flush=True)
                                    full_content += delta
                            except:
                                pass
                    print(f"\n\n完整响应: {full_content}")
                    return True
                else:
                    print(f"\n错误: {await response.aread()}")
                    return False
                
    except Exception as e:
        print(f"\n流式请求失败: {e}")
        return False


async def run_all_tests():
    """运行所有集成测试"""
    print("=" * 60)
    print("协议转换器集成测试 - MiniMax API")
    print("=" * 60)
    
    await test_openai_chat_request()
    await test_anthropic_request()
    await test_openai_responses_request()
    
    success = await test_with_minimax()
    if success:
        await test_stream_with_minimax()
    
    print("\n" + "=" * 60)
    print("集成测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
