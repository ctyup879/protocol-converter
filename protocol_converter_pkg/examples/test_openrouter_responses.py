"""
集成测试 - OpenRouter Responses API (OpenAI Responses 格式)
"""

import asyncio
import json
from protocol_converter import (
    ConverterConfig,
    OpenAIResponsesConverter,
    OpenAIChatConverter,
    AnthropicConverter,
)


# OpenRouter Responses API 配置
OPENROUTER_RESPONSES_CONFIG = ConverterConfig(
    backend_type="openrouter",
    backend_url="https://openrouter.ai/api/v1/responses",
    api_key="REDACTED_OPENROUTER_API_KEY",
    default_model="openai/gpt-oss-120b:free",
    timeout=60.0,
)


async def test_openrouter_responses_direct():
    """测试 OpenRouter - 直接发送 OpenAI Responses 格式"""
    print("\n" + "=" * 60)
    print("测试 1: OpenRouter Responses API (直接发送 Responses 格式)")
    print("=" * 60)
    
    request = {
        "model": "openai/gpt-oss-120b:free",
        "input": "Tell me a joke",
    }
    
    print(f"\n请求: {json.dumps(request, indent=2, ensure_ascii=False)}")
    
    try:
        import httpx
        
        headers = {
            "Content-Type": "application/json",
            **OPENROUTER_RESPONSES_CONFIG.get_auth_headers()
        }
        
        async with httpx.AsyncClient(timeout=OPENROUTER_RESPONSES_CONFIG.timeout) as client:
            response = await client.post(
                OPENROUTER_RESPONSES_CONFIG.backend_url,
                headers=headers,
                json=request
            )
            
            print(f"\n状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n✅ 成功!")
                print(f"\n响应: {json.dumps(result, indent=2, ensure_ascii=False)[:600]}...")
                return True
            else:
                print(f"\n❌ 错误: {response.text[:300]}")
                return False
                
    except Exception as e:
        print(f"\n❌ 请求失败: {e}")
        return False


async def test_openrouter_from_openai_chat():
    """测试 OpenRouter - 接收 OpenAI Chat 请求并转换为 Responses 格式"""
    print("\n" + "=" * 60)
    print("测试 2: OpenRouter (OpenAI Chat -> Responses 转换)")
    print("=" * 60)
    
    # 原始 OpenAI Chat 请求
    chat_request = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Tell me a joke"}
        ]
    }
    
    print(f"\n原始 Chat 请求: {json.dumps(chat_request, indent=2, ensure_ascii=False)}")
    
    # 转换为 Responses 格式
    responses_request = OpenAIResponsesConverter.from_openai_chat_request(chat_request)
    responses_request["model"] = "openai/gpt-oss-120b:free"
    
    print(f"\n转换为 Responses 格式: {json.dumps(responses_request, indent=2, ensure_ascii=False)}")
    
    try:
        import httpx
        
        headers = {
            "Content-Type": "application/json",
            **OPENROUTER_RESPONSES_CONFIG.get_auth_headers()
        }
        
        async with httpx.AsyncClient(timeout=OPENROUTER_RESPONSES_CONFIG.timeout) as client:
            response = await client.post(
                OPENROUTER_RESPONSES_CONFIG.backend_url,
                headers=headers,
                json=responses_request
            )
            
            print(f"\n状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n✅ 成功!")
                print(f"\nResponses 响应: {json.dumps(result, indent=2, ensure_ascii=False)[:600]}...")
                return True
            else:
                print(f"\n❌ 错误: {response.text[:300]}")
                return False
                
    except Exception as e:
        print(f"\n❌ 请求失败: {e}")
        return False


async def test_openrouter_from_anthropic():
    """测试 OpenRouter - 接收 Anthropic 请求并转换为 Responses 格式"""
    print("\n" + "=" * 60)
    print("测试 3: OpenRouter (Anthropic -> Responses 转换)")
    print("=" * 60)
    
    # Anthropic 请求
    anthropic_request = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "Tell me a joke"}
        ]
    }
    
    print(f"\n原始 Anthropic 请求: {json.dumps(anthropic_request, indent=2, ensure_ascii=False)}")
    
    # Anthropic -> OpenAI Chat
    chat_request = AnthropicConverter.to_openai_chat(anthropic_request)
    print(f"\n转换为 Chat 格式: {json.dumps(chat_request, indent=2, ensure_ascii=False)[:300]}...")
    
    # Chat -> Responses
    responses_request = OpenAIResponsesConverter.from_openai_chat_request(chat_request)
    responses_request["model"] = "openai/gpt-oss-120b:free"
    
    print(f"\n转换为 Responses 格式: {json.dumps(responses_request, indent=2, ensure_ascii=False)}")
    
    try:
        import httpx
        
        headers = {
            "Content-Type": "application/json",
            **OPENROUTER_RESPONSES_CONFIG.get_auth_headers()
        }
        
        async with httpx.AsyncClient(timeout=OPENROUTER_RESPONSES_CONFIG.timeout) as client:
            response = await client.post(
                OPENROUTER_RESPONSES_CONFIG.backend_url,
                headers=headers,
                json=responses_request
            )
            
            print(f"\n状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n✅ 成功!")
                print(f"\nResponses 响应: {json.dumps(result, indent=2, ensure_ascii=False)[:600]}...")
                return True
            else:
                print(f"\n❌ 错误: {response.text[:300]}")
                return False
                
    except Exception as e:
        print(f"\n❌ 请求失败: {e}")
        return False


async def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("协议转换器 - OpenRouter Responses API 测试")
    print("端点: https://openrouter.ai/api/v1/responses")
    print("=" * 60)
    
    await test_openrouter_responses_direct()
    await test_openrouter_from_openai_chat()
    await test_openrouter_from_anthropic()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
