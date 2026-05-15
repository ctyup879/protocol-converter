"""
集成测试 - 使用 Anthropic 兼容后端进行端到端测试
"""

import asyncio
import json
from protocol_converter import (
    ProtocolConverterEngine,
    ConverterConfig,
    Protocol,
    OpenAIChatConverter,
    AnthropicConverter,
)


# ============================================================
# 方式1: Anthropic 兼容后端 (API 直接使用 Anthropic 格式)
# ============================================================
# Anthropic 兼容后端使用 /v1/messages 端点
# 可以直接发送 Anthropic 格式请求，无需转换

CONFIG_ANTHROPIC = ConverterConfig(
    backend_url="https://api.minimaxi.com/anthropic/v1/messages",
    api_key="REDACTED_MINIMAX_API_KEY",
    default_model="MiniMax-M2.7",
    timeout=60.0,
)


# ============================================================
# 方式2: OpenAI 兼容后端 (需要协议转换)
# ============================================================
CONFIG_OPENAI = ConverterConfig(
    backend_url="https://api.minimaxi.com/v1/chat/completions",
    api_key="REDACTED_MINIMAX_API_KEY",
    default_model="MiniMax-M2.7",
    timeout=60.0,
    model_mapping={
        "claude-sonnet-4-20250514": "MiniMax-M2.7",
        "claude-opus-4-6": "MiniMax-M2.7",
        "gpt-4o": "MiniMax-M2.7",
    }
)


async def test_anthropic_backend_direct():
    """测试 Anthropic 兼容后端 - 直接发送 Anthropic 格式"""
    print("\n" + "=" * 60)
    print("测试 1: Anthropic 兼容后端（直接发送 Anthropic 格式）")
    print("=" * 60)
    
    engine = ProtocolConverterEngine(CONFIG_ANTHROPIC)
    
    # Anthropic 格式请求
    anthropic_request = {
        "model": "MiniMax-M2.7",
        "max_tokens": 100,
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": "Say hello in one sentence."}
        ]
    }
    
    # 检测协议
    protocol = engine.detect_protocol(anthropic_request)
    print(f"检测到的协议: {protocol}")
    
    # Anthropic 后端不需要转换，直接发送
    print("\n直接发送 Anthropic 格式请求到:", CONFIG_ANTHROPIC.backend_url)
    print(json.dumps(anthropic_request, indent=2, ensure_ascii=False))
    
    try:
        import httpx
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": CONFIG_ANTHROPIC.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        async with httpx.AsyncClient(timeout=CONFIG_ANTHROPIC.timeout) as client:
            response = await client.post(
                CONFIG_ANTHROPIC.backend_url,
                headers=headers,
                json=anthropic_request
            )
            
            print(f"\n状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("\n响应内容:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return True
            else:
                print(f"\n错误: {response.text}")
                return False
                
    except Exception as e:
        print(f"\n请求失败: {e}")
        return False


async def test_anthropic_backend_with_conversion():
    """测试 Anthropic 兼容后端 - 模拟客户端发送不同协议请求"""
    print("\n" + "=" * 60)
    print("测试 2: Anthropic 兼容后端（协议转换）")
    print("=" * 60)
    
    # 模拟收到 OpenAI Chat 请求，需要转换为 Anthropic 格式
    openai_request = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one sentence."}
        ]
    }
    
    print("\n原始 OpenAI Chat 请求:")
    print(json.dumps(openai_request, indent=2, ensure_ascii=False))
    
    # 转换为 Anthropic 格式
    anthropic_request = AnthropicConverter.to_openai_chat(openai_request)
    anthropic_request["model"] = "MiniMax-M2.7"
    
    print("\n转换为 Anthropic 格式:")
    print(json.dumps(anthropic_request, indent=2, ensure_ascii=False))
    
    try:
        import httpx
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": CONFIG_ANTHROPIC.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        async with httpx.AsyncClient(timeout=CONFIG_ANTHROPIC.timeout) as client:
            response = await client.post(
                CONFIG_ANTHROPIC.backend_url,
                headers=headers,
                json=anthropic_request
            )
            
            print(f"\n状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("\nAnthropic 响应:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return True
            else:
                print(f"\n错误: {response.text}")
                return False
                
    except Exception as e:
        print(f"\n请求失败: {e}")
        return False


async def test_stream_anthropic_backend():
    """测试 Anthropic 兼容后端 - 流式请求"""
    print("\n" + "=" * 60)
    print("测试 3: Anthropic 兼容后端（流式请求）")
    print("=" * 60)
    
    anthropic_request = {
        "model": "MiniMax-M2.7",
        "max_tokens": 200,
        "stream": True,
        "messages": [
            {"role": "user", "content": "Count from 1 to 3."}
        ]
    }
    
    print("\n流式请求:")
    print(json.dumps(anthropic_request, indent=2, ensure_ascii=False))
    
    try:
        import httpx
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": CONFIG_ANTHROPIC.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        async with httpx.AsyncClient(timeout=CONFIG_ANTHROPIC.timeout) as client:
            async with client.stream(
                "POST",
                CONFIG_ANTHROPIC.backend_url,
                headers=headers,
                json=anthropic_request
            ) as response:
                print(f"\n状态码: {response.status_code}")
                
                if response.status_code == 200:
                    full_content = ""
                    print("\n流式响应 (SSE):")
                    async for line in response.aiter_lines():
                        if line.startswith("event:"):
                            print(f"  {line}")
                        elif line.startswith("data:"):
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                # Anthropic 流式事件
                                event_type = chunk.get("type", "")
                                if event_type == "content_block_delta":
                                    delta = chunk.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        text = delta.get("text", "")
                                        print(text, end="", flush=True)
                                        full_content += text
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
    """运行所有测试"""
    print("=" * 60)
    print("协议转换器集成测试 - Anthropic 兼容后端")
    print("=" * 60)
    
    success1 = await test_anthropic_backend_direct()
    if success1:
        await test_anthropic_backend_with_conversion()
        await test_stream_anthropic_backend()
    
    print("\n" + "=" * 60)
    print("集成测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
