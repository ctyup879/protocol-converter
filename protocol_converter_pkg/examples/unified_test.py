"""
统一集成测试 - 展示两种后端模式
"""

import asyncio
import json
from protocol_converter import (
    ProtocolConverterEngine,
    ConverterConfig,
    Protocol,
    AnthropicConverter,
)


# ============================================================
# 后端配置示例
# ============================================================

# 模式1: OpenAI 兼容后端 (需要协议转换)
OPENAI_BACKEND = ConverterConfig(
    backend_type="openai",
    backend_url="https://api.minimaxi.com/v1/chat/completions",
    api_key="sk-cp-xxx",
    default_model="MiniMax-M2.7",
    model_mapping={
        "claude-sonnet-4-20250514": "MiniMax-M2.7",
        "claude-opus-4-6": "MiniMax-M2.7",
        "gpt-4o": "MiniMax-M2.7",
    }
)

# 模式2: Anthropic 兼容后端 (直接发送 Anthropic 格式)
ANTHROPIC_BACKEND = ConverterConfig(
    backend_type="anthropic",
    backend_url="https://api.minimaxi.com/anthropic/v1/messages",
    api_key="REDACTED_MINIMAX_API_KEY",
    default_model="MiniMax-M2.7",
)


async def send_request(config: ConverterConfig, request: dict, description: str):
    """发送请求到指定后端"""
    import httpx
    
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")
    print(f"后端类型: {config.backend_type}")
    print(f"URL: {config.backend_url}")
    
    headers = {
        "Content-Type": "application/json",
        **config.get_auth_headers()
    }
    
    async with httpx.AsyncClient(timeout=config.timeout) as client:
        response = await client.post(
            config.backend_url,
            headers=headers,
            json=request
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ 成功!")
            print(f"响应类型: {result.get('type', 'N/A')}")
            
            if config.backend_type == "anthropic":
                # Anthropic 响应
                content = result.get("content", [])
                for block in content:
                    if block.get("type") == "text":
                        print(f"文本: {block.get('text', '')}")
            else:
                # OpenAI 响应
                choices = result.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    print(f"文本: {msg.get('content', '')}")
            return True
        else:
            print(f"\n❌ 失败: {response.status_code}")
            print(f"错误: {response.text[:200]}")
            return False


async def demo_unified():
    """统一演示"""
    print("=" * 60)
    print("协议转换器 - 统一集成测试")
    print("=" * 60)
    
    # 模拟收到的不同协议请求
    requests = {
        "openai_chat": {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Say hello in 5 words."}
            ]
        },
        "anthropic": {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 50,
            "messages": [
                {"role": "user", "content": "Say hello in 5 words."}
            ]
        }
    }
    
    engine = ProtocolConverterEngine(OPENAI_BACKEND)
    
    # 场景1: OpenAI 后端 - 接收 Anthropic 请求
    print("\n" + "🔄 " + "场景1: OpenAI 后端接收 Anthropic 请求".center(54))
    protocol = engine.detect_protocol(requests["anthropic"])
    print(f"检测协议: {protocol.value}")
    
    converted = engine.convert_request(requests["anthropic"])
    print(f"转换请求: {json.dumps(converted, indent=2, ensure_ascii=False)[:200]}...")
    
    # 注意: 这里使用真实 API key 需要替换 OPENAI_BACKEND.api_key
    # await send_request(OPENAI_BACKEND, converted, "OpenAI 后端处理 Anthropic 请求")
    
    # 场景2: Anthropic 后端 - 直接处理 Anthropic 请求
    print("\n" + "🔄 " + "场景2: Anthropic 后端直接处理".center(54))
    await send_request(ANTHROPIC_BACKEND, requests["anthropic"], "Anthropic 后端处理 Anthropic 请求")
    
    # 场景3: Anthropic 后端 - 接收 OpenAI 请求并转换
    print("\n" + "🔄 " + "场景3: Anthropic 后端接收 OpenAI 请求".center(54))
    openai_to_anthropic = AnthropicConverter.to_openai_chat(requests["openai_chat"])
    openai_to_anthropic["model"] = ANTHROPIC_BACKEND.default_model
    await send_request(ANTHROPIC_BACKEND, openai_to_anthropic, "Anthropic 后端处理转换后的 OpenAI 请求")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo_unified())
