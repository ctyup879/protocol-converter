"""
集成测试 - 使用 MiniMax API 进行端到端测试

测试场景:
1. OpenAI Chat / Anthropic / Responses 请求检测与转换
2. 实际 API 调用（非流式 + 流式）
3. Tool calling / function calling 转换
4. Thinking / reasoning_effort 参数映射
5. 响应回转（后端响应 → 客户端协议格式）
6. 多模态内容转换
7. 跨协议流式请求（Responses→Chat 流式、Anthropic→Chat 流式）
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
    backend_type="openai",
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


# ============================================================
# 1. 协议检测与请求转换
# ============================================================

async def test_protocol_detection():
    """测试协议检测"""
    print("\n" + "=" * 60)
    print("测试 1: 协议检测")
    print("=" * 60)
    
    # OpenAI Chat
    chat_req = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}
    assert engine.detect_protocol(chat_req) == Protocol.OPENAI_CHAT
    print(f"  OpenAI Chat: {engine.detect_protocol(chat_req)}")
    
    # Anthropic
    anthropic_req = {"model": "claude-sonnet-4-20250514", "max_tokens": 1024, "messages": [{"role": "user", "content": "Hello"}]}
    assert engine.detect_protocol(anthropic_req) == Protocol.ANTHROPIC
    print(f"  Anthropic:   {engine.detect_protocol(anthropic_req)}")
    
    # OpenAI Responses
    responses_req = {"model": "gpt-4o", "input": "Hello"}
    assert engine.detect_protocol(responses_req) == Protocol.OPENAI_RESPONSES
    print(f"  Responses:   {engine.detect_protocol(responses_req)}")
    
    # 带 thinking 参数的 Anthropic
    thinking_req = {"model": "claude-opus-4-6", "max_tokens": 1024, "thinking": {"type": "enabled", "budget_tokens": 10000}, "messages": [{"role": "user", "content": "Think"}]}
    assert engine.detect_protocol(thinking_req) == Protocol.ANTHROPIC
    print(f"  Thinking:    {engine.detect_protocol(thinking_req)}")


async def test_request_conversion():
    """测试请求转换"""
    print("\n" + "=" * 60)
    print("测试 2: 请求转换")
    print("=" * 60)
    
    # Anthropic → OpenAI Chat
    anthropic_req = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7
    }
    converted = engine.convert_request(anthropic_req)
    print(f"\n  Anthropic → Chat:")
    print(f"    model: {converted['model']}")
    print(f"    messages: {len(converted['messages'])} 条")
    print(f"    max_completion_tokens: {converted.get('max_completion_tokens')}")
    
    # OpenAI Responses → OpenAI Chat
    responses_req = {
        "model": "gpt-4o",
        "input": [{"type": "message", "role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        "instructions": "You are helpful.",
        "max_output_tokens": 1024
    }
    converted = engine.convert_request(responses_req)
    print(f"\n  Responses → Chat:")
    print(f"    model: {converted['model']}")
    print(f"    messages: {len(converted['messages'])} 条")
    print(f"    max_completion_tokens: {converted.get('max_completion_tokens')}")


# ============================================================
# 2. 实际 API 调用
# ============================================================

async def test_api_call():
    """测试实际 API 调用（Anthropic 请求 → OpenAI Chat 后端）"""
    print("\n" + "=" * 60)
    print("测试 3: 实际 API 调用")
    print("=" * 60)
    
    try:
        import httpx
        
        anthropic_request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say hello in one sentence."}]
        }
        
        # 转换为 OpenAI Chat
        converted = engine.convert_request(anthropic_request)
        
        headers = {
            "Content-Type": "application/json",
            **CONFIG.get_auth_headers()
        }
        
        async with httpx.AsyncClient(timeout=CONFIG.timeout) as client:
            response = await client.post(CONFIG.backend_url, headers=headers, json=converted)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                print(f"  后端响应: {content}")
                
                # 转换回 Anthropic 格式
                anthropic_resp = engine.convert_response(result, Protocol.ANTHROPIC)
                print(f"  转回 Anthropic: type={anthropic_resp['type']}, stop_reason={anthropic_resp['stop_reason']}")
                print(f"  内容: {anthropic_resp['content'][0]['text']}")
                return True
            else:
                print(f"  错误: {response.status_code} {response.text[:200]}")
                return False
    except Exception as e:
        print(f"  请求失败: {e}")
        return False


async def test_streaming():
    """测试流式请求"""
    print("\n" + "=" * 60)
    print("测试 4: 流式请求")
    print("=" * 60)
    
    try:
        import httpx
        
        anthropic_request = {
            "model": "MiniMax-M2.7",
            "max_tokens": 200,
            "stream": True,
            "messages": [{"role": "user", "content": "Count from 1 to 5."}]
        }
        
        converted = engine.convert_request(anthropic_request)
        converted["stream"] = True
        converted["stream_options"] = {"include_usage": True}
        
        headers = {
            "Content-Type": "application/json",
            **CONFIG.get_auth_headers()
        }
        
        async with httpx.AsyncClient(timeout=CONFIG.timeout) as client:
            async with client.stream("POST", CONFIG.backend_url, headers=headers, json=converted) as response:
                if response.status_code == 200:
                    full_content = ""
                    print("  流式响应: ", end="", flush=True)
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
                    print(f"\n  完整响应: {full_content}")
                    return True
                else:
                    print(f"  错误: {response.status_code}")
                    return False
    except Exception as e:
        print(f"  流式请求失败: {e}")
        return False


# ============================================================
# 3. Tool calling 转换
# ============================================================

async def test_tool_calling():
    """测试工具调用转换"""
    print("\n" + "=" * 60)
    print("测试 5: Tool Calling 转换")
    print("=" * 60)
    
    # Anthropic 工具定义 → OpenAI Chat
    anthropic_req = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "What's the weather in Beijing?"}],
        "tools": [{
            "name": "get_weather",
            "description": "Get current weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}}
            }
        }],
        "tool_choice": "auto"
    }
    
    converted = engine.convert_request(anthropic_req)
    print(f"  tools: {json.dumps(converted.get('tools', []), ensure_ascii=False)[:200]}")
    print(f"  tool_choice: {converted.get('tool_choice')}")
    
    # 验证工具格式
    assert converted["tools"][0]["type"] == "function"
    assert converted["tools"][0]["function"]["name"] == "get_weather"
    assert "parameters" in converted["tools"][0]["function"]
    assert converted["tool_choice"] == "auto"
    
    # 测试 Anthropic tool_choice: "any" → Chat: "required"
    anthropic_req["tool_choice"] = "any"
    converted2 = engine.convert_request(anthropic_req)
    print(f"  tool_choice 'any' → '{converted2.get('tool_choice')}'")
    assert converted2["tool_choice"] == "required"
    
    # 测试 Anthropic tool_choice: {"type":"tool","name":"x"} → Chat
    anthropic_req["tool_choice"] = {"type": "tool", "name": "get_weather"}
    converted3 = engine.convert_request(anthropic_req)
    print(f"  tool_choice {{type:tool}} → {json.dumps(converted3.get('tool_choice'), ensure_ascii=False)}")
    assert converted3["tool_choice"]["type"] == "function"
    
    # 测试 tool_result 转换
    anthropic_with_result = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {"city": "Beijing"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_123", "content": "Sunny, 25°C"}
            ]}
        ],
        "tools": [{"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}]
    }
    
    converted4 = engine.convert_request(anthropic_with_result)
    roles = [m["role"] for m in converted4["messages"]]
    print(f"\n  tool_result 转换后角色序列: {roles}")
    assert "tool" in roles  # tool_result 应转换为 role=tool 的消息
    
    # 找到 tool 消息
    tool_msg = [m for m in converted4["messages"] if m["role"] == "tool"][0]
    assert tool_msg["tool_call_id"] == "toolu_123"
    assert tool_msg["content"] == "Sunny, 25°C"
    print(f"  tool 消息验证通过: tool_call_id=toolu_123, content=Sunny, 25°C")


# ============================================================
# 4. Thinking / Reasoning 转换
# ============================================================

async def test_thinking_reasoning():
    """测试 thinking ↔ reasoning_effort 转换"""
    print("\n" + "=" * 60)
    print("测试 6: Thinking / Reasoning_effort 转换")
    print("=" * 60)
    
    # Anthropic thinking → OpenAI reasoning_effort
    test_cases = [
        ({"type": "disabled"}, "none"),
        ({"type": "enabled", "budget_tokens": 1024}, "low"),
        ({"type": "enabled", "budget_tokens": 10000}, "medium"),
        ({"type": "enabled", "budget_tokens": 32000}, "high"),
        ({"type": "adaptive", "budget_tokens": 5000}, "low"),  # adaptive 按预算映射（5000 < 10000 → low）
    ]
    
    for thinking, expected_effort in test_cases:
        req = {
            "model": "claude-opus-4-6",
            "max_tokens": 16000,
            "thinking": thinking,
            "messages": [{"role": "user", "content": "Think hard"}]
        }
        result = AnthropicConverter.to_openai_chat(req)
        effort = result.get("reasoning_effort", "N/A")
        status = "✓" if effort == expected_effort else "✗"
        print(f"  {status} thinking {json.dumps(thinking)} → reasoning_effort={effort} (expected: {expected_effort})")


# ============================================================
# 5. 响应回转测试
# ============================================================

async def test_response_conversion():
    """测试响应格式回转"""
    print("\n" + "=" * 60)
    print("测试 7: 响应格式回转")
    print("=" * 60)
    
    # 模拟 OpenAI Chat 响应
    chat_response = {
        "id": "chatcmpl-123",
        "model": "gpt-4o",
        "choices": [{
            "message": {"content": "Hello! How can I help you?"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
    }
    
    # Chat → Anthropic
    anthropic_resp = engine.convert_response(chat_response, Protocol.ANTHROPIC)
    print(f"  Chat → Anthropic:")
    print(f"    type: {anthropic_resp['type']}, stop_reason: {anthropic_resp['stop_reason']}")
    print(f"    content: {anthropic_resp['content'][0]['text']}")
    assert anthropic_resp["type"] == "message"
    assert anthropic_resp["stop_reason"] == "end_turn"
    
    # Chat → Responses
    responses_resp = engine.convert_response(chat_response, Protocol.OPENAI_RESPONSES)
    print(f"  Chat → Responses:")
    print(f"    object: {responses_resp['object']}, status: {responses_resp['status']}")
    assert responses_resp["object"] == "response"
    assert responses_resp["status"] == "completed"
    
    # 带 tool_calls 的响应
    chat_tool_response = {
        "id": "chatcmpl-456",
        "model": "gpt-4o",
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35}
    }
    
    # Chat tool_calls → Anthropic tool_use
    anthropic_tool_resp = engine.convert_response(chat_tool_response, Protocol.ANTHROPIC)
    print(f"\n  Chat tool_calls → Anthropic:")
    print(f"    stop_reason: {anthropic_tool_resp['stop_reason']}")
    print(f"    tool_use: name={anthropic_tool_resp['content'][0]['name']}")
    assert anthropic_tool_resp["stop_reason"] == "tool_use"
    assert anthropic_tool_resp["content"][0]["type"] == "tool_use"
    
    # Chat tool_calls → Responses function_call
    responses_tool_resp = engine.convert_response(chat_tool_response, Protocol.OPENAI_RESPONSES)
    print(f"  Chat tool_calls → Responses:")
    func_call = [o for o in responses_tool_resp["output"] if o["type"] == "function_call"]
    print(f"    function_call: name={func_call[0]['name'] if func_call else 'N/A'}")


# ============================================================
# 6. 多模态内容转换
# ============================================================

async def test_multimodal():
    """测试多模态内容转换"""
    print("\n" + "=" * 60)
    print("测试 8: 多模态内容转换")
    print("=" * 60)
    
    # Anthropic image (base64) → OpenAI image_url
    anthropic_image = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}
            ]
        }]
    }
    
    converted = engine.convert_request(anthropic_image)
    content = converted["messages"][-1]["content"]
    if isinstance(content, list):
        image_blocks = [b for b in content if b.get("type") == "image_url"]
        print(f"  Anthropic image → Chat image_url: {len(image_blocks)} 个图片块")
        if image_blocks:
            print(f"    URL 前缀: {image_blocks[0]['image_url']['url'][:30]}...")
    
    # Anthropic image (URL) → OpenAI image_url
    anthropic_url_image = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "url", "url": "https://example.com/photo.jpg"}}
            ]
        }]
    }
    
    converted2 = engine.convert_request(anthropic_url_image)
    content2 = converted2["messages"][-1]["content"]
    if isinstance(content2, list):
        image_blocks2 = [b for b in content2 if b.get("type") == "image_url"]
        if image_blocks2:
            print(f"  Anthropic URL image → Chat: url={image_blocks2[0]['image_url']['url']}")
    
    # Responses input_image → Chat image_url
    responses_image = {
        "model": "gpt-4o",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe this image"},
                {"type": "input_image", "image_url": "https://example.com/photo.jpg"}
            ]
        }]
    }
    
    converted3 = OpenAIResponsesConverter.to_openai_chat(responses_image)
    msg_content = converted3["messages"][-1]["content"]
    if isinstance(msg_content, list):
        img_blocks = [b for b in msg_content if b.get("type") == "image_url"]
        if img_blocks:
            print(f"  Responses input_image → Chat image_url: {len(img_blocks)} 个")


async def test_streaming_responses_to_chat():
    """测试 Responses 请求 → Chat 后端流式"""
    print("\n" + "=" * 60)
    print("测试 9: Responses 请求 → Chat 后端 (流式)")
    print("=" * 60)
    
    try:
        import httpx
        
        responses_request = {
            "model": "gpt-4o",
            "input": [{"type": "message", "role": "user", "content": [{"type": "text", "text": "Count from 1 to 3."}]}],
            "instructions": "You are helpful.",
        }
        
        converted = engine.convert_request(responses_request)
        converted["stream"] = True
        converted["stream_options"] = {"include_usage": True}
        
        headers = {
            "Content-Type": "application/json",
            **CONFIG.get_auth_headers()
        }
        
        async with httpx.AsyncClient(timeout=CONFIG.timeout) as client:
            async with client.stream("POST", CONFIG.backend_url, headers=headers, json=converted) as response:
                if response.status_code == 200:
                    full_content = ""
                    print("  流式响应: ", end="", flush=True)
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
                    print(f"\n  完整响应: {full_content}")
                    return True
                else:
                    print(f"  错误: {response.status_code}")
                    return False
    except Exception as e:
        print(f"  流式请求失败: {e}")
        return False


async def test_streaming_anthropic_to_chat():
    """测试 Anthropic 请求 → Chat 后端流式"""
    print("\n" + "=" * 60)
    print("测试 10: Anthropic 请求 → Chat 后端 (流式)")
    print("=" * 60)
    
    try:
        import httpx
        
        anthropic_request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 200,
            "system": "You are helpful.",
            "stream": True,
            "messages": [{"role": "user", "content": "Count from 1 to 3."}]
        }
        
        converted = engine.convert_request(anthropic_request)
        converted["stream"] = True
        converted["stream_options"] = {"include_usage": True}
        
        headers = {
            "Content-Type": "application/json",
            **CONFIG.get_auth_headers()
        }
        
        async with httpx.AsyncClient(timeout=CONFIG.timeout) as client:
            async with client.stream("POST", CONFIG.backend_url, headers=headers, json=converted) as response:
                if response.status_code == 200:
                    full_content = ""
                    print("  流式响应: ", end="", flush=True)
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
                    print(f"\n  完整响应: {full_content}")
                    return True
                else:
                    print(f"  错误: {response.status_code}")
                    return False
    except Exception as e:
        print(f"  流式请求失败: {e}")
        return False


# ============================================================
# 主函数
# ============================================================

async def run_all_tests():
    """运行所有集成测试"""
    print("=" * 60)
    print("协议转换器集成测试 - MiniMax API (OpenAI Chat 后端)")
    print("=" * 60)
    
    await test_protocol_detection()
    await test_request_conversion()
    await test_tool_calling()
    await test_thinking_reasoning()
    await test_response_conversion()
    await test_multimodal()
    
    success = await test_api_call()
    if success:
        await test_streaming()
        await test_streaming_responses_to_chat()
        await test_streaming_anthropic_to_chat()
    
    print("\n" + "=" * 60)
    print("集成测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
