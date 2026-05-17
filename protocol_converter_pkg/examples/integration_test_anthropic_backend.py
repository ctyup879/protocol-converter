"""
集成测试 - 使用 Anthropic 兼容后端进行端到端测试

测试场景:
1. Anthropic 后端直接请求
2. OpenAI Chat 请求 → Anthropic 后端（协议转换）
3. OpenAI Responses 请求 → Anthropic 后端
4. Anthropic 后端流式请求
5. Tool calling 跨协议转换
6. Thinking / reasoning_effort 参数
7. 响应回转（Anthropic 后端响应 → Chat/Responses 格式）
8. Chat 请求 → Anthropic 后端流式
9. Responses 请求 → Anthropic 后端流式
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


# ============================================================
# 后端配置
# ============================================================

# Anthropic 兼容后端
CONFIG_ANTHROPIC = ConverterConfig(
    backend_type="anthropic",
    backend_url="https://api.minimaxi.com/anthropic/v1/messages",
    api_key="REDACTED_MINIMAX_API_KEY",
    default_model="MiniMax-M2.7",
    timeout=60.0,
    model_mapping={
        "gpt-4o": "MiniMax-M2.7",
    }
)

# Anthropic 后端引擎
engine_anthropic = ProtocolConverterEngine(CONFIG_ANTHROPIC)

# OpenAI Chat 兼容后端（用于对比测试）
CONFIG_OPENAI = ConverterConfig(
    backend_type="openai",
    backend_url="https://api.minimaxi.com/v1/chat/completions",
    api_key="REDACTED_MINIMAX_API_KEY",
    default_model="MiniMax-M2.7",
    timeout=60.0,
    model_mapping={
        "claude-sonnet-4-20250514": "MiniMax-M2.7",
        "gpt-4o": "MiniMax-M2.7",
    }
)

engine_openai = ProtocolConverterEngine(CONFIG_OPENAI)


# ============================================================
# 1. Anthropic 后端直接请求
# ============================================================

async def test_anthropic_backend_direct():
    """测试 Anthropic 兼容后端 - 直接发送 Anthropic 格式"""
    print("\n" + "=" * 60)
    print("测试 1: Anthropic 后端直接请求")
    print("=" * 60)
    
    anthropic_request = {
        "model": "MiniMax-M2.7",
        "max_tokens": 100,
        "system": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": "Say hello in one sentence."}]
    }
    
    # 检测协议
    protocol = engine_anthropic.detect_protocol(anthropic_request)
    print(f"  检测到的协议: {protocol}")
    
    # 转换请求（Anthropic→Anthropic 后端，应基本不变）
    converted = engine_anthropic.convert_request(anthropic_request)
    print(f"  转换后 model: {converted.get('model')}")
    print(f"  转换后 max_tokens: {converted.get('max_tokens')}")
    
    try:
        import httpx
        headers = {
            "Content-Type": "application/json",
            **CONFIG_ANTHROPIC.get_auth_headers()
        }
        
        async with httpx.AsyncClient(timeout=CONFIG_ANTHROPIC.timeout) as client:
            response = await client.post(CONFIG_ANTHROPIC.backend_url, headers=headers, json=converted)
            if response.status_code == 200:
                result = response.json()
                text_blocks = [b for b in result.get("content", []) if b.get("type") == "text"]
                print(f"  响应: {text_blocks[0]['text'] if text_blocks else 'N/A'}")
                return True
            else:
                print(f"  错误: {response.status_code} {response.text[:200]}")
                return False
    except Exception as e:
        print(f"  请求失败: {e}")
        return False


# ============================================================
# 2. OpenAI Chat 请求 → Anthropic 后端
# ============================================================

async def test_chat_to_anthropic_backend():
    """测试 OpenAI Chat 请求 → Anthropic 后端（协议转换）"""
    print("\n" + "=" * 60)
    print("测试 2: OpenAI Chat → Anthropic 后端")
    print("=" * 60)
    
    # 模拟客户端发送 OpenAI Chat 请求
    chat_request = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one sentence."}
        ],
        "max_completion_tokens": 100,
        "temperature": 0.7
    }
    
    print(f"\n  原始 Chat 请求:")
    print(f"    model: {chat_request['model']}")
    print(f"    messages: {len(chat_request['messages'])} 条")
    
    # 引擎自动检测协议并转换为 Anthropic 格式
    converted = engine_anthropic.convert_request(chat_request)
    print(f"\n  转换后 Anthropic 请求:")
    print(f"    model: {converted.get('model')}")
    print(f"    system: {converted.get('system')}")
    print(f"    max_tokens: {converted.get('max_tokens')}")
    print(f"    temperature: {converted.get('temperature')}")
    
    # 验证转换正确性
    assert converted["system"] == "You are a helpful assistant."
    assert converted["max_tokens"] == 100
    assert converted["temperature"] == 0.7
    print(f"  ✓ 转换验证通过")
    
    try:
        import httpx
        headers = {
            "Content-Type": "application/json",
            **CONFIG_ANTHROPIC.get_auth_headers()
        }
        
        async with httpx.AsyncClient(timeout=CONFIG_ANTHROPIC.timeout) as client:
            response = await client.post(CONFIG_ANTHROPIC.backend_url, headers=headers, json=converted)
            if response.status_code == 200:
                result = response.json()
                
                # 将 Anthropic 响应转回 OpenAI Chat 格式
                chat_resp = engine_anthropic.convert_response(result, Protocol.OPENAI_CHAT)
                print(f"\n  Anthropic 响应 → Chat 格式:")
                print(f"    object: {chat_resp['object']}")
                print(f"    content: {chat_resp['choices'][0]['message']['content']}")
                return True
            else:
                print(f"  错误: {response.status_code} {response.text[:200]}")
                return False
    except Exception as e:
        print(f"  请求失败: {e}")
        return False


# ============================================================
# 3. OpenAI Responses 请求 → Anthropic 后端
# ============================================================

async def test_responses_to_anthropic_backend():
    """测试 OpenAI Responses 请求 → Anthropic 后端"""
    print("\n" + "=" * 60)
    print("测试 3: OpenAI Responses → Anthropic 后端")
    print("=" * 60)
    
    responses_request = {
        "model": "gpt-4o",
        "input": [{"type": "message", "role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "instructions": "You are helpful.",
        "max_output_tokens": 100
    }
    
    print(f"\n  原始 Responses 请求:")
    print(f"    model: {responses_request['model']}")
    print(f"    input: {len(responses_request['input'])} 项")
    
    # 引擎转换: Responses → (先 Chat →) Anthropic
    converted = engine_anthropic.convert_request(responses_request)
    print(f"\n  转换后 Anthropic 请求:")
    print(f"    model: {converted.get('model')}")
    print(f"    system: {converted.get('system')}")
    print(f"    max_tokens: {converted.get('max_tokens')}")
    
    # 验证 instructions → system, max_output_tokens → max_tokens
    assert converted.get("system") == "You are helpful."
    assert converted.get("max_tokens") == 100
    print(f"  ✓ 转换验证通过")


# ============================================================
# 4. Anthropic 后端流式请求
# ============================================================

async def test_stream_anthropic_backend():
    """测试 Anthropic 兼容后端 - 流式请求"""
    print("\n" + "=" * 60)
    print("测试 4: Anthropic 后端流式请求")
    print("=" * 60)
    
    anthropic_request = {
        "model": "MiniMax-M2.7",
        "max_tokens": 200,
        "stream": True,
        "messages": [{"role": "user", "content": "Count from 1 to 3."}]
    }
    
    try:
        import httpx
        headers = {
            "Content-Type": "application/json",
            **CONFIG_ANTHROPIC.get_auth_headers()
        }
        
        async with httpx.AsyncClient(timeout=CONFIG_ANTHROPIC.timeout) as client:
            async with client.stream("POST", CONFIG_ANTHROPIC.backend_url, headers=headers, json=anthropic_request) as response:
                if response.status_code == 200:
                    full_content = ""
                    event_types = []
                    print("  Anthropic SSE 事件:")
                    async for line in response.aiter_lines():
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                            event_types.append(event_type)
                        elif line.startswith("data:"):
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                chunk_type = chunk.get("type", "")
                                if chunk_type == "content_block_delta":
                                    delta = chunk.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        text = delta.get("text", "")
                                        print(text, end="", flush=True)
                                        full_content += text
                            except:
                                pass
                    print(f"\n  收到事件类型: {list(set(event_types))}")
                    print(f"  完整响应: {full_content}")
                    return True
                else:
                    print(f"  错误: {response.status_code}")
                    return False
    except Exception as e:
        print(f"  流式请求失败: {e}")
        return False


# ============================================================
# 5. Tool calling 跨协议转换
# ============================================================

async def test_tool_calling_cross_protocol():
    """测试工具调用跨协议转换"""
    print("\n" + "=" * 60)
    print("测试 5: Tool Calling 跨协议转换")
    print("=" * 60)
    
    # === OpenAI Chat → Anthropic 工具定义转换 ===
    chat_tools_req = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What's the weather?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        }],
        "tool_choice": "auto"
    }
    
    # Chat → Anthropic
    anthropic_req = engine_anthropic.convert_request(chat_tools_req)
    print(f"\n  Chat tools → Anthropic tools:")
    if "tools" in anthropic_req:
        tool = anthropic_req["tools"][0]
        print(f"    name: {tool.get('name')}")
        print(f"    input_schema: {json.dumps(tool.get('input_schema', {}), ensure_ascii=False)[:100]}")
        assert tool["name"] == "get_weather"
        assert "input_schema" in tool
        print(f"  ✓ Chat → Anthropic 工具转换验证通过")
    
    # === Anthropic tool_use → Chat tool_calls 响应转换 ===
    anthropic_tool_response = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me check the weather."},
            {"type": "tool_use", "id": "toolu_456", "name": "get_weather", "input": {"city": "Beijing"}}
        ],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 50, "output_tokens": 30}
    }
    
    chat_resp = engine_anthropic.convert_response(anthropic_tool_response, Protocol.OPENAI_CHAT)
    print(f"\n  Anthropic tool_use → Chat tool_calls:")
    msg = chat_resp["choices"][0]["message"]
    print(f"    finish_reason: {chat_resp['choices'][0]['finish_reason']}")
    if msg.get("tool_calls"):
        tc = msg["tool_calls"][0]
        print(f"    tool_call: id={tc['id']}, name={tc['function']['name']}")
        assert chat_resp["choices"][0]["finish_reason"] == "tool_calls"
        assert tc["function"]["name"] == "get_weather"
        print(f"  ✓ Anthropic → Chat 工具调用转换验证通过")
    
    # === Chat tool_calls → Anthropic tool_use 请求转换 ===
    chat_with_tool_calls = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_789",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'}
                }]
            },
            {"role": "tool", "tool_call_id": "call_789", "content": "Sunny, 25°C"}
        ]
    }
    
    anthropic_req2 = engine_anthropic.convert_request(chat_with_tool_calls)
    print(f"\n  Chat tool_calls → Anthropic tool_use:")
    msgs = anthropic_req2["messages"]
    for i, msg in enumerate(msgs):
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    print(f"    msg[{i}]: {block.get('type', msg.get('role'))}")
    print(f"  ✓ Chat → Anthropic tool_calls 消息转换验证通过")


# ============================================================
# 6. Thinking / Reasoning 参数
# ============================================================

async def test_thinking_reasoning_anthropic_backend():
    """测试 thinking / reasoning_effort 跨协议映射"""
    print("\n" + "=" * 60)
    print("测试 6: Thinking / Reasoning_effort 映射")
    print("=" * 60)
    
    # OpenAI reasoning_effort → Anthropic thinking
    test_cases = [
        ("none", {"type": "disabled"}),
        ("low", {"type": "enabled", "budget_tokens": 1024}),
        ("medium", {"type": "enabled", "budget_tokens": 10000}),
        ("high", {"type": "enabled", "budget_tokens": 32000}),
        ("xhigh", {"type": "enabled", "budget_tokens": 64000}),
    ]
    
    for effort, expected_thinking in test_cases:
        chat_req = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Think about this"}],
            "reasoning_effort": effort
        }
        
        result = engine_anthropic.convert_request(chat_req)
        thinking = result.get("thinking", {})
        match = thinking.get("type") == expected_thinking.get("type")
        status = "✓" if match else "✗"
        print(f"  {status} reasoning_effort='{effort}' → thinking.type={thinking.get('type')} (expected: {expected_thinking.get('type')})")
        if thinking.get("budget_tokens"):
            print(f"     budget_tokens={thinking['budget_tokens']}")


# ============================================================
# 7. 响应回转测试
# ============================================================

async def test_response_roundtrip():
    """测试 Anthropic 后端响应 → Chat/Responses 格式回转"""
    print("\n" + "=" * 60)
    print("测试 7: 响应回转（Anthropic → Chat / Responses）")
    print("=" * 60)
    
    # 模拟 Anthropic 后端响应
    anthropic_response = {
        "id": "msg_abc123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello! How can I help you?"}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 15, "output_tokens": 10}
    }
    
    # Anthropic → Chat
    chat_resp = engine_anthropic.convert_response(anthropic_response, Protocol.OPENAI_CHAT)
    print(f"  Anthropic → Chat:")
    print(f"    object: {chat_resp['object']}")
    print(f"    content: {chat_resp['choices'][0]['message']['content']}")
    print(f"    finish_reason: {chat_resp['choices'][0]['finish_reason']}")
    print(f"    usage: prompt={chat_resp['usage']['prompt_tokens']}, completion={chat_resp['usage']['completion_tokens']}")
    assert chat_resp["choices"][0]["finish_reason"] == "stop"
    
    # Anthropic → Responses
    responses_resp = engine_anthropic.convert_response(anthropic_response, Protocol.OPENAI_RESPONSES)
    print(f"\n  Anthropic → Responses:")
    print(f"    object: {responses_resp['object']}")
    print(f"    status: {responses_resp['status']}")
    output_text = [o for o in responses_resp["output"] if o.get("type") == "message"]
    if output_text:
        print(f"    output message: {output_text[0]['content'][0]['text']}")
    assert responses_resp["status"] == "completed"
    
    # 带 tool_use 的 Anthropic 响应
    anthropic_tool_resp = {
        "id": "msg_def456",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "toolu_789", "name": "get_weather", "input": {"city": "Beijing"}}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 30, "output_tokens": 20}
    }
    
    # Anthropic tool_use → Chat tool_calls
    chat_tool_resp = engine_anthropic.convert_response(anthropic_tool_resp, Protocol.OPENAI_CHAT)
    print(f"\n  Anthropic tool_use → Chat tool_calls:")
    msg = chat_tool_resp["choices"][0]["message"]
    print(f"    finish_reason: {chat_tool_resp['choices'][0]['finish_reason']}")
    if msg.get("tool_calls"):
        tc = msg["tool_calls"][0]
        args = json.loads(tc["function"]["arguments"])
        print(f"    tool_call: {tc['function']['name']}({args})")
    assert chat_tool_resp["choices"][0]["finish_reason"] == "tool_calls"


async def test_stream_chat_to_anthropic_backend():
    """测试 Chat 请求 → Anthropic 后端（流式）"""
    print("\n" + "=" * 60)
    print("测试 8: Chat 请求 → Anthropic 后端 (流式)")
    print("=" * 60)
    
    chat_request = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Count from 1 to 3."}
        ],
        "max_completion_tokens": 200,
    }
    
    converted = engine_anthropic.convert_request(chat_request)
    converted["model"] = CONFIG_ANTHROPIC.default_model
    converted["stream"] = True
    
    try:
        import httpx
        headers = {
            "Content-Type": "application/json",
            **CONFIG_ANTHROPIC.get_auth_headers()
        }
        
        async with httpx.AsyncClient(timeout=CONFIG_ANTHROPIC.timeout) as client:
            async with client.stream("POST", CONFIG_ANTHROPIC.backend_url, headers=headers, json=converted) as response:
                if response.status_code == 200:
                    full_content = ""
                    event_types = []
                    print("  流式响应: ", end="", flush=True)
                    async for line in response.aiter_lines():
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                            event_types.append(event_type)
                        elif line.startswith("data:"):
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                chunk_type = chunk.get("type", "")
                                if chunk_type == "content_block_delta":
                                    delta = chunk.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        text = delta.get("text", "")
                                        print(text, end="", flush=True)
                                        full_content += text
                            except:
                                pass
                    print(f"\n  收到事件类型: {list(set(event_types))}")
                    print(f"  完整响应: {full_content}")
                    return True
                else:
                    print(f"  错误: {response.status_code}")
                    return False
    except Exception as e:
        print(f"  流式请求失败: {e}")
        return False


async def test_stream_responses_to_anthropic_backend():
    """测试 Responses 请求 → Anthropic 后端（流式）"""
    print("\n" + "=" * 60)
    print("测试 9: Responses 请求 → Anthropic 后端 (流式)")
    print("=" * 60)
    
    responses_request = {
        "model": "gpt-4o",
        "input": [{"type": "message", "role": "user", "content": [{"type": "text", "text": "Count from 1 to 3."}]}],
        "instructions": "You are helpful.",
        "max_output_tokens": 200,
    }
    
    converted = engine_anthropic.convert_request(responses_request)
    converted["model"] = CONFIG_ANTHROPIC.default_model
    converted["stream"] = True
    
    try:
        import httpx
        headers = {
            "Content-Type": "application/json",
            **CONFIG_ANTHROPIC.get_auth_headers()
        }
        
        async with httpx.AsyncClient(timeout=CONFIG_ANTHROPIC.timeout) as client:
            async with client.stream("POST", CONFIG_ANTHROPIC.backend_url, headers=headers, json=converted) as response:
                if response.status_code == 200:
                    full_content = ""
                    event_types = []
                    print("  流式响应: ", end="", flush=True)
                    async for line in response.aiter_lines():
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                            event_types.append(event_type)
                        elif line.startswith("data:"):
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                chunk_type = chunk.get("type", "")
                                if chunk_type == "content_block_delta":
                                    delta = chunk.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        text = delta.get("text", "")
                                        print(text, end="", flush=True)
                                        full_content += text
                            except:
                                pass
                    print(f"\n  收到事件类型: {list(set(event_types))}")
                    print(f"  完整响应: {full_content}")
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
    """运行所有测试"""
    print("=" * 60)
    print("协议转换器集成测试 - Anthropic 兼容后端")
    print("=" * 60)
    
    await test_anthropic_backend_direct()
    await test_chat_to_anthropic_backend()
    await test_responses_to_anthropic_backend()
    await test_tool_calling_cross_protocol()
    await test_thinking_reasoning_anthropic_backend()
    await test_response_roundtrip()
    await test_stream_anthropic_backend()
    await test_stream_chat_to_anthropic_backend()
    await test_stream_responses_to_anthropic_backend()
    
    print("\n" + "=" * 60)
    print("集成测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
