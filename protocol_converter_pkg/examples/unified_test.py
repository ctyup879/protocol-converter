"""
统一集成测试 - 展示三种后端模式的完整转换流程

测试场景:
1. 三种后端模式（OpenAI Chat / OpenAI Responses / Anthropic）
2. 6 种转换路径的请求转换
3. 响应回转验证
4. Tool calling 全链路
5. 流式请求
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
# 三种后端配置
# ============================================================

# 模式1: OpenAI Chat 兼容后端
OPENAI_BACKEND = ConverterConfig(
    backend_type="openai",
    backend_url="https://api.minimaxi.com/v1/chat/completions",
    api_key="sk-cp-xxx",
    default_model="MiniMax-M2.7",
    model_mapping={
        "claude-sonnet-4-20250514": "MiniMax-M2.7",
        "gpt-4o": "MiniMax-M2.7",
    }
)

# 模式2: Anthropic 兼容后端
ANTHROPIC_BACKEND = ConverterConfig(
    backend_type="anthropic",
    backend_url="https://api.minimaxi.com/anthropic/v1/messages",
    api_key="sk-cp-xxx",
    default_model="MiniMax-M2.7",
    model_mapping={
        "gpt-4o": "MiniMax-M2.7",
    }
)

# 模式3: OpenAI Responses 兼容后端
RESPONSES_BACKEND = ConverterConfig(
    backend_type="openai_responses",
    backend_url="https://openrouter.ai/api/v1/responses",
    api_key="sk-or-v1-xxx",
    default_model="openai/gpt-oss-120b:free",
)

engine_openai = ProtocolConverterEngine(OPENAI_BACKEND)
engine_anthropic = ProtocolConverterEngine(ANTHROPIC_BACKEND)
engine_responses = ProtocolConverterEngine(RESPONSES_BACKEND)


# ============================================================
# 1. 三种协议检测
# ============================================================

async def test_protocol_detection():
    """测试三种协议的检测"""
    print("\n" + "=" * 60)
    print("测试 1: 协议检测")
    print("=" * 60)
    
    test_requests = {
        "OpenAI Chat": {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        "Anthropic": {"model": "claude-sonnet-4-20250514", "max_tokens": 100, "messages": [{"role": "user", "content": "Hi"}]},
        "Responses": {"model": "gpt-4o", "input": "Hi"},
        "Anthropic+thinking": {"model": "claude-opus-4-6", "max_tokens": 100, "thinking": {"type": "enabled", "budget_tokens": 5000}, "messages": [{"role": "user", "content": "Hi"}]},
        "Responses+reasoning": {"model": "o3", "input": "Think", "reasoning": {"effort": "high"}},
    }
    
    for name, req in test_requests.items():
        protocol = engine_openai.detect_protocol(req)
        print(f"  {name}: {protocol.value}")


# ============================================================
# 2. 6 种转换路径
# ============================================================

async def test_all_conversion_paths():
    """测试所有 6 种转换路径"""
    print("\n" + "=" * 60)
    print("测试 2: 6 种转换路径")
    print("=" * 60)
    
    # --- 到 OpenAI Chat 后端 ---
    print("\n  --- 目标: OpenAI Chat ---")
    
    # Anthropic → Chat
    anthropic_req = {"model": "claude-sonnet-4-20250514", "max_tokens": 100, "system": "Be helpful.", "messages": [{"role": "user", "content": "Hi"}]}
    r1 = engine_openai.convert_request(anthropic_req)
    print(f"    Anthropic→Chat: messages={len(r1['messages'])}, has max_completion_tokens={'max_completion_tokens' in r1}")
    
    # Responses → Chat
    responses_req = {"model": "gpt-4o", "input": [{"type": "message", "role": "user", "content": [{"type": "text", "text": "Hi"}]}], "instructions": "Be helpful.", "max_output_tokens": 100}
    r2 = engine_openai.convert_request(responses_req)
    print(f"    Responses→Chat: messages={len(r2['messages'])}, max_completion_tokens={r2.get('max_completion_tokens')}")
    
    # --- 到 Anthropic 后端 ---
    print("\n  --- 目标: Anthropic ---")
    
    # Chat → Anthropic
    chat_req = {"model": "gpt-4o", "messages": [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "Hi"}], "max_completion_tokens": 100}
    r3 = engine_anthropic.convert_request(chat_req)
    print(f"    Chat→Anthropic: system={r3.get('system')}, max_tokens={r3.get('max_tokens')}")
    
    # Responses → Anthropic (先 Chat 再 Anthropic)
    r4 = engine_anthropic.convert_request(responses_req)
    print(f"    Responses→Anthropic: system={r4.get('system')}, max_tokens={r4.get('max_tokens')}")
    
    # --- 到 Responses 后端 ---
    print("\n  --- 目标: OpenAI Responses ---")
    
    # Chat → Responses
    r5 = engine_responses.convert_request(chat_req)
    print(f"    Chat→Responses: instructions={r5.get('instructions')}, max_output_tokens={r5.get('max_output_tokens')}")
    
    # Anthropic → Responses (先 Chat 再 Responses)
    r6 = engine_responses.convert_request(anthropic_req)
    print(f"    Anthropic→Responses: instructions={r6.get('instructions')}, max_output_tokens={r6.get('max_output_tokens')}")


# ============================================================
# 3. 响应回转验证
# ============================================================

async def test_response_roundtrip():
    """测试响应格式回转"""
    print("\n" + "=" * 60)
    print("测试 3: 响应回转验证")
    print("=" * 60)
    
    # Chat 响应 → Anthropic / Responses
    chat_resp = {
        "id": "chatcmpl-123",
        "model": "gpt-4o",
        "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }
    
    anthropic_resp = AnthropicConverter.from_openai_chat(chat_resp)
    responses_resp = OpenAIResponsesConverter.from_openai_chat(chat_resp)
    
    print(f"  Chat → Anthropic: type={anthropic_resp['type']}, stop_reason={anthropic_resp['stop_reason']}")
    print(f"  Chat → Responses: object={responses_resp['object']}, status={responses_resp['status']}")
    
    # Anthropic 响应 → Chat / Responses
    anthropic_backend_resp = {
        "id": "msg_456",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello from Claude!"}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5}
    }
    
    chat_from_anthropic = engine_anthropic.convert_response(anthropic_backend_resp, Protocol.OPENAI_CHAT)
    responses_from_anthropic = engine_anthropic.convert_response(anthropic_backend_resp, Protocol.OPENAI_RESPONSES)
    
    print(f"\n  Anthropic → Chat: object={chat_from_anthropic['object']}, content={chat_from_anthropic['choices'][0]['message']['content']}")
    print(f"  Anthropic → Responses: object={responses_from_anthropic['object']}, status={responses_from_anthropic['status']}")
    
    # Responses 响应 → Chat / Anthropic
    responses_backend_resp = {
        "id": "resp_789",
        "object": "response",
        "status": "completed",
        "model": "gpt-4o",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello from Responses!"}]}],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    }
    
    chat_from_responses = engine_responses.convert_response(responses_backend_resp, Protocol.OPENAI_CHAT)
    anthropic_from_responses = engine_responses.convert_response(responses_backend_resp, Protocol.ANTHROPIC)
    
    print(f"\n  Responses → Chat: object={chat_from_responses['object']}, content={chat_from_responses['choices'][0]['message']['content']}")
    print(f"  Responses → Anthropic: type={anthropic_from_responses['type']}, stop_reason={anthropic_from_responses['stop_reason']}")


# ============================================================
# 4. Tool calling 全链路
# ============================================================

async def test_tool_calling_e2e():
    """测试工具调用全链路转换"""
    print("\n" + "=" * 60)
    print("测试 4: Tool Calling 全链路")
    print("=" * 60)
    
    # Anthropic 完整对话 → Chat
    anthropic_conversation = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_001", "name": "get_weather", "input": {"city": "Beijing"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_001", "content": "Sunny, 25°C"}
            ]}
        ],
        "tools": [{
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
        }],
        "tool_choice": "auto"
    }
    
    # 1) Anthropic → Chat
    chat_req = engine_openai.convert_request(anthropic_conversation)
    print(f"\n  Anthropic → Chat:")
    roles = [m["role"] for m in chat_req["messages"]]
    print(f"    角色序列: {roles}")
    print(f"    tools: {len(chat_req.get('tools', []))} 个")
    print(f"    tool_choice: {chat_req.get('tool_choice')}")
    
    # 验证 tool_use → tool_calls, tool_result → tool 消息
    assistant_msgs = [m for m in chat_req["messages"] if m["role"] == "assistant"]
    tool_msgs = [m for m in chat_req["messages"] if m["role"] == "tool"]
    assert any(m.get("tool_calls") for m in assistant_msgs), "assistant 应有 tool_calls"
    assert len(tool_msgs) > 0, "应有 tool 消息"
    assert tool_msgs[0]["tool_call_id"] == "toolu_001"
    print(f"    ✓ tool_use→tool_calls, tool_result→tool 验证通过")
    
    # 2) Chat → Anthropic（反向）
    chat_with_tools = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'}}]},
            {"role": "tool", "tool_call_id": "call_123", "content": "Sunny, 25°C"}
        ],
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
        "tool_choice": "auto"
    }
    
    anthropic_req = engine_anthropic.convert_request(chat_with_tools)
    print(f"\n  Chat → Anthropic:")
    # 检查 tool_use 和 tool_result 是否正确转换
    has_tool_use = False
    has_tool_result = False
    for msg in anthropic_req["messages"]:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "tool_use":
                        has_tool_use = True
                    if block.get("type") == "tool_result":
                        has_tool_result = True
    
    print(f"    has tool_use: {has_tool_use}")
    print(f"    has tool_result: {has_tool_result}")
    print(f"    tools: name={anthropic_req['tools'][0]['name']}, has input_schema={'input_schema' in anthropic_req['tools'][0]}")
    
    # 3) Responses → Chat 工具转换
    responses_with_tools = {
        "model": "gpt-4o",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "text", "text": "Check weather"}]},
            {"type": "function_call", "call_id": "call_456", "name": "get_weather", "arguments": '{"city": "NYC"}'},
            {"type": "function_call_output", "call_id": "call_456", "output": "Sunny, 72F"}
        ],
        "tools": [{"type": "function", "name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]
    }
    
    chat_from_responses = OpenAIResponsesConverter.to_openai_chat(responses_with_tools)
    roles2 = [m["role"] for m in chat_from_responses["messages"]]
    print(f"\n  Responses → Chat 工具:")
    print(f"    角色序列: {roles2}")
    print(f"    tools: {len(chat_from_responses.get('tools', []))} 个")


# ============================================================
# 5. 流式请求
# ============================================================

async def test_streaming():
    """测试 Anthropic 后端流式请求"""
    print("\n" + "=" * 60)
    print("测试 5: 流式请求")
    print("=" * 60)
    
    try:
        import httpx
        
        # 使用 Anthropic 后端流式
        anthropic_request = {
            "model": "MiniMax-M2.7",
            "max_tokens": 200,
            "stream": True,
            "messages": [{"role": "user", "content": "Count from 1 to 3."}]
        }
        
        headers = {
            "Content-Type": "application/json",
            **ANTHROPIC_BACKEND.get_auth_headers()
        }
        
        print("  Anthropic SSE 流:")
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", ANTHROPIC_BACKEND.backend_url, headers=headers, json=anthropic_request) as response:
                if response.status_code == 200:
                    full_content = ""
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                if chunk.get("type") == "content_block_delta":
                                    delta = chunk.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        text = delta.get("text", "")
                                        print(text, end="", flush=True)
                                        full_content += text
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
    """运行所有测试"""
    print("=" * 60)
    print("协议转换器 - 统一集成测试")
    print("=" * 60)
    
    await test_protocol_detection()
    await test_all_conversion_paths()
    await test_response_roundtrip()
    await test_tool_calling_e2e()
    
    # 流式测试需要有效 API Key
    # await test_streaming()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
