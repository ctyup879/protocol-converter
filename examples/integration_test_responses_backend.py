"""
集成测试 - OpenAI Responses 协议

使用 OpenAI Responses 兼容后端进行端到端测试。
可配置任意 OpenAI Responses 兼容的 API 端点（如 OpenAI 官方、OpenRouter 等）。

测试场景:
1. 直接发送 OpenAI Responses 格式
2. OpenAI Chat → Responses 转换
3. Anthropic → Responses 转换
4. Tool calling / function calling 跨协议
5. Responses 响应 → Chat / Anthropic 格式回转
6. 流式 Responses 请求
7. Responses 特有参数 (reasoning, text, truncation)
8. Chat 请求 → Responses 后端流式
9. Anthropic 请求 → Responses 后端流式
"""

import asyncio
import json
import os
from pathlib import Path
from protocol_converter import (
    ProtocolConverterEngine,
    ConverterConfig,
    Protocol,
    OpenAIResponsesConverter,
    OpenAIChatConverter,
    AnthropicConverter,
)


def _load_env():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

_load_env()


# ============================================================
# OpenAI Responses 后端配置
# ============================================================
# 请根据实际使用的 API 提供方修改以下配置：
# - OpenAI 官方:   backend_url="https://api.openai.com/v1/responses"
# - OpenRouter:    backend_url="https://openrouter.ai/api/v1/responses"
# - 其他兼容端点:  backend_url="https://your-provider/v1/responses"

RESPONSES_CONFIG = ConverterConfig(
    backend_type="openai_responses",
    backend_url="https://openrouter.ai/api/v1/responses",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    default_model="openai/gpt-oss-120b:free",
    timeout=60.0,
    model_mapping={
        "gpt-4o": "openai/gpt-oss-120b:free",
        "gpt-4o-mini": "openai/gpt-oss-120b:free",
        "claude-sonnet-4-20250514": "openai/gpt-oss-120b:free",
        "claude-opus-4-6": "openai/gpt-oss-120b:free",
        "MiniMax-M2.7": "openai/gpt-oss-120b:free",
    },
)

engine = ProtocolConverterEngine(RESPONSES_CONFIG)


# ============================================================
# 1. 直接发送 Responses 格式
# ============================================================

async def test_responses_direct():
    """测试直接发送 OpenAI Responses 格式"""
    print("\n" + "=" * 60)
    print("测试 1: 直接发送 Responses 格式")
    print("=" * 60)
    
    request = {
        "model": RESPONSES_CONFIG.default_model,
        "input": "Tell me a joke",
    }
    
    print(f"  请求: {json.dumps(request, ensure_ascii=False)}")
    
    try:
        import httpx
        headers = {"Content-Type": "application/json", **RESPONSES_CONFIG.get_auth_headers()}
        
        async with httpx.AsyncClient(timeout=RESPONSES_CONFIG.timeout) as client:
            response = await client.post(RESPONSES_CONFIG.backend_url, headers=headers, json=request)
            
            if response.status_code == 200:
                result = response.json()
                # 提取文本输出
                output_text = ""
                for item in result.get("output", []):
                    if item.get("type") == "message":
                        for c in item.get("content", []):
                            if c.get("type") == "output_text":
                                output_text += c.get("text", "")
                print(f"  响应: {output_text[:200]}")
                return True
            else:
                print(f"  错误: {response.status_code} {response.text[:200]}")
                return False
    except Exception as e:
        print(f"  请求失败: {e}")
        return False


# ============================================================
# 2. OpenAI Chat → Responses 转换
# ============================================================

async def test_chat_to_responses():
    """测试 OpenAI Chat 请求转换为 Responses 格式"""
    print("\n" + "=" * 60)
    print("测试 2: OpenAI Chat → Responses")
    print("=" * 60)
    
    chat_request = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Tell me a joke"}
        ],
        "max_completion_tokens": 200
    }
    
    # 转换为 Responses 格式
    responses_req = OpenAIResponsesConverter.from_openai_chat_request(chat_request)
    responses_req["model"] = RESPONSES_CONFIG.default_model
    
    print(f"  原始 Chat: model={chat_request['model']}, messages={len(chat_request['messages'])}")
    print(f"  转换后 Responses: model={responses_req['model']}, input type={type(responses_req['input']).__name__}")
    print(f"  instructions: {responses_req.get('instructions')}")
    print(f"  max_output_tokens: {responses_req.get('max_output_tokens')}")
    
    assert responses_req.get("instructions") == "You are helpful."
    assert responses_req.get("max_output_tokens") == 200
    
    try:
        import httpx
        headers = {"Content-Type": "application/json", **RESPONSES_CONFIG.get_auth_headers()}
        
        async with httpx.AsyncClient(timeout=RESPONSES_CONFIG.timeout) as client:
            response = await client.post(RESPONSES_CONFIG.backend_url, headers=headers, json=responses_req)
            
            if response.status_code == 200:
                result = response.json()
                # 转回 Chat 格式
                chat_resp = OpenAIResponsesConverter.to_chat_response(result)
                print(f"  Responses → Chat 回转: object={chat_resp.get('object')}")
                return True
            else:
                print(f"  错误: {response.status_code}")
                return False
    except Exception as e:
        print(f"  请求失败: {e}")
        return False


# ============================================================
# 3. Anthropic → Responses 转换
# ============================================================

async def test_anthropic_to_responses():
    """测试 Anthropic 请求转换为 Responses 格式"""
    print("\n" + "=" * 60)
    print("测试 3: Anthropic → Responses")
    print("=" * 60)
    
    anthropic_request = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "Tell me a joke"}]
    }
    
    # Anthropic → Chat → Responses
    chat_req = AnthropicConverter.to_openai_chat(anthropic_request)
    responses_req = OpenAIResponsesConverter.from_openai_chat_request(chat_req)
    responses_req["model"] = RESPONSES_CONFIG.default_model
    
    print(f"  原始 Anthropic: model={anthropic_request['model']}")
    print(f"  转换后 Responses: model={responses_req['model']}")
    print(f"  instructions: {responses_req.get('instructions')}")
    print(f"  max_output_tokens: {responses_req.get('max_output_tokens')}")
    
    # 验证 system → instructions, max_tokens → max_output_tokens
    assert responses_req.get("instructions") == "You are helpful."
    assert responses_req.get("max_output_tokens") == 100
    print(f"  ✓ 转换验证通过")


# ============================================================
# 4. Tool calling 跨协议
# ============================================================

async def test_tool_calling():
    """测试工具调用跨协议转换"""
    print("\n" + "=" * 60)
    print("测试 4: Tool Calling 跨协议转换")
    print("=" * 60)
    
    # === Chat 工具定义 → Responses 工具定义 ===
    chat_tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }
    }]
    
    responses_tools = OpenAIResponsesConverter._convert_chat_tools_to_responses(chat_tools)
    print(f"  Chat tools → Responses tools:")
    print(f"    type: {responses_tools[0]['type']}")
    print(f"    name: {responses_tools[0]['name']}")
    assert responses_tools[0]["type"] == "function"
    assert responses_tools[0]["name"] == "get_weather"
    
    # === Responses function_call 输入 → Chat tool_calls ===
    responses_input = [
        {"type": "message", "role": "user", "content": [{"type": "text", "text": "What's the weather?"}]},
        {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": '{"city": "NYC"}'},
        {"type": "function_call_output", "call_id": "call_1", "output": "Sunny, 72F"}
    ]
    
    chat_req = OpenAIResponsesConverter.to_openai_chat({"model": "gpt-4o", "input": responses_input})
    roles = [m["role"] for m in chat_req["messages"]]
    print(f"\n  Responses function_call → Chat:")
    print(f"    角色序列: {roles}")
    assert "assistant" in roles  # function_call → assistant + tool_calls
    assert "tool" in roles       # function_call_output → tool
    
    # === Anthropic 工具 → Responses 工具 ===
    anthropic_tools = [{
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
    }]
    
    # Anthropic tools → Chat tools → Responses tools
    chat_tools_from_anthropic = AnthropicConverter._convert_tools(anthropic_tools)
    responses_tools_from_anthropic = OpenAIResponsesConverter._convert_chat_tools_to_responses(chat_tools_from_anthropic)
    print(f"\n  Anthropic tools → Responses tools:")
    print(f"    type: {responses_tools_from_anthropic[0]['type']}")
    print(f"    name: {responses_tools_from_anthropic[0]['name']}")


# ============================================================
# 5. Responses 响应 → Chat / Anthropic 格式回转
# ============================================================

async def test_response_roundtrip():
    """测试 Responses 响应格式回转"""
    print("\n" + "=" * 60)
    print("测试 5: Responses 响应 → Chat / Anthropic 格式回转")
    print("=" * 60)
    
    # 模拟 Responses 响应
    responses_response = {
        "id": "resp_123",
        "object": "response",
        "status": "completed",
        "created_at": 1234567890,
        "model": "gpt-4o",
        "output": [
            {
                "type": "message",
                "id": "msg_abc",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello! How can I help you?"}]
            }
        ],
        "usage": {"input_tokens": 15, "output_tokens": 10, "total_tokens": 25}
    }
    
    # Responses → Chat
    chat_resp = OpenAIResponsesConverter.to_chat_response(responses_response)
    print(f"  Responses → Chat:")
    print(f"    object: {chat_resp['object']}")
    print(f"    content: {chat_resp['choices'][0]['message']['content']}")
    print(f"    finish_reason: {chat_resp['choices'][0]['finish_reason']}")
    assert chat_resp["choices"][0]["message"]["content"] == "Hello! How can I help you?"
    
    # Responses → Chat → Anthropic
    anthropic_resp = AnthropicConverter.from_openai_chat(chat_resp)
    print(f"\n  Responses → Chat → Anthropic:")
    print(f"    type: {anthropic_resp['type']}")
    print(f"    stop_reason: {anthropic_resp['stop_reason']}")
    print(f"    content: {anthropic_resp['content'][0]['text']}")
    assert anthropic_resp["type"] == "message"
    assert anthropic_resp["stop_reason"] == "end_turn"
    
    # 带 function_call 的 Responses 响应
    responses_tool_response = {
        "id": "resp_456",
        "object": "response",
        "status": "completed",
        "model": "gpt-4o",
        "output": [
            {
                "type": "function_call",
                "id": "fc_789",
                "call_id": "fc_789",
                "name": "get_weather",
                "arguments": '{"city": "Beijing"}',
                "status": "completed"
            }
        ],
        "usage": {"input_tokens": 20, "output_tokens": 15, "total_tokens": 35}
    }
    
    # Responses function_call → Chat tool_calls
    chat_tool_resp = OpenAIResponsesConverter.to_chat_response(responses_tool_response)
    print(f"\n  Responses function_call → Chat tool_calls:")
    msg = chat_tool_resp["choices"][0]["message"]
    if msg.get("tool_calls"):
        tc = msg["tool_calls"][0]
        print(f"    tool_call: {tc['function']['name']}({tc['function']['arguments']})")
    
    # Responses function_call → Chat → Anthropic tool_use
    anthropic_tool_resp = AnthropicConverter.from_openai_chat(chat_tool_resp)
    tool_use_blocks = [b for b in anthropic_tool_resp["content"] if b.get("type") == "tool_use"]
    if tool_use_blocks:
        print(f"  Responses → Chat → Anthropic tool_use:")
        print(f"    name: {tool_use_blocks[0]['name']}")
        print(f"    input: {tool_use_blocks[0]['input']}")


# ============================================================
# 6. 流式 Responses 请求
# ============================================================

async def test_streaming():
    """测试流式 Responses 请求"""
    print("\n" + "=" * 60)
    print("测试 6: 流式 Responses 请求")
    print("=" * 60)
    
    request = {
        "model": RESPONSES_CONFIG.default_model,
        "input": "Count from 1 to 3",
        "stream": True
    }
    
    try:
        import httpx
        headers = {"Content-Type": "application/json", **RESPONSES_CONFIG.get_auth_headers()}
        
        async with httpx.AsyncClient(timeout=RESPONSES_CONFIG.timeout) as client:
            async with client.stream("POST", RESPONSES_CONFIG.backend_url, headers=headers, json=request) as response:
                if response.status_code == 200:
                    full_content = ""
                    event_types = set()
                    print("  流式响应: ", end="", flush=True)
                    async for line in response.aiter_lines():
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                            event_types.add(event_type)
                        elif line.startswith("data:"):
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                chunk_type = chunk.get("type", "")
                                event_types.add(chunk_type)
                                # 提取文本增量
                                if chunk_type == "response.output_text.delta":
                                    delta = chunk.get("delta", "")
                                    print(delta, end="", flush=True)
                                    full_content += delta
                            except:
                                pass
                    print(f"\n  收到事件类型: {sorted(event_types)}")
                    return True
                else:
                    print(f"  错误: {response.status_code}")
                    return False
    except Exception as e:
        print(f"  流式请求失败: {e}")
        return False


# ============================================================
# 7. Responses 特有参数
# ============================================================

async def test_responses_specific_params():
    """测试 Responses 特有参数转换"""
    print("\n" + "=" * 60)
    print("测试 7: Responses 特有参数转换")
    print("=" * 60)
    
    # reasoning 参数 → Chat reasoning_effort
    req_with_reasoning = {
        "model": "o3",
        "input": "Solve this",
        "reasoning": {"effort": "high", "summary": "auto"}
    }
    chat_req = OpenAIResponsesConverter.to_openai_chat(req_with_reasoning)
    print(f"  reasoning → reasoning_effort: {chat_req.get('reasoning_effort')}")
    assert chat_req.get("reasoning_effort") == "high"
    
    # text.format → response_format
    req_with_text_format = {
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
    chat_req2 = OpenAIResponsesConverter.to_openai_chat(req_with_text_format)
    print(f"  text.format → response_format: {chat_req2.get('response_format', {}).get('type')}")
    assert chat_req2.get("response_format", {}).get("type") == "json_schema"
    
    # max_output_tokens → max_completion_tokens
    req_with_max_output = {
        "model": "gpt-4o",
        "input": "Hello",
        "max_output_tokens": 1024
    }
    chat_req3 = OpenAIResponsesConverter.to_openai_chat(req_with_max_output)
    print(f"  max_output_tokens → max_completion_tokens: {chat_req3.get('max_completion_tokens')}")
    assert chat_req3.get("max_completion_tokens") == 1024
    
    # instructions → developer message
    req_with_instructions = {
        "model": "gpt-4o",
        "input": "Hello",
        "instructions": "You are a poet."
    }
    chat_req4 = OpenAIResponsesConverter.to_openai_chat(req_with_instructions)
    system_msgs = [m for m in chat_req4["messages"] if m["role"] == "developer"]
    print(f"  instructions → developer message: {system_msgs[0]['content'] if system_msgs else 'N/A'}")
    assert system_msgs[0]["content"] == "You are a poet."
    
    # Chat reasoning_effort → Responses reasoning
    chat_reasoning = {"model": "o3", "messages": [{"role": "user", "content": "Solve"}], "reasoning_effort": "high"}
    responses_req = OpenAIResponsesConverter.from_openai_chat_request(chat_reasoning)
    print(f"\n  Chat reasoning_effort → Responses reasoning: {responses_req.get('reasoning')}")
    assert responses_req.get("reasoning", {}).get("effort") == "high"
    
    # Chat response_format → Responses text
    chat_format = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Generate"}], "response_format": {"type": "json_object"}}
    responses_req2 = OpenAIResponsesConverter.from_openai_chat_request(chat_format)
    print(f"  Chat response_format → Responses text: {responses_req2.get('text')}")
    assert responses_req2.get("text", {}).get("format", {}).get("type") == "json_object"
    
    print(f"\n  ✓ 所有参数转换验证通过")


# ============================================================
# 8. Chat 请求 → Responses 后端流式
# ============================================================

async def test_stream_chat_to_responses_backend():
    """测试 Chat 请求 → Responses 后端（流式）"""
    print("\n" + "=" * 60)
    print("测试 8: Chat 请求 → Responses 后端 (流式)")
    print("=" * 60)
    
    chat_request = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Count from 1 to 3."}
        ],
        "max_completion_tokens": 200,
    }
    
    converted = engine.convert_request(chat_request)
    converted["stream"] = True
    
    try:
        import httpx
        headers = {"Content-Type": "application/json", **RESPONSES_CONFIG.get_auth_headers()}
        
        async with httpx.AsyncClient(timeout=RESPONSES_CONFIG.timeout) as client:
            async with client.stream("POST", RESPONSES_CONFIG.backend_url, headers=headers, json=converted) as response:
                if response.status_code == 200:
                    full_content = ""
                    event_types = set()
                    print("  流式响应: ", end="", flush=True)
                    async for line in response.aiter_lines():
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                            event_types.add(event_type)
                        elif line.startswith("data:"):
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                chunk_type = chunk.get("type", "")
                                event_types.add(chunk_type)
                                if chunk_type == "response.output_text.delta":
                                    delta = chunk.get("delta", "")
                                    print(delta, end="", flush=True)
                                    full_content += delta
                            except:
                                pass
                    print(f"\n  收到事件类型: {sorted(event_types)}")
                    print(f"  完整响应: {full_content}")
                    return True
                else:
                    error_body = await response.aread()
                    print(f"  错误: {response.status_code} {error_body.decode()[:300]}")
                    return False
    except Exception as e:
        print(f"  流式请求失败: {e}")
        return False


# ============================================================
# 9. Anthropic 请求 → Responses 后端流式
# ============================================================

async def test_stream_anthropic_to_responses_backend():
    """测试 Anthropic 请求 → Responses 后端（流式）"""
    print("\n" + "=" * 60)
    print("测试 9: Anthropic 请求 → Responses 后端 (流式)")
    print("=" * 60)
    
    anthropic_request = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 200,
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "Count from 1 to 3."}]
    }
    
    converted = engine.convert_request(anthropic_request)
    converted["stream"] = True
    
    try:
        import httpx
        headers = {"Content-Type": "application/json", **RESPONSES_CONFIG.get_auth_headers()}
        
        async with httpx.AsyncClient(timeout=RESPONSES_CONFIG.timeout) as client:
            async with client.stream("POST", RESPONSES_CONFIG.backend_url, headers=headers, json=converted) as response:
                if response.status_code == 200:
                    full_content = ""
                    event_types = set()
                    print("  流式响应: ", end="", flush=True)
                    async for line in response.aiter_lines():
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                            event_types.add(event_type)
                        elif line.startswith("data:"):
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                chunk_type = chunk.get("type", "")
                                event_types.add(chunk_type)
                                if chunk_type == "response.output_text.delta":
                                    delta = chunk.get("delta", "")
                                    print(delta, end="", flush=True)
                                    full_content += delta
                            except:
                                pass
                    print(f"\n  收到事件类型: {sorted(event_types)}")
                    print(f"  完整响应: {full_content}")
                    return True
                else:
                    error_body = await response.aread()
                    print(f"  错误: {response.status_code} {error_body.decode()[:300]}")
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
    print("协议转换器 - OpenAI Responses 协议测试")
    print(f"后端: {RESPONSES_CONFIG.backend_url}")
    print("=" * 60)
    
    await test_responses_direct()
    await test_chat_to_responses()
    await test_anthropic_to_responses()
    await test_tool_calling()
    await test_response_roundtrip()
    await test_responses_specific_params()
    await test_streaming()
    await test_stream_chat_to_responses_backend()
    await test_stream_anthropic_to_responses_backend()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
