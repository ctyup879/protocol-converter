"""
9 路全量集成测试 - 3 种协议请求 × 3 种后端

验证矩阵:
                    Chat后端    Responses后端    Anthropic后端
Chat 请求:            ①            ②               ③
Responses 请求:       ④            ⑤               ⑥
Anthropic 请求:       ⑦            ⑧               ⑨
"""

import asyncio
import json
import httpx
from protocol_converter import (
    ProtocolConverterEngine,
    ConverterConfig,
    Protocol,
    OpenAIResponsesConverter,
    AnthropicConverter,
)


# ============================================================
# 三种后端配置
# ============================================================

# Chat 后端 (MiniMax)
CHAT_CONFIG = ConverterConfig(
    backend_type="openai",
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

# Responses 后端 (OpenRouter)
RESPONSES_CONFIG = ConverterConfig(
    backend_type="openai_responses",
    backend_url="https://openrouter.ai/api/v1/responses",
    api_key="REDACTED_OPENROUTER_API_KEY",
    default_model="openai/gpt-oss-120b:free",
    timeout=60.0,
    model_mapping={
        "gpt-4o": "openai/gpt-oss-120b:free",
        "claude-sonnet-4-20250514": "openai/gpt-oss-120b:free",
        "MiniMax-M2.7": "openai/gpt-oss-120b:free",
    }
)

# Anthropic 后端 (MiniMax)
ANTHROPIC_CONFIG = ConverterConfig(
    backend_type="anthropic",
    backend_url="https://api.minimaxi.com/anthropic/v1/messages",
    api_key="REDACTED_MINIMAX_API_KEY",
    default_model="MiniMax-M2.7",
    timeout=60.0,
    model_mapping={
        "gpt-4o": "MiniMax-M2.7",
        "claude-sonnet-4-20250514": "MiniMax-M2.7",
    }
)

engine_chat = ProtocolConverterEngine(CHAT_CONFIG)
engine_responses = ProtocolConverterEngine(RESPONSES_CONFIG)
engine_anthropic = ProtocolConverterEngine(ANTHROPIC_CONFIG)


# ============================================================
# 通用请求定义
# ============================================================

CHAT_REQUEST = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Say hello in one sentence."}
    ],
    "max_completion_tokens": 100,
}

RESPONSES_REQUEST = {
    "model": "gpt-4o",
    "input": "Say hello in one sentence.",
    "instructions": "You are helpful.",
    "max_output_tokens": 100,
}

ANTHROPIC_REQUEST = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 100,
    "system": "You are helpful.",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
}


# ============================================================
# 辅助函数
# ============================================================

def extract_text_from_chat_response(resp):
    """从 Chat 响应提取文本"""
    choices = resp.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "")
    return ""

def extract_text_from_responses_response(resp):
    """从 Responses 响应提取文本"""
    for item in resp.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") in ("output_text", "text"):
                    return c.get("text", "")
    return ""

def extract_text_from_anthropic_response(resp):
    """从 Anthropic 响应提取文本"""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            return block.get("text", "")
    return ""


async def call_backend(config, engine, request, target_protocol):
    """
    通用后端调用: 转换请求 → 调用后端 → 转换响应
    
    Returns:
        (success, original_response, converted_response)
    """
    # 转换请求
    converted_req = engine.convert_request(request)
    
    headers = {"Content-Type": "application/json", **config.get_auth_headers()}
    
    async with httpx.AsyncClient(timeout=config.timeout) as client:
        response = await client.post(config.backend_url, headers=headers, json=converted_req)
        
        if response.status_code != 200:
            return False, None, f"HTTP {response.status_code}: {response.text[:200]}"
        
        result = response.json()
        
        # 转换响应回源协议格式
        converted_resp = engine.convert_response(result, target_protocol)
        return True, result, converted_resp


# ============================================================
# ① Chat 请求 → Chat 后端
# ============================================================

async def test_chat_to_chat():
    print("\n" + "=" * 60)
    print("① Chat 请求 → Chat 后端")
    print("=" * 60)
    
    try:
        success, raw, resp = await call_backend(
            CHAT_CONFIG, engine_chat, CHAT_REQUEST, Protocol.OPENAI_CHAT
        )
        if not success:
            print(f"  ✗ 失败: {resp}")
            return False
        text = extract_text_from_chat_response(resp)
        print(f"  请求: Chat ({len(CHAT_REQUEST['messages'])} messages)")
        print(f"  后端: Chat (直通)")
        print(f"  响应: {text[:100]}")
        print(f"  ✓ 通过" if text else "  ✗ 无文本")
        return bool(text)
    except Exception as e:
        print(f"  ✗ 异常: {e}")
        return False


# ============================================================
# ② Chat 请求 → Responses 后端
# ============================================================

async def test_chat_to_responses():
    print("\n" + "=" * 60)
    print("② Chat 请求 → Responses 后端")
    print("=" * 60)
    
    try:
        success, raw, resp = await call_backend(
            RESPONSES_CONFIG, engine_responses, CHAT_REQUEST, Protocol.OPENAI_CHAT
        )
        if not success:
            print(f"  ✗ 失败: {resp}")
            return False
        # raw 是 Responses 格式，resp 是转回 Chat 的
        text_from_raw = extract_text_from_responses_response(raw)
        text_from_converted = extract_text_from_chat_response(resp)
        print(f"  请求: Chat ({len(CHAT_REQUEST['messages'])} messages)")
        print(f"  后端: Responses")
        print(f"  原始响应: {text_from_raw[:100]}")
        print(f"  回转Chat: {text_from_converted[:100]}")
        print(f"  ✓ 通过" if text_from_raw and text_from_converted else "  ✗ 无文本")
        return bool(text_from_raw and text_from_converted)
    except Exception as e:
        print(f"  ✗ 异常: {e}")
        return False


# ============================================================
# ③ Chat 请求 → Anthropic 后端
# ============================================================

async def test_chat_to_anthropic():
    print("\n" + "=" * 60)
    print("③ Chat 请求 → Anthropic 后端")
    print("=" * 60)
    
    try:
        success, raw, resp = await call_backend(
            ANTHROPIC_CONFIG, engine_anthropic, CHAT_REQUEST, Protocol.OPENAI_CHAT
        )
        if not success:
            print(f"  ✗ 失败: {resp}")
            return False
        text_from_raw = extract_text_from_anthropic_response(raw)
        text_from_converted = extract_text_from_chat_response(resp)
        print(f"  请求: Chat ({len(CHAT_REQUEST['messages'])} messages)")
        print(f"  后端: Anthropic")
        print(f"  原始响应: {text_from_raw[:100]}")
        print(f"  回转Chat: {text_from_converted[:100]}")
        print(f"  ✓ 通过" if text_from_raw and text_from_converted else "  ✗ 无文本")
        return bool(text_from_raw and text_from_converted)
    except Exception as e:
        print(f"  ✗ 异常: {e}")
        return False


# ============================================================
# ④ Responses 请求 → Chat 后端
# ============================================================

async def test_responses_to_chat():
    print("\n" + "=" * 60)
    print("④ Responses 请求 → Chat 后端")
    print("=" * 60)
    
    try:
        success, raw, resp = await call_backend(
            CHAT_CONFIG, engine_chat, RESPONSES_REQUEST, Protocol.OPENAI_RESPONSES
        )
        if not success:
            print(f"  ✗ 失败: {resp}")
            return False
        text_from_raw = extract_text_from_chat_response(raw)
        text_from_converted = extract_text_from_responses_response(resp)
        print(f"  请求: Responses (input={RESPONSES_REQUEST['input'][:30]}...)")
        print(f"  后端: Chat")
        print(f"  原始响应: {text_from_raw[:100]}")
        print(f"  回转Responses: {text_from_converted[:100]}")
        print(f"  ✓ 通过" if text_from_raw and text_from_converted else "  ✗ 无文本")
        return bool(text_from_raw and text_from_converted)
    except Exception as e:
        print(f"  ✗ 异常: {e}")
        return False


# ============================================================
# ⑤ Responses 请求 → Responses 后端
# ============================================================

async def test_responses_to_responses():
    print("\n" + "=" * 60)
    print("⑤ Responses 请求 → Responses 后端")
    print("=" * 60)
    
    try:
        success, raw, resp = await call_backend(
            RESPONSES_CONFIG, engine_responses, RESPONSES_REQUEST, Protocol.OPENAI_RESPONSES
        )
        if not success:
            print(f"  ✗ 失败: {resp}")
            return False
        text = extract_text_from_responses_response(raw)
        print(f"  请求: Responses (input={RESPONSES_REQUEST['input'][:30]}...)")
        print(f"  后端: Responses (直通)")
        print(f"  响应: {text[:100]}")
        print(f"  ✓ 通过" if text else "  ✗ 无文本")
        return bool(text)
    except Exception as e:
        print(f"  ✗ 异常: {e}")
        return False


# ============================================================
# ⑥ Responses 请求 → Anthropic 后端
# ============================================================

async def test_responses_to_anthropic():
    print("\n" + "=" * 60)
    print("⑥ Responses 请求 → Anthropic 后端")
    print("=" * 60)
    
    try:
        success, raw, resp = await call_backend(
            ANTHROPIC_CONFIG, engine_anthropic, RESPONSES_REQUEST, Protocol.OPENAI_RESPONSES
        )
        if not success:
            print(f"  ✗ 失败: {resp}")
            return False
        text_from_raw = extract_text_from_anthropic_response(raw)
        text_from_converted = extract_text_from_responses_response(resp)
        print(f"  请求: Responses (input={RESPONSES_REQUEST['input'][:30]}...)")
        print(f"  后端: Anthropic")
        print(f"  原始响应: {text_from_raw[:100]}")
        print(f"  回转Responses: {text_from_converted[:100]}")
        print(f"  ✓ 通过" if text_from_raw and text_from_converted else "  ✗ 无文本")
        return bool(text_from_raw and text_from_converted)
    except Exception as e:
        print(f"  ✗ 异常: {e}")
        return False


# ============================================================
# ⑦ Anthropic 请求 → Chat 后端
# ============================================================

async def test_anthropic_to_chat():
    print("\n" + "=" * 60)
    print("⑦ Anthropic 请求 → Chat 后端")
    print("=" * 60)
    
    try:
        success, raw, resp = await call_backend(
            CHAT_CONFIG, engine_chat, ANTHROPIC_REQUEST, Protocol.ANTHROPIC
        )
        if not success:
            print(f"  ✗ 失败: {resp}")
            return False
        text_from_raw = extract_text_from_chat_response(raw)
        text_from_converted = extract_text_from_anthropic_response(resp)
        print(f"  请求: Anthropic (model={ANTHROPIC_REQUEST['model']})")
        print(f"  后端: Chat")
        print(f"  原始响应: {text_from_raw[:100]}")
        print(f"  回转Anthropic: {text_from_converted[:100]}")
        print(f"  ✓ 通过" if text_from_raw and text_from_converted else "  ✗ 无文本")
        return bool(text_from_raw and text_from_converted)
    except Exception as e:
        print(f"  ✗ 异常: {e}")
        return False


# ============================================================
# ⑧ Anthropic 请求 → Responses 后端
# ============================================================

async def test_anthropic_to_responses():
    print("\n" + "=" * 60)
    print("⑧ Anthropic 请求 → Responses 后端")
    print("=" * 60)
    
    try:
        success, raw, resp = await call_backend(
            RESPONSES_CONFIG, engine_responses, ANTHROPIC_REQUEST, Protocol.ANTHROPIC
        )
        if not success:
            print(f"  ✗ 失败: {resp}")
            return False
        text_from_raw = extract_text_from_responses_response(raw)
        text_from_converted = extract_text_from_anthropic_response(resp)
        print(f"  请求: Anthropic (model={ANTHROPIC_REQUEST['model']})")
        print(f"  后端: Responses")
        print(f"  原始响应: {text_from_raw[:100]}")
        print(f"  回转Anthropic: {text_from_converted[:100]}")
        print(f"  ✓ 通过" if text_from_raw and text_from_converted else "  ✗ 无文本")
        return bool(text_from_raw and text_from_converted)
    except Exception as e:
        print(f"  ✗ 异常: {e}")
        return False


# ============================================================
# ⑨ Anthropic 请求 → Anthropic 后端
# ============================================================

async def test_anthropic_to_anthropic():
    print("\n" + "=" * 60)
    print("⑨ Anthropic 请求 → Anthropic 后端")
    print("=" * 60)
    
    try:
        success, raw, resp = await call_backend(
            ANTHROPIC_CONFIG, engine_anthropic, ANTHROPIC_REQUEST, Protocol.ANTHROPIC
        )
        if not success:
            print(f"  ✗ 失败: {resp}")
            return False
        text = extract_text_from_anthropic_response(raw)
        print(f"  请求: Anthropic (model={ANTHROPIC_REQUEST['model']})")
        print(f"  后端: Anthropic (直通)")
        print(f"  响应: {text[:100]}")
        print(f"  ✓ 通过" if text else "  ✗ 无文本")
        return bool(text)
    except Exception as e:
        print(f"  ✗ 异常: {e}")
        return False


# ============================================================
# 主函数
# ============================================================

async def run_all_tests():
    print("=" * 60)
    print("9 路全量集成测试 - 3 协议 × 3 后端")
    print("=" * 60)
    print(f"  Chat 后端:      {CHAT_CONFIG.backend_url}")
    print(f"  Responses 后端: {RESPONSES_CONFIG.backend_url}")
    print(f"  Anthropic 后端: {ANTHROPIC_CONFIG.backend_url}")
    
    results = {}
    
    # 第一行: Chat 请求 → 三种后端
    results["①Chat→Chat"] = await test_chat_to_chat()
    results["②Chat→Responses"] = await test_chat_to_responses()
    results["③Chat→Anthropic"] = await test_chat_to_anthropic()
    
    # 第二行: Responses 请求 → 三种后端
    results["④Responses→Chat"] = await test_responses_to_chat()
    results["⑤Responses→Responses"] = await test_responses_to_responses()
    results["⑥Responses→Anthropic"] = await test_responses_to_anthropic()
    
    # 第三行: Anthropic 请求 → 三种后端
    results["⑦Anthropic→Chat"] = await test_anthropic_to_chat()
    results["⑧Anthropic→Responses"] = await test_anthropic_to_responses()
    results["⑨Anthropic→Anthropic"] = await test_anthropic_to_anthropic()
    
    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print("-" * 60)
    
    pass_count = 0
    for name, ok in results.items():
        status = "✓ 通过" if ok else "✗ 失败"
        print(f"  {name}: {status}")
        if ok:
            pass_count += 1
    
    print("-" * 60)
    print(f"  通过: {pass_count}/{len(results)}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
