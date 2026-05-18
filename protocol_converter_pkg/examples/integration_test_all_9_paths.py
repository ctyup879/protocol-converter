"""
9 路全量集成测试 - 3 种协议请求 × 3 种后端（非流式 + 流式）

验证矩阵:
                    Chat后端    Responses后端    Anthropic后端
Chat 请求:            ①            ②               ③
Responses 请求:       ④            ⑤               ⑥
Anthropic 请求:       ⑦            ⑧               ⑨

每条路径同时测试非流式和流式两种模式。
"""

import asyncio
import json
import os
from pathlib import Path
import httpx
from protocol_converter import (
    ProtocolConverterEngine,
    ConverterConfig,
    Protocol,
    OpenAIResponsesConverter,
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
# 三种后端配置
# ============================================================

# Chat 后端 (MiniMax)
CHAT_CONFIG = ConverterConfig(
    backend_type="openai",
    backend_url="https://api.minimaxi.com/v1/chat/completions",
    api_key=os.environ.get("MINIMAX_API_KEY", ""),
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
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
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
    api_key=os.environ.get("MINIMAX_API_KEY", ""),
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
    通用后端调用（非流式）: 转换请求 → 调用后端 → 转换响应
    
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


async def call_backend_stream(config, engine, request, target_protocol):
    """
    通用后端调用（流式）: 转换请求 → 调用后端流式 → 解析SSE → 提取文本
    
    Returns:
        (success, full_text, event_types)
    """
    # 转换请求并设置 stream=True
    converted_req = engine.convert_request(request)
    converted_req["stream"] = True
    
    # 对 Chat 后端添加 stream_options
    if config.backend_type == "openai" and "stream_options" not in converted_req:
        converted_req["stream_options"] = {"include_usage": True}
    
    headers = {"Content-Type": "application/json", **config.get_auth_headers()}
    full_text = ""
    event_types = set()
    
    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            async with client.stream("POST", config.backend_url, headers=headers, json=converted_req) as response:
                if response.status_code != 200:
                    return False, "", set()
                
                pending_event_type = None
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    
                    # 处理 event: 行
                    if line.startswith("event:"):
                        pending_event_type = line[6:].strip()
                        continue
                    
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        
                        current_event_type = pending_event_type
                        pending_event_type = None
                        
                        try:
                            chunk = json.loads(data_str)
                            
                            # Anthropic 后端 SSE
                            if config.backend_type == "anthropic":
                                evt_type = current_event_type or chunk.get("type", "")
                                event_types.add(evt_type)
                                if evt_type == "content_block_delta":
                                    delta = chunk.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        full_text += delta.get("text", "")
                            
                            # Responses 后端 SSE
                            elif config.backend_type == "openai_responses":
                                evt_type = current_event_type or chunk.get("type", "")
                                event_types.add(evt_type)
                                if evt_type == "response.output_text.delta":
                                    full_text += chunk.get("delta", "")
                            
                            # Chat 后端 SSE
                            else:
                                chunk_type = chunk.get("object", "")
                                event_types.add(chunk_type)
                                choices = chunk.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        full_text += content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass
        
        return True, full_text, event_types
    except Exception as e:
        print(f"    流式异常: {e}")
        return False, "", set()


# ============================================================
# ① Chat 请求 → Chat 后端
# ============================================================

async def test_chat_to_chat():
    print("\n" + "=" * 60)
    print("① Chat 请求 → Chat 后端")
    print("=" * 60)
    
    # 非流式
    try:
        success, raw, resp = await call_backend(
            CHAT_CONFIG, engine_chat, CHAT_REQUEST, Protocol.OPENAI_CHAT
        )
        if not success:
            print(f"  ✗ 非流式失败: {resp}")
            non_stream_ok = False
        else:
            text = extract_text_from_chat_response(resp)
            print(f"  [非流式] 请求: Chat → Chat (直通)")
            print(f"  [非流式] 响应: {text[:100]}")
            non_stream_ok = bool(text)
    except Exception as e:
        print(f"  ✗ 非流式异常: {e}")
        non_stream_ok = False
    
    # 流式
    try:
        success, full_text, event_types = await call_backend_stream(
            CHAT_CONFIG, engine_chat, CHAT_REQUEST, Protocol.OPENAI_CHAT
        )
        if not success:
            print(f"  ✗ 流式失败")
            stream_ok = False
        else:
            print(f"  [流式]   请求: Chat → Chat (直通)")
            print(f"  [流式]   响应: {full_text[:100]}")
            print(f"  [流式]   事件: {sorted(event_types)}")
            stream_ok = bool(full_text)
    except Exception as e:
        print(f"  ✗ 流式异常: {e}")
        stream_ok = False
    
    print(f"  {'✓ 通过' if non_stream_ok and stream_ok else '✗ 失败'} (非流式={'✓' if non_stream_ok else '✗'}, 流式={'✓' if stream_ok else '✗'})")
    return non_stream_ok and stream_ok


# ============================================================
# ② Chat 请求 → Responses 后端
# ============================================================

async def test_chat_to_responses():
    print("\n" + "=" * 60)
    print("② Chat 请求 → Responses 后端")
    print("=" * 60)
    
    # 非流式
    try:
        success, raw, resp = await call_backend(
            RESPONSES_CONFIG, engine_responses, CHAT_REQUEST, Protocol.OPENAI_CHAT
        )
        if not success:
            print(f"  ✗ 非流式失败: {resp}")
            non_stream_ok = False
        else:
            text_from_raw = extract_text_from_responses_response(raw)
            text_from_converted = extract_text_from_chat_response(resp)
            print(f"  [非流式] 请求: Chat → Responses")
            print(f"  [非流式] 原始: {text_from_raw[:80]}")
            print(f"  [非流式] 回转: {text_from_converted[:80]}")
            non_stream_ok = bool(text_from_raw and text_from_converted)
    except Exception as e:
        print(f"  ✗ 非流式异常: {e}")
        non_stream_ok = False
    
    # 流式
    try:
        success, full_text, event_types = await call_backend_stream(
            RESPONSES_CONFIG, engine_responses, CHAT_REQUEST, Protocol.OPENAI_CHAT
        )
        if not success:
            print(f"  ✗ 流式失败")
            stream_ok = False
        else:
            print(f"  [流式]   请求: Chat → Responses")
            print(f"  [流式]   响应: {full_text[:80]}")
            print(f"  [流式]   事件: {sorted(event_types)}")
            stream_ok = bool(full_text)
    except Exception as e:
        print(f"  ✗ 流式异常: {e}")
        stream_ok = False
    
    print(f"  {'✓ 通过' if non_stream_ok and stream_ok else '✗ 失败'} (非流式={'✓' if non_stream_ok else '✗'}, 流式={'✓' if stream_ok else '✗'})")
    return non_stream_ok and stream_ok


# ============================================================
# ③ Chat 请求 → Anthropic 后端
# ============================================================

async def test_chat_to_anthropic():
    print("\n" + "=" * 60)
    print("③ Chat 请求 → Anthropic 后端")
    print("=" * 60)
    
    # 非流式
    try:
        success, raw, resp = await call_backend(
            ANTHROPIC_CONFIG, engine_anthropic, CHAT_REQUEST, Protocol.OPENAI_CHAT
        )
        if not success:
            print(f"  ✗ 非流式失败: {resp}")
            non_stream_ok = False
        else:
            text_from_raw = extract_text_from_anthropic_response(raw)
            text_from_converted = extract_text_from_chat_response(resp)
            print(f"  [非流式] 请求: Chat → Anthropic")
            print(f"  [非流式] 原始: {text_from_raw[:80]}")
            print(f"  [非流式] 回转: {text_from_converted[:80]}")
            non_stream_ok = bool(text_from_raw and text_from_converted)
    except Exception as e:
        print(f"  ✗ 非流式异常: {e}")
        non_stream_ok = False
    
    # 流式
    try:
        success, full_text, event_types = await call_backend_stream(
            ANTHROPIC_CONFIG, engine_anthropic, CHAT_REQUEST, Protocol.OPENAI_CHAT
        )
        if not success:
            print(f"  ✗ 流式失败")
            stream_ok = False
        else:
            print(f"  [流式]   请求: Chat → Anthropic")
            print(f"  [流式]   响应: {full_text[:80]}")
            print(f"  [流式]   事件: {sorted(event_types)}")
            stream_ok = bool(full_text)
    except Exception as e:
        print(f"  ✗ 流式异常: {e}")
        stream_ok = False
    
    print(f"  {'✓ 通过' if non_stream_ok and stream_ok else '✗ 失败'} (非流式={'✓' if non_stream_ok else '✗'}, 流式={'✓' if stream_ok else '✗'})")
    return non_stream_ok and stream_ok


# ============================================================
# ④ Responses 请求 → Chat 后端
# ============================================================

async def test_responses_to_chat():
    print("\n" + "=" * 60)
    print("④ Responses 请求 → Chat 后端")
    print("=" * 60)
    
    # 非流式
    try:
        success, raw, resp = await call_backend(
            CHAT_CONFIG, engine_chat, RESPONSES_REQUEST, Protocol.OPENAI_RESPONSES
        )
        if not success:
            print(f"  ✗ 非流式失败: {resp}")
            non_stream_ok = False
        else:
            text_from_raw = extract_text_from_chat_response(raw)
            text_from_converted = extract_text_from_responses_response(resp)
            print(f"  [非流式] 请求: Responses → Chat")
            print(f"  [非流式] 原始: {text_from_raw[:80]}")
            print(f"  [非流式] 回转: {text_from_converted[:80]}")
            non_stream_ok = bool(text_from_raw and text_from_converted)
    except Exception as e:
        print(f"  ✗ 非流式异常: {e}")
        non_stream_ok = False
    
    # 流式
    try:
        success, full_text, event_types = await call_backend_stream(
            CHAT_CONFIG, engine_chat, RESPONSES_REQUEST, Protocol.OPENAI_RESPONSES
        )
        if not success:
            print(f"  ✗ 流式失败")
            stream_ok = False
        else:
            print(f"  [流式]   请求: Responses → Chat")
            print(f"  [流式]   响应: {full_text[:80]}")
            print(f"  [流式]   事件: {sorted(event_types)}")
            stream_ok = bool(full_text)
    except Exception as e:
        print(f"  ✗ 流式异常: {e}")
        stream_ok = False
    
    print(f"  {'✓ 通过' if non_stream_ok and stream_ok else '✗ 失败'} (非流式={'✓' if non_stream_ok else '✗'}, 流式={'✓' if stream_ok else '✗'})")
    return non_stream_ok and stream_ok


# ============================================================
# ⑤ Responses 请求 → Responses 后端
# ============================================================

async def test_responses_to_responses():
    print("\n" + "=" * 60)
    print("⑤ Responses 请求 → Responses 后端")
    print("=" * 60)
    
    # 非流式
    try:
        success, raw, resp = await call_backend(
            RESPONSES_CONFIG, engine_responses, RESPONSES_REQUEST, Protocol.OPENAI_RESPONSES
        )
        if not success:
            print(f"  ✗ 非流式失败: {resp}")
            non_stream_ok = False
        else:
            text = extract_text_from_responses_response(raw)
            print(f"  [非流式] 请求: Responses → Responses (直通)")
            print(f"  [非流式] 响应: {text[:100]}")
            non_stream_ok = bool(text)
    except Exception as e:
        print(f"  ✗ 非流式异常: {e}")
        non_stream_ok = False
    
    # 流式
    try:
        success, full_text, event_types = await call_backend_stream(
            RESPONSES_CONFIG, engine_responses, RESPONSES_REQUEST, Protocol.OPENAI_RESPONSES
        )
        if not success:
            print(f"  ✗ 流式失败")
            stream_ok = False
        else:
            print(f"  [流式]   请求: Responses → Responses (直通)")
            print(f"  [流式]   响应: {full_text[:100]}")
            print(f"  [流式]   事件: {sorted(event_types)}")
            stream_ok = bool(full_text)
    except Exception as e:
        print(f"  ✗ 流式异常: {e}")
        stream_ok = False
    
    print(f"  {'✓ 通过' if non_stream_ok and stream_ok else '✗ 失败'} (非流式={'✓' if non_stream_ok else '✗'}, 流式={'✓' if stream_ok else '✗'})")
    return non_stream_ok and stream_ok


# ============================================================
# ⑥ Responses 请求 → Anthropic 后端
# ============================================================

async def test_responses_to_anthropic():
    print("\n" + "=" * 60)
    print("⑥ Responses 请求 → Anthropic 后端")
    print("=" * 60)
    
    # 非流式
    try:
        success, raw, resp = await call_backend(
            ANTHROPIC_CONFIG, engine_anthropic, RESPONSES_REQUEST, Protocol.OPENAI_RESPONSES
        )
        if not success:
            print(f"  ✗ 非流式失败: {resp}")
            non_stream_ok = False
        else:
            text_from_raw = extract_text_from_anthropic_response(raw)
            text_from_converted = extract_text_from_responses_response(resp)
            print(f"  [非流式] 请求: Responses → Anthropic")
            print(f"  [非流式] 原始: {text_from_raw[:80]}")
            print(f"  [非流式] 回转: {text_from_converted[:80]}")
            non_stream_ok = bool(text_from_raw and text_from_converted)
    except Exception as e:
        print(f"  ✗ 非流式异常: {e}")
        non_stream_ok = False
    
    # 流式
    try:
        success, full_text, event_types = await call_backend_stream(
            ANTHROPIC_CONFIG, engine_anthropic, RESPONSES_REQUEST, Protocol.OPENAI_RESPONSES
        )
        if not success:
            print(f"  ✗ 流式失败")
            stream_ok = False
        else:
            print(f"  [流式]   请求: Responses → Anthropic")
            print(f"  [流式]   响应: {full_text[:80]}")
            print(f"  [流式]   事件: {sorted(event_types)}")
            stream_ok = bool(full_text)
    except Exception as e:
        print(f"  ✗ 流式异常: {e}")
        stream_ok = False
    
    print(f"  {'✓ 通过' if non_stream_ok and stream_ok else '✗ 失败'} (非流式={'✓' if non_stream_ok else '✗'}, 流式={'✓' if stream_ok else '✗'})")
    return non_stream_ok and stream_ok


# ============================================================
# ⑦ Anthropic 请求 → Chat 后端
# ============================================================

async def test_anthropic_to_chat():
    print("\n" + "=" * 60)
    print("⑦ Anthropic 请求 → Chat 后端")
    print("=" * 60)
    
    # 非流式
    try:
        success, raw, resp = await call_backend(
            CHAT_CONFIG, engine_chat, ANTHROPIC_REQUEST, Protocol.ANTHROPIC
        )
        if not success:
            print(f"  ✗ 非流式失败: {resp}")
            non_stream_ok = False
        else:
            text_from_raw = extract_text_from_chat_response(raw)
            text_from_converted = extract_text_from_anthropic_response(resp)
            print(f"  [非流式] 请求: Anthropic → Chat")
            print(f"  [非流式] 原始: {text_from_raw[:80]}")
            print(f"  [非流式] 回转: {text_from_converted[:80]}")
            non_stream_ok = bool(text_from_raw and text_from_converted)
    except Exception as e:
        print(f"  ✗ 非流式异常: {e}")
        non_stream_ok = False
    
    # 流式
    try:
        success, full_text, event_types = await call_backend_stream(
            CHAT_CONFIG, engine_chat, ANTHROPIC_REQUEST, Protocol.ANTHROPIC
        )
        if not success:
            print(f"  ✗ 流式失败")
            stream_ok = False
        else:
            print(f"  [流式]   请求: Anthropic → Chat")
            print(f"  [流式]   响应: {full_text[:80]}")
            print(f"  [流式]   事件: {sorted(event_types)}")
            stream_ok = bool(full_text)
    except Exception as e:
        print(f"  ✗ 流式异常: {e}")
        stream_ok = False
    
    print(f"  {'✓ 通过' if non_stream_ok and stream_ok else '✗ 失败'} (非流式={'✓' if non_stream_ok else '✗'}, 流式={'✓' if stream_ok else '✗'})")
    return non_stream_ok and stream_ok


# ============================================================
# ⑧ Anthropic 请求 → Responses 后端
# ============================================================

async def test_anthropic_to_responses():
    print("\n" + "=" * 60)
    print("⑧ Anthropic 请求 → Responses 后端")
    print("=" * 60)
    
    # 非流式
    try:
        success, raw, resp = await call_backend(
            RESPONSES_CONFIG, engine_responses, ANTHROPIC_REQUEST, Protocol.ANTHROPIC
        )
        if not success:
            print(f"  ✗ 非流式失败: {resp}")
            non_stream_ok = False
        else:
            text_from_raw = extract_text_from_responses_response(raw)
            text_from_converted = extract_text_from_anthropic_response(resp)
            print(f"  [非流式] 请求: Anthropic → Responses")
            print(f"  [非流式] 原始: {text_from_raw[:80]}")
            print(f"  [非流式] 回转: {text_from_converted[:80]}")
            non_stream_ok = bool(text_from_raw and text_from_converted)
    except Exception as e:
        print(f"  ✗ 非流式异常: {e}")
        non_stream_ok = False
    
    # 流式
    try:
        success, full_text, event_types = await call_backend_stream(
            RESPONSES_CONFIG, engine_responses, ANTHROPIC_REQUEST, Protocol.ANTHROPIC
        )
        if not success:
            print(f"  ✗ 流式失败")
            stream_ok = False
        else:
            print(f"  [流式]   请求: Anthropic → Responses")
            print(f"  [流式]   响应: {full_text[:80]}")
            print(f"  [流式]   事件: {sorted(event_types)}")
            stream_ok = bool(full_text)
    except Exception as e:
        print(f"  ✗ 流式异常: {e}")
        stream_ok = False
    
    print(f"  {'✓ 通过' if non_stream_ok and stream_ok else '✗ 失败'} (非流式={'✓' if non_stream_ok else '✗'}, 流式={'✓' if stream_ok else '✗'})")
    return non_stream_ok and stream_ok


# ============================================================
# ⑨ Anthropic 请求 → Anthropic 后端
# ============================================================

async def test_anthropic_to_anthropic():
    print("\n" + "=" * 60)
    print("⑨ Anthropic 请求 → Anthropic 后端")
    print("=" * 60)
    
    # 非流式
    try:
        success, raw, resp = await call_backend(
            ANTHROPIC_CONFIG, engine_anthropic, ANTHROPIC_REQUEST, Protocol.ANTHROPIC
        )
        if not success:
            print(f"  ✗ 非流式失败: {resp}")
            non_stream_ok = False
        else:
            text = extract_text_from_anthropic_response(raw)
            print(f"  [非流式] 请求: Anthropic → Anthropic (直通)")
            print(f"  [非流式] 响应: {text[:100]}")
            non_stream_ok = bool(text)
    except Exception as e:
        print(f"  ✗ 非流式异常: {e}")
        non_stream_ok = False
    
    # 流式
    try:
        success, full_text, event_types = await call_backend_stream(
            ANTHROPIC_CONFIG, engine_anthropic, ANTHROPIC_REQUEST, Protocol.ANTHROPIC
        )
        if not success:
            print(f"  ✗ 流式失败")
            stream_ok = False
        else:
            print(f"  [流式]   请求: Anthropic → Anthropic (直通)")
            print(f"  [流式]   响应: {full_text[:100]}")
            print(f"  [流式]   事件: {sorted(event_types)}")
            stream_ok = bool(full_text)
    except Exception as e:
        print(f"  ✗ 流式异常: {e}")
        stream_ok = False
    
    print(f"  {'✓ 通过' if non_stream_ok and stream_ok else '✗ 失败'} (非流式={'✓' if non_stream_ok else '✗'}, 流式={'✓' if stream_ok else '✗'})")
    return non_stream_ok and stream_ok


# ============================================================
# 主函数
# ============================================================

async def run_all_tests():
    print("=" * 60)
    print("9 路全量集成测试 - 3 协议 × 3 后端 (非流式 + 流式)")
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
    
    # 同协议透传 + 无 model_mapping 测试
    await test_same_protocol_passthrough_no_mapping()


# ============================================================
# 同协议透传 + 无 model_mapping 集成测试
# ============================================================

async def test_same_protocol_passthrough_no_mapping():
    """测试同协议透传时 model_mapping 为空，model 原样透传"""
    print("\n" + "=" * 60)
    print("同协议透传 + 无 model_mapping 测试")
    print("=" * 60)
    
    results = {}
    
    # --- Chat → Chat 后端 (无 model_mapping) ---
    print("\n  [Chat→Chat 无model_mapping]")
    chat_config_no_mapping = ConverterConfig(
        backend_type="openai",
        backend_url=CHAT_CONFIG.backend_url,
        api_key=CHAT_CONFIG.api_key,
        default_model=CHAT_CONFIG.default_model,
        timeout=CHAT_CONFIG.timeout,
        # model_mapping 不设置
    )
    engine_no_mapping = ProtocolConverterEngine(chat_config_no_mapping)
    
    # 请求中直接使用后端支持的 model 名称
    chat_req_native = {
        "model": "MiniMax-M2.7",
        "messages": [{"role": "user", "content": "Say hi"}],
    }
    
    # 非流式
    ok_nonstream = False
    try:
        converted = engine_no_mapping.convert_request(chat_req_native)
        assert converted["model"] == "MiniMax-M2.7", f"model_mapping为空时应原样透传, 实际: {converted['model']}"
        print(f"    [非流式] model透传: {converted['model']} ✓")
        
        headers = {"Content-Type": "application/json", **chat_config_no_mapping.get_auth_headers()}
        async with httpx.AsyncClient(timeout=chat_config_no_mapping.timeout) as client:
            resp = await client.post(chat_config_no_mapping.backend_url, headers=headers, json=converted)
            ok_nonstream = resp.status_code == 200
            print(f"    [非流式] 后端响应: {resp.status_code} {'✓' if ok_nonstream else '✗'}")
    except Exception as e:
        print(f"    [非流式] 失败: {e}")
    
    # 流式
    ok_stream = False
    try:
        converted = engine_no_mapping.convert_request(chat_req_native)
        converted["stream"] = True
        converted["stream_options"] = {"include_usage": True}
        assert converted["model"] == "MiniMax-M2.7", f"流式model透传失败, 实际: {converted['model']}"
        
        headers = {"Content-Type": "application/json", **chat_config_no_mapping.get_auth_headers()}
        async with httpx.AsyncClient(timeout=chat_config_no_mapping.timeout) as client:
            async with client.stream("POST", chat_config_no_mapping.backend_url, headers=headers, json=converted) as resp:
                if resp.status_code == 200:
                    full_content = ""
                    async for line in resp.aiter_lines():
                        if line.startswith("data:"):
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if delta:
                                    full_content += delta
                            except:
                                pass
                    ok_stream = True
                    print(f"    [流式]   model透传 + 流式响应: ✓")
                else:
                    print(f"    [流式]   错误: {resp.status_code}")
    except Exception as e:
        print(f"    [流式]   失败: {e}")
    
    results["Chat→Chat无mapping"] = ok_nonstream and ok_stream
    
    # --- Responses → Responses 后端 (无 model_mapping) ---
    print("\n  [Responses→Responses 无model_mapping]")
    responses_config_no_mapping = ConverterConfig(
        backend_type="openai_responses",
        backend_url=RESPONSES_CONFIG.backend_url,
        api_key=RESPONSES_CONFIG.api_key,
        default_model=RESPONSES_CONFIG.default_model,
        timeout=RESPONSES_CONFIG.timeout,
        # model_mapping 不设置
    )
    engine_resp_no_mapping = ProtocolConverterEngine(responses_config_no_mapping)
    
    responses_req_native = {
        "model": "openai/gpt-oss-120b:free",
        "input": "Say hi",
    }
    
    # 非流式
    ok_nonstream = False
    try:
        converted = engine_resp_no_mapping.convert_request(responses_req_native)
        assert converted["model"] == "openai/gpt-oss-120b:free", f"model_mapping为空时应原样透传, 实际: {converted['model']}"
        print(f"    [非流式] model透传: {converted['model']} ✓")
        
        headers = {"Content-Type": "application/json", **responses_config_no_mapping.get_auth_headers()}
        async with httpx.AsyncClient(timeout=responses_config_no_mapping.timeout) as client:
            resp = await client.post(responses_config_no_mapping.backend_url, headers=headers, json=converted)
            ok_nonstream = resp.status_code == 200
            print(f"    [非流式] 后端响应: {resp.status_code} {'✓' if ok_nonstream else '✗'}")
    except Exception as e:
        print(f"    [非流式] 失败: {e}")
    
    # 流式
    ok_stream = False
    try:
        converted = engine_resp_no_mapping.convert_request(responses_req_native)
        converted["stream"] = True
        assert converted["model"] == "openai/gpt-oss-120b:free"
        
        headers = {"Content-Type": "application/json", **responses_config_no_mapping.get_auth_headers()}
        async with httpx.AsyncClient(timeout=responses_config_no_mapping.timeout) as client:
            async with client.stream("POST", responses_config_no_mapping.backend_url, headers=headers, json=converted) as resp:
                if resp.status_code == 200:
                    full_content = ""
                    event_types = set()
                    async for line in resp.aiter_lines():
                        if line.startswith("event:"):
                            event_types.add(line[6:].strip())
                        elif line.startswith("data:"):
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                chunk_type = chunk.get("type", "")
                                if chunk_type == "response.output_text.delta":
                                    full_content += chunk.get("delta", "")
                            except:
                                pass
                    ok_stream = True
                    print(f"    [流式]   model透传 + 流式响应: ✓")
                else:
                    print(f"    [流式]   错误: {resp.status_code}")
    except Exception as e:
        print(f"    [流式]   失败: {e}")
    
    results["Responses→Responses无mapping"] = ok_nonstream and ok_stream
    
    # --- Anthropic → Anthropic 后端 (无 model_mapping) ---
    print("\n  [Anthropic→Anthropic 无model_mapping]")
    anthropic_config_no_mapping = ConverterConfig(
        backend_type="anthropic",
        backend_url=ANTHROPIC_CONFIG.backend_url,
        api_key=ANTHROPIC_CONFIG.api_key,
        default_model=ANTHROPIC_CONFIG.default_model,
        timeout=ANTHROPIC_CONFIG.timeout,
        # model_mapping 不设置
    )
    engine_anth_no_mapping = ProtocolConverterEngine(anthropic_config_no_mapping)
    
    anthropic_req_native = {
        "model": "MiniMax-M2.7",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Say hi"}],
    }
    
    # 非流式
    ok_nonstream = False
    try:
        converted = engine_anth_no_mapping.convert_request(anthropic_req_native)
        assert converted["model"] == "MiniMax-M2.7", f"model_mapping为空时应原样透传, 实际: {converted['model']}"
        print(f"    [非流式] model透传: {converted['model']} ✓")
        
        headers = {"Content-Type": "application/json", **anthropic_config_no_mapping.get_auth_headers()}
        async with httpx.AsyncClient(timeout=anthropic_config_no_mapping.timeout) as client:
            resp = await client.post(anthropic_config_no_mapping.backend_url, headers=headers, json=converted)
            ok_nonstream = resp.status_code == 200
            print(f"    [非流式] 后端响应: {resp.status_code} {'✓' if ok_nonstream else '✗'}")
    except Exception as e:
        print(f"    [非流式] 失败: {e}")
    
    # 流式
    ok_stream = False
    try:
        converted = engine_anth_no_mapping.convert_request(anthropic_req_native)
        converted["stream"] = True
        assert converted["model"] == "MiniMax-M2.7"
        
        headers = {"Content-Type": "application/json", **anthropic_config_no_mapping.get_auth_headers()}
        async with httpx.AsyncClient(timeout=anthropic_config_no_mapping.timeout) as client:
            async with client.stream("POST", anthropic_config_no_mapping.backend_url, headers=headers, json=converted) as resp:
                if resp.status_code == 200:
                    full_content = ""
                    event_types = []
                    async for line in resp.aiter_lines():
                        if line.startswith("event:"):
                            event_types.append(line[6:].strip())
                        elif line.startswith("data:"):
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                if chunk.get("type") == "content_block_delta":
                                    delta = chunk.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        full_content += delta.get("text", "")
                            except:
                                pass
                    ok_stream = True
                    print(f"    [流式]   model透传 + 流式响应: ✓")
                else:
                    print(f"    [流式]   错误: {resp.status_code}")
    except Exception as e:
        print(f"    [流式]   失败: {e}")
    
    results["Anthropic→Anthropic无mapping"] = ok_nonstream and ok_stream
    
    # 汇总
    print("\n" + "-" * 60)
    print("同协议透传 + 无 model_mapping 测试结果:")
    for name, ok in results.items():
        status = "✓ 通过" if ok else "✗ 失败"
        print(f"  {name}: {status}")
    print("-" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
