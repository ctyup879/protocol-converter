"""
Anthropic Messages API 协议转换器

参考: https://docs.anthropic.com/en/api/messages
参考 SDK: anthropic-sdk-python

Anthropic Messages API 请求参数:
- model (必填): 模型名称
- max_tokens (必填): 最大生成 token 数
- messages (必填): 消息列表
- system: 系统提示词 (str | TextBlockParam[])
- tools: 工具定义 (ToolUnionParam[])
- tool_choice: 工具选择策略
- temperature: 温度 (0.0-1.0, 默认 1.0)
- top_p: nucleus 采样
- top_k: Top-K 采样
- stop_sequences: 停止序列
- stream: 是否流式
- thinking: 扩展思考配置
- metadata: 元数据 {"user_id": "..."}
- cache_control: 缓存控制
- service_tier: 服务层级 ("auto" | "standard_only")
- container: 容器标识
- output_config: 输出配置

Anthropic 消息内容块类型:
- text: 文本
- image: 图片 (source: base64 | url)
- document: 文档 (PDF等)
- tool_use: 工具调用
- tool_result: 工具结果
- thinking: 扩展思考
- redacted_thinking: 脱敏思考
- server_tool_use: 服务器工具调用
- web_search_tool_result: 网页搜索结果
- web_fetch_tool_result: 网页获取结果
- code_execution_tool_result: 代码执行结果
- search_result: 搜索结果

Anthropic 流式事件类型:
- message_start: 消息开始 (包含完整 message 对象)
- content_block_start: 内容块开始
- content_block_delta: 内容增量 (text_delta | input_json_delta | thinking_delta | citations_delta | signature_delta)
- content_block_stop: 内容块结束
- message_delta: 消息级增量 (stop_reason, stop_sequence, usage)
- message_stop: 消息结束
- ping: 心跳
- error: 错误

Anthropic 响应字段:
- id, type, role, content, model, stop_reason, stop_sequence, stop_details, usage
- usage: input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens,
         server_tool_use, service_tier, cache_creation, inference_geo

Anthropic 停止原因:
- end_turn: 自然停止
- max_tokens: 达到最大 token 数
- stop_sequence: 遇到自定义停止序列
- tool_use: 工具调用
- pause_turn: 暂停长运行
- refusal: 流式分类器介入
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union, Iterator


class AnthropicConverter:
    """Anthropic Messages API 协议转换器"""
    
    PROTOCOL_NAME = "anthropic"
    
    # Anthropic -> OpenAI 角色映射
    ROLE_MAP = {
        "user": "user",
        "assistant": "assistant",
    }
    
    # OpenAI -> Anthropic 停止原因映射
    STOP_REASON_MAP = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "refusal",
        "function_call": "end_turn",
    }
    
    # Anthropic -> OpenAI 停止原因映射
    REVERSE_STOP_REASON_MAP = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
        "pause_turn": "stop",
        "refusal": "content_filter",
    }
    
    # Anthropic tool_choice -> OpenAI tool_choice 映射
    TOOL_CHOICE_MAP = {
        "auto": "auto",
        "any": "required",
        "none": "none",
    }
    
    # Anthropic 服务层级映射
    SERVICE_TIER_MAP = {
        "auto": "auto",
        "standard_only": "default",
    }
    
    # ================================================================
    # 请求转换：Anthropic -> OpenAI Chat
    # ================================================================
    
    @classmethod
    def to_openai_chat(cls, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anthropic Messages 请求转换为 OpenAI Chat 格式
        
        Args:
            request: Anthropic Messages 格式请求
            
        Returns:
            Dict: OpenAI Chat 格式请求
        """
        messages = []
        
        # 1. 处理 system prompt（Anthropic 顶级参数 -> OpenAI system 消息）
        system_prompt = request.get("system")
        if system_prompt:
            if isinstance(system_prompt, str):
                messages.append({"role": "system", "content": system_prompt})
            elif isinstance(system_prompt, list):
                system_content = cls._convert_system_blocks(system_prompt)
                if system_content:
                    if isinstance(system_content, str):
                        messages.append({"role": "system", "content": system_content})
                    else:
                        # 复杂内容块（含图片等）
                        messages.append({"role": "system", "content": system_content})
        
        # 2. 处理 messages
        anthropic_messages = request.get("messages", [])
        for msg in anthropic_messages:
            converted_list = cls._convert_message(msg)
            messages.extend(converted_list)
        
        # 3. 转换 tools
        tools = cls._convert_tools(request.get("tools", []))
        
        # 4. 转换 tool_choice
        tool_choice = cls._convert_tool_choice(request.get("tool_choice"))
        
        # 5. 构建 Chat 请求
        chat_request = {
            "model": request.get("model", "claude-sonnet-4-20250514"),
            "messages": messages,
        }
        
        # 6. 映射参数
        if request.get("stream") is not None:
            chat_request["stream"] = request["stream"]
        if request.get("temperature") is not None:
            chat_request["temperature"] = request["temperature"]
        if request.get("top_p") is not None:
            chat_request["top_p"] = request["top_p"]
        if request.get("max_tokens") is not None:
            # Anthropic max_tokens -> OpenAI max_completion_tokens (推荐) / max_tokens (兼容)
            chat_request["max_completion_tokens"] = request["max_tokens"]
        if tools:
            chat_request["tools"] = tools
        if tool_choice is not None:
            chat_request["tool_choice"] = tool_choice
        if request.get("stop_sequences"):
            chat_request["stop"] = request["stop_sequences"]
        if request.get("metadata"):
            user_id = request["metadata"].get("user_id")
            if user_id:
                chat_request["user"] = user_id
        if request.get("service_tier"):
            mapped = cls.SERVICE_TIER_MAP.get(request["service_tier"])
            if mapped:
                chat_request["service_tier"] = mapped
        
        # 7. 处理 thinking 参数 -> OpenAI reasoning_effort
        thinking = request.get("thinking")
        if thinking and isinstance(thinking, dict):
            thinking_type = thinking.get("type")
            if thinking_type == "enabled":
                # 启用思考：使用 reasoning_effort (如果后端支持)
                # OpenAI reasoning_effort: none, minimal, low, medium, high, xhigh
                budget = thinking.get("budget_tokens", 0)
                if budget >= 32000:
                    chat_request["reasoning_effort"] = "high"
                elif budget >= 10000:
                    chat_request["reasoning_effort"] = "medium"
                else:
                    chat_request["reasoning_effort"] = "low"
            elif thinking_type == "disabled":
                chat_request["reasoning_effort"] = "none"
        
        # 8. Anthropic 特有参数（OpenAI 不直接支持的，放入 extra_body）
        extra = {}
        if request.get("top_k") is not None:
            extra["top_k"] = request["top_k"]
        # thinking 参数也保留原始值以防后端需要
        if thinking and isinstance(thinking, dict):
            extra["thinking"] = thinking
        if request.get("cache_control") is not None:
            extra["cache_control"] = request["cache_control"]
        if request.get("container") is not None:
            extra["container"] = request["container"]
        if request.get("output_config") is not None:
            extra["output_config"] = request["output_config"]
        if extra:
            chat_request["extra_body"] = extra
        
        return chat_request
    
    @classmethod
    def _convert_system_blocks(cls, blocks: List[Dict]) -> Union[str, List[Dict], None]:
        """转换 Anthropic system 内容块列表"""
        has_complex = False
        text_parts = []
        content_parts = []
        
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")
            
            if block_type == "text":
                text_parts.append(block.get("text", ""))
                content_parts.append({"type": "text", "text": block.get("text", "")})
            else:
                has_complex = True
        
        if has_complex:
            return content_parts if content_parts else None
        elif text_parts:
            return "\n".join(text_parts)
        return None
    
    @classmethod
    def _convert_message(cls, msg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        转换单条 Anthropic 消息，可能返回多条 OpenAI 消息
        
        Anthropic assistant 消息可能包含 tool_use 块，需要拆分：
        - text 块 -> content
        - tool_use 块 -> tool_calls
        - thinking 块 -> 保留思考内容或跳过
        Anthropic user 消息可能包含 tool_result 块，需要拆分为 tool 消息
        """
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role not in cls.ROLE_MAP and role != "user":
            return []
        
        openai_role = cls.ROLE_MAP.get(role, role)
        result = []
        
        if isinstance(content, str):
            # 简单字符串内容
            result.append({"role": openai_role, "content": content})
        
        elif isinstance(content, list):
            # 内容块数组 - 需要分类处理
            text_parts = []
            content_parts = []  # OpenAI 多模态内容块
            tool_calls = []
            tool_results = []
            has_multimodal = False
            
            for block in content:
                block_type = block.get("type", "")
                
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                    content_parts.append({"type": "text", "text": block.get("text", "")})
                
                elif block_type == "tool_use":
                    # assistant 消息中的工具调用
                    tool_calls.append({
                        "id": block.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {}), ensure_ascii=False)
                        }
                    })
                
                elif block_type == "tool_result":
                    # user 消息中的工具结果
                    tool_results.append(block)
                
                elif block_type == "thinking":
                    # thinking 块 - OpenAI 不直接支持相同格式
                    # 跳过，但记录存在（如果后端支持 reasoning_effort 会自动处理）
                    pass
                
                elif block_type == "redacted_thinking":
                    # 脱敏思考块 - 跳过
                    pass
                
                elif block_type == "image":
                    # 图片块 - 正确转换为 OpenAI image_url 格式
                    has_multimodal = True
                    image_url = cls._convert_image_source(block.get("source", {}))
                    content_parts.append({
                        "type": "image_url",
                        "image_url": image_url
                    })
                
                elif block_type == "document":
                    # 文档块 (PDF等) - 转为 OpenAI file 格式或占位
                    has_multimodal = True
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        media_type = source.get("media_type", "application/pdf")
                        data = source.get("data", "")
                        # OpenAI Chat API 支持 file 类型的内容块
                        content_parts.append({
                            "type": "file",
                            "file": {
                                "mime_type": media_type,
                                "file_data": f"data:{media_type};base64,{data}"
                            }
                        })
                    elif source.get("type") == "url":
                        content_parts.append({
                            "type": "file",
                            "file": {
                                "mime_type": source.get("media_type", "application/pdf"),
                                "file_data": source.get("url", "")
                            }
                        })
                    elif source.get("type") == "text":
                        text_parts.append(source.get("text", ""))
                        content_parts.append({"type": "text", "text": source.get("text", "")})
                
                elif block_type == "search_result":
                    # 搜索结果块 - 转为文本表示
                    search_text = block.get("content", "")
                    if isinstance(search_text, str):
                        text_parts.append(f"[Search Result: {search_text}]")
                    elif isinstance(search_content := search_text, list):
                        for item in search_content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_parts.append(f"[Search Result: {item.get('text', '')}]")
                
                elif block_type == "server_tool_use":
                    # 服务器工具调用 - 转为普通工具调用
                    tool_calls.append({
                        "id": block.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {}), ensure_ascii=False)
                        }
                    })
                
                elif block_type == "web_search_tool_result":
                    # 网页搜索工具结果 - 转为工具结果
                    tool_results.append({
                        "tool_use_id": block.get("tool_use_id", ""),
                        "content": block.get("content", ""),
                    })
                
                elif block_type == "web_fetch_tool_result":
                    # 网页获取工具结果 - 转为工具结果
                    tool_results.append({
                        "tool_use_id": block.get("tool_use_id", ""),
                        "content": block.get("content", ""),
                    })
                
                elif block_type == "code_execution_tool_result":
                    # 代码执行工具结果 - 转为工具结果
                    tool_results.append({
                        "tool_use_id": block.get("tool_use_id", ""),
                        "content": block.get("content", ""),
                    })
            
            # 构建 assistant 消息（可能带 tool_calls）
            if role == "assistant":
                assistant_msg = {"role": "assistant"}
                if has_multimodal and content_parts:
                    # 多模态内容
                    text_content_parts = [p for p in content_parts if p.get("type") == "text"]
                    if text_content_parts:
                        assistant_msg["content"] = "\n".join(p.get("text", "") for p in text_content_parts)
                    else:
                        assistant_msg["content"] = None
                elif text_parts:
                    assistant_msg["content"] = "\n".join(text_parts)
                else:
                    assistant_msg["content"] = None
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                result.append(assistant_msg)
            
            # 构建 user 消息
            elif role == "user":
                if has_multimodal and content_parts:
                    # 多模态内容使用 OpenAI 的 content 数组格式
                    result.append({"role": "user", "content": content_parts})
                elif text_parts:
                    result.append({"role": "user", "content": "\n".join(text_parts)})
                
                # tool_result -> role=tool 消息
                for tr in tool_results:
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                    }
                    tr_content = tr.get("content", "")
                    if isinstance(tr_content, str):
                        tool_msg["content"] = tr_content
                    elif isinstance(tr_content, list):
                        # 提取文本
                        text_list = []
                        for b in tr_content:
                            if isinstance(b, dict) and b.get("type") == "text":
                                text_list.append(b.get("text", ""))
                        tool_msg["content"] = "\n".join(text_list) if text_list else ""
                    else:
                        tool_msg["content"] = str(tr_content)
                    result.append(tool_msg)
        
        else:
            result.append({"role": openai_role, "content": str(content)})
        
        return result
    
    @classmethod
    def _convert_image_source(cls, source: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换 Anthropic 图片源为 OpenAI image_url 格式
        
        Anthropic source types:
        - base64: {"type": "base64", "media_type": "...", "data": "..."}
        - url: {"type": "url", "url": "..."}
        
        OpenAI image_url format:
        - {"url": "data:image/png;base64,..." | "https://..."}
        """
        source_type = source.get("type", "")
        
        if source_type == "base64":
            media_type = source.get("media_type", "image/png")
            data = source.get("data", "")
            return {"url": f"data:{media_type};base64,{data}"}
        elif source_type == "url":
            return {"url": source.get("url", "")}
        
        return {"url": ""}
    
    @classmethod
    def _convert_tools(cls, tools: List[Dict]) -> List[Dict]:
        """
        转换 Anthropic 工具定义到 OpenAI 格式
        
        Anthropic: {"name", "description", "input_schema"}
        OpenAI: {"type": "function", "function": {"name", "description", "parameters"}}
        """
        converted = []
        for tool in tools:
            if "name" in tool:
                func_def = {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                }
                # Anthropic 用 input_schema, OpenAI 用 parameters
                if "input_schema" in tool:
                    func_def["parameters"] = tool["input_schema"]
                converted.append({
                    "type": "function",
                    "function": func_def
                })
        return converted
    
    @classmethod
    def _convert_tool_choice(cls, tool_choice: Any) -> Optional[Any]:
        """
        转换 Anthropic tool_choice 到 OpenAI 格式
        
        Anthropic: "auto" | "any" | "none" | {"type": "tool", "name": "..."}
        OpenAI: "auto" | "none" | "required" | {"type": "function", "function": {"name": "..."}}
        """
        if tool_choice is None:
            return None
        
        if isinstance(tool_choice, str):
            return cls.TOOL_CHOICE_MAP.get(tool_choice, tool_choice)
        
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "tool":
                return {
                    "type": "function",
                    "function": {"name": tool_choice.get("name", "")}
                }
            if tool_choice.get("type") == "auto":
                return "auto"
            if tool_choice.get("type") == "any":
                return "required"
        
        return None
    
    # ================================================================
    # 响应转换：OpenAI Chat -> Anthropic
    # ================================================================
    
    @classmethod
    def from_openai_chat(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI Chat 响应转换为 Anthropic 格式
        
        OpenAI: {"id", "model", "choices": [{"message": {"content", "tool_calls"}, "finish_reason"}], "usage"}
        Anthropic: {"id", "type", "role", "content": [{"type": "text"}|{"type": "tool_use"}], 
                    "model", "stop_reason", "stop_sequence", "stop_details", "usage"}
        """
        choices = response.get("choices", [])
        content_blocks = []
        stop_reason = "end_turn"
        stop_details = None
        
        for choice in choices:
            message = choice.get("message", {})
            content = message.get("content")
            tool_calls = message.get("tool_calls", [])
            
            # 处理文本内容
            if content:
                content_blocks.append({
                    "type": "text",
                    "text": content
                })
            
            # 处理工具调用
            for tc in tool_calls:
                tc_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    args_obj = json.loads(args_str)
                except (json.JSONDecodeError, TypeError):
                    args_obj = {}
                
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc_id,
                    "name": func.get("name", ""),
                    "input": args_obj
                })
            
            # 获取停止原因
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                stop_reason = cls.STOP_REASON_MAP.get(finish_reason, "end_turn")
            
            # 处理 refusal 内容
            refusal = message.get("refusal")
            if refusal:
                stop_reason = "refusal"
                stop_details = {"reason": "content_policy", "message": refusal}
        
        # 如果没有任何内容块，添加空文本
        if not content_blocks:
            content_blocks.append({"type": "text", "text": ""})
        
        # 转换 usage - 包含完整的 Anthropic usage 字段
        usage = response.get("usage", {})
        
        return {
            "id": response.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": response.get("model", ""),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": stop_details,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cache_creation_input_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) if isinstance(usage.get("prompt_tokens_details"), dict) else 0,
                "cache_read_input_tokens": 0,
            }
        }
    
    @classmethod
    def _map_stop_reason(cls, openai_reason: str) -> str:
        """映射 OpenAI 停止原因到 Anthropic"""
        return cls.STOP_REASON_MAP.get(openai_reason, "end_turn")
    
    # ================================================================
    # 流式转换：OpenAI Chat Stream -> Anthropic SSE
    # ================================================================
    
    @classmethod
    def convert_stream_chunk(cls, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换 OpenAI Chat 流式块到 Anthropic SSE 格式
        
        Anthropic SSE 事件类型（严格按照官方 SDK 定义）:
        - message_start: 包含完整 message 对象（id, type, role, model, usage）
        - content_block_start: 新内容块开始 {"type": "text"|"tool_use", ...}
        - content_block_delta: 内容增量 {"type": "text_delta"|"input_json_delta"|"thinking_delta", ...}
        - content_block_stop: 内容块结束
        - message_delta: 消息级增量 (stop_reason, stop_details, stop_sequence, usage)
        - message_stop: 消息结束
        - ping: 心跳
        
        注意：Anthropic 的流式事件需要严格按顺序：
        message_start -> [content_block_start -> content_block_delta* -> content_block_stop]* -> message_delta -> message_stop
        """
        choices = chunk.get("choices", [])
        
        if not choices:
            return {"type": "ping"}
        
        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")
        
        # 判断事件类型
        if delta.get("role") == "assistant":
            # 第一个块 - message_start
            usage = chunk.get("usage", {})
            return {
                "type": "message_start",
                "message": {
                    "id": chunk.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": chunk.get("model", ""),
                    "stop_reason": None,
                    "stop_sequence": None,
                    "stop_details": None,
                    "usage": {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    }
                }
            }
        
        # 处理 thinking/reasoning 内容（如果后端返回了 reasoning_token）
        # OpenAI 在 stream_options.include_usage 时可能在 delta 中包含 reasoning
        reasoning_content = delta.get("reasoning_content")
        if reasoning_content is not None:
            return {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "thinking_delta",
                    "thinking": reasoning_content
                }
            }
        
        if delta.get("content") is not None and delta["content"] != "":
            # 文本增量
            return {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": delta["content"]
                }
            }
        
        tool_calls = delta.get("tool_calls", [])
        if tool_calls:
            tc = tool_calls[0]
            tc_idx = tc.get("index", 0)
            tc_func = tc.get("function", {})
            
            # 如果有 id 和 name，是 tool_use 开始 -> content_block_start
            if tc.get("id"):
                return {
                    "type": "content_block_start",
                    "index": tc_idx + 1,  # +1 因为 text 通常是 index 0
                    "content_block": {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc_func.get("name", ""),
                        "input": {}
                    }
                }
            
            # 否则是 input_json_delta -> content_block_delta
            args_part = tc_func.get("arguments", "")
            if args_part:
                return {
                    "type": "content_block_delta",
                    "index": tc_idx + 1,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": args_part
                    }
                }
        
        if finish_reason:
            # 消息结束 -> message_delta
            usage = chunk.get("usage", {})
            stop_reason = cls._map_stop_reason(finish_reason)
            
            # 构建 stop_details
            stop_details = None
            if stop_reason == "refusal":
                stop_details = {"reason": "content_policy"}
            
            return {
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                    "stop_details": stop_details,
                },
                "usage": {
                    "output_tokens": usage.get("completion_tokens", 0)
                }
            }
        
        return {"type": "ping"}
    
    # ================================================================
    # 响应转换：Anthropic -> OpenAI Responses
    # ================================================================
    
    @classmethod
    def to_openai_responses(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anthropic 响应转换为 OpenAI Responses 格式
        """
        output = []
        content = response.get("content", [])
        
        for block in content:
            block_type = block.get("type")
            
            if block_type == "text":
                output.append({
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": block.get("text", "")
                    }]
                })
            
            elif block_type == "tool_use":
                output.append({
                    "type": "function_call",
                    "id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "call_id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                    "status": "completed"
                })
        
        usage = response.get("usage", {})
        return {
            "id": response.get("id", f"resp_{uuid.uuid4().hex[:24]}"),
            "object": "response",
            "status": "completed",
            "created_at": time.time(),
            "model": response.get("model", ""),
            "output": output,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            }
        }


class AnthropicStreamingParser:
    """Anthropic 流式响应解析器"""
    
    @staticmethod
    def parse_sse_line(line: str) -> Optional[tuple]:
        """解析 SSE 行"""
        if not line or line.startswith(":"):
            return None
        
        if line.startswith("event:"):
            event_type = line[6:].strip()
            return (event_type, None)
        
        if line.startswith("data:"):
            data = line[5:].strip()
            if data == "[DONE]":
                return None
            try:
                return (None, json.loads(data))
            except json.JSONDecodeError:
                return (None, data)
        
        return None
    
    @staticmethod
    def parse_sse_chunk(chunk: bytes) -> Iterator[Dict[str, Any]]:
        """解析 SSE 块"""
        text = chunk.decode("utf-8")
        for line in text.split("\n"):
            result = AnthropicStreamingParser.parse_sse_line(line)
            if result:
                event_type, data = result
                if data:
                    yield {"event": event_type, "data": data}
