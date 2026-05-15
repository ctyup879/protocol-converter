"""
Anthropic Messages API 协议转换器

参考: https://docs.anthropic.com/en/api/messages
参考 SDK: anthropic-sdk-python

Anthropic Messages API 请求参数:
- model (必填): 模型名称
- max_tokens (必填): 最大生成 token 数 (设为0仅预热缓存)
- messages (必填): 消息列表
- system: 系统提示词 (str | TextBlockParam[])
- tools: 工具定义 (ToolUnionParam[])
  - 客户端工具: {"name", "description", "input_schema"}
  - 服务器工具: type 为 bash_20250124, text_editor_*, code_execution_*, web_search_*, web_fetch_*, memory_*, tool_search_*
- tool_choice: 工具选择策略 ("auto" | "any" | "none" | {"type": "tool", "name": "..."})
- temperature: 温度 (0.0-1.0, 默认 1.0)
- top_p: nucleus 采样
- top_k: Top-K 采样
- stop_sequences: 停止序列
- stream: 是否流式
- thinking: 扩展思考配置
  - {"type": "enabled", "budget_tokens": N, "display": "summarized"|"omitted"}
    - display: "summarized" = 正常返回thinking, "omitted" = 脱敏但返回signature
  - {"type": "disabled"}
  - {"type": "adaptive", "budget_tokens": N, "display": "summarized"|"omitted"}
- metadata: 元数据 {"user_id": "..."}
- cache_control: 缓存控制 (CacheControlEphemeralParam)
- service_tier: 服务层级 ("auto" | "standard_only")
- container: 容器标识
- output_config: 输出配置
- inference_geo: 推理地理区域

Anthropic 消息内容块类型:
- text: 文本
- image: 图片 (source: base64 | url)
- document: 文档 (PDF等, source: base64 | url | text)
- tool_use: 工具调用
- tool_result: 工具结果
- thinking: 扩展思考 (含 thinking + signature)
- redacted_thinking: 脱敏思考 (仅含 signature)
- server_tool_use: 服务器工具调用
- web_search_tool_result: 网页搜索结果
- web_fetch_tool_result: 网页获取结果
- code_execution_tool_result: 代码执行结果
- bash_code_execution_tool_result: Bash代码执行结果
- text_editor_code_execution_tool_result: 文本编辑器代码执行结果
- tool_search_tool_result: 工具搜索结果
- container_upload: 容器上传
- search_result: 搜索结果

Anthropic 流式事件类型 (严格顺序):
  message_start -> [content_block_start -> content_block_delta* -> content_block_stop]* -> message_delta -> message_stop
- message_start: 消息开始 (包含完整 message 对象含 usage)
- content_block_start: 内容块开始 (含 index + content_block)
- content_block_delta: 内容增量 (含 index + delta)
  - text_delta: {type: "text_delta", text: "..."}
  - input_json_delta: {type: "input_json_delta", partial_json: "..."}
  - thinking_delta: {type: "thinking_delta", thinking: "..."}
  - signature_delta: {type: "signature_delta", signature: "..."}
  - citations_delta: {type: "citations_delta", citation: {...}}
- content_block_stop: 内容块结束 (含 index)
- message_delta: 消息级增量 (delta: stop_reason, stop_sequence, stop_details + usage: output_tokens)
- message_stop: 消息结束
- ping: 心跳
- error: 错误

Anthropic 响应字段:
- id, type, role, content, model, stop_reason, stop_sequence, stop_details, usage, container
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
    
    # Anthropic 请求服务层级映射 (Anthropic request -> Chat)
    SERVICE_TIER_MAP = {
        "auto": "auto",
        "standard_only": "default",
    }
    
    # Anthropic 响应 usage 服务层级映射 (Anthropic response usage -> Chat)
    # Anthropic response: "standard", "priority", "batch"
    # Chat response: "auto", "default", "flex", "scale", "priority"
    RESPONSE_SERVICE_TIER_MAP = {
        "standard": "default",
        "priority": "priority",
        "batch": "default",  # Anthropic batch 无 Chat 等价项，映射为 default
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
        raw_tools = request.get("tools", [])
        tools = cls._convert_tools(raw_tools)
        
        # 3.5 保留 Anthropic 服务器工具在 extra_body 中
        server_tools = []
        ANTHROPIC_SERVER_TOOL_PREFIXES = (
            "bash_", "text_editor_", "code_execution_",
            "web_search_", "web_fetch_", "memory_",
            "tool_search_",
        )
        for tool in raw_tools:
            if isinstance(tool, dict):
                tool_type = tool.get("type", "")
                if any(tool_type.startswith(prefix) for prefix in ANTHROPIC_SERVER_TOOL_PREFIXES):
                    server_tools.append(tool)
        
        # 4. 转换 tool_choice（同时提取 disable_parallel_tool_use）
        tool_choice, parallel_tool_calls_from_tc = cls._convert_tool_choice(request.get("tool_choice"))
        
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
        if parallel_tool_calls_from_tc is not None:
            chat_request["parallel_tool_calls"] = parallel_tool_calls_from_tc
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
                budget = thinking.get("budget_tokens", 0)
                if budget >= 32000:
                    chat_request["reasoning_effort"] = "high"
                elif budget >= 10000:
                    chat_request["reasoning_effort"] = "medium"
                else:
                    chat_request["reasoning_effort"] = "low"
            elif thinking_type == "adaptive":
                # adaptive 类型 - 模型自主决定是否思考，无 budget_tokens 字段
                # 参考 SDK: ThinkingConfigAdaptiveParam 仅有 type + display，无 budget_tokens
                # 对于 adaptive，模型自行决定推理深度，映射为 "medium"（o系列模型默认值）
                budget = thinking.get("budget_tokens")
                if budget is not None:
                    if budget >= 32000:
                        chat_request["reasoning_effort"] = "high"
                    elif budget >= 10000:
                        chat_request["reasoning_effort"] = "medium"
                    else:
                        chat_request["reasoning_effort"] = "low"
                else:
                    # adaptive 无 budget_tokens，默认 medium
                    chat_request["reasoning_effort"] = "medium"
            elif thinking_type == "disabled":
                chat_request["reasoning_effort"] = "none"
        
        # 8. Anthropic 特有参数（OpenAI 不直接支持的，放入 extra_body）
        extra = {}
        if request.get("top_k") is not None:
            extra["top_k"] = request["top_k"]
        # thinking 参数保留原始值（含 display, adaptive 等子字段）以防后端需要
        # display: "summarized" | "omitted" - 控制 thinking 内容的显示方式
        if thinking and isinstance(thinking, dict):
            extra["thinking"] = thinking
        if request.get("cache_control") is not None:
            extra["cache_control"] = request["cache_control"]
        if request.get("container") is not None:
            extra["container"] = request["container"]
        if request.get("output_config") is not None:
            extra["output_config"] = request["output_config"]
        if request.get("inference_geo") is not None:
            extra["inference_geo"] = request["inference_geo"]
        if request.get("metadata") is not None:
            # 保留完整 metadata (不仅仅是 user_id)
            extra["metadata"] = request["metadata"]
        # 保留 Anthropic 服务器工具
        if server_tools:
            extra["anthropic_server_tools"] = server_tools
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
                    text_part = {"type": "text", "text": block.get("text", "")}
                    # 保留 citations 字段（Anthropic SDK TextBlock 支持 citations）
                    if block.get("citations"):
                        text_part["citations"] = block["citations"]
                    content_parts.append(text_part)
                
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
                    # Anthropic tool_result 支持 is_error 字段
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
                        # Chat file part 仅支持 file_data/file_id/filename；不支持 mime_type。
                        file_obj = {
                            "file_data": f"data:{media_type};base64,{data}"
                        }
                        if block.get("title"):
                            file_obj["filename"] = block["title"]
                        content_parts.append({
                            "type": "file",
                            "file": file_obj
                        })
                    elif source.get("type") == "url":
                        # Chat Completions 不支持 file_url，降级为文本引用。
                        file_url_text = f"[File URL: {source.get('url', '')}]"
                        text_parts.append(file_url_text)
                        content_parts.append({"type": "text", "text": file_url_text})
                    elif source.get("type") == "text":
                        text_parts.append(source.get("text", ""))
                        content_parts.append({"type": "text", "text": source.get("text", "")})
                
                elif block_type == "search_result":
                    # 搜索结果块 - 转为文本表示
                    search_text = block.get("content", "")
                    if isinstance(search_text, str):
                        text_parts.append(f"[Search Result: {search_text}]")
                        content_parts.append({"type": "text", "text": f"[Search Result: {search_text}]"})
                    elif isinstance(search_content := search_text, list):
                        for item in search_content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                sr_text = f"[Search Result: {item.get('text', '')}]"
                                text_parts.append(sr_text)
                                content_parts.append({"type": "text", "text": sr_text})
                
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
                
                elif block_type == "bash_code_execution_tool_result":
                    # Bash代码执行工具结果 - 转为工具结果
                    tool_results.append({
                        "tool_use_id": block.get("tool_use_id", ""),
                        "content": block.get("content", ""),
                    })
                
                elif block_type == "text_editor_code_execution_tool_result":
                    # 文本编辑器代码执行工具结果 - 转为工具结果
                    tool_results.append({
                        "tool_use_id": block.get("tool_use_id", ""),
                        "content": block.get("content", ""),
                    })
                
                elif block_type == "tool_search_tool_result":
                    # 工具搜索结果 - 转为工具结果
                    tool_results.append({
                        "tool_use_id": block.get("tool_use_id", ""),
                        "content": block.get("content", ""),
                    })
                
                elif block_type == "container_upload":
                    # 容器上传块 - 转为文本表示
                    text_parts.append(f"[Container upload: {block.get('id', '')}]")
            
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
                    # 保留 is_error 标记（Chat API 无直接等价，用 content 前缀标记）
                    if tr.get("is_error"):
                        tool_msg["content"] = f"[Error] {tool_msg['content']}"
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
        Anthropic 服务器工具: {"type": "bash_20250124", ...}, {"type": "web_search_20250305", ...} 等
        OpenAI: {"type": "function", "function": {"name", "description", "parameters"}}
        """
        # Anthropic 服务器工具类型 (type 格式: name_YYYYMMDD)
        ANTHROPIC_SERVER_TOOL_PREFIXES = (
            "bash_", "text_editor_", "code_execution_",
            "web_search_", "web_fetch_", "memory_",
            "tool_search_",
        )
        
        converted = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            
            tool_type = tool.get("type", "")
            
            # 检查是否为 Anthropic 服务器工具 (type 为 name_YYYYMMDD 格式)
            is_server_tool = any(tool_type.startswith(prefix) for prefix in ANTHROPIC_SERVER_TOOL_PREFIXES)
            
            if is_server_tool:
                # 服务器工具 - 放入 extra_body 保留原始定义
                continue
            
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
    def _convert_tool_choice(cls, tool_choice: Any) -> tuple:
        """
        转换 Anthropic tool_choice 到 OpenAI 格式
        
        Anthropic: "auto" | "any" | "none" | {"type": "tool", "name": "..."}
        Anthropic dict 类型还支持 disable_parallel_tool_use: bool
        
        OpenAI: "auto" | "none" | "required" | {"type": "function", "function": {"name": "..."}}
        
        Returns:
            tuple: (tool_choice_value, parallel_tool_calls_value)
                parallel_tool_calls 为 None 表示不设置，False 表示禁用并行
        """
        if tool_choice is None:
            return None, None
        
        if isinstance(tool_choice, str):
            return cls.TOOL_CHOICE_MAP.get(tool_choice, tool_choice), None
        
        if isinstance(tool_choice, dict):
            disable_parallel = tool_choice.get("disable_parallel_tool_use", False)
            parallel_tool_calls = False if disable_parallel else None
            
            if tool_choice.get("type") == "tool":
                return {
                    "type": "function",
                    "function": {"name": tool_choice.get("name", "")}
                }, parallel_tool_calls
            if tool_choice.get("type") == "auto":
                return "auto", parallel_tool_calls
            if tool_choice.get("type") == "any":
                return "required", parallel_tool_calls
        
        return None, None
    
    # ================================================================
    # 响应转换：OpenAI Chat -> Anthropic
    # ================================================================
    
    @classmethod
    def from_openai_chat(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI Chat 响应转换为 Anthropic 格式
        
        OpenAI: {"id", "model", "choices": [{"message": {"content", "tool_calls", "reasoning_content"}, "finish_reason"}], "usage"}
        Anthropic: {"id", "type", "role", "content": [{"type": "text"}|{"type": "thinking"}|{"type": "tool_use"}], 
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
            reasoning_content = message.get("reasoning_content")
            
            # 处理推理内容 -> thinking 块 (必须在文本块之前)
            # Anthropic SDK: ThinkingBlock 必须包含 thinking + signature 字段
            if reasoning_content:
                content_blocks.append({
                    "type": "thinking",
                    "thinking": reasoning_content,
                    "signature": "",  # Chat API 不提供 signature，设为空字符串
                })
            
            # 处理文本内容（空字符串 "" 也是有效内容，有别于 None）
            if content is not None:
                text_block = {
                    "type": "text",
                    "text": content
                }
                # 保留 annotations -> citations 映射
                if message.get("annotations"):
                    text_block["citations"] = message["annotations"]
                content_blocks.append(text_block)
            
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
        
        anthropic_usage = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        
        # 从 prompt_tokens_details 提取缓存信息
        prompt_tokens_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_tokens_details, dict):
            cached_tokens = prompt_tokens_details.get("cached_tokens", 0)
            if cached_tokens:
                anthropic_usage["cache_read_input_tokens"] = cached_tokens
            # 提取 audio_tokens（Chat API 特有）
            audio_tokens = prompt_tokens_details.get("audio_tokens", 0)
            if audio_tokens:
                anthropic_usage["audio_tokens"] = audio_tokens
        
        # 从 completion_tokens_details 提取推理 token 信息
        completion_tokens_details = usage.get("completion_tokens_details")
        if isinstance(completion_tokens_details, dict):
            reasoning_tokens = completion_tokens_details.get("reasoning_tokens", 0)
            if reasoning_tokens:
                anthropic_usage["output_tokens"] = usage.get("completion_tokens", 0)
            # 提取 audio_tokens 输出
            audio_tokens_output = completion_tokens_details.get("audio_tokens", 0)
            if audio_tokens_output:
                anthropic_usage["audio_output_tokens"] = audio_tokens_output
        
        # 从 Chat usage 额外字段中恢复 Anthropic 特有的 usage 数据（用于往返转换场景）
        if usage.get("cache_creation_input_tokens"):
            anthropic_usage["cache_creation_input_tokens"] = usage["cache_creation_input_tokens"]
        if usage.get("cache_read_input_tokens"):
            anthropic_usage["cache_read_input_tokens"] = usage["cache_read_input_tokens"]
        
        # Chat service_tier -> Anthropic usage.service_tier
        # Chat: "auto"|"default"|"flex"|"scale"|"priority"
        # Anthropic: "standard"|"priority"|"batch"
        chat_service_tier = usage.get("service_tier")
        if chat_service_tier:
            chat_to_anthropic_tier = {
                "default": "standard",
                "auto": "standard",
                "flex": "standard",
                "scale": "standard",
                "priority": "priority",
            }
            anthropic_usage["service_tier"] = chat_to_anthropic_tier.get(
                chat_service_tier, chat_service_tier
            )
        
        result = {
            "id": response.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": response.get("model", ""),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": stop_details,
            "usage": anthropic_usage
        }
        
        # 保留 container 字段（从 Chat usage 或直接来源中恢复）
        if usage.get("container"):
            result["container"] = usage["container"]
        
        return result
    
    @classmethod
    def _map_stop_reason(cls, openai_reason: str) -> str:
        """映射 OpenAI 停止原因到 Anthropic"""
        return cls.STOP_REASON_MAP.get(openai_reason, "end_turn")
    
    # ================================================================
    # 流式转换：OpenAI Chat Stream -> Anthropic SSE
    # ================================================================
    
    # 流式状态跟踪（用于维护 content_block 索引和生命周期）
    _stream_state = {
        "text_block_started": False,
        "text_block_index": 0,
        "tool_block_index": 1,  # text 块默认 index 0, tool 块从 1 开始
        "started_tool_ids": set(),  # 已发出 content_block_start 的 tool id
        "thinking_block_started": False,
        "thinking_block_index": None,  # thinking 块的索引（如果有）
        "next_block_index": 0,  # 下一个可用块索引
    }
    
    @classmethod
    def reset_stream_state(cls):
        """重置流式状态（每次新流式请求开始时调用）"""
        cls._stream_state = {
            "text_block_started": False,
            "text_block_index": 0,
            "tool_block_index": 1,
            "started_tool_ids": set(),
            "thinking_block_started": False,
            "thinking_block_index": None,
            "next_block_index": 0,
        }
    
    @classmethod
    def convert_stream_chunk(cls, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        转换 OpenAI Chat 流式块到 Anthropic SSE 事件列表
        
        Anthropic SSE 事件需要严格按顺序：
        message_start -> [content_block_start -> content_block_delta* -> content_block_stop]* -> message_delta -> message_stop
        
        注意：一个 OpenAI chunk 可能需要生成多个 Anthropic 事件
        （例如：第一个文本 chunk 需要先发 content_block_start 再发 content_block_delta）
        
        Returns:
            List[Dict]: Anthropic SSE 事件列表
        """
        events = []
        choices = chunk.get("choices", [])
        
        if not choices:
            return [{"type": "ping"}]
        
        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")
        
        # 1. message_start 事件
        if delta.get("role") == "assistant":
            cls.reset_stream_state()
            usage = chunk.get("usage", {})
            events.append({
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
            })
            return events
        
        # 2. 处理 thinking/reasoning 内容
        reasoning_content = delta.get("reasoning_content")
        if reasoning_content is not None:
            # 如果还没开始 thinking 块，先发 content_block_start
            if not cls._stream_state["thinking_block_started"]:
                cls._stream_state["thinking_block_started"] = True
                thinking_idx = cls._stream_state["next_block_index"]
                cls._stream_state["thinking_block_index"] = thinking_idx
                cls._stream_state["next_block_index"] = thinking_idx + 1
                events.append({
                    "type": "content_block_start",
                    "index": thinking_idx,
                    "content_block": {
                        "type": "thinking",
                        "thinking": "",
                        "signature": ""  # Anthropic SDK: ThinkingBlock 必须包含 signature
                    }
                })
            events.append({
                "type": "content_block_delta",
                "index": cls._stream_state["thinking_block_index"],
                "delta": {
                    "type": "thinking_delta",
                    "thinking": reasoning_content
                }
            })
            # Anthropic SDK 要求 thinking 流包含 signature_delta 事件
            # 用于多轮对话的 thinking 连续性，Chat API 不提供真实 signature
            events.append({
                "type": "content_block_delta",
                "index": cls._stream_state["thinking_block_index"],
                "delta": {
                    "type": "signature_delta",
                    "signature": ""  # Chat API 不提供 signature
                }
            })
            return events
        
        # 3. 处理文本内容
        if delta.get("content") is not None and delta["content"] != "":
            # 如果 thinking 块已开始但未关闭，先关闭它
            if cls._stream_state["thinking_block_started"]:
                cls._stream_state["thinking_block_started"] = False
                events.append({
                    "type": "content_block_stop",
                    "index": cls._stream_state["thinking_block_index"]
                })
            
            # 如果还没开始文本块，先发 content_block_start
            if not cls._stream_state["text_block_started"]:
                cls._stream_state["text_block_started"] = True
                text_idx = cls._stream_state["next_block_index"]
                cls._stream_state["text_block_index"] = text_idx
                cls._stream_state["next_block_index"] = text_idx + 1
                events.append({
                    "type": "content_block_start",
                    "index": text_idx,
                    "content_block": {
                        "type": "text",
                        "text": ""
                    }
                })
            events.append({
                "type": "content_block_delta",
                "index": cls._stream_state["text_block_index"],
                "delta": {
                    "type": "text_delta",
                    "text": delta["content"]
                }
            })
            return events
        
        # 4. 处理工具调用
        tool_calls = delta.get("tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                tc_idx_raw = tc.get("index", 0)
                tc_func = tc.get("function", {})
                tc_id = tc.get("id", "")
                
                # 如果有 id，是 tool_use 开始 -> content_block_start
                if tc_id and tc_id not in cls._stream_state["started_tool_ids"]:
                    # 如果 thinking 块已开始，先关闭它
                    if cls._stream_state["thinking_block_started"]:
                        cls._stream_state["thinking_block_started"] = False
                        events.append({
                            "type": "content_block_stop",
                            "index": cls._stream_state["thinking_block_index"]
                        })
                    
                    # 如果文本块已开始但还没关闭，先关闭文本块
                    if cls._stream_state["text_block_started"]:
                        cls._stream_state["text_block_started"] = False
                        events.append({
                            "type": "content_block_stop",
                            "index": cls._stream_state["text_block_index"]
                        })
                    
                    # 计算此 tool 块的索引
                    tool_idx = cls._stream_state["next_block_index"]
                    cls._stream_state["started_tool_ids"].add(tc_id)
                    cls._stream_state[f"_tool_idx_{tc_id}"] = tool_idx
                    cls._stream_state["next_block_index"] = tool_idx + 1
                    
                    events.append({
                        "type": "content_block_start",
                        "index": tool_idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc_id,
                            "name": tc_func.get("name", ""),
                            "input": {}
                        }
                    })
                
                # input_json_delta -> content_block_delta
                args_part = tc_func.get("arguments", "")
                if args_part and tc_id:
                    tool_idx = cls._stream_state.get(f"_tool_idx_{tc_id}", tc_idx_raw + 1)
                    events.append({
                        "type": "content_block_delta",
                        "index": tool_idx,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": args_part
                        }
                    })
            
            return events
        
        # 5. 消息结束
        if finish_reason:
            # 处理 content_filter 错误 -> error 事件
            if finish_reason == "content_filter":
                events.append({
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "message": "Content filtered by safety system"
                    }
                })
            
            # 关闭所有未关闭的 content blocks
            if cls._stream_state["thinking_block_started"]:
                cls._stream_state["thinking_block_started"] = False
                events.append({
                    "type": "content_block_stop",
                    "index": cls._stream_state["thinking_block_index"]
                })
            
            if cls._stream_state["text_block_started"]:
                cls._stream_state["text_block_started"] = False
                events.append({
                    "type": "content_block_stop",
                    "index": cls._stream_state["text_block_index"]
                })
            
            # 关闭所有 tool blocks
            for tc_id in cls._stream_state["started_tool_ids"]:
                tool_idx = cls._stream_state.get(f"_tool_idx_{tc_id}", 1)
                events.append({
                    "type": "content_block_stop",
                    "index": tool_idx
                })
            
            # message_delta
            usage = chunk.get("usage", {})
            stop_reason = cls._map_stop_reason(finish_reason)
            
            stop_details = None
            if stop_reason == "refusal":
                stop_details = {"reason": "content_policy"}
            
            events.append({
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                    "stop_details": stop_details,
                },
                "usage": {
                    "output_tokens": usage.get("completion_tokens", 0)
                }
            })
            
            # message_stop 事件（Anthropic SSE 严格序列要求）
            events.append({
                "type": "message_stop"
            })
            
            return events
        
        return [{"type": "ping"}]
    
    # ================================================================
    # 响应转换：Anthropic -> OpenAI Responses
    # ================================================================
    
    @classmethod
    def to_openai_responses(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anthropic 响应转换为 OpenAI Responses 格式
        
        内容块映射:
        - thinking -> reasoning 输出项
        - text -> message 输出项 (output_text)
        - tool_use -> function_call 输出项
        - redacted_thinking -> reasoning 输出项 (仅含 encrypted_content)
        """
        output = []
        content = response.get("content", [])
        
        for block in content:
            block_type = block.get("type")
            
            if block_type == "thinking":
                # thinking 块 -> reasoning 输出项
                reasoning_content = []
                thinking_text = block.get("thinking", "")
                if thinking_text:
                    reasoning_content.append({
                        "type": "reasoning_text",
                        "text": thinking_text
                    })
                reasoning_item = {
                    "type": "reasoning",
                    "id": f"rs_{uuid.uuid4().hex[:24]}",
                    "content": reasoning_content,
                    "summary": [],  # Responses API: summary 是必填字段 (List[Summary])
                    "status": "completed",
                }
                # 如果有 signature，添加 encrypted_content
                # Anthropic SDK ThinkingBlock.signature: str
                signature = block.get("signature")
                if signature:
                    reasoning_item["encrypted_content"] = signature
                output.append(reasoning_item)
            
            elif block_type == "redacted_thinking":
                # 脱敏思考块 -> reasoning 输出项 (仅含 encrypted_content)
                # Anthropic SDK RedactedThinkingBlock: data 字段 (非 signature)
                redacted_data = block.get("data") or block.get("signature")
                if redacted_data:
                    output.append({
                        "type": "reasoning",
                        "id": f"rs_{uuid.uuid4().hex[:24]}",
                        "content": [],
                        "summary": [],
                        "encrypted_content": redacted_data,
                        "status": "completed",
                    })
            
            elif block_type == "text":
                output_text = {
                    "type": "output_text",
                    "text": block.get("text", "")
                }
                # 保留 citations（Responses API output_text 支持 annotations）
                if block.get("citations"):
                    output_text["annotations"] = block["citations"]
                output.append({
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [output_text]
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
            
            elif block_type == "server_tool_use":
                # 服务器工具调用 - 转为 function_call
                output.append({
                    "type": "function_call",
                    "id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "call_id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                    "status": "completed"
                })
        
        # 映射停止原因
        stop_reason = response.get("stop_reason", "end_turn")
        status = "completed"
        incomplete_details = None
        if stop_reason == "max_tokens":
            status = "incomplete"
            incomplete_details = {"reason": "max_output_tokens"}
        elif stop_reason == "refusal":
            status = "incomplete"
            incomplete_details = {"reason": "content_filter"}
        elif stop_reason == "pause_turn":
            status = "incomplete"
            incomplete_details = {"reason": "max_output_tokens"}
        
        usage = response.get("usage", {})
        
        # 构建 Responses usage (含 input_tokens_details 和 output_tokens_details)
        response_usage = {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        }
        
        # input_tokens_details
        cached_tokens = usage.get("cache_read_input_tokens", 0)
        if cached_tokens:
            response_usage["input_tokens_details"] = {"cached_tokens": cached_tokens}
        
        # output_tokens_details - Anthropic cache_creation_input_tokens 不直接映射到 reasoning_tokens
        # 但 Anthropic thinking token 信息可以映射
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        # Anthropic 没有直接暴露 reasoning_tokens，可以从 output_tokens 和 usage 推断
        # 保守处理：不设置 reasoning_tokens 除非有明确的来源
        
        result = {
            "id": response.get("id", f"resp_{uuid.uuid4().hex[:24]}"),
            "object": "response",
            "status": status,
            "created_at": time.time(),
            "model": response.get("model", ""),
            "output": output,
            "usage": response_usage,
        }
        
        if incomplete_details:
            result["incomplete_details"] = incomplete_details
        
        return result


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
