# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Simplified ReAct orchestrator for memory updates - single LLM call with tool use.

Reference: bot/vikingbot/agent/loop.py AgentLoop structure
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Set, Tuple

from openviking.models.vlm.base import VLMBase
from openviking.server.identity import RequestContext
from openviking.session.memory.dataclass import MemoryOperations
from openviking.session.memory.schema_model_generator import (
    SchemaModelGenerator,
)
from openviking.session.memory.tools import (
    MEMORY_TOOLS_REGISTRY,
    add_tool_call_pair_to_messages,
    get_tool,
)
from openviking.session.memory.utils import (
    parse_json_with_stability,
    parse_memory_file_with_fields,
    pretty_print_messages,
    validate_operations_uris,
)
from openviking.storage.viking_fs import VikingFS, get_viking_fs
from openviking_cli.utils import get_logger

logger = get_logger(__name__)


class ExtractLoop:
    """
    Simplified ReAct orchestrator for memory updates.

    Workflow:
    0. Pre-fetch: System performs ls + read .overview.md + search (via strategy)
    1. LLM call with tools: Model decides to either use tools OR output final operations
    2. If tools used: Execute and continue loop
    3. If operations output: Return and finish
    """

    def __init__(
        self,
        vlm: VLMBase,
        viking_fs: Optional[VikingFS] = None,
        model: Optional[str] = None,
        max_iterations: int = 3,
        ctx: Optional[RequestContext] = None,
        context_provider: Optional[Any] = None,  # ExtractContextProvider
        extraction_text_mode: bool = False,
    ):
        """
        Initialize the ExtractLoop.

        Args:
            vlm: VLM instance (from openviking.models.vlm.base)
            viking_fs: VikingFS instance for storage operations
            model: Model name to use
            max_iterations: Maximum number of ReAct iterations (default: 5)
            ctx: Request context
            context_provider: ExtractContextProvider - 必须提供（由 provider 加载 schema）
            extraction_text_mode: If True, never pass `tools=` to the LLM. The
                model emits structured operations as JSON in `content` only,
                parsed via parse_json_with_stability. Used for small / quantized
                models where llama.cpp grammar-constrained sampling wedges on
                the extraction schema.
        """
        self.vlm = vlm
        self.viking_fs = viking_fs or get_viking_fs()
        self.model = model or self.vlm.model
        self.max_iterations = max_iterations
        self.ctx = ctx
        self.context_provider = context_provider
        self.extraction_text_mode = extraction_text_mode

        # Schema 生成器（在 run() 中初始化）
        self.schema_model_generator = None
        self._json_schema = None

        # 预计算：避免每次迭代重复计算
        self._tool_schemas: Optional[List[Dict[str, Any]]] = None
        self._expected_fields: Optional[List[str]] = None
        self._operations_model: Optional[Any] = None

        # Track files read during ReAct for refetch detection
        self._read_files: Set[str] = set()
        # Transaction handle for file locking
        self._transaction_handle = None

    def _build_prefetch_summary(self, tool_call_messages: list) -> str:
        """Build a summary of prefetched context to nudge direct JSON output."""
        # Count what was found in prefetch
        found_items = 0
        for msg in tool_call_messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and "result" in content:
                    found_items += 1

        if found_items == 0:
            return (
                "Memory directories are empty. Analyze the conversation above and output your "
                "extraction JSON directly. Extract ALL new facts as ADD operations."
            )
        return (
            f"Pre-fetched {found_items} existing memory files. They cover PRIOR conversations. "
            "Analyze the conversation above and output your extraction JSON directly. "
            "Extract NEW facts from THIS conversation as ADD operations. Use EDIT only when a "
            "new fact extends an existing entity card."
        )

    @staticmethod
    def _is_placeholder_query(query: str) -> bool:
        """Detect placeholder search queries that small models copy from prompts."""
        placeholders = {"[keywords]", "[keyword]", "keywords", "[query]", "[search]"}
        return query.strip().lower() in placeholders

    def _should_use_compact_schema(self) -> bool:
        """Check if compact schema should be used based on config.

        Opt-in via `memory.small_model_mode` (recommended for ~8B models).
        Falls back to the legacy `memory.compact_schema` flag if present.

        Name-based auto-detection is intentionally disabled: it false-positives
        on capable MoE models like qwen/qwen3.5-35b-a3b that handle the full
        schema fine. Users must opt in explicitly for small-model adaptations.
        """
        from openviking_cli.utils.config import get_openviking_config

        try:
            config = get_openviking_config()
            if getattr(config.memory, "small_model_mode", False):
                return True
            if getattr(config.memory, "compact_schema", False):
                return True
        except Exception:
            pass
        return False

    def _is_small_model_mode(self) -> bool:
        """Returns True if small_model_mode is enabled in config."""
        from openviking_cli.utils.config import get_openviking_config

        try:
            config = get_openviking_config()
            return bool(getattr(config.memory, "small_model_mode", False))
        except Exception:
            return False

    async def run(self) -> Tuple[Optional[MemoryOperations], List[Dict[str, Any]]]:
        """
        Run the simplified ReAct loop for memory updates.

        Returns:
            Tuple of (final MemoryOperations, tools_used list)
        """
        iteration = 0
        max_iterations = self.max_iterations
        final_operations = None
        tools_used: List[Dict[str, Any]] = []

        # 从 provider 获取 schemas（内部自动加载 registry）
        schemas = self.context_provider.get_memory_schemas(self.ctx)

        # 初始化 schema 生成器（使用 schemas 而非 registry）
        self.schema_model_generator = SchemaModelGenerator(schemas)
        self.schema_model_generator.generate_all_models()

        # Compact schema opt-in via memory.small_model_mode (~1200 → ~200 tokens,
        # strips PATCH/SEARCH-REPLACE types). Default off: the full schema keeps
        # field descriptions that large instruction-tuned models benefit from.
        use_compact = self._should_use_compact_schema()
        if use_compact:
            logger.info(
                f"[ExtractLoop] Using compact JSON schema for model={self.model} "
                f"(small_model_mode enabled)"
            )
        self._json_schema = self.schema_model_generator.get_llm_json_schema(
            compact=use_compact
        )

        # 预计算工具 schemas
        allowed_tools = self.context_provider.get_tools()
        self._tool_schemas = [
            tool.to_schema()
            for tool in MEMORY_TOOLS_REGISTRY.values()
            if tool.name in allowed_tools
        ]

        # 预计算 expected_fields
        self._expected_fields = ["reasoning", "edit_overview_uris", "delete_uris"]

        # 获取 ExtractContext（整个流程复用）
        self._extract_context = self.context_provider.get_extract_context()
        if self._extract_context is None:
            raise ValueError("Failed to get ExtractContext from provider")
        for schema in schemas:
            self._expected_fields.append(schema.memory_type)

        # 预计算 operations_model
        self._operations_model = self.schema_model_generator.create_structured_operations_model()

        # Reset read files tracking for this run
        self._read_files.clear()

        # Build initial messages from provider
        import json

        schema_str = json.dumps(self._json_schema, ensure_ascii=False)

        messages = []
        # Single system message (some models reject multiple system messages)
        instruction_text = self.context_provider.instruction()

        # Small-model output-quality preamble: compact schema strips field
        # descriptions to save tokens, which makes small models emit empty
        # `content` fields and skip secondary entities. This block tells them
        # in natural language what the stripped descriptions used to say.
        # Only injected when small_model_mode is on (gated on use_compact).
        quality_block = ""
        if use_compact:
            memory_type_list = ", ".join(s.memory_type for s in schemas) or "(none)"
            quality_block = f"""

## Extraction Rules (read before generating)

**WHAT to extract** (conversational memories):
- `entities`: one card per named person (BOTH speakers, separately — never merge speakers or cross-attribute facts). Also significant objects, places, organizations mentioned.
- `events`: things that happened, with dates. Required `summary` (who did what, where, outcome, ~80 chars) + `goal` + `date`.
- `preferences`: stated likes, habits, values. One per speaker-topic pair.
- `episodes`: high-level narrative summary of this session.
- `profile`: brief user-level summary.

**LEAVE EMPTY — do NOT fabricate these** (they are for AI-agent tool usage, not human conversation):
- `skills`: `[]`
- `tools`: `[]`
- `cases`: `[]`
- `patterns`: `[]`
Never invent `call_count`, `success_time`, `problem`/`solution` from conversational content. If the conversation has no tool calls or agent actions, these stay empty arrays.

**Field format** (compact schema strips descriptions — follow these rules):
- `date`: "YYYY-MM-DD". Convert "yesterday"/"last week" to absolute using session date.
- `ranges`: message INDEX range, format "start-end" like "0-10". NEVER a date. If unsure, use "0-999".
- `event_name`: lowercase_with_underscores, ≤3 words, no dates in name.
- Text fields (`content`, `summary`, `goal`): be specific but concise, ~80–150 chars. Never empty/null — if you can't fill it, skip the whole memory item.

**Speaker attribution**: `[Name]:` prefixes identify who said what. Facts belong to whoever said them. Do not merge two speakers into one entity.

**Memory types available**: {memory_type_list}

## Worked example (study the SHAPE, use only the real conversation's facts)

Conversation:
```
[Alex]: I started rock climbing last month at the new gym downtown. Loving it!
[Sam]: Wow, nice. I'm more of a cyclist myself — did a 50-mile ride yesterday.
[Alex]: 50 miles?! Insane. The rock gym has a great bouldering wall.
[Sam]: I prefer endurance to bursts. Different brains, I guess.
```

Expected JSON output:
```json
{{
  "reasoning": "Alex started rock climbing recently; Sam is a cyclist. Both have stated hobby preferences. The downtown rock gym is a notable place.",
  "skills": [],
  "cases": [],
  "tools": [],
  "patterns": [],
  "entities": [
    {{"name": "Alex", "content": "Started rock climbing last month at the new downtown rock gym, enjoying it. Prefers high-burst activities."}},
    {{"name": "Sam", "content": "Cyclist who completed a 50-mile ride yesterday. Prefers endurance activities over bursts."}},
    {{"name": "downtown rock gym", "content": "New rock climbing gym downtown where Alex started climbing. Has a great bouldering wall."}}
  ],
  "events": [
    {{"event_name": "alex_started_climbing", "date": "2026-03-08", "goal": "take up a new hobby", "summary": "Alex started rock climbing last month at the new downtown rock gym.", "ranges": "0-3"}},
    {{"event_name": "sam_50_mile_ride", "date": "2026-04-07", "goal": "endurance training", "summary": "Sam completed a 50-mile cycling ride yesterday.", "ranges": "1-3"}}
  ],
  "preferences": [
    {{"user": "Alex", "topic": "rock climbing", "content": "Alex enjoys rock climbing as a recently-started hobby."}},
    {{"user": "Sam", "topic": "endurance cycling", "content": "Sam prefers endurance cycling over high-burst activities."}}
  ],
  "episodes": [
    {{"episode_title": "Comparing fitness hobbies", "session_id": "example", "session_time": "2026-04-08", "content": "Alex shares about starting rock climbing at the new downtown gym. Sam contrasts with cycling, having just done a 50-mile ride."}}
  ],
  "profile": {{"content": "Alex is a new rock climber. Sam is an endurance cyclist."}},
  "edit_overview_uris": [],
  "delete_uris": []
}}
```

The example uses fictional names (Alex, Sam, rock climbing, cycling, gym, 50-mile ride). Match its SHAPE — 3 entities (both speakers + a secondary place), 2 events, 2 preferences, 1 episode, 1 profile, all 4 agent categories empty — but with facts from the REAL conversation below. Never copy "Alex", "Sam", "rock climbing", "cycling", "50-mile ride" or "gym" into your output; those are example data only.

Output ONLY a single JSON object — no schema echo, no prose."""

        messages.append(
            {
                "role": "system",
                "content": f"""{instruction_text}

## Output Format
See the complete JSON Schema below:
```json
{schema_str}
```{quality_block}""",
            }
        )

        await self._mark_cache_breakpoint(messages)
        # Pre-fetch context via provider
        tool_call_messages = await self.context_provider.prefetch(
            ctx=self.ctx,
            viking_fs=self.viking_fs,
            transaction_handle=self._transaction_handle,
            vlm=self.vlm,
        )
        messages.extend(tool_call_messages)

        # Add a summary nudge after prefetch to encourage direct JSON output
        prefetch_summary = self._build_prefetch_summary(tool_call_messages)
        if prefetch_summary:
            messages.append({
                "role": "user",
                "content": prefetch_summary,
            })

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"ReAct iteration {iteration}/{max_iterations}")

            # Check if this is the last iteration - force final result
            is_last_iteration = iteration >= max_iterations

            # If last iteration, add a message telling the model to return result directly
            if is_last_iteration:
                messages.append(
                    {
                        "role": "user",
                        "content": "You have reached the maximum number of tool call iterations. Do not call any more tools - return your final result directly now.",
                    }
                )

            # Call LLM with tools - model decides: tool calls OR final operations
            pretty_print_messages(messages)
            tool_calls, operations = await self._call_llm(messages, force_final=is_last_iteration)

            if tool_calls:
                await self._execute_tool_calls(messages, tool_calls, tools_used)
                continue

            # If model returned final operations, check if refetch is needed
            if operations is not None:
                # Check if any write_uris target existing files that weren't read
                refetch_uris = await self._check_unread_existing_files(operations)
                if refetch_uris:
                    logger.info(f"Found unread existing files: {refetch_uris}, refetching...")
                    # Add refetch results to messages and continue loop
                    await self._add_refetch_results_to_messages(messages, refetch_uris)
                    # Allow one extra iteration for refetch (max 2 extensions total)
                    hard_cap = self.max_iterations + 2
                    if iteration >= max_iterations and max_iterations < hard_cap:
                        max_iterations += 1
                        logger.info(f"Extended max_iterations to {max_iterations} for refetch (hard cap: {hard_cap})")

                    continue

                final_operations = operations
                break
            # If no tool calls either, continue to next iteration (don't break!)
            logger.warning(
                f"LLM returned neither tool calls nor operations (iteration {iteration}/{max_iterations})"
            )
            # If it's the last iteration, use empty operations
            if is_last_iteration:
                final_operations = MemoryOperations()
                break
            # Otherwise continue and try again
            continue

        if final_operations is None:
            if iteration >= max_iterations:
                raise RuntimeError(f"Reached {max_iterations} iterations without completion")
            else:
                raise RuntimeError("ReAct loop completed but no operations generated")

        logger.info(f"final_operations={final_operations.model_dump_json(indent=4)}")

        return final_operations, tools_used

    async def _execute_tool_calls(self, messages, tool_calls, tools_used):
        # Filter out placeholder tool calls (small models copy "[Keywords]" from prompts)
        valid_tool_calls = []
        for tc in tool_calls:
            if tc.name == "search" and tc.arguments:
                query = tc.arguments.get("query", "")
                if self._is_placeholder_query(query):
                    logger.warning(f"Skipping placeholder search query: '{query}'")
                    add_tool_call_pair_to_messages(
                        messages, call_id=tc.id, tool_name=tc.name,
                        params=tc.arguments,
                        result={"error": "Use specific keywords from the conversation, not placeholders like '[Keywords]'."},
                    )
                    continue
            valid_tool_calls.append(tc)

        if not valid_tool_calls:
            # All tool calls were placeholders — nudge model to output JSON
            messages.append({
                "role": "user",
                "content": "Your search queries were placeholders. Stop making tool calls — output your extraction JSON directly now.",
            })
            return

        # Execute all valid tool calls in parallel
        async def execute_single_tool_call(idx: int, tool_call):
            """Execute a single tool call."""
            result = await self._execute_tool(tool_call)
            return idx, tool_call, result

        action_tasks = [
            execute_single_tool_call(idx, tool_call) for idx, tool_call in enumerate(valid_tool_calls)
        ]
        results = await self._execute_in_parallel(action_tasks)

        # Process results and add to messages
        for _idx, tool_call, result in results:
            # Skip if arguments is None
            if tool_call.arguments is None:
                logger.warning(f"Tool call {tool_call.name} has no arguments, skipping")
                continue

            tools_used.append(
                {
                    "tool_name": tool_call.name,
                    "params": tool_call.arguments,
                    "result": result,
                }
            )

            # Track read tool calls for refetch detection
            if tool_call.name == "read" and tool_call.arguments.get("uri"):
                self._read_files.add(tool_call.arguments["uri"])

            add_tool_call_pair_to_messages(
                messages,
                call_id=tool_call.id,
                tool_name=tool_call.name,
                params=tool_call.arguments,
                result=result,
            )

    def _validate_operations(self, operations: MemoryOperations) -> None:
        """
        Validate that all operations have allowed URIs.

        Args:
            operations: The MemoryOperations to validate

        Raises:
            ValueError: If any operation has a disallowed URI
        """
        # Get registry from provider (internal method)
        registry = self.context_provider._get_registry()
        schemas = self.context_provider.get_memory_schemas(self.ctx)

        # Use pre-initialized extract_context
        if not hasattr(self, '_extract_context') or self._extract_context is None:
            raise ValueError("ExtractContext not initialized")

        is_valid, errors = validate_operations_uris(
            operations,
            schemas,
            registry,
            user_space="default",
            agent_space="default",
            extract_context=self._extract_context,
        )
        if not is_valid:
            error_msg = "Invalid memory operations:\n" + "\n".join(f"  - {err}" for err in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

    async def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        force_final: bool = False,
    ) -> Tuple[Optional[List], Optional[MemoryOperations]]:
        """
        Call LLM with tools. Returns either tool calls OR final operations.

        Args:
            messages: Message list
            force_final: If True, force model to return final result (not tool calls)

        Returns:
            Tuple of (tool_calls, operations) - one will be None, the other set
        """
        # 标记 cache breakpoint
        await self._mark_cache_breakpoint(messages)

        # Call LLM with tools - use tools from strategy
        tool_choice = "none" if force_final else None

        call_kwargs: dict = dict(messages=messages)
        # Skip tools entirely on the last iteration (force_final) OR when
        # extraction_text_mode is enabled. In text mode the model is expected
        # to output structured operations as JSON in `content`, parsed via
        # parse_json_with_stability — no tool calls, no llama.cpp grammar
        # constraints, no wedge on small models.
        if force_final or self.extraction_text_mode:
            # Small-model path: ask for JSON-object response format so backends
            # that honor it (OpenAI, llama.cpp) enforce valid JSON output. We
            # tested json_schema strict grammar with Bonsai 1-bit and it
            # collapsed the model to minimum valid output (1 item per session
            # because the schema has no `required` arrays at top level — `{}`
            # satisfies it and the grammar gives the model an easy exit).
            # json_object alone gives the model more breathing room to actually
            # populate memory categories while still avoiding schema-echo.
            if self.extraction_text_mode:
                call_kwargs["response_format"] = {"type": "json_object"}
        else:
            call_kwargs["tools"] = self._tool_schemas
            call_kwargs["tool_choice"] = tool_choice
        # Only pass max_retries if the backend supports it (e.g. Volcengine).
        if hasattr(self.vlm, "max_retries"):
            import inspect
            sig = inspect.signature(self.vlm.get_completion_async)
            if "max_retries" in sig.parameters:
                call_kwargs["max_retries"] = self.vlm.max_retries
        response = await self.vlm.get_completion_async(**call_kwargs)
        # print(f'response={response}')
        # Log cache hit info
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            prompt_tokens = usage.get("prompt_tokens", 0)
            cached_tokens = (
                usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                if isinstance(usage.get("prompt_tokens_details"), dict)
                else 0
            )
            if prompt_tokens > 0:
                cache_hit_rate = (cached_tokens / prompt_tokens) * 100
                logger.info(
                    f"[KVCache] prompt_tokens={prompt_tokens}, cached_tokens={cached_tokens}, cache_hit_rate={cache_hit_rate:.1f}%"
                )
            else:
                logger.info(
                    f"[KVCache] prompt_tokens={prompt_tokens}, cached_tokens={cached_tokens}"
                )

        # Case 1: LLM returned tool calls
        if hasattr(response, "has_tool_calls") and response.has_tool_calls:
            # Format tool calls nicely for debug logging
            for tc in response.tool_calls:
                logger.info(f"[assistant tool_call] (id={tc.id}, name={tc.name})")
                logger.info(f"  {json.dumps(tc.arguments, indent=2, ensure_ascii=False)}")
            return (response.tool_calls, None)

        # Case 2: Try to parse MemoryOperations from content with stability
        content = (response.content if hasattr(response, "content") else str(response)) or ""
        if content:
            try:
                logger.info(f"[assistant] response length={len(content)} chars, first 500: {content[:500]}")
                logger.debug(f"[assistant]\n{content}")

                # Use cached operations_model and expected_fields
                operations, error = parse_json_with_stability(
                    content=content,
                    model_class=self._operations_model,
                    expected_fields=self._expected_fields,
                )

                if error is not None:
                    logger.warning(f"Failed to parse memory operations: {error}")
                    logger.info(f"Raw content (first 1000): {content[:1000]}")
                    return (None, None)

                # Validate URIs — non-fatal, skip invalid operations but keep valid ones
                try:
                    self._validate_operations(operations)
                except Exception as ve:
                    logger.warning(f"URI validation failed (non-fatal, keeping operations): {ve}")

                return (None, operations)
            except Exception as e:
                logger.exception(f"Error parsing operations: {e}")

        # Case 3: No tool calls and no parsable operations
        print("No tool calls or operations parsed")
        return (None, None)

    async def _execute_tool(
        self,
        tool_call,
    ) -> Any:
        """Execute a single read action (read/search/ls/tree)."""
        if not self.viking_fs:
            return {"error": "VikingFS not available"}

        tool = get_tool(tool_call.name)
        if not tool:
            return {"error": f"Unknown tool: {tool_call.name}"}

        # 创建 ToolContext
        from openviking.server.identity import ToolContext

        tool_ctx = ToolContext(request_ctx=self.ctx, transaction_handle=self._transaction_handle)

        try:
            result = await tool.execute(self.viking_fs, tool_ctx, **tool_call.arguments)
            return result
        except Exception as e:
            logger.error(f"Failed to execute {tool_call.name}: {e}")
            return {"error": str(e)}

    async def _execute_in_parallel(
        self,
        tasks: List[Any],
    ) -> List[Any]:
        """Execute tasks in parallel, similar to AgentLoop."""
        return await asyncio.gather(*tasks)

    async def _check_unread_existing_files(
        self,
        operations: MemoryOperations,
    ) -> List[str]:
        """Check if write operations target existing files that weren't read during ReAct."""
        memory_type_fields = getattr(operations, "_memory_type_fields", None)
        if not memory_type_fields:
            return []

        from openviking.session.memory.utils.uri import resolve_flat_model_uri

        registry = self.context_provider._get_registry()
        refetch_uris = []

        for field_name in memory_type_fields:
            value = getattr(operations, field_name, None)
            if value is None:
                continue
            items = value if isinstance(value, list) else [value]
            for item in items:
                # Convert to dict
                item_dict = dict(item) if hasattr(item, "model_dump") else dict(item)
                try:
                    uri = resolve_flat_model_uri(
                        item_dict, registry, "default", "default",
                        memory_type=field_name,
                        extract_context=self._extract_context,
                    )
                except Exception as e:
                    logger.warning(f"Failed to resolve URI for {item}: {e}")
                    continue

                if uri in self._read_files:
                    continue
                try:
                    await self.viking_fs.read_file(uri, ctx=self.ctx)
                    refetch_uris.append(uri)
                except Exception:
                    pass
        return refetch_uris

    async def _add_refetch_results_to_messages(
        self,
        messages: List[Dict[str, Any]],
        refetch_uris: List[str],
    ) -> None:
        """Add existing file content as read tool results to messages."""
        # Calculate call_id based on existing tool messages
        call_id_seq = len([m for m in messages if m.get("role") == "tool"]) + 1000

        for uri in refetch_uris:
            try:
                content = await self.viking_fs.read_file(uri, ctx=self.ctx)
                parsed = parse_memory_file_with_fields(content)

                # Add as read tool call + result
                add_tool_call_pair_to_messages(
                    messages=messages,
                    call_id=call_id_seq,
                    tool_name="read",
                    params={"uri": uri},
                    result=parsed,
                )
                call_id_seq += 1

                # Mark as read
                self._read_files.add(uri)
            except Exception as e:
                logger.warning(f"Failed to refetch {uri}: {e}")

        # Add reminder message for the model
        messages.append(
            {
                "role": "user",
                "content": "Note: The files above were automatically read because they exist and you didn't read them before deciding to write. Please consider the existing content when making write decisions. You can now output updated operations.",
            }
        )

    async def _mark_cache_breakpoint(self, messages):
        # 支持 dict 消息和 object 消息
        last_msg = messages[-1]
        last_msg["cache_control"] = {"type": "ephemeral"}
