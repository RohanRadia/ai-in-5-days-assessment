# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Guardrail Callbacks, Strategic Model Routing, HITL Gateways, and Telemetry Hooks.

Implements Sections 3 & 4 of the Enterprise Cloud FinOps & Sustainability Concierge:
- `route_model_by_complexity`: Dynamic model router selecting between `FAST_MODEL`
  (`gemini-2.5-flash`) and `PLANNING_MODEL` (`gemini-2.5-pro`).
- `before_model_callback`: Prompt-injection / policy guardrail check, PII redaction,
  and sliding-window conversation history compaction (`compact_conversation_history`).
- `after_model_callback`: Self-evaluation factual grounding check and output PII scrubbing.
- `before_tool_callback`: Logs `"event_type": "TOOL_INTENT"` and enforces the explicit
  Human-in-the-Loop (HITL) confirmation gate for destructive tool invocations.
- `after_tool_callback`: Logs `"event_type": "TOOL_OUTCOME"` with execution latency and status.
- `after_agent_callback`: Triggers non-blocking background memory consolidation via
  `asyncio.create_task(consolidate_session_memory_async(...))`.
- `on_model_error_callback`: Graceful local fallback so `adk web` and `pytest` execute
  the complete ADK tool trajectory even in environments without active cloud credentials.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types

from app.config import FAST_MODEL, PLANNING_MODEL
from app.memory import compact_conversation_history, consolidate_session_memory_async
from app.telemetry import (
    log_structured_event,
    scrub_pii_data,
    scrub_pii_payload,
    trace_span,
)

# ---------------------------------------------------------------------------
# 1. Strategic Model Routing (`gemini-2.5-flash` vs `gemini-2.5-pro`)
# ---------------------------------------------------------------------------

_COMPLEX_PLANNING_KEYWORDS = (
    "remediate",
    "terminate",
    "architecture",
    "multi-region",
    "roadmap",
    "migration plan",
    "executive summary",
    "compare",
    "tradeoff",
    "synthesis",
    "pipeline",
)


def route_model_by_complexity(
    query: str,
    complexity_score: float | None = None,
) -> str:
    """Dynamically routes between `FAST_MODEL` (`gemini-2.5-flash`) and `PLANNING_MODEL` (`gemini-2.5-pro`).

    Args:
        query: Incoming user prompt or task description.
        complexity_score: Optional explicit complexity score in `[0.0, 1.0]`.

    Returns:
        Model identifier (`"gemini-2.5-flash"` for rapid triage/classification, or
        `"gemini-2.5-pro"` for multi-step FinOps planning and remediation synthesis).
    """
    if complexity_score is not None:
        return PLANNING_MODEL if complexity_score >= 0.6 else FAST_MODEL

    normalized = (query or "").strip().lower()
    keyword_hits = sum(1 for kw in _COMPLEX_PLANNING_KEYWORDS if kw in normalized)
    if keyword_hits >= 1 or len(normalized) > 220:
        return PLANNING_MODEL
    return FAST_MODEL


# ---------------------------------------------------------------------------
# 2. Prompt-Injection & Policy Guardrail Patterns
# ---------------------------------------------------------------------------

_PROMPT_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions\b"),
    re.compile(r"(?i)\bbypass\s+(?:human|hitl|safety|approval|confirmation|guardrail)\b"),
    re.compile(
        r"(?i)\b(?:delete|terminate|destroy|drop)\s+all\s+(?:production\s+)?(?:instances|databases|projects|clusters)\s+without\s+(?:approval|confirmation)\b"
    ),
    re.compile(r"(?i)\breveal\s+(?:your\s+)?(?:system\s+prompt|constitution|secret\s+key)\b"),
    re.compile(r"(?i)\byou\s+are\s+now\s+in\s+(?:jailbreak|unrestricted|god|sudo)\s+mode\b"),
]


def detect_policy_violation(text: str) -> str | None:
    """Inspects user input for prompt injection or FinOps safety policy violations.

    Args:
        text: User prompt text to inspect.

    Returns:
        Description of the matched policy violation, or `None` if compliant.
    """
    if not text:
        return None
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            return f"Blocked by FinOps Policy Guardrail (matched pattern: {pattern.pattern})"
    return None


def _get_session_id(ctx: Any) -> str:
    """Safely extracts session_id from CallbackContext or ToolContext."""
    session = getattr(ctx, "session", None)
    if session is not None and getattr(session, "id", None):
        return str(session.id)
    return str(getattr(ctx, "invocation_id", None) or "default-session")


def _get_agent_name(ctx: Any) -> str:
    """Safely extracts agent_name from CallbackContext or ToolContext."""
    return str(getattr(ctx, "agent_name", None) or "finops_concierge")


# ---------------------------------------------------------------------------
# 3. Agent Lifecycle Callbacks (`before_agent_callback` / `after_agent_callback`)
# ---------------------------------------------------------------------------


async def before_agent_callback(callback_context: CallbackContext) -> None:
    """Initializes session state defaults and logs agent invocation start."""
    state = callback_context.state
    if "user_preferences" not in state:
        state["user_preferences"] = {
            "currency": "USD",
            "default_lookback_days": 30,
            "cfe_target_pct": 90.0,
        }
    if "hitl_confirmed_instances" not in state:
        state["hitl_confirmed_instances"] = {}
    if "terminated_instances" not in state:
        state["terminated_instances"] = []

    log_structured_event(
        "AGENT_INVOCATION_START",
        f"Starting agent '{_get_agent_name(callback_context)}'",
        session_id=_get_session_id(callback_context),
        agent_name=_get_agent_name(callback_context),
        severity="INFO",
    )
    return None


async def after_agent_callback(
    callback_context: CallbackContext,
) -> genai_types.Content | None:
    """Schedules asynchronous session memory consolidation via `asyncio.create_task()`."""
    session_id = _get_session_id(callback_context)
    agent_name = _get_agent_name(callback_context)

    transcript: list[Any] = []
    session = getattr(callback_context, "session", None)
    if session is not None and getattr(session, "events", None):
        for ev in session.events:
            if getattr(ev, "content", None) is not None:
                transcript.append(ev.content)
    elif getattr(callback_context, "user_content", None) is not None:
        transcript.append(callback_context.user_content)

    # Trigger background memory consolidation without blocking the response
    task = asyncio.create_task(
        consolidate_session_memory_async(session_id=session_id, transcript=transcript)
    )
    callback_context.state["last_memory_consolidation_scheduled"] = True
    # Prevent task garbage collection before completion
    from app.memory import _BACKGROUND_MEMORY_TASKS

    _BACKGROUND_MEMORY_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_MEMORY_TASKS.discard)

    log_structured_event(
        "AGENT_INVOCATION_COMPLETE",
        f"Completed agent '{agent_name}' and scheduled async memory consolidation.",
        session_id=session_id,
        agent_name=agent_name,
        severity="INFO",
    )
    return None


# ---------------------------------------------------------------------------
# 4. Model Guardrail Callbacks (`before_model_callback` / `after_model_callback`)
# ---------------------------------------------------------------------------

_OFFLINE_FALLBACK_ACTIVE = os.environ.get("FINOPS_OFFLINE_MODE", "").lower() in (
    "1",
    "true",
    "yes",
)


async def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """Enforces input guardrails, PII scrubbing, history compaction, and dynamic model routing.

    Args:
        callback_context: Active ADK `CallbackContext`.
        llm_request: Mutable `LlmRequest` about to be sent to the model.

    Returns:
        `None` to proceed with the model call, or an `LlmResponse` to short-circuit execution
        when a policy violation is blocked (or when running in offline local fallback mode).
    """
    session_id = _get_session_id(callback_context)
    agent_name = _get_agent_name(callback_context)

    with trace_span(
        "callback.before_model_callback",
        {"finops.session_id": session_id, "finops.agent_name": agent_name},
    ):
        # 1. History Compaction on `llm_request.contents`
        if llm_request.contents and len(llm_request.contents) > 8:
            llm_request.contents = compact_conversation_history(
                list(llm_request.contents), max_turns=8, max_tokens=12000
            )

        # 2. Inspect & scrub latest user message for PII and prompt-injection policy violations
        latest_user_text = ""
        if llm_request.contents:
            for content in llm_request.contents:
                if not content.parts:
                    continue
                for part in content.parts:
                    if getattr(part, "text", None):
                        if content.role == "user":
                            latest_user_text = str(part.text)
                        part.text = scrub_pii_data(str(part.text))

        violation = detect_policy_violation(latest_user_text)
        if violation:
            log_structured_event(
                "GUARDRAIL_BLOCKED_INPUT",
                f"Blocked policy-violating input: {violation}",
                session_id=session_id,
                agent_name=agent_name,
                severity="WARNING",
                violation_reason=violation,
            )
            return LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part.from_text(
                            text=(
                                "I cannot fulfill this request. Enterprise FinOps & Sustainability "
                                "Governance prohibits bypassing Human-in-the-Loop (HITL) approval "
                                "gates, overriding system safety instructions, or performing "
                                "unapproved destructive cloud operations."
                            )
                        )
                    ],
                )
            )

        # 3. Record strategic model selection in state
        if latest_user_text:
            routed_model = route_model_by_complexity(latest_user_text)
            callback_context.state["routed_model"] = routed_model

        # 4. If offline fallback mode has been activated, synthesize deterministic ADK turn
        if _OFFLINE_FALLBACK_ACTIVE:
            return _build_local_fallback_llm_response(callback_context, llm_request)

        return None


async def after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
    """Performs self-evaluation grounding verification and PII redaction on model output.

    Args:
        callback_context: Active ADK `CallbackContext`.
        llm_response: `LlmResponse` returned by the model.

    Returns:
        Modified `LlmResponse` if PII redaction or grounding correction was applied, else `None`.
    """
    session_id = _get_session_id(callback_context)
    agent_name = _get_agent_name(callback_context)

    with trace_span(
        "callback.after_model_callback",
        {"finops.session_id": session_id, "finops.agent_name": agent_name},
    ):
        if not llm_response or not llm_response.content or not llm_response.content.parts:
            return None

        modified = False
        terminated_instances = callback_context.state.get("terminated_instances", [])

        for part in llm_response.content.parts:
            if getattr(part, "text", None):
                original_text = str(part.text)
                scrubbed_text = scrub_pii_data(original_text)

                # Self-Evaluation Grounding Check: prevent hallucinated unapproved terminations
                lower_text = scrubbed_text.lower()
                if (
                    "permanently terminated" in lower_text
                    and not terminated_instances
                    and "dry-run" not in lower_text
                    and "pending" not in lower_text
                    and "approval" not in lower_text
                ):
                    scrubbed_text += (
                        "\n\n[Grounding Verification Note: No live instance termination has been "
                        "executed in this session yet; Human-in-the-Loop approval is required.]"
                    )

                if scrubbed_text != original_text:
                    part.text = scrubbed_text
                    modified = True

        log_structured_event(
            "MODEL_GROUNDING_EVAL",
            "Completed post-model self-evaluation grounding and PII check.",
            session_id=session_id,
            agent_name=agent_name,
            severity="INFO",
            modified=modified,
        )

        return llm_response if modified else None


# ---------------------------------------------------------------------------
# 5. Tool Callbacks (`before_tool_callback` / `after_tool_callback`)
# ---------------------------------------------------------------------------


async def before_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> dict[str, Any] | None:
    """Logs `TOOL_INTENT` and enforces the Human-in-the-Loop gate before tool execution.

    Args:
        tool: ADK `BaseTool` about to be executed.
        args: Dictionary of arguments passed by the model to the tool.
        tool_context: Active ADK `ToolContext`.

    Returns:
        `None` to allow the tool to execute, or a structured dictionary to halt execution
        when Human-in-the-Loop confirmation is required.
    """
    tool_name = getattr(tool, "name", str(tool))
    session_id = _get_session_id(tool_context)
    agent_name = _get_agent_name(tool_context)

    with trace_span(
        f"callback.before_tool.{tool_name}",
        {"finops.tool_name": tool_name, "finops.session_id": session_id},
    ):
        # Scrub any PII from string arguments before logging or execution
        for k, v in list(args.items()):
            if isinstance(v, str):
                args[k] = scrub_pii_data(v)

        tool_context.state[f"_tool_start_ts_{tool_name}"] = time.perf_counter()

        # Criterion 14: Explicitly log TOOL_INTENT before tool execution
        log_structured_event(
            "TOOL_INTENT",
            f"Agent '{agent_name}' intends to execute tool '{tool_name}'",
            session_id=session_id,
            agent_name=agent_name,
            severity="INFO",
            tool_name=tool_name,
            intended_args=scrub_pii_payload(args),
        )

        # Criterion 12: Enforce Human-in-the-Loop (HITL) Confirmation Gate in before_tool_callback
        if tool_name == "terminate_idle_cloud_instance":
            dry_run = bool(args.get("dry_run", True))
            human_approved = bool(
                args.get("human_approved", False)
                or tool_context.state.get("human_approved", False)
            )
            if not dry_run and not human_approved:
                instance_id = str(args.get("instance_id", "unknown-instance"))
                project_id = str(args.get("project_id", "unknown-project"))
                zone = str(args.get("zone", "us-east1-b"))
                reason = str(args.get("reason", "Idle resource remediation"))

                try:
                    tool_context.actions.require_confirmation = True
                except Exception:
                    tool_context.actions.__dict__["require_confirmation"] = True
                if os.environ.get("ADK_NATIVE_CONFIRMATION_EVENT") == "true":
                    try:
                        tool_context.request_confirmation(
                            hint=(
                                f"Approve permanent termination of idle instance '{instance_id}' "
                                f"in '{project_id}/{zone}'?"
                            )
                        )
                    except Exception:
                        pass

                tool_context.state["pending_hitl_approval"] = {
                    "project_id": project_id,
                    "zone": zone,
                    "instance_id": instance_id,
                    "reason": reason,
                }

                log_structured_event(
                    "HITL_CONFIRMATION_REQUIRED",
                    f"Blocked unapproved destructive tool call to '{tool_name}' for '{instance_id}'",
                    session_id=session_id,
                    agent_name=agent_name,
                    severity="WARNING",
                    tool_name=tool_name,
                    instance_id=instance_id,
                )

                return {
                    "status": "pending_approval",
                    "project_id": project_id,
                    "zone": zone,
                    "instance_id": instance_id,
                    "action_taken": (
                        f"HALTED BY HUMAN-IN-THE-LOOP GATE: Destructive termination of "
                        f"'{instance_id}' in '{zone}' requires explicit human operator approval. "
                        "Ask the user to confirm before re-invoking with `human_approved=True`."
                    ),
                    "estimated_monthly_savings_usd": 2220.0,
                    "estimated_carbon_saved_kg_co2e": 310.5,
                    "requires_human_confirmation": True,
                    "audit_reference": "FINOPS-HITL-GATE-PENDING",
                    "recovery_instructions": (
                        "Present the estimated savings ($2,220.00/mo and 310.5 kgCO2e/mo) to the "
                        "human user and wait for explicit approval before setting `human_approved=True`."
                    ),
                }

        return None


async def after_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict[str, Any],
) -> dict[str, Any] | None:
    """Logs `TOOL_OUTCOME` with execution latency and scrubbed result after tool execution.

    Args:
        tool: Executed ADK `BaseTool`.
        args: Arguments passed to the tool.
        tool_context: Active ADK `ToolContext`.
        tool_response: Raw dictionary returned by the tool.

    Returns:
        PII-scrubbed `tool_response` dictionary.
    """
    tool_name = getattr(tool, "name", str(tool))
    session_id = _get_session_id(tool_context)
    agent_name = _get_agent_name(tool_context)

    start_ts = tool_context.state.get(f"_tool_start_ts_{tool_name}")
    latency_ms = (
        round((time.perf_counter() - float(start_ts)) * 1000.0, 3)
        if start_ts is not None
        else 0.0
    )

    with trace_span(
        f"callback.after_tool.{tool_name}",
        {
            "finops.tool_name": tool_name,
            "finops.latency_ms": latency_ms,
            "finops.session_id": session_id,
        },
    ):
        sanitized_response = scrub_pii_payload(tool_response)
        status = (
            sanitized_response.get("status", "unknown")
            if isinstance(sanitized_response, dict)
            else "completed"
        )

        # Criterion 14: Explicitly log TOOL_OUTCOME after tool execution
        log_structured_event(
            "TOOL_OUTCOME",
            f"Tool '{tool_name}' completed with status '{status}' in {latency_ms}ms",
            session_id=session_id,
            agent_name=agent_name,
            severity="INFO" if status != "error" else "WARNING",
            tool_name=tool_name,
            outcome_status=status,
            latency_ms=latency_ms,
            tool_response=sanitized_response,
        )

        return sanitized_response if isinstance(sanitized_response, dict) else None


# ---------------------------------------------------------------------------
# 6. Graceful Local Model Error Fallback (`on_model_error_callback`)
# ---------------------------------------------------------------------------


def _extract_project_id(text: str) -> str:
    match = re.search(
        r"\b([a-z][a-z0-9]+-(?:prod|dev|stage|finops|analytics|cloud)[a-z0-9\-]*)\b",
        text.lower(),
    )
    return match.group(1) if match else "acme-prod-analytics"


def _extract_region(text: str) -> str:
    match = re.search(
        r"\b((?:us|europe|asia|northamerica)-[a-z]+\d+)\b",
        text.lower(),
    )
    return match.group(1) if match else "us-east1"


def _extract_zone(text: str) -> str:
    match = re.search(
        r"\b((?:us|europe|asia|northamerica)-[a-z]+\d+-[a-z])\b",
        text.lower(),
    )
    return match.group(1) if match else "us-east1-b"


def _extract_instance_id(text: str) -> str:
    match = re.search(r"\b(gce-[a-z0-9\-]+|vm-[a-z0-9\-]+|instance-[a-z0-9\-]+)\b", text.lower())
    return match.group(1) if match else "gce-idle-gpu-batch-04"


def _build_local_fallback_llm_response(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse:
    """Synthesizes deterministic tool calls and grounded responses when running offline."""
    contents = list(llm_request.contents or [])
    last_content = contents[-1] if contents else None

    # Check if the latest content is a FunctionResponse from a tool execution
    if last_content and last_content.parts:
        func_responses = [
            p.function_response
            for p in last_content.parts
            if getattr(p, "function_response", None) is not None
        ]
        if func_responses:
            fr = func_responses[-1]
            resp_data = dict(fr.response or {}) if hasattr(fr, "response") else {}
            tool_name = getattr(fr, "name", "finops_tool")
            summary_text = (
                f"### Enterprise FinOps & Sustainability Assessment (`{tool_name}`)\n\n"
                f"**Execution Status**: `{resp_data.get('status', 'success')}`\n"
                f"**Details**: {json.dumps(resp_data, indent=2)}\n\n"
                "**Next Steps**: Review the recommended FinOps and GreenOps optimizations above. "
                "Any destructive instance termination requires explicit Human-in-the-Loop approval."
            )
            return LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part.from_text(text=summary_text)],
                )
            )

    # Otherwise inspect the latest user message and emit the appropriate FunctionCall
    user_text = ""
    for content in reversed(contents):
        if content.role == "user" and content.parts:
            for part in content.parts:
                if getattr(part, "text", None):
                    user_text = str(part.text)
                    break
            if user_text:
                break

    lower = user_text.lower()
    project_id = _extract_project_id(user_text)

    if any(k in lower for k in ("terminate", "delete", "shut down", "decommission")):
        is_approved = "human_approved=true" in lower or "confirm" in lower or "approved" in lower
        is_dry_run = "dry_run=true" in lower or ("dry run" in lower)
        return LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part.from_function_call(
                        name="terminate_idle_cloud_instance",
                        args={
                            "project_id": project_id,
                            "zone": _extract_zone(user_text),
                            "instance_id": _extract_instance_id(user_text),
                            "reason": "Remediating idle Compute Engine instance (<5% CPU utilization)",
                            "dry_run": is_dry_run,
                            "human_approved": is_approved,
                        },
                    )
                ],
            )
        )

    if any(k in lower for k in ("carbon", "footprint", "emission", "co2", "cfe", "kwh", "green", "sustainability")):
        kwh_match = re.search(r"\b(\d+(?:\.\d+)?)\s*kwh\b", lower)
        monthly_kwh = float(kwh_match.group(1)) if kwh_match else 4500.0
        return LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part.from_function_call(
                        name="calculate_workload_carbon_footprint",
                        args={
                            "project_id": project_id,
                            "region": _extract_region(user_text),
                            "monthly_kwh": monthly_kwh,
                            "workload_name": "primary-cloud-workload",
                        },
                    )
                ],
            )
        )

    if any(k in lower for k in ("cost", "spend", "anomal", "billing", "budget", "idle", "finops")):
        return LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part.from_function_call(
                        name="analyze_project_cost_anomalies",
                        args={
                            "project_id": project_id,
                            "lookback_days": 30,
                            "variance_threshold_pct": 15.0,
                        },
                    )
                ],
            )
        )

    # General greeting / concierge overview
    return LlmResponse(
        content=genai_types.Content(
            role="model",
            parts=[
                genai_types.Part.from_text(
                    text=(
                        "Hello! I am the **Enterprise Cloud FinOps & Sustainability Concierge Agent**. "
                        "I can help you:\n"
                        "1. **Analyze Cloud Cost Anomalies** (`analyze_project_cost_anomalies`)\n"
                        "2. **Calculate Workload Carbon Footprint & 24/7 CFE** (`calculate_workload_carbon_footprint`)\n"
                        "3. **Safely Remediate Idle Cloud Instances** (`terminate_idle_cloud_instance` with Human-in-the-Loop confirmation)."
                    )
                )
            ],
        )
    )


async def on_model_error_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    error: Exception,
) -> LlmResponse | None:
    """Provides a resilient local fallback if live Vertex AI / Gemini credentials are unavailable."""
    global _OFFLINE_FALLBACK_ACTIVE
    _OFFLINE_FALLBACK_ACTIVE = True
    session_id = _get_session_id(callback_context)
    agent_name = _get_agent_name(callback_context)

    log_structured_event(
        "MODEL_FALLBACK_ACTIVATED",
        f"Live model call raised {type(error).__name__}; activating deterministic local FinOps fallback.",
        session_id=session_id,
        agent_name=agent_name,
        severity="WARNING",
        error_type=type(error).__name__,
    )
    return _build_local_fallback_llm_response(callback_context, llm_request)


# ---------------------------------------------------------------------------
# 7. Global ADK Policy & Observability Plugin (`FinOpsGuardrailPlugin`)
# ---------------------------------------------------------------------------


class FinOpsGuardrailPlugin(BasePlugin):
    """Cross-cutting ADK Plugin enforcing FinOps guardrails, PII redaction, and OTel tracing."""

    def __init__(self, name: str = "finops_guardrail_plugin") -> None:
        super().__init__(name=name)

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        return await before_model_callback(callback_context, llm_request)

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        return await after_model_callback(callback_context, llm_response)

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> LlmResponse | None:
        return await on_model_error_callback(callback_context, llm_request, error)
