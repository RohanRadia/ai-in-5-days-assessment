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

"""Multi-agent orchestration for the Enterprise Cloud FinOps & Sustainability Concierge.

Implements:
- Strategic model routing (`FAST_MODEL = "gemini-2.5-flash"`,
  `PLANNING_MODEL = "gemini-2.5-pro"`, and `route_model_by_complexity`)
- Multi-agent topology combining a coordinator `LlmAgent` (`root_agent`),
  a fast triage sub-agent (`cost_triage_agent`), and a `SequentialAgent`
  (`remediation_pipeline_agent` with analysis + execution stages)
- Context caching (`ContextCacheConfig`), event history compaction
  (`EventsCompactionConfig` + `compact_conversation_history`), persistent
  session state (`DatabaseSessionService`), RAG memory (`EnterpriseRagMemoryStore`),
  and non-blocking background memory consolidation (`consolidate_session_memory_async`)
- Guardrails, PII scrubbing, HITL confirmation hooks, and structured telemetry
"""

from __future__ import annotations

import os
from typing import Any

from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.plugins.bigquery_agent_analytics_plugin import (
    BigQueryAgentAnalyticsPlugin,
    BigQueryLoggerConfig,
)
from google.genai import types

from app.callbacks import (
    FinOpsGuardrailPlugin,
    after_agent_callback,
    after_model_callback,
    after_tool_callback,
    before_agent_callback,
    before_model_callback,
    before_tool_callback,
    on_model_error_callback,
    route_model_by_complexity,
)
from app.config import (
    FAST_MODEL as _CONFIG_FAST_MODEL,
)
from app.config import (
    PLANNING_MODEL as _CONFIG_PLANNING_MODEL,
)
from app.config import (
    get_config,
)
from app.memory import (
    SYSTEM_CONSTITUTION,
    EnterpriseRagMemoryStore,
    compact_conversation_history,
    consolidate_session_memory_async,
    create_persistent_session_service,
    get_context_cache_config,
    get_events_compaction_config,
    get_memory_store,
)
from app.telemetry import get_structured_logger, init_telemetry, log_structured_event
from app.tools import (
    FINOPS_TOOLS,
    analyze_project_cost_anomalies,
    calculate_workload_carbon_footprint,
    terminate_idle_cloud_instance,
)

# Explicit strategic model routing constants
FAST_MODEL: str = _CONFIG_FAST_MODEL  # "gemini-2.5-flash"
PLANNING_MODEL: str = _CONFIG_PLANNING_MODEL  # "gemini-2.5-pro"

logger = get_structured_logger("finops_concierge.agent")
init_telemetry()

# Persistent session service & RAG memory service instances
session_service = create_persistent_session_service()
memory_service: EnterpriseRagMemoryStore = get_memory_store()


def build_gemini_model(model_name: str) -> Gemini:
    """Construct a resilient ADK `Gemini` wrapper with retry configuration."""
    return Gemini(
        model=model_name,
        retry_options=types.HttpRetryOptions(attempts=2),
    )


# ---------------------------------------------------------------------------
# Sub-Agent 1: Fast Cost & Sustainability Triage Agent (gemini-2.5-flash)
# ---------------------------------------------------------------------------
cost_triage_agent = LlmAgent(
    name="cost_triage_agent",
    description=(
        "Fast FinOps and sustainability triage specialist that classifies cost "
        "spikes and computes regional Scope 2/3 carbon emissions."
    ),
    model=build_gemini_model(FAST_MODEL),
    instruction=(
        f"{SYSTEM_CONSTITUTION}\n\n"
        "You are `cost_triage_agent`, a high-speed FinOps & GreenOps triage "
        "specialist running on `gemini-2.5-flash`. Rapidly inspect project billing "
        "variance using `analyze_project_cost_anomalies` and evaluate regional "
        "emissions using `calculate_workload_carbon_footprint`. Summarize severity "
        "and carbon reduction opportunities concisely."
    ),
    tools=[
        analyze_project_cost_anomalies,
        calculate_workload_carbon_footprint,
    ],
    output_key="cost_triage_summary",
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
    on_model_error_callback=on_model_error_callback,
)


# ---------------------------------------------------------------------------
# Sub-Agent 2: Sequential Remediation Pipeline Agent (Analysis -> Execution)
# ---------------------------------------------------------------------------
idle_resource_analyzer_agent = LlmAgent(
    name="idle_resource_analyzer_agent",
    description=(
        "Analyzes anomalous cloud spend and identifies idle Compute Engine VMs "
        "eligible for remediation."
    ),
    model=build_gemini_model(FAST_MODEL),
    instruction=(
        f"{SYSTEM_CONSTITUTION}\n\n"
        "Step 1 of Remediation Pipeline: Verify cost anomalies and idle VM "
        "metrics using `analyze_project_cost_anomalies`. Identify target "
        "instance IDs, zones, and utilization levels, and store the findings "
        "for the remediation execution step."
    ),
    tools=[analyze_project_cost_anomalies],
    output_key="idle_resource_analysis",
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
    on_model_error_callback=on_model_error_callback,
)

remediation_execution_agent = LlmAgent(
    name="remediation_execution_agent",
    description=(
        "Executes governed remediation of idle cloud instances while strictly "
        "enforcing the Human-in-the-Loop (HITL) approval gate."
    ),
    model=build_gemini_model(PLANNING_MODEL),
    instruction=(
        f"{SYSTEM_CONSTITUTION}\n\n"
        "Step 2 of Remediation Pipeline: Based on the analysis in "
        "state['idle_resource_analysis'], invoke `terminate_idle_cloud_instance` "
        "when remediation is requested. NEVER set `human_approved=True` unless "
        "the user has explicitly confirmed termination in the conversation."
    ),
    tools=[terminate_idle_cloud_instance],
    output_key="remediation_execution_result",
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
    on_model_error_callback=on_model_error_callback,
)

remediation_pipeline_agent = SequentialAgent(
    name="remediation_pipeline_agent",
    description=(
        "Sequential FinOps remediation pipeline that chains idle resource "
        "analysis with HITL-governed instance termination."
    ),
    sub_agents=[
        idle_resource_analyzer_agent,
        remediation_execution_agent,
    ],
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
)


# ---------------------------------------------------------------------------
# Root Coordinator Agent (gemini-2.5-pro)
# ---------------------------------------------------------------------------
root_agent = LlmAgent(
    # Keep in sync with create_params.root_agent_name in agents-cli-manifest.yaml
    name="finops_concierge",
    description=(
        "Enterprise Cloud FinOps & Sustainability Concierge Coordinator that "
        "analyzes cloud spend anomalies, calculates workload carbon footprint, "
        "and orchestrates governed idle resource remediation."
    ),
    model=build_gemini_model(PLANNING_MODEL),
    instruction=SYSTEM_CONSTITUTION,
    tools=FINOPS_TOOLS,
    sub_agents=[
        cost_triage_agent,
        remediation_pipeline_agent,
    ],
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
    on_model_error_callback=on_model_error_callback,
)


def _build_plugins() -> list[Any]:
    """Assemble ADK plugins including FinOps guardrails and optional BigQuery analytics."""
    plugins: list[Any] = [FinOpsGuardrailPlugin()]
    cfg = get_config()
    if cfg.project_id and os.environ.get("ENABLE_BQ_ANALYTICS", "false").lower() == "true":
        try:
            from google.cloud import bigquery

            bq = bigquery.Client(project=cfg.project_id)
            bq.create_dataset(f"{cfg.project_id}.{cfg.bq_dataset_id}", exists_ok=True)
            plugins.append(
                BigQueryAgentAnalyticsPlugin(
                    project_id=cfg.project_id,
                    dataset_id=cfg.bq_dataset_id,
                    location=cfg.location,
                    config=BigQueryLoggerConfig(
                        gcs_bucket_name=os.environ.get("BQ_ANALYTICS_GCS_BUCKET"),
                        connection_id=os.environ.get("BQ_ANALYTICS_CONNECTION_ID"),
                    ),
                )
            )
        except Exception as exc:
            log_structured_event(
                "BQ_ANALYTICS_FALLBACK",
                f"BigQuery Analytics plugin skipped: {exc}",
                severity="WARNING",
                agent_name="finops_concierge",
            )
    return plugins


app = App(
    root_agent=root_agent,
    name="app",
    context_cache_config=get_context_cache_config(),
    events_compaction_config=get_events_compaction_config(),
    plugins=_build_plugins(),
)

__all__ = [
    "FAST_MODEL",
    "PLANNING_MODEL",
    "SYSTEM_CONSTITUTION",
    "Agent",
    "analyze_project_cost_anomalies",
    "app",
    "calculate_workload_carbon_footprint",
    "compact_conversation_history",
    "consolidate_session_memory_async",
    "cost_triage_agent",
    "idle_resource_analyzer_agent",
    "memory_service",
    "remediation_execution_agent",
    "remediation_pipeline_agent",
    "root_agent",
    "route_model_by_complexity",
    "session_service",
    "terminate_idle_cloud_instance",
]
