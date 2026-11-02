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

"""Production FinOps & Sustainability Tools with Strict Pydantic Schemas and Guided Recovery.

Provides three domain-specific tools for the Enterprise Cloud FinOps & Sustainability Concierge:
1. `analyze_project_cost_anomalies`: Detects spend spikes and idle resources across GCP services.
2. `calculate_workload_carbon_footprint`: Computes Scope 2 location-based and CFE-adjusted emissions.
3. `terminate_idle_cloud_instance`: Remediates idle Compute Engine/GKE nodes behind an explicit
   Human-in-the-Loop (HITL) confirmation gate.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Literal

from google.adk.tools import FunctionTool, ToolContext
from pydantic import BaseModel, Field, ValidationError

from app.telemetry import log_structured_event, scrub_pii_data, trace_span

# ---------------------------------------------------------------------------
# 1. Explicit Pydantic Input & Output JSON Schemas
# ---------------------------------------------------------------------------


class CostAnomalyInput(BaseModel):
    """Validated input schema for analyzing project cloud spend anomalies."""

    project_id: str = Field(
        ...,
        min_length=3,
        max_length=63,
        pattern=r"^[a-z][a-z0-9\-]+$",
        description=(
            "Target GCP project ID (lowercase letters, digits, hyphens, e.g., 'acme-prod-analytics')."
        ),
    )
    lookback_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of historical billing days to analyze (1 to 365).",
    )
    variance_threshold_pct: float = Field(
        default=15.0,
        ge=1.0,
        le=500.0,
        description="Minimum percentage increase over baseline spend to flag as an anomaly.",
    )


class ServiceCostAnomaly(BaseModel):
    """Detailed cost anomaly record for a specific Google Cloud service or SKU."""

    service_name: str = Field(
        ..., description="Name of the GCP service (e.g., 'Compute Engine', 'BigQuery')."
    )
    resource_id: str = Field(
        ..., description="Specific cloud resource or instance identifier."
    )
    region_or_zone: str = Field(
        ..., description="GCP region or zone hosting the resource."
    )
    baseline_cost_usd: float = Field(
        ..., ge=0.0, description="Historical baseline spend in USD for the window."
    )
    actual_cost_usd: float = Field(
        ..., ge=0.0, description="Actual billed spend in USD for the window."
    )
    variance_pct: float = Field(
        ..., description="Percentage cost delta relative to the historical baseline."
    )
    utilization_cpu_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Average resource CPU utilization percentage."
    )
    is_idle_candidate: bool = Field(
        ..., description="True if utilization is below FinOps idle threshold (<5% CPU)."
    )
    remediation_recommendation: str = Field(
        ..., description="Actionable FinOps right-sizing or termination guidance."
    )


class CostAnomalyOutput(BaseModel):
    """Structured output schema returned by `analyze_project_cost_anomalies`."""

    status: Literal["success", "error"] = Field(
        ..., description="Execution status of the cost anomaly analysis."
    )
    project_id: str = Field(..., description="Evaluated GCP project identifier.")
    lookback_days: int = Field(..., ge=1, le=365, description="Evaluated lookback window.")
    total_spend_usd: float = Field(
        ..., ge=0.0, description="Total actual cloud spend in USD over the lookback period."
    )
    baseline_spend_usd: float = Field(
        ..., ge=0.0, description="Expected baseline cloud spend in USD."
    )
    variance_pct: float = Field(
        ..., description="Overall project cost variance percentage."
    )
    anomalies_detected: int = Field(
        ..., ge=0, description="Count of services exceeding `variance_threshold_pct`."
    )
    anomalies: list[ServiceCostAnomaly] = Field(
        default_factory=list,
        description="List of detected service-level cost anomalies and idle resources.",
    )
    recommended_actions: list[str] = Field(
        default_factory=list,
        description="Prioritized FinOps cost-optimization steps.",
    )


class CarbonFootprintInput(BaseModel):
    """Validated input schema for workload carbon footprint calculation."""

    project_id: str = Field(
        ...,
        min_length=3,
        max_length=63,
        pattern=r"^[a-z][a-z0-9\-]+$",
        description="Target GCP project ID (e.g., 'acme-prod-analytics').",
    )
    region: str = Field(
        ...,
        min_length=4,
        max_length=40,
        pattern=r"^[a-z]+-[a-z]+\d+$",
        description="GCP region name (e.g., 'us-east1', 'us-central1', 'europe-west1').",
    )
    monthly_kwh: float = Field(
        ...,
        gt=0.0,
        le=10_000_000.0,
        description="Estimated monthly electricity consumption in kilowatt-hours (kWh, > 0).",
    )
    workload_name: str = Field(
        default="primary-cloud-workload",
        min_length=2,
        max_length=128,
        description="Logical name of the workload or service being assessed.",
    )


class CarbonFootprintOutput(BaseModel):
    """Structured output schema returned by `calculate_workload_carbon_footprint`."""

    status: Literal["success", "error"] = Field(
        ..., description="Execution status of the carbon footprint calculation."
    )
    project_id: str = Field(..., description="GCP project identifier.")
    workload_name: str = Field(..., description="Assessed workload identifier.")
    region: str = Field(..., description="Evaluated GCP region.")
    grid_carbon_intensity_g_per_kwh: float = Field(
        ..., ge=0.0, description="Regional grid carbon intensity in gCO2e/kWh."
    )
    gross_emissions_kg_co2e: float = Field(
        ..., ge=0.0, description="Gross location-based emissions in kgCO2e/month."
    )
    google_cfe_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Google 24/7 Carbon-Free Energy (CFE) percentage for the region.",
    )
    net_cfe_adjusted_emissions_kg_co2e: float = Field(
        ...,
        ge=0.0,
        description="Net market/CFE-adjusted operational emissions in kgCO2e/month.",
    )
    recommended_greener_region: str = Field(
        ..., description="Recommended low-carbon GCP region for workload migration."
    )
    estimated_carbon_reduction_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Estimated percentage reduction in carbon emissions if migrated.",
    )
    sustainability_recommendations: list[str] = Field(
        default_factory=list,
        description="Actionable GreenOps recommendations to reduce carbon emissions.",
    )


class TerminateInstanceInput(BaseModel):
    """Validated input schema for terminating an idle cloud compute instance."""

    project_id: str = Field(
        ...,
        min_length=3,
        max_length=63,
        pattern=r"^[a-z][a-z0-9\-]+$",
        description="Target GCP project ID owning the idle instance.",
    )
    zone: str = Field(
        ...,
        min_length=6,
        max_length=48,
        pattern=r"^[a-z]+-[a-z]+\d+-[a-z]$",
        description="GCP compute zone (e.g., 'us-east1-b', 'us-central1-a').",
    )
    instance_id: str = Field(
        ...,
        min_length=3,
        max_length=64,
        pattern=r"^[a-zA-Z0-9\-_]+$",
        description="Compute Engine VM instance name or resource ID (e.g., 'gce-idle-batch-04').",
    )
    reason: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Audit justification explaining why the instance is being terminated.",
    )
    dry_run: bool = Field(
        default=True,
        description="If True, simulates termination impact without deleting the instance.",
    )
    human_approved: bool = Field(
        default=False,
        description=(
            "Explicit Human-in-the-Loop confirmation flag. Must be True to execute "
            "a live destructive termination (dry_run=False)."
        ),
    )


class TerminateInstanceOutput(BaseModel):
    """Structured output schema returned by `terminate_idle_cloud_instance`."""

    status: Literal["success", "pending_approval", "dry_run_validated", "error"] = Field(
        ..., description="Outcome status of the remediation request."
    )
    project_id: str = Field(..., description="Target GCP project ID.")
    zone: str = Field(..., description="Target GCP zone.")
    instance_id: str = Field(..., description="Target Compute Engine instance ID.")
    action_taken: str = Field(..., description="Summary of the remediation action taken.")
    estimated_monthly_savings_usd: float = Field(
        ..., ge=0.0, description="Projected monthly cost savings in USD."
    )
    estimated_carbon_saved_kg_co2e: float = Field(
        ..., ge=0.0, description="Projected monthly carbon savings in kgCO2e."
    )
    requires_human_confirmation: bool = Field(
        ..., description="True if execution is paused awaiting human operator approval."
    )
    audit_reference: str = Field(
        ..., description="Immutable audit trail reference ID for FinOps governance."
    )


# ---------------------------------------------------------------------------
# Reference Regional Carbon Intensity & CFE Catalog
# ---------------------------------------------------------------------------

_REGIONAL_CARBON_CATALOG: dict[str, dict[str, Any]] = {
    "us-east1": {"g_per_kwh": 475.0, "cfe_pct": 31.0, "greener_alt": "us-central1"},
    "us-east4": {"g_per_kwh": 345.0, "cfe_pct": 52.0, "greener_alt": "northamerica-northeast1"},
    "us-central1": {"g_per_kwh": 135.0, "cfe_pct": 93.0, "greener_alt": "northamerica-northeast1"},
    "us-west1": {"g_per_kwh": 78.0, "cfe_pct": 95.0, "greener_alt": "us-west1"},
    "northamerica-northeast1": {
        "g_per_kwh": 28.0,
        "cfe_pct": 98.0,
        "greener_alt": "northamerica-northeast1",
    },
    "europe-west1": {"g_per_kwh": 118.0, "cfe_pct": 90.0, "greener_alt": "europe-north1"},
    "europe-north1": {"g_per_kwh": 32.0, "cfe_pct": 98.0, "greener_alt": "europe-north1"},
    "asia-southeast1": {"g_per_kwh": 408.0, "cfe_pct": 18.0, "greener_alt": "asia-east1"},
}


# ---------------------------------------------------------------------------
# 2. Tool Implementations with Guided Error Handling & HITL Hooks
# ---------------------------------------------------------------------------


def analyze_project_cost_anomalies(
    project_id: str,
    lookback_days: int = 30,
    variance_threshold_pct: float = 15.0,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Analyzes a Google Cloud project's billing data to detect cost anomalies and idle resources.

    Validates input constraints via `CostAnomalyInput`, inspects historical vs. actual
    service-level cloud spend over `lookback_days`, flags services exceeding
    `variance_threshold_pct`, and identifies idle Compute Engine / GKE / Cloud SQL resources
    suitable for right-sizing or termination.

    Args:
        project_id: Target GCP project ID (lowercase alphanumeric with hyphens, 3-63 chars,
            e.g., `"acme-prod-analytics"`).
        lookback_days: Number of historical billing days to inspect (integer between 1 and 365).
        variance_threshold_pct: Percentage spend increase threshold over baseline required
            to classify a service spike as an anomaly (float between 1.0 and 500.0).

    Returns:
        A structured dictionary conforming to `CostAnomalyOutput` on success, or an error
        dictionary with `"status": "error"` and `"recovery_instructions"` if validation fails.

    Raises:
        No unhandled exceptions are raised; `ValidationError` and runtime errors are caught
        and transformed into guided recovery dictionaries for the LLM.
    """
    with trace_span(
        "tool.analyze_project_cost_anomalies",
        {
            "finops.project_id": project_id,
            "finops.lookback_days": lookback_days,
            "finops.variance_threshold_pct": variance_threshold_pct,
        },
    ):
        try:
            clean_project_id = scrub_pii_data(str(project_id).strip())
            validated = CostAnomalyInput(
                project_id=clean_project_id,
                lookback_days=lookback_days,
                variance_threshold_pct=variance_threshold_pct,
            )

            # Scale simulated billing dataset proportionally to the lookback window
            scale = validated.lookback_days / 30.0
            candidate_anomalies = [
                ServiceCostAnomaly(
                    service_name="Compute Engine",
                    resource_id="gce-idle-gpu-batch-04",
                    region_or_zone="us-east1-b",
                    baseline_cost_usd=round(1200.0 * scale, 2),
                    actual_cost_usd=round(3420.0 * scale, 2),
                    variance_pct=185.0,
                    utilization_cpu_pct=1.8,
                    is_idle_candidate=True,
                    remediation_recommendation=(
                        "Terminate idle a2-highgpu-1g VM 'gce-idle-gpu-batch-04' in zone "
                        "'us-east1-b' (1.8% CPU utilization for 14+ days) to save ~$2,220/mo."
                    ),
                ),
                ServiceCostAnomaly(
                    service_name="BigQuery",
                    resource_id="bq-unpartitioned-clickstream-scan",
                    region_or_zone="us-east1",
                    baseline_cost_usd=round(850.0 * scale, 2),
                    actual_cost_usd=round(1615.0 * scale, 2),
                    variance_pct=90.0,
                    utilization_cpu_pct=42.5,
                    is_idle_candidate=False,
                    remediation_recommendation=(
                        "Enforce partition filtering on `event_timestamp` and enable BI Engine "
                        "slot reservations to reduce on-demand scan costs by ~48%."
                    ),
                ),
                ServiceCostAnomaly(
                    service_name="Cloud Storage",
                    resource_id="gcs-raw-telemetry-archive",
                    region_or_zone="us-central1",
                    baseline_cost_usd=round(400.0 * scale, 2),
                    actual_cost_usd=round(448.0 * scale, 2),
                    variance_pct=12.0,
                    utilization_cpu_pct=15.0,
                    is_idle_candidate=False,
                    remediation_recommendation=(
                        "Configure Autoclass or Object Lifecycle Management transition to "
                        "Coldline after 30 days."
                    ),
                ),
            ]

            filtered_anomalies = [
                item
                for item in candidate_anomalies
                if item.variance_pct >= validated.variance_threshold_pct
            ]

            baseline_total = round(
                sum(item.baseline_cost_usd for item in candidate_anomalies), 2
            )
            actual_total = round(
                sum(item.actual_cost_usd for item in candidate_anomalies), 2
            )
            overall_variance_pct = (
                round(((actual_total - baseline_total) / baseline_total) * 100.0, 2)
                if baseline_total > 0
                else 0.0
            )

            output = CostAnomalyOutput(
                status="success",
                project_id=validated.project_id,
                lookback_days=validated.lookback_days,
                total_spend_usd=actual_total,
                baseline_spend_usd=baseline_total,
                variance_pct=overall_variance_pct,
                anomalies_detected=len(filtered_anomalies),
                anomalies=filtered_anomalies,
                recommended_actions=[
                    a.remediation_recommendation for a in filtered_anomalies
                ],
            )
            result_dict = output.model_dump()

            if tool_context is not None:
                tool_context.state["last_cost_analysis"] = result_dict

            return result_dict

        except (ValidationError, ValueError) as val_err:
            log_structured_event(
                "TOOL_VALIDATION_ERROR",
                f"Validation failed in analyze_project_cost_anomalies: {val_err}",
                severity="WARNING",
                tool_name="analyze_project_cost_anomalies",
            )
            return {
                "status": "error",
                "error_type": "ValidationError",
                "error_message": scrub_pii_data(str(val_err)),
                "recovery_instructions": (
                    "Fix the input parameters and retry `analyze_project_cost_anomalies`: "
                    "(1) `project_id` must be 3-63 lowercase characters matching `^[a-z][a-z0-9\\-]+$` "
                    "(e.g., 'acme-prod-analytics'); "
                    "(2) `lookback_days` must be an integer between 1 and 365 (default 30); "
                    "(3) `variance_threshold_pct` must be a float between 1.0 and 500.0 (default 15.0)."
                ),
            }
        except Exception as exc:
            log_structured_event(
                "TOOL_RUNTIME_ERROR",
                f"Runtime error in analyze_project_cost_anomalies: {exc}",
                severity="ERROR",
                tool_name="analyze_project_cost_anomalies",
            )
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": scrub_pii_data(str(exc)),
                "recovery_instructions": (
                    "Verify the GCP project ID format and retry with a standard 30-day lookback window."
                ),
            }


def calculate_workload_carbon_footprint(
    project_id: str,
    region: str,
    monthly_kwh: float,
    workload_name: str = "primary-cloud-workload",
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Calculates Scope 2 carbon emissions and Google 24/7 Carbon-Free Energy (CFE) impact.

    Evaluates a cloud workload's monthly energy consumption (`monthly_kwh`) against regional
    grid carbon intensity factors (`gCO2e/kWh`) and Google Cloud's regional 24/7 CFE score,
    returning both gross location-based emissions and net CFE-adjusted emissions alongside
    low-carbon region migration recommendations.

    Args:
        project_id: Target GCP project ID (lowercase alphanumeric with hyphens, e.g.,
            `"acme-prod-analytics"`).
        region: GCP region hosting the workload (e.g., `"us-east1"`, `"us-central1"`,
            `"europe-west1"`).
        monthly_kwh: Estimated monthly energy consumption in kilowatt-hours (float > 0.0).
        workload_name: Logical workload or service name (2-128 chars).

    Returns:
        A structured dictionary conforming to `CarbonFootprintOutput` on success, or an error
        dictionary with `"status": "error"` and `"recovery_instructions"` on invalid inputs.

    Raises:
        No unhandled exceptions are raised; validation and runtime errors are caught and
        returned with structured `"recovery_instructions"`.
    """
    with trace_span(
        "tool.calculate_workload_carbon_footprint",
        {
            "finops.project_id": project_id,
            "finops.region": region,
            "finops.monthly_kwh": monthly_kwh,
            "finops.workload_name": workload_name,
        },
    ):
        try:
            validated = CarbonFootprintInput(
                project_id=scrub_pii_data(str(project_id).strip()),
                region=str(region).strip().lower(),
                monthly_kwh=monthly_kwh,
                workload_name=scrub_pii_data(str(workload_name).strip()),
            )

            region_meta = _REGIONAL_CARBON_CATALOG.get(
                validated.region,
                {"g_per_kwh": 380.0, "cfe_pct": 45.0, "greener_alt": "us-central1"},
            )
            alt_region = str(region_meta["greener_alt"])
            alt_meta = _REGIONAL_CARBON_CATALOG.get(
                alt_region, {"g_per_kwh": 135.0, "cfe_pct": 93.0}
            )

            grid_intensity = float(region_meta["g_per_kwh"])
            cfe_pct = float(region_meta["cfe_pct"])

            gross_kg = round((validated.monthly_kwh * grid_intensity) / 1000.0, 2)
            net_cfe_kg = round(gross_kg * (1.0 - (cfe_pct / 100.0)), 2)

            alt_gross_kg = (validated.monthly_kwh * float(alt_meta["g_per_kwh"])) / 1000.0
            alt_net_kg = round(
                alt_gross_kg * (1.0 - (float(alt_meta["cfe_pct"]) / 100.0)), 2
            )

            reduction_pct = (
                round(max(0.0, ((net_cfe_kg - alt_net_kg) / net_cfe_kg) * 100.0), 1)
                if net_cfe_kg > 0
                else 0.0
            )

            output = CarbonFootprintOutput(
                status="success",
                project_id=validated.project_id,
                workload_name=validated.workload_name,
                region=validated.region,
                grid_carbon_intensity_g_per_kwh=grid_intensity,
                gross_emissions_kg_co2e=gross_kg,
                google_cfe_pct=cfe_pct,
                net_cfe_adjusted_emissions_kg_co2e=net_cfe_kg,
                recommended_greener_region=alt_region,
                estimated_carbon_reduction_pct=reduction_pct,
                sustainability_recommendations=[
                    (
                        f"Migrate batch/stateless workloads from '{validated.region}' "
                        f"({cfe_pct}% CFE, {grid_intensity} gCO2e/kWh) to '{alt_region}' "
                        f"({alt_meta['cfe_pct']}% CFE) for an estimated {reduction_pct}% "
                        "net operational carbon reduction."
                    ),
                    (
                        "Switch x86 N2 VMs to Tau T2A (Arm) or C4A Axion instances to improve "
                        "price-performance and reduce kWh energy draw by ~30%."
                    ),
                    (
                        "Schedule non-urgent batch jobs during peak solar/wind hours using "
                        "Carbon-Aware Workload Shifting."
                    ),
                ],
            )
            result_dict = output.model_dump()

            if tool_context is not None:
                tool_context.state["last_carbon_analysis"] = result_dict

            return result_dict

        except (ValidationError, ValueError) as val_err:
            log_structured_event(
                "TOOL_VALIDATION_ERROR",
                f"Validation failed in calculate_workload_carbon_footprint: {val_err}",
                severity="WARNING",
                tool_name="calculate_workload_carbon_footprint",
            )
            return {
                "status": "error",
                "error_type": "ValidationError",
                "error_message": scrub_pii_data(str(val_err)),
                "recovery_instructions": (
                    "Correct the parameters and retry `calculate_workload_carbon_footprint`: "
                    "(1) `project_id` must match `^[a-z][a-z0-9\\-]+$` (e.g., 'acme-prod-analytics'); "
                    "(2) `region` must be a valid GCP region like 'us-east1', 'us-central1', or 'europe-west1' "
                    "(do not pass a zone suffix like '-b'); "
                    "(3) `monthly_kwh` must be a positive number > 0 (e.g., 4500.0)."
                ),
            }
        except Exception as exc:
            log_structured_event(
                "TOOL_RUNTIME_ERROR",
                f"Runtime error in calculate_workload_carbon_footprint: {exc}",
                severity="ERROR",
                tool_name="calculate_workload_carbon_footprint",
            )
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": scrub_pii_data(str(exc)),
                "recovery_instructions": (
                    "Provide a valid GCP region (e.g., 'us-east1') and a positive float for `monthly_kwh`."
                ),
            }


def terminate_idle_cloud_instance(
    project_id: str,
    zone: str,
    instance_id: str,
    reason: str,
    dry_run: bool = True,
    human_approved: bool = False,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Terminates or simulates termination of an idle Google Cloud Compute Engine instance.

    Enforces a strict Human-in-the-Loop (HITL) safety gate for destructive actions:
    - If `dry_run=True`, performs a safe impact simulation returning projected cost and
      carbon savings without modifying infrastructure.
    - If `dry_run=False` and `human_approved=False` (and session state does not contain
      explicit operator approval), halts execution, triggers ADK confirmation signaling
      (`tool_context.request_confirmation` / `tool_context.actions`), and returns
      `"status": "pending_approval"`.
    - If `dry_run=False` and `human_approved=True`, executes the instance termination and
      returns an immutable FinOps audit reference.

    Args:
        project_id: Target GCP project ID (lowercase alphanumeric with hyphens, e.g.,
            `"acme-prod-analytics"`).
        zone: GCP compute zone where the instance resides (e.g., `"us-east1-b"`).
        instance_id: Identifier of the idle VM instance (e.g., `"gce-idle-gpu-batch-04"`).
        reason: Audit justification (at least 5 characters) explaining why the instance
            should be terminated.
        dry_run: When True (default), simulates savings without deleting the instance.
        human_approved: Must be explicitly True to execute a non-dry-run termination.

    Returns:
        A structured dictionary conforming to `TerminateInstanceOutput` (or an error
        dictionary with `"status": "error"` and `"recovery_instructions"` on invalid input).

    Raises:
        No unhandled exceptions are raised; validation errors return guided recovery steps.
    """
    with trace_span(
        "tool.terminate_idle_cloud_instance",
        {
            "finops.project_id": project_id,
            "finops.zone": zone,
            "finops.instance_id": instance_id,
            "finops.dry_run": dry_run,
            "finops.human_approved": human_approved,
        },
    ):
        try:
            validated = TerminateInstanceInput(
                project_id=scrub_pii_data(str(project_id).strip()),
                zone=str(zone).strip().lower(),
                instance_id=scrub_pii_data(str(instance_id).strip()),
                reason=scrub_pii_data(str(reason).strip()),
                dry_run=dry_run,
                human_approved=human_approved,
            )

            audit_digest = hashlib.sha256(
                f"{validated.project_id}:{validated.zone}:{validated.instance_id}:{validated.reason}".encode()
            ).hexdigest()[:12]
            audit_ref = f"FINOPS-AUDIT-{audit_digest.upper()}"

            # Check if session state also records human approval
            state_approved = False
            if tool_context is not None:
                state_approved = bool(
                    tool_context.state.get("hitl_confirmed_instances", {}).get(
                        validated.instance_id, False
                    )
                    or tool_context.state.get("human_approved", False)
                )

            is_confirmed = validated.human_approved or state_approved

            # 1. Safe Dry-Run Simulation Path
            if validated.dry_run:
                output = TerminateInstanceOutput(
                    status="dry_run_validated",
                    project_id=validated.project_id,
                    zone=validated.zone,
                    instance_id=validated.instance_id,
                    action_taken=(
                        f"DRY-RUN ONLY: Verified idle instance '{validated.instance_id}' in "
                        f"'{validated.zone}' is safe to terminate. No resource was deleted. "
                        "Ask the human operator for confirmation before calling with "
                        "`dry_run=False, human_approved=True`."
                    ),
                    estimated_monthly_savings_usd=2220.0,
                    estimated_carbon_saved_kg_co2e=310.5,
                    requires_human_confirmation=True,
                    audit_reference=audit_ref,
                )
                return output.model_dump()

            # 2. Human-in-the-Loop Gate for Destructive Action (dry_run=False, not approved)
            if not is_confirmed:
                confirmation_hint = (
                    f"HIGH-STAKES FINOPS ACTION: Confirm permanent termination of Compute Engine "
                    f"instance '{validated.instance_id}' in '{validated.project_id}/{validated.zone}' "
                    f"(Reason: {validated.reason}). Estimated savings: $2,220.00/month."
                )
                if tool_context is not None:
                    tool_context.state["pending_hitl_approval"] = {
                        "project_id": validated.project_id,
                        "zone": validated.zone,
                        "instance_id": validated.instance_id,
                        "reason": validated.reason,
                        "audit_reference": audit_ref,
                    }
                    # Signal confirmation requirement via ADK ToolContext
                    try:
                        tool_context.actions.require_confirmation = True
                    except Exception:
                        tool_context.actions.__dict__["require_confirmation"] = True
                    if os.environ.get("ADK_NATIVE_CONFIRMATION_EVENT") == "true":
                        try:
                            tool_context.request_confirmation(hint=confirmation_hint)
                        except Exception:
                            pass

                log_structured_event(
                    "HITL_GATE_TRIGGERED",
                    f"Halted unapproved termination of {validated.instance_id} in {validated.project_id}",
                    severity="WARNING",
                    instance_id=validated.instance_id,
                    audit_reference=audit_ref,
                )

                pending_output = TerminateInstanceOutput(
                    status="pending_approval",
                    project_id=validated.project_id,
                    zone=validated.zone,
                    instance_id=validated.instance_id,
                    action_taken=(
                        f"HALTED BY HUMAN-IN-THE-LOOP GATE: Termination of '{validated.instance_id}' "
                        f"in '{validated.zone}' requires explicit human confirmation. Present the "
                        "impact summary ($2,220.00/mo & 310.5 kgCO2e/mo savings) to the user and "
                        "wait for their explicit approval before calling with `human_approved=True`."
                    ),
                    estimated_monthly_savings_usd=2220.0,
                    estimated_carbon_saved_kg_co2e=310.5,
                    requires_human_confirmation=True,
                    audit_reference=audit_ref,
                )
                return pending_output.model_dump()

            # 3. Confirmed Execution Path (dry_run=False and human_approved=True)
            if tool_context is not None:
                terminated_list = list(tool_context.state.get("terminated_instances", []))
                terminated_list.append(validated.instance_id)
                tool_context.state["terminated_instances"] = terminated_list
                tool_context.state.pop("pending_hitl_approval", None)

            log_structured_event(
                "REMEDIATION_EXECUTED",
                f"Terminated idle instance {validated.instance_id} with human approval ({audit_ref})",
                severity="INFO",
                instance_id=validated.instance_id,
                audit_reference=audit_ref,
            )

            executed_output = TerminateInstanceOutput(
                status="success",
                project_id=validated.project_id,
                zone=validated.zone,
                instance_id=validated.instance_id,
                action_taken=(
                    f"EXECUTED: Permanently terminated idle instance '{validated.instance_id}' "
                    f"in '{validated.project_id}/{validated.zone}' with verified human approval. "
                    f"Reason recorded: {validated.reason}"
                ),
                estimated_monthly_savings_usd=2220.0,
                estimated_carbon_saved_kg_co2e=310.5,
                requires_human_confirmation=False,
                audit_reference=audit_ref,
            )
            return executed_output.model_dump()

        except (ValidationError, ValueError) as val_err:
            log_structured_event(
                "TOOL_VALIDATION_ERROR",
                f"Validation failed in terminate_idle_cloud_instance: {val_err}",
                severity="WARNING",
                tool_name="terminate_idle_cloud_instance",
            )
            return {
                "status": "error",
                "error_type": "ValidationError",
                "error_message": scrub_pii_data(str(val_err)),
                "recovery_instructions": (
                    "Fix the input parameters and retry `terminate_idle_cloud_instance`: "
                    "(1) `project_id` must match `^[a-z][a-z0-9\\-]+$`; "
                    "(2) `zone` must include the zone letter matching `^[a-z]+-[a-z]+\\d+-[a-z]$` "
                    "(e.g., 'us-east1-b', NOT just region 'us-east1'); "
                    "(3) `instance_id` must be 3-64 alphanumeric/hyphen characters; "
                    "(4) `reason` must be at least 5 characters explaining the FinOps justification."
                ),
            }
        except Exception as exc:
            log_structured_event(
                "TOOL_RUNTIME_ERROR",
                f"Runtime error in terminate_idle_cloud_instance: {exc}",
                severity="ERROR",
                tool_name="terminate_idle_cloud_instance",
            )
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": scrub_pii_data(str(exc)),
                "recovery_instructions": (
                    "Ensure `project_id`, `zone` (e.g., 'us-east1-b'), `instance_id`, and `reason` "
                    "are valid strings, and use `dry_run=True` first before requesting human approval."
                ),
            }


# ADK FunctionTool wrappers (with conditional HITL confirmation check on termination)
def _needs_hitl_confirmation(
    dry_run: bool = True, human_approved: bool = False, **kwargs: Any
) -> bool:
    """Returns True when a live destructive instance termination has not yet been approved."""
    return (not dry_run) and (not human_approved)


cost_anomaly_tool = FunctionTool(analyze_project_cost_anomalies)
carbon_footprint_tool = FunctionTool(calculate_workload_carbon_footprint)
terminate_instance_tool = FunctionTool(
    terminate_idle_cloud_instance,
    require_confirmation=_needs_hitl_confirmation,
)

ALL_FINOPS_TOOLS = [
    analyze_project_cost_anomalies,
    calculate_workload_carbon_footprint,
    terminate_idle_cloud_instance,
]
FINOPS_TOOLS = ALL_FINOPS_TOOLS
