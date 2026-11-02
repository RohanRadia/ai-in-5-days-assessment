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

"""Unit tests for core FinOps & Sustainability Concierge domain logic."""

from app.agent import FAST_MODEL, PLANNING_MODEL, root_agent
from app.telemetry import scrub_pii_data
from app.tools import (
    analyze_project_cost_anomalies,
    calculate_workload_carbon_footprint,
    terminate_idle_cloud_instance,
)


def test_agent_configuration() -> None:
    """Verify root agent and model constants are properly configured."""
    assert root_agent.name == "finops_concierge"
    assert FAST_MODEL == "gemini-2.5-flash"
    assert PLANNING_MODEL == "gemini-2.5-pro"


def test_finops_tools_smoke() -> None:
    """Verify core FinOps tools return structured dictionaries."""
    anomalies = analyze_project_cost_anomalies(project_id="acme-prod-analytics")
    assert anomalies["status"] == "success"

    carbon = calculate_workload_carbon_footprint(
        project_id="acme-prod-analytics",
        region="us-central1",
        monthly_kwh=1200.0,
    )
    assert carbon["status"] == "success"

    dry_run = terminate_idle_cloud_instance(
        project_id="acme-prod-analytics",
        zone="us-east1-b",
        instance_id="gce-idle-gpu-batch-04",
        reason="Idle GPU VM under 5% CPU utilization",
        dry_run=True,
        human_approved=False,
    )
    assert dry_run["status"] == "dry_run_validated"


def test_pii_redaction_smoke() -> None:
    """Verify PII scrubber masks sensitive patterns."""
    redacted = scrub_pii_data("Email admin@corp.example.com or call 415-555-0199")
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted
