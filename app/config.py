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

"""Configuration and secure secret management for the Cloud FinOps & Sustainability Agent.

Uses Google Cloud Secret Manager (`SecretManagerServiceClient`) as the primary
source of truth for sensitive configuration values, with a graceful fallback
to environment variables for local development and CI/CD test environments.
Zero API keys or secrets are hardcoded in the repository.
"""

from __future__ import annotations

import functools
import logging
import os

from dotenv import load_dotenv
from google.cloud import secretmanager
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

# Strategic Model Routing Constants
FAST_MODEL = "gemini-2.5-flash"
PLANNING_MODEL = "gemini-2.5-pro"


@functools.lru_cache(maxsize=64)
def get_secret(
    secret_id: str,
    project_id: str | None = None,
    version_id: str = "latest",
    default: str | None = None,
) -> str:
    """Retrieves a secret value from Google Cloud Secret Manager with env fallback.

    First checks if an explicit environment variable matching `secret_id` (or its
    uppercase snake_case form) is set for fast local overrides. Otherwise queries
    Google Cloud Secret Manager (`projects/{project}/secrets/{secret_id}/versions/{version}`).
    Falls back gracefully to environment variables or `default` when running
    locally without GCP Secret Manager access.

    Args:
        secret_id: Identifier of the secret in GCP Secret Manager (e.g.,
            `"finops-billing-api-token"` or `"DB_PASS"`).
        project_id: Optional GCP project ID. Defaults to `GOOGLE_CLOUD_PROJECT`.
        version_id: Secret version to access. Defaults to `"latest"`.
        default: Default fallback value if neither Secret Manager nor environment
            variables provide the secret.

    Returns:
        The resolved secret string (empty string if not found and no default set).
    """
    env_key_candidates = [
        secret_id,
        secret_id.upper().replace("-", "_"),
        f"FINOPS_SECRET_{secret_id.upper().replace('-', '_')}",
    ]

    # 1. Attempt Google Cloud Secret Manager if a project_id is configured and
    # Secret Manager is not explicitly disabled for offline unit tests.
    resolved_project = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    disable_gcp_sm = os.environ.get("FINOPS_DISABLE_SECRET_MANAGER", "").lower() in (
        "1",
        "true",
        "yes",
    )

    if resolved_project and not disable_gcp_sm:
        try:
            client = secretmanager.SecretManagerServiceClient()
            resource_name = (
                f"projects/{resolved_project}/secrets/{secret_id}/versions/{version_id}"
            )
            response = client.access_secret_version(request={"name": resource_name})
            payload = response.payload.data.decode("UTF-8")
            if payload:
                return payload
        except Exception as exc:
            logger.debug(
                "Secret Manager lookup for '%s' in project '%s' fell back to env: %s",
                secret_id,
                resolved_project,
                type(exc).__name__,
            )

    # 2. Graceful fallback to environment variables
    for env_key in env_key_candidates:
        env_val = os.environ.get(env_key)
        if env_val is not None and env_val != "":
            return env_val

    if default is not None:
        return default

    return ""


class FinOpsConfig(BaseModel):
    """Typed runtime configuration for the Enterprise Cloud FinOps & Sustainability Agent."""

    project_id: str = Field(
        default_factory=lambda: os.environ.get(
            "GOOGLE_CLOUD_PROJECT", "enterprise-finops-local"
        ),
        description="Target Google Cloud Project ID.",
    )
    location: str = Field(
        default_factory=lambda: os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        description="Google Cloud region or global endpoint for Vertex AI.",
    )
    fast_model: str = Field(
        default=FAST_MODEL,
        description="Low-latency Gemini model for triage and classification.",
    )
    planning_model: str = Field(
        default=PLANNING_MODEL,
        description="High-reasoning Gemini model for complex FinOps synthesis and planning.",
    )
    session_db_url: str = Field(
        default_factory=lambda: os.environ.get(
            "FINOPS_SESSION_DB_URL", "sqlite+aiosqlite:///./.finops_sessions.db"
        ),
        description="SQLAlchemy database URL for persistent ADK session storage.",
    )
    rag_corpus_id: str | None = Field(
        default_factory=lambda: os.environ.get("VERTEX_RAG_CORPUS_ID"),
        description="Optional Vertex AI RAG Engine Corpus resource name for long-term memory.",
    )
    anomaly_variance_threshold_pct: float = Field(
        default=15.0,
        ge=1.0,
        le=500.0,
        description="Percentage spend variance threshold to flag a FinOps cost anomaly.",
    )
    high_stakes_cost_threshold_usd: float = Field(
        default=500.0,
        ge=0.0,
        description="Monthly cost impact (USD) above which human approval is strictly required.",
    )
    enable_cloud_dlp: bool = Field(
        default_factory=lambda: os.environ.get("ENABLE_CLOUD_DLP", "false").lower()
        in ("1", "true", "yes"),
        description="Whether to call Google Cloud DLP API in addition to regex PII scrubbing.",
    )
    enable_cloud_trace: bool = Field(
        default_factory=lambda: os.environ.get("ENABLE_CLOUD_TRACE", "false").lower()
        in ("1", "true", "yes"),
        description="Whether to export OpenTelemetry spans to Google Cloud Trace.",
    )
    enable_cloud_logging: bool = Field(
        default_factory=lambda: os.environ.get("ENABLE_CLOUD_LOGGING", "false").lower()
        in ("1", "true", "yes"),
        description="Whether to attach the Google Cloud Logging handler.",
    )

    @property
    def billing_export_secret(self) -> str:
        """Resolves the billing export secret dynamically via Secret Manager / env."""
        return get_secret(
            "finops-billing-export-token",
            project_id=self.project_id,
            default="local-dev-non-sensitive-placeholder",
        )


@functools.lru_cache(maxsize=1)
def get_config() -> FinOpsConfig:
    """Returns the singleton FinOpsConfig instance."""
    return FinOpsConfig()
