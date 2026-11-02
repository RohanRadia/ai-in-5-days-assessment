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

"""Process-wide ADK session/artifact services shared by every serving surface.

Registered under ``shared://`` so the ADK web routes, the A2A path, and the
reasoning_engine adapter share one instance: a session created on any surface
is visible to the others.
"""

from __future__ import annotations

import functools
import os

from google.adk.artifacts import GcsArtifactService, InMemoryArtifactService
from google.adk.cli.service_registry import get_service_registry
from google.adk.cli.utils.service_factory import create_session_service_from_options

SESSION_SERVICE_URI = "shared://session"
ARTIFACT_SERVICE_URI = "shared://artifact"

_AGENT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


@functools.cache
def get_session_service():
    """Process-wide session service shared across every serving surface."""
    from urllib.parse import quote

    db_user = os.environ.get("DB_USER", "postgres")
    db_name = os.environ.get("DB_NAME", "postgres")
    db_pass = os.environ.get("DB_PASS")
    instance_connection_name = os.environ.get("INSTANCE_CONNECTION_NAME")

    if instance_connection_name and db_pass:
        # URL-encode credentials/instance; '[' would otherwise trigger IPv6 parsing.
        encoded_user = quote(db_user, safe="")
        encoded_pass = quote(db_pass, safe="")
        encoded_instance = instance_connection_name.replace(":", "%3A")
        session_service_uri = (
            f"postgresql+asyncpg://{encoded_user}:{encoded_pass}@"
            f"/{db_name}?host=/cloudsql/{encoded_instance}"
        )
        return create_session_service_from_options(
            base_dir=_AGENT_DIR, session_service_uri=session_service_uri
        )

    if os.environ.get("FINOPS_USE_DB_SESSION", "false").lower() == "true":
        from app.memory import create_persistent_session_service

        return create_persistent_session_service()

    from google.adk.sessions.in_memory_session_service import InMemorySessionService

    return InMemorySessionService()


@functools.cache
def get_artifact_service():
    """Process-wide artifact service: GCS when a bucket is set, else in-memory."""
    if bucket := os.environ.get("LOGS_BUCKET_NAME"):
        return GcsArtifactService(bucket_name=bucket)
    return InMemoryArtifactService()


_registry = get_service_registry()
_registry.register_session_service("shared", lambda uri, **kw: get_session_service())
_registry.register_artifact_service("shared", lambda uri, **kw: get_artifact_service())
