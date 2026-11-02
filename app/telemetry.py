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

"""Observability, Distributed Tracing, Structured JSON Logging, and PII Redaction.

Implements enterprise telemetry standards for the Cloud FinOps & Sustainability Agent:
- Active PII scrubbing (`scrub_pii_data`) using regex patterns + optional Google Cloud DLP
  (`google.cloud.dlp_v2`) prior to any log emission or memory persistence.
- Structured JSON logging (`google.cloud.logging` + `python-json-logger`) with mandatory
  `session_id`, `agent_name`, `severity`, and `event_type` fields.
- OpenTelemetry distributed tracing (`TracerProvider`, `BatchSpanProcessor`,
  `CloudTraceSpanExporter` with local fallback) and linked span helpers.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import logging
import os
import re
import sys
from collections import deque
from collections.abc import Iterator
from typing import Any

import google.cloud.logging as cloud_logging
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from pythonjsonlogger.json import JsonFormatter

from app.config import get_config

try:
    from google.cloud import dlp_v2
except ImportError:  # pragma: no cover
    dlp_v2 = None  # type: ignore[assignment]

try:
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
except ImportError:  # pragma: no cover
    CloudTraceSpanExporter = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 1. Active PII Redaction (Regex + Optional Google Cloud DLP)
# ---------------------------------------------------------------------------

_PII_REGEX_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Google API keys, OAuth tokens, OpenAI/GitHub/AWS tokens, generic secret assignments
    (
        re.compile(
            r"\b(?:AIza[0-9A-Za-z\-_]{20,}|ya29\.[0-9A-Za-z\-_]+|sk-[0-9A-Za-z\-_]{16,}|ghp_[0-9A-Za-z]{20,}|AKIA[0-9A-Z]{16})\b"
        ),
        "[REDACTED_API_KEY]",
    ),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|bearer)\s*[:=]\s*['\"]?([A-Za-z0-9\-_\.]{12,})['\"]?"
        ),
        r"\1=[REDACTED_API_KEY]",
    ),
    # Credit Card Numbers (13-16 digits, optional dashes or spaces)
    (
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{1,4}\b"),
        "[REDACTED_CREDIT_CARD]",
    ),
    # US Social Security Numbers
    (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[REDACTED_SSN]",
    ),
    # Email Addresses
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    # Phone Numbers (E.164 and North American formats)
    (
        re.compile(
            r"(?:(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}|\+\d{1,3}[-.\s]\d{2,4}[-.\s]\d{3,4}[-.\s]\d{3,4})\b"
        ),
        "[REDACTED_PHONE]",
    ),
]


def scrub_pii_data(text: str, use_cloud_dlp: bool | None = None) -> str:
    """Scrubs sensitive PII and credentials from text before logging or memory storage.

    Applies deterministic regex masks for emails, phone numbers, credit cards, SSNs,
    and API keys/tokens, and optionally invokes Google Cloud Data Loss Prevention
    (`google.cloud.dlp_v2`) when enabled in configuration.

    Args:
        text: Input string that may contain sensitive personal or credential data.
        use_cloud_dlp: Optional override to enable/disable Google Cloud DLP API call.

    Returns:
        Sanitized string with all detected PII replaced by redaction tokens.
    """
    if not isinstance(text, str) or not text:
        return text

    sanitized = text
    for pattern, replacement in _PII_REGEX_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)

    cfg = get_config()
    should_call_dlp = cfg.enable_cloud_dlp if use_cloud_dlp is None else use_cloud_dlp

    if should_call_dlp and dlp_v2 is not None and cfg.project_id:
        try:
            dlp_client = dlp_v2.DlpServiceClient()
            parent = f"projects/{cfg.project_id}/locations/global"
            inspect_config = {
                "info_types": [
                    {"name": "EMAIL_ADDRESS"},
                    {"name": "PHONE_NUMBER"},
                    {"name": "CREDIT_CARD_NUMBER"},
                    {"name": "GCP_API_KEY"},
                    {"name": "AUTH_TOKEN"},
                ]
            }
            deidentify_config = {
                "info_type_transformations": {
                    "transformations": [
                        {
                            "primitive_transformation": {
                                "replace_with_info_type_config": {}
                            }
                        }
                    ]
                }
            }
            response = dlp_client.deidentify_content(
                request={
                    "parent": parent,
                    "inspect_config": inspect_config,
                    "deidentify_config": deidentify_config,
                    "item": {"value": sanitized},
                }
            )
            if response and response.item and response.item.value:
                sanitized = response.item.value
        except Exception:
            # Graceful fallback: regex redaction has already scrubbed the payload.
            pass

    return sanitized


def scrub_pii_payload(payload: Any) -> Any:
    """Recursively scrubs PII across strings, dictionaries, lists, and tuples."""
    if isinstance(payload, str):
        return scrub_pii_data(payload)
    if isinstance(payload, dict):
        return {str(k): scrub_pii_payload(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [scrub_pii_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return tuple(scrub_pii_payload(item) for item in payload)
    return payload


# ---------------------------------------------------------------------------
# 2. Structured JSON Logging (Cloud Logging + python-json-logger)
# ---------------------------------------------------------------------------

RECENT_TELEMETRY_EVENTS: deque[dict[str, Any]] = deque(maxlen=500)


class FinOpsJsonFormatter(JsonFormatter):
    """Custom JSON log formatter guaranteeing session_id, agent_name, and severity."""

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = datetime.datetime.now(
            datetime.UTC
        ).isoformat()
        log_record["severity"] = record.levelname
        log_record["session_id"] = getattr(record, "session_id", None) or log_record.get(
            "session_id", "default-session"
        )
        log_record["agent_name"] = getattr(record, "agent_name", None) or log_record.get(
            "agent_name", "finops_concierge"
        )
        log_record["event_type"] = getattr(record, "event_type", None) or log_record.get(
            "event_type", "AGENT_LOG"
        )
        if "message" in log_record and isinstance(log_record["message"], str):
            log_record["message"] = scrub_pii_data(log_record["message"])


def configure_structured_logging(logger_name: str = "finops_concierge") -> logging.Logger:
    """Configures structured JSON logging with Google Cloud Logging and stream fallback.

    Args:
        logger_name: Name of the application logger to configure.

    Returns:
        Configured `logging.Logger` instance emitting structured JSON payloads.
    """
    structured_logger = logging.getLogger(logger_name)
    structured_logger.setLevel(logging.INFO)
    structured_logger.propagate = False

    if not structured_logger.handlers:
        cfg = get_config()
        if cfg.enable_cloud_logging:
            try:
                gcp_logging_client = cloud_logging.Client(project=cfg.project_id)
                cloud_handler = gcp_logging_client.get_default_handler()
                cloud_handler.setFormatter(
                    FinOpsJsonFormatter(
                        "%(timestamp)s %(severity)s %(session_id)s %(agent_name)s %(event_type)s %(message)s"
                    )
                )
                structured_logger.addHandler(cloud_handler)
            except Exception:
                # Fall back to standard stream JSON handler if Cloud Logging is unreachable
                pass

        stream_handler = logging.StreamHandler(stream=sys.stderr)
        stream_handler.setFormatter(
            FinOpsJsonFormatter(
                "%(timestamp)s %(severity)s %(session_id)s %(agent_name)s %(event_type)s %(message)s"
            )
        )
        structured_logger.addHandler(stream_handler)

    return structured_logger


_STRUCTURED_LOGGER = configure_structured_logging()


def log_structured_event(
    event_type: str,
    message: str,
    *,
    session_id: str = "default-session",
    agent_name: str = "finops_concierge",
    severity: str = "INFO",
    **extra_fields: Any,
) -> dict[str, Any]:
    """Emits a PII-scrubbed structured JSON log entry with mandatory metadata fields.

    Args:
        event_type: Semantic event classification (e.g., `"TOOL_INTENT"`,
            `"TOOL_OUTCOME"`, `"GUARDRAIL_CHECK"`, `"MEMORY_CONSOLIDATION"`).
        message: Human-readable summary of the event.
        session_id: Active conversation session identifier.
        agent_name: Name of the agent emitting the event.
        severity: Log severity (`"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`).
        **extra_fields: Additional structured key-value telemetry fields.

    Returns:
        The sanitized structured dictionary that was logged.
    """
    clean_message = scrub_pii_data(message)
    clean_extra = scrub_pii_payload(extra_fields)

    payload: dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "severity": severity.upper(),
        "session_id": scrub_pii_data(str(session_id)),
        "agent_name": scrub_pii_data(str(agent_name)),
        "event_type": event_type,
        "message": clean_message,
        **clean_extra,
    }

    RECENT_TELEMETRY_EVENTS.append(payload)

    level = getattr(logging, severity.upper(), logging.INFO)
    _STRUCTURED_LOGGER.log(
        level,
        clean_message,
        extra={
            "session_id": payload["session_id"],
            "agent_name": payload["agent_name"],
            "event_type": event_type,
            **clean_extra,
        },
    )
    return payload


# ---------------------------------------------------------------------------
# 3. Distributed Tracing (OpenTelemetry + Cloud Trace Exporter with Fallback)
# ---------------------------------------------------------------------------


class InMemorySpanBufferExporter(SpanExporter):
    """Lightweight in-memory span exporter for local development and unit test verification."""

    def __init__(self, maxlen: int = 200) -> None:
        self.spans: deque[Any] = deque(maxlen=maxlen)

    def export(self, spans: Any) -> SpanExportResult:
        for span in spans:
            self.spans.append(span)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


SPAN_BUFFER_EXPORTER = InMemorySpanBufferExporter()
_TRACER_INITIALIZED = False


def initialize_tracing(service_name: str = "finops-concierge-agent") -> trace.Tracer:
    """Initializes OpenTelemetry TracerProvider with CloudTraceSpanExporter and fallback.

    Args:
        service_name: Logical OTel service name attached to exported trace spans.

    Returns:
        Configured `opentelemetry.trace.Tracer` instance.
    """
    global _TRACER_INITIALIZED
    if not _TRACER_INITIALIZED:
        cfg = get_config()
        resource = Resource.create(
            {
                "service.name": os.environ.get("OTEL_SERVICE_NAME", service_name),
                "cloud.account.id": cfg.project_id,
                "cloud.region": cfg.location,
            }
        )
        provider = TracerProvider(resource=resource)

        # Always attach the lightweight in-memory exporter for deterministic local verification
        provider.add_span_processor(SimpleSpanProcessor(SPAN_BUFFER_EXPORTER))

        # Attach CloudTraceSpanExporter when enabled and available, with console fallback
        if cfg.enable_cloud_trace and CloudTraceSpanExporter is not None:
            try:
                cloud_exporter = CloudTraceSpanExporter(project_id=cfg.project_id)
                provider.add_span_processor(BatchSpanProcessor(cloud_exporter))
            except Exception as exc:
                log_structured_event(
                    "TRACING_FALLBACK",
                    f"CloudTraceSpanExporter unavailable ({type(exc).__name__}); using local span buffer.",
                    severity="WARNING",
                )
        elif os.environ.get("OTEL_LOG_SPANS_CONSOLE", "").lower() in ("1", "true"):
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        # Set global tracer provider if not already overridden
        try:
            trace.set_tracer_provider(provider)
        except Exception:
            pass
        _TRACER_INITIALIZED = True

    return trace.get_tracer(service_name)


def get_tracer() -> trace.Tracer:
    """Returns the application OpenTelemetry tracer."""
    return initialize_tracing()


# Convenience aliases used across modules
get_structured_logger = configure_structured_logging
init_telemetry = initialize_tracing


@contextlib.contextmanager
def trace_span(
    span_name: str, attributes: dict[str, Any] | None = None
) -> Iterator[trace.Span]:
    """Wraps an agent or tool operation in a linked OpenTelemetry span.

    Args:
        span_name: Name of the span (e.g., `"tool.analyze_project_cost_anomalies"`).
        attributes: Key-value attributes to record on the span (automatically PII-scrubbed).

    Yields:
        Active OpenTelemetry `Span`.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(span_name) as span:
        if attributes:
            clean_attrs = scrub_pii_payload(attributes)
            for key, val in clean_attrs.items():
                if val is not None:
                    span.set_attribute(
                        str(key),
                        val
                        if isinstance(val, (bool, int, float, str))
                        else json.dumps(val, default=str),
                    )
        yield span
