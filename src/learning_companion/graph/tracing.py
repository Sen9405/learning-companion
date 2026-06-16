"""OpenTelemetry tracing for Phoenix."""

from __future__ import annotations

import os


def setup_tracing(
    endpoint: str | None = None,
    service_name: str = "learning-companion",
) -> bool:
    """Setup OpenTelemetry tracing for Phoenix.

    Returns True if tracing was configured, False otherwise.
    """
    try:
        from opentelemetry import trace as trace_api
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        endpoint = endpoint or os.environ.get(
            "PHOENIX_COLLECTOR_ENDPOINT",
            "http://localhost:4317",
        )

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace_api.set_tracer_provider(provider)

        # Patch popular libraries
        try:
            from opentelemetry.instrumentation.requests import RequestsInstrumentor
            RequestsInstrumentor().instrument()
        except Exception:
            pass

        print(f"[Tracing] Phoenix OTLP endpoint: {endpoint}")
        return True
    except ImportError:
        print("[Tracing] OpenTelemetry packages not installed, skipping")
        return False
    except Exception as e:
        print(f"[Tracing] Setup error: {e}")
        return False
