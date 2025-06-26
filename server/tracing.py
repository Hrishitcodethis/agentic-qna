from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
import os

def setup_tracing(service_name: str):
    """
    Set up OpenTelemetry tracing for the service.
    Args:
        service_name: Name of the service for tracing
    """
    # Create a resource with service information
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: service_name,
        ResourceAttributes.SERVICE_VERSION: "0.1.0",
    })

    # Create and set the tracer provider
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    return trace.get_tracer(service_name)

def get_tracer(service_name: str):
    """
    Get a tracer for the service. If tracing is not set up, set it up first.
    Args:
        service_name: Name of the service for tracing
    Returns:
        An OpenTelemetry tracer instance
    """
    if not trace.get_tracer_provider():
        setup_tracing(service_name)
    return trace.get_tracer(service_name) 