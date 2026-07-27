"""Matched reference/candidate benchmarking through normal execution adapters."""

from nerve.representation_optimizer.benchmarking.contracts import (
    BENCHMARK_OBSERVATION_SCHEMA,
    BENCHMARK_EVIDENCE_INTEGRITY_SCHEMA,
    BENCHMARK_PLAN_SCHEMA,
    BENCHMARK_RECORD_SCHEMA,
    BENCHMARK_RESIDENCY_EVENT_SCHEMA,
    BENCHMARK_RUN_SCHEMA,
    BENCHMARK_WORKLOAD_SCHEMA,
    BenchmarkContractError,
    BenchmarkObservation,
    BenchmarkPlan,
    BenchmarkResidencyEvent,
    BenchmarkRun,
    BenchmarkWorkload,
)
from nerve.representation_optimizer.benchmarking.executor_adapter import (
    ResidentComponentExecutionAdapter,
)

__all__ = [
    "BENCHMARK_OBSERVATION_SCHEMA",
    "BENCHMARK_EVIDENCE_INTEGRITY_SCHEMA",
    "BENCHMARK_PLAN_SCHEMA",
    "BENCHMARK_RECORD_SCHEMA",
    "BENCHMARK_RESIDENCY_EVENT_SCHEMA",
    "BENCHMARK_RUN_SCHEMA",
    "BENCHMARK_WORKLOAD_SCHEMA",
    "BenchmarkContractError",
    "BenchmarkObservation",
    "BenchmarkPlan",
    "BenchmarkResidencyEvent",
    "BenchmarkRun",
    "BenchmarkWorkload",
    "ResidentComponentExecutionAdapter",
]
