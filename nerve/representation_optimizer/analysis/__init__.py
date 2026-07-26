"""Architecture-neutral algebraic and structural analysis."""

from nerve.representation_optimizer.analysis.context import (
    ActivationTrace,
    AnalysisBudget,
    ScopeAnalysisContext,
)
from nerve.representation_optimizer.analysis.engine import (
    AnalysisRun,
    analyze_scope,
    builtin_analyzers,
)
from nerve.representation_optimizer.analysis.evidence import (
    ANALYSIS_RUN_SCHEMA,
    validate_analysis_run_directory,
)
from nerve.representation_optimizer.analysis.tensor_repository import (
    InMemoryTensorRepository,
    PackageTensorRepository,
    TensorObservation,
)

__all__ = [
    "ActivationTrace",
    "ANALYSIS_RUN_SCHEMA",
    "AnalysisBudget",
    "AnalysisRun",
    "InMemoryTensorRepository",
    "PackageTensorRepository",
    "ScopeAnalysisContext",
    "TensorObservation",
    "analyze_scope",
    "builtin_analyzers",
    "validate_analysis_run_directory",
]
