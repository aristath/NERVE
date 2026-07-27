"""Deterministic unattended representation-optimization coordination."""

from nerve.representation_optimizer.automation.contracts import (
    CandidateResourceCost,
    OptimizationBudget,
)
from nerve.representation_optimizer.automation.command import (
    OptimizePackageOutcome,
    optimize_compiled_package,
)
from nerve.representation_optimizer.automation.orchestrator import (
    run_automated_optimizer,
)
from nerve.representation_optimizer.automation.report import (
    AutomatedOptimizationOutcome,
    validate_report_directory,
)
from nerve.representation_optimizer.automation.target import (
    CandidateToolchain,
    CandidateToolchainResolver,
    DeviceLeaseManager,
    NoDeviceLeaseManager,
    OptimizationTarget,
    VerifiedDeviceLeaseManager,
)

__all__ = [
    "AutomatedOptimizationOutcome",
    "CandidateResourceCost",
    "CandidateToolchain",
    "CandidateToolchainResolver",
    "DeviceLeaseManager",
    "NoDeviceLeaseManager",
    "OptimizePackageOutcome",
    "OptimizationBudget",
    "OptimizationTarget",
    "VerifiedDeviceLeaseManager",
    "run_automated_optimizer",
    "optimize_compiled_package",
    "validate_report_directory",
]
