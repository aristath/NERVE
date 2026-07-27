"""Deterministic unattended representation-optimization coordination."""

from nerve.representation_optimizer.automation.contracts import (
    CandidateResourceCost,
    OptimizationBudget,
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
    "OptimizationBudget",
    "OptimizationTarget",
    "VerifiedDeviceLeaseManager",
    "run_automated_optimizer",
    "validate_report_directory",
]
