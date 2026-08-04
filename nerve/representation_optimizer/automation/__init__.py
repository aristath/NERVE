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
    CapacityLeaseState,
    CandidateToolchain,
    CandidateToolchainResolver,
    DeviceLeaseManager,
    NoDeviceLeaseManager,
    OptimizationTarget,
    VerifiedCapacityLeaseManager,
)

__all__ = [
    "AutomatedOptimizationOutcome",
    "CapacityLeaseState",
    "CandidateResourceCost",
    "CandidateToolchain",
    "CandidateToolchainResolver",
    "DeviceLeaseManager",
    "NoDeviceLeaseManager",
    "OptimizationBudget",
    "OptimizationTarget",
    "VerifiedCapacityLeaseManager",
    "run_automated_optimizer",
    "validate_report_directory",
]
