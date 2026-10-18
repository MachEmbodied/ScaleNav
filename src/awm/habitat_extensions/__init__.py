"""AWM Habitat-Sim evaluation extensions for VLN benchmarks."""

from typing import Any

__all__ = ["AWMAgent", "AWMHabitatGSEvaluator"]


def __getattr__(name: str) -> Any:
    """Keep the public API without importing the full model stack eagerly."""
    if name == "AWMAgent":
        from awm.habitat_extensions.awm_agent import AWMAgent
        return AWMAgent
    if name == "AWMHabitatGSEvaluator":
        from awm.habitat_extensions.awm_habitat_gs_evaluator import AWMHabitatGSEvaluator
        return AWMHabitatGSEvaluator
    raise AttributeError(name)
