from __future__ import annotations

from enum import Enum


class Phase(str, Enum):
    FEASIBILITY = "FEASIBILITY"
    BASELINE = "BASELINE"
    MESH = "MESH"
    GROUPING = "GROUPING"
    LIFECYCLE = "LIFECYCLE"
    PLACEMENT = "PLACEMENT"
    OFFLOAD = "OFFLOAD"


def next_phase(current: Phase, improved: bool) -> Phase:
    if improved:
        return current
    if current == Phase.FEASIBILITY:
        return Phase.FEASIBILITY
    if current == Phase.BASELINE:
        return Phase.MESH
    if current == Phase.MESH:
        return Phase.GROUPING
    if current == Phase.GROUPING:
        return Phase.LIFECYCLE
    if current == Phase.LIFECYCLE:
        return Phase.PLACEMENT
    if current == Phase.PLACEMENT:
        return Phase.OFFLOAD
    return Phase.OFFLOAD
