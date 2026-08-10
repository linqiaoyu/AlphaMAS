"""Frozen experiment configuration definitions shared with the interactive CLI."""

RESEARCH_DEPTH_ROUNDS = {"shallow": 1, "medium": 3, "deep": 5}


def resolve_research_rounds(depth: str) -> int:
    try:
        return RESEARCH_DEPTH_ROUNDS[depth.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown research depth: {depth}") from exc
