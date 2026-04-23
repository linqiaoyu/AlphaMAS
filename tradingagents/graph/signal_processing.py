# TradingAgents/graph/signal_processing.py

import re
from typing import Any


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: Any):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted rating (BUY, HOLD, or SELL)
        """
        if not full_signal:
            return "HOLD"

        normalized = str(full_signal).upper()
        patterns = [
            r"RATING\s*:\s*\**\s*(BUY|HOLD|SELL)\b",
            r"FINAL TRANSACTION PROPOSAL:\s*\**\s*(BUY|HOLD|SELL)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                return match.group(1)

        return "HOLD"
