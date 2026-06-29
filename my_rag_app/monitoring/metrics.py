"""
In-memory metrics counters for guardrail activity.
Simple, process-local counters — no external metrics backend yet.
"""

from dataclasses import dataclass, field


@dataclass
class GuardrailMetrics:
    pii_detections: int = 0
    blocked_requests: int = 0

    def record_pii_detection(self, count: int = 1) -> None:
        self.pii_detections += count

    def record_blocked_request(self) -> None:
        self.blocked_requests += 1

    def snapshot(self) -> dict:
        return {"pii_detections": self.pii_detections, "blocked_requests": self.blocked_requests}


# Process-wide singleton — imported by pii.py and validation.py to record events
metrics = GuardrailMetrics()