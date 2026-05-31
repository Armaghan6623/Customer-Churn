"""customer_crunch.aiops

Autonomous Operations layer (data drift monitoring, self-healing, etc.).

Exports:
- ks_drift_test
- monitor_drift
- self_heal_if_needed
"""

from .drift import ks_drift_test, monitor_drift
from .self_heal import self_heal_if_needed

__all__ = [
    "ks_drift_test",
    "monitor_drift",
    "self_heal_if_needed",
]


