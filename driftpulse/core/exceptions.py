"""Custom exception classes for DriftPulse-AI."""

from __future__ import annotations


class DriftPulseError(Exception):
    """Base exception for all DriftPulse errors."""
    pass


class ProfilingError(DriftPulseError):
    """Raised when data profiling fails due to incompatible data format or empty samples."""
    pass


class StatisticalTestError(DriftPulseError):
    """Raised when a mathematical or statistical calculation fails."""
    pass


class IncompatibleSchemaError(DriftPulseError):
    """Raised when baseline and production datasets have mismatched feature sets."""
    pass


class ConfigurationError(DriftPulseError):
    """Raised when threshold or pipeline configuration parameters are invalid."""
    pass


class AlertDispatchError(DriftPulseError):
    """Raised when dispatching monitoring alerts fails."""
    pass
