"""Custom exceptions for wbia-core."""


class WbiaCoreError(Exception):
    """Base exception for all wbia-core errors."""


class ConfigError(WbiaCoreError):
    """Invalid configuration."""


class DataError(WbiaCoreError):
    """Invalid data or data format."""


class FeatureExtractionError(WbiaCoreError):
    """Feature extraction failed."""


class IndexError(WbiaCoreError):
    """Index build or query failed."""


class SpatialVerificationError(WbiaCoreError):
    """Spatial verification failed."""


class NotInstalledError(WbiaCoreError, ImportError):
    """Optional dependency not installed."""
