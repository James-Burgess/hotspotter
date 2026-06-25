"""Custom exceptions for hotspotter."""


class HotspotterError(Exception):
    """Base exception for all hotspotter errors."""


class ConfigError(HotspotterError):
    """Invalid configuration."""


class DataError(HotspotterError):
    """Invalid data or data format."""


class FeatureExtractionError(HotspotterError):
    """Feature extraction failed."""


class IndexError(HotspotterError):
    """Index build or query failed."""


class SpatialVerificationError(HotspotterError):
    """Spatial verification failed."""


class NotInstalledError(HotspotterError, ImportError):
    """Optional dependency not installed."""
