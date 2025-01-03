"""Custom exceptions for topo_explorer."""

class TopoExplorerError(Exception):
    """Base exception for topo_explorer."""
    pass

class ManifoldError(TopoExplorerError):
    """Exception raised for errors in manifold operations."""
    pass

class ConfigurationError(TopoExplorerError):
    """Exception raised for errors in configuration."""
    pass

class VisualizationError(TopoExplorerError):
    """Exception raised for errors in visualization."""
    pass

class GeometricError(TopoExplorerError):
    """Exception raised for errors in geometric computations."""
    pass

class ParallelTransportError(GeometricError):
    """Exception raised for errors in parallel transport."""
    pass

class ProjectionError(GeometricError):
    """Exception raised for errors in projection operations."""
    pass

class ParametrizationError(GeometricError):
    """Exception raised for errors in manifold parametrization."""
    pass