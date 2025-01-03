"""Mathematical utility functions for geometric computations."""

import numpy as np
from typing import Tuple, Optional, List, Union, Callable
from scipy.integrate import solve_ivp

def normalize_vector(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize a vector with numerical stability.
    
    Args:
        v: Vector to normalize
        eps: Small number for numerical stability
    
    Returns:
        Normalized vector
        
    Raises:
        ValueError: If vector has near-zero norm
    """
    norm = np.linalg.norm(v)
    if norm < eps:
        raise ValueError("Vector has near-zero norm")
    return v / norm

def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    """
    Orthonormalize a set of vectors using modified Gram-Schmidt process.
    
    Args:
        vectors: Array of shape (n, d) containing n d-dimensional vectors
    
    Returns:
        Array of orthonormalized vectors
        
    Raises:
        ValueError: If input vectors are linearly dependent
    """
    n = len(vectors)
    orthogonal = np.zeros_like(vectors)
    
    for i in range(n):
        orthogonal[i] = vectors[i].copy()
        for j in range(i):
            projection = np.dot(vectors[i], orthogonal[j]) * orthogonal[j]
            orthogonal[i] = orthogonal[i] - projection
        try:
            orthogonal[i] = normalize_vector(orthogonal[i])
        except ValueError:
            raise ValueError(f"Vector {i} is linearly dependent")
    
    return orthogonal

def parallel_transport_matrix(v1: np.ndarray, 
                            v2: np.ndarray, 
                            eps: float = 1e-8) -> np.ndarray:
    """
    Compute parallel transport matrix between two unit vectors.
    
    Uses Rodrigues rotation formula to compute the transformation matrix
    that parallel transports vectors from v1 to v2.
    
    Args:
        v1: Initial unit vector
        v2: Final unit vector
        eps: Small number for numerical stability
    
    Returns:
        3x3 parallel transport matrix
    """
    v1 = normalize_vector(v1)
    v2 = normalize_vector(v2)
    
    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    
    if abs(cos_theta - 1.0) < eps:  
        return np.eye(3)
        
    if abs(cos_theta + 1.0) < eps:    
        axis = np.array([1.0, 0.0, 0.0]) if abs(v1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = normalize_vector(axis - np.dot(axis, v1) * v1)
    else:
        axis = normalize_vector(np.cross(v1, v2))
    
    theta = np.arccos(cos_theta)
    
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - cos_theta) * K @ K

def christoffel_symbols(metric_tensor: Callable[[np.ndarray], np.ndarray],
                       point: np.ndarray,
                       h: float = 1e-5) -> np.ndarray:
    """
    Compute Christoffel symbols using finite differences.
    
    Args:
        metric_tensor: Function that computes metric tensor at a point
        point: Point at which to compute Christoffel symbols
        h: Step size for finite differences
    
    Returns:
        Array of shape (d, d, d) containing Christoffel symbols
    """
    d = len(point)
    gamma = np.zeros((d, d, d))
    
    g = metric_tensor(point)
    g_inv = np.linalg.inv(g)
    dg = np.zeros((d, d, d))
    
    for k in range(d):
        point_plus = point.copy()
        point_minus = point.copy()
        point_plus[k] += h
        point_minus[k] -= h
        dg[:,:,k] = (metric_tensor(point_plus) - metric_tensor(point_minus)) / (2*h)
    
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    gamma[i,j,k] += 0.5 * g_inv[i,l] * (
                        dg[l,j,k] + dg[l,k,j] - dg[j,k,l]
                    )
    
    return gamma

def geodesic_equation(metric_tensor: Callable[[np.ndarray], np.ndarray],
                     t: float,
                     state: np.ndarray) -> np.ndarray:
    """
    Compute right-hand side of geodesic equation.
    
    Args:
        metric_tensor: Function that computes metric tensor at a point
        t: Current time parameter
        state: Current state [position, velocity]
    
    Returns:
        Time derivative of state [velocity, acceleration]
    """
    d = len(state) // 2
    x = state[:d]  
    v = state[d:]  
    
    gamma = christoffel_symbols(metric_tensor, x)
    
    a = np.zeros(d)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                a[i] -= gamma[i,j,k] * v[j] * v[k]
    
    return np.concatenate([v, a])

def compute_geodesic(start_point: np.ndarray,
                    start_velocity: np.ndarray,
                    metric_tensor: Callable[[np.ndarray], np.ndarray],
                    t_span: Tuple[float, float],
                    n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute geodesic curve by solving the geodesic equation.
    
    Args:
        start_point: Starting point of geodesic
        start_velocity: Initial velocity vector
        metric_tensor: Function that computes metric tensor at a point
        t_span: (t_start, t_end) for integration
        n_points: Number of points to return
    
    Returns:
        Tuple of (times, points) along geodesic
    """
    initial_state = np.concatenate([start_point, start_velocity])
    
    solution = solve_ivp(
        lambda t, y: geodesic_equation(metric_tensor, t, y),
        t_span,
        initial_state,
        t_evaluation=np.linspace(t_span[0], t_span[1], n_points)
    )
    
    d = len(start_point)
    points = solution.y[:d].T
    
    return solution.t, points

def riemann_curvature_tensor(metric_tensor: Callable[[np.ndarray], np.ndarray],
                           point: np.ndarray,
                           h: float = 1e-5) -> np.ndarray:
    """
    Compute Riemann curvature tensor at a point.
    
    Args:
        metric_tensor: Function that computes metric tensor at a point
        point: Point at which to compute curvature
        h: Step size for finite differences
    
    Returns:
        Array of shape (d, d, d, d) containing Riemann tensor components
    """
    d = len(point)
    R = np.zeros((d, d, d, d))
    
    gamma = christoffel_symbols(metric_tensor, point)
    dgamma = np.zeros((d, d, d, d))
    
    for l in range(d):
        point_plus = point.copy()
        point_minus = point.copy()
        point_plus[l] += h
        point_minus[l] -= h
        gamma_plus = christoffel_symbols(metric_tensor, point_plus)
        gamma_minus = christoffel_symbols(metric_tensor, point_minus)
        dgamma[:,:,:,l] = (gamma_plus - gamma_minus) / (2*h)
    
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    R[i,j,k,l] = dgamma[i,l,j,k] - dgamma[i,k,j,l]
                    for m in range(d):
                        R[i,j,k,l] += (
                            gamma[i,l,m] * gamma[m,k,j] -
                            gamma[i,k,m] * gamma[m,l,j]
                        )
    
    return R

def sectional_curvature(metric_tensor: Callable[[np.ndarray], np.ndarray],
                       point: np.ndarray,
                       v1: np.ndarray,
                       v2: np.ndarray) -> float:
    """
    Compute sectional curvature for a 2-plane spanned by two vectors.
    
    Args:
        metric_tensor: Function that computes metric tensor at a point
        point: Point at which to compute curvature
        v1, v2: Two vectors spanning the 2-plane
    
    Returns:
        Sectional curvature value
    """
    g = metric_tensor(point)
    R = riemann_curvature_tensor(metric_tensor, point)
    
    numerator = 0
    for i in range(len(point)):
        for j in range(len(point)):
            for k in range(len(point)):
                for l in range(len(point)):
                    numerator += R[i,j,k,l] * v1[i] * v2[j] * v1[k] * v2[l]
    
    g11 = sum(g[i,j] * v1[i] * v1[j] for i in range(len(point)) 
              for j in range(len(point)))
    g12 = sum(g[i,j] * v1[i] * v2[j] for i in range(len(point)) 
              for j in range(len(point)))
    g22 = sum(g[i,j] * v2[i] * v2[j] for i in range(len(point)) 
              for j in range(len(point)))
    denominator = g11 * g22 - g12 * g12
    
    return numerator / denominator

def exponential_map(point: np.ndarray,
                   vector: np.ndarray,
                   metric_tensor: Callable[[np.ndarray], np.ndarray],
                   t_max: float = 1.0) -> np.ndarray:
    """
    Compute exponential map of a vector at a point.
    
    Args:
        point: Base point
        vector: Tangent vector
        metric_tensor: Function that computes metric tensor at a point
        t_max: Parameter value at which to evaluate exponential map
    
    Returns:
        Point reached by exponential map
    """
    _, points = compute_geodesic(
        point,
        vector,
        metric_tensor,
        (0, t_max),
        n_points=2
    )
    return points[-1]

def parallel_transport_along_curve(vector: np.ndarray,
                                curve: np.ndarray,
                                metric_tensor: Callable[[np.ndarray], np.ndarray],
                                n_steps: int = 100) -> List[np.ndarray]:
    """
    Parallel transport a vector along a curve.
    
    Args:
        vector: Initial vector to transport
        curve: Array of points defining the curve
        metric_tensor: Function that computes metric tensor at a point
        n_steps: Number of steps for numerical integration
    
    Returns:
        List of transported vectors at each point along the curve
    """
    result = [vector]
    
    for i in range(len(curve)-1):
        tangent = curve[i+1] - curve[i]
        
        gamma = christoffel_symbols(metric_tensor, curve[i])
        
        transported = result[-1].copy()
        for j in range(len(vector)):
            for k in range(len(vector)):
                for l in range(len(vector)):
                    transported[j] -= gamma[j,k,l] * tangent[k] * result[-1][l]
        
        result.append(normalize_vector(transported))
    
    return result