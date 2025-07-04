import numpy as np

def generate_matrix(N, r):
    """Create circulant tridiagonal matrix for periodic BC"""
    A = np.eye(N)
    for i in range(N):
        A[i, (i+1) % N] = r
        A[i, (i-1) % N] = -r
    return A

def validate_parameters(C, dx, dt):
    """Check CFL condition for stability"""
    cfl = C * dt / dx
    if abs(cfl) > 1.0:
        raise ValueError(f"CFL={cfl:.2f} > 1.0 (unstable)")

def initial_condition(x, type="gaussian"):
    """Generate initial profile"""
    if type == "gaussian":
        return np.exp(-40 * (x - 0.5)**2)
    elif type == "square":
        return np.where((x > 0.3) & (x < 0.7), 1.0, 0.0)
    else:
        raise ValueError("Invalid IC type")
