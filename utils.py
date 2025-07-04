import numpy as np

def cfl_condition(C, dx, dt):
    """Calculate CFL number"""
    return C * dt / dx

def l2_error(u, u_ref):
    """Compute relative L2 error"""
    return np.linalg.norm(u - u_ref) / np.linalg.norm(u_ref)

def save_solution(step, x, u_quantum, u_classic):
    """Save solution snapshots"""
    np.savez(f"results/step_{step:04d}.npz", 
             x=x, quantum=u_quantum, classic=u_classic)
