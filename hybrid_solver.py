import numpy as np
import matplotlib.pyplot as plt
import time
from classical_fd import generate_matrix, initial_condition, validate_parameters
from quantum_vqls import solve_linear_system

# Physics parameters
C = 0.5       # Reduced wave speed for stability
L = 1.0       # Domain length
T_final = 0.1 # Reduced simulation time

# Discretization parameters
N = 4         # Grid points (must be power of 2)
dx = L / N
dt = 0.005    # Reduced time step
steps = int(T_final / dt)
r = C * dt / (2 * dx)  # Stability parameter

print(f"Using parameters: r={r:.4f}, steps={steps}")

# Validate parameters
validate_parameters(C, dx, dt)

# Generate system matrix
A = generate_matrix(N, r)
print("System matrix A:\n", np.round(A, 4))

# Spatial grid and initial condition
x = np.linspace(0, L, N, endpoint=False)
u_classic = initial_condition(x, "square")  # Easier to solve
u_quantum = u_classic.copy()
print("Initial condition:", np.round(u_quantum, 4))

# Time-stepping
quantum_times, classic_times = [], []
errors = []

for step in range(steps):
    print(f"\n--- Step {step+1}/{steps} ---")
    
    # Classical solve (reference)
    t0 = time.time()
    u_classic = np.linalg.solve(A, u_classic)
    classic_time = time.time() - t0
    classic_times.append(classic_time)
    print(f"Classical solved in {classic_time:.4f}s")
    
    # Quantum solve
    t0 = time.time()
    try:
        # Normalize and solve
        norm_b = np.linalg.norm(u_quantum)
        b = u_quantum / norm_b
        u_quantum = solve_linear_system(A, b, num_layers=1, max_iter=30)
        
        # Rescale to physical magnitude
        u_quantum = norm_b * u_quantum
        quantum_time = time.time() - t0
        quantum_times.append(quantum_time)
        print(f"Quantum solved in {quantum_time:.4f}s")
        
        # Calculate error
        error = np.linalg.norm(u_quantum - u_classic)
        errors.append(error)
        print(f"Error: {error:.6f}")
    except Exception as e:
        print(f"Quantum solve failed: {str(e)}")
        break

# Plot results
plt.figure(figsize=(12, 5))

# Solution plot
plt.subplot(1, 2, 1)
plt.plot(x, u_classic, 'o-', label="Classical")
plt.plot(x, u_quantum, 'x--', label="Quantum")
plt.title(f"Solution at t={T_final:.2f}")
plt.xlabel("Position")
plt.ylabel("u(x)")
plt.legend()
plt.grid(True)

# Error plot
plt.subplot(1, 2, 2)
plt.plot(errors, 'r-')
plt.title("Error Evolution")
plt.xlabel("Time Step")
plt.ylabel("L2 Error")
plt.grid(True)
plt.yscale('log')

plt.tight_layout()
plt.savefig("results/comparison.png")
print("Saved results/comparison.png")

# Performance summary
print("\n=== Performance Summary ===")
print(f"Classical avg time: {np.mean(classic_times):.6f} ± {np.std(classic_times):.6f} s")
print(f"Quantum avg time: {np.mean(quantum_times):.6f} ± {np.std(quantum_times):.6f} s")
if len(errors) > 0:
    print(f"Final solution error: {errors[-1]:.6f}")
