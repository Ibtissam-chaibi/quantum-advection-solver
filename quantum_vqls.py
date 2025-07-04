import numpy as np
import matplotlib.pyplot as plt
import time
from classical_fd import generate_matrix, initial_condition, validate_parameters
from quantum_vqls import solve_linear_system

# Physics parameters
C = 1.0       # Wave speed
L = 1.0       # Domain length
T_final = 0.5 # Simulation time

# Discretization parameters
N = 4         # Grid points (must be power of 2)
dx = L / N
dt = 0.01
steps = int(T_final / dt)
r = C * dt / (2 * dx)  # Stability parameter

# Validate parameters
validate_parameters(C, dx, dt)

# Generate system matrix
A = generate_matrix(N, r)
print("System matrix A:\n", A)

# Spatial grid and initial condition
x = np.linspace(0, L, N, endpoint=False)
u_classic = u_quantum = initial_condition(x, "gaussian")
print("Initial condition:", u_quantum)

# Time-stepping
quantum_times, classic_times = [], []
for step in range(steps):
    # Classical solve (reference)
    t0 = time.time()
    u_classic = np.linalg.solve(A, u_classic)
    classic_times.append(time.time() - t0)
    
    # Quantum solve
    t0 = time.time()
    b = u_quantum / np.linalg.norm(u_quantum)  # Normalize
    u_quantum = solve_linear_system(A, b)
    u_quantum *= np.linalg.norm(b)  # Rescale to physical magnitude
    quantum_times.append(time.time() - t0)
    
    if step % 10 == 0:
        error = np.linalg.norm(u_quantum - u_classic)
        print(f"Step {step:4d}/{steps}: Error = {error:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(x, u_classic, 'o-', label="Classical")
plt.plot(x, u_quantum, 'x--', label="Quantum")
plt.title("Final Solution")
plt.xlabel("Position")
plt.ylabel("u(x)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(classic_times, label="Classical")
plt.plot(quantum_times, label="Quantum")
plt.title("Compute Time per Step")
plt.xlabel("Time Step")
plt.ylabel("Seconds")
plt.yscale('log')
plt.legend()

plt.tight_layout()
plt.savefig("results/comparison.png")

# Performance summary
print("\n=== Performance Summary ===")
print(f"Classical avg time: {np.mean(classic_times):.4f} ± {np.std(classic_times):.4f} s")
print(f"Quantum avg time: {np.mean(quantum_times):.4f} ± {np.std(quantum_times):.4f} s")
print(f"Final solution error: {np.linalg.norm(u_quantum - u_classic):.6f}")
