import pennylane as qml
from pennylane import numpy as np

def ansatz(theta, wires):
    """Hardware-efficient ansatz with trainable rotations"""
    n_qubits = len(wires)
    layers = len(theta) // (2 * n_qubits)
    idx = 0
    
    for _ in range(layers):
        # First set of rotations
        for i in wires:
            qml.RY(theta[idx], wires=i)
            idx += 1
        
        # Forward entanglement
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[wires[i], wires[i+1]])
        
        # Second set of rotations
        for i in wires:
            qml.RY(theta[idx], wires=i)
            idx += 1
        
        # Reverse entanglement (safe implementation)
        if n_qubits > 1:
            for i in range(n_qubits - 2, -1, -1):
                qml.CNOT(wires=[wires[i], wires[i+1]])

def solve_linear_system(A, b, num_layers=1, max_iter=30):
    """VQLS solver for Au = b"""
    n_qubits = int(np.log2(len(b)))
    dev = qml.device("default.qubit", wires=n_qubits)
    num_params = 2 * n_qubits * num_layers
    
    @qml.qnode(dev)
    def circuit(theta):
        ansatz(theta, wires=range(n_qubits))
        return qml.state()
    
    def cost(theta):
        psi = circuit(theta)
        A_psi = A @ psi
        
        # Numerically stable cost calculation
        residual = A_psi - b
        cost_value = np.vdot(residual, residual).real
        return cost_value + 1e-6 * np.vdot(psi, psi).real  # Regularization
    
    # Initialize parameters
    theta = np.random.uniform(0, 2*np.pi, num_params, requires_grad=True)
    opt = qml.AdamOptimizer(0.05)
    
    # Optimization loop
    for i in range(max_iter):
        theta, loss = opt.step_and_cost(cost, theta)
        if i % 5 == 0:
            print(f"Iter {i:3d}: Loss = {loss:.6f}")
    
    # Compute robust scaling
    psi_opt = circuit(theta)
    numerator = np.vdot(b, A @ psi_opt)
    denominator = np.vdot(psi_opt, A.conj().T @ A @ psi_opt)
    
    if abs(denominator) > 1e-8:
        scale = (numerator / denominator).real
    else:
        scale = 1.0
        print("Warning: Small denominator, using scale=1")
    
    return scale * np.real(psi_opt)
