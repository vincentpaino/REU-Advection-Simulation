import numpy as np
import matplotlib.pyplot as plt

# --- 1. Problem Setup ---
# Define the physical and numerical parameters for the simulation.
# These parameters correspond to the problem defined in the report [cite: 71-78, 184].
NX = 100  # Number of spatial grid points
T = 1.0  # Final time
DT = 0.01  # Time step size
X_MIN, X_MAX = -1.0, 1.0  # Domain limits
A = 1.0  # Advection speed [cite: 74]
BETA = 1e-4 # Regularization parameter for the cost function [cite: 184]
LEARNING_RATE = 5e-5 # Step size for gradient descent
ITERATIONS = 500 # Number of optimization iterations

# Calculate derived parameters
NT = int(T / DT)  # Number of time steps
X = np.linspace(X_MIN, X_MAX, NX)  # Spatial grid
DX = (X_MAX - X_MIN) / (NX - 1)  # Spatial step size

# --- 2. Generate True Data ---
# To test the method, we first generate "observed" data.
# We define a "true" initial condition, solve the advection equation forward,
# and use the solution at the final time T as our observation data `u_d` [cite: 71-72].
def true_initial_condition(x):
    """The actual initial condition we want to recover[cite: 185]."""
    return np.where((x > -0.5) & (x < 0), 1.0, 0.5)

def forward_solve(u0):
    """
    Solves the 1D advection equation u_t + a*u_x = 0 using a forward-time,
    backward-space discretization scheme [cite: 77-78].
    This function solves the governing equation N(u,φ)=0[cite: 74].
    
    Args:
        u0: The initial condition at t=0.

    Returns:
        A matrix containing the solution u(x,t) for all time steps.
    """
    u_hist = np.zeros((NT + 1, NX))
    u_hist[0, :] = u0
    
    for k in range(1, NT + 1):  # Time loop from step 1 to m
        for i in range(1, NX):  # Space loop
            # Discretization: u_i^k = u_i^(k-1) - a * dt/dx * (u_i^(k-1) - u_(i-1)^(k-1))
            # This is a rearrangement of Eq. on page 6 [cite: 78]
            u_hist[k, i] = u_hist[k-1, i] - A * DT / DX * (u_hist[k-1, i] - u_hist[k-1, i-1])
        # Apply boundary condition at the left boundary (x=-1) [cite: 74]
        # In this example, we assume c=0.5, similar to the nonlinear example's g(x) base.
        u_hist[k, 0] = 0.5 
    return u_hist

# Generate the "observed" data
U_TRUE_0 = true_initial_condition(X)
U_HISTORY_TRUE = forward_solve(U_TRUE_0)
U_OBSERVED = U_HISTORY_TRUE[-1, :] # Data at final time T [cite: 71]

# --- 3. Adjoint Method Implementation ---

def calculate_cost(u_final, u_observed, u0):
    """
    Calculates the cost function J.
    The cost function measures the mismatch between the solution at the final
    time and the observed data, plus a regularization term[cite: 72, 184].
    J = ∫(u(x,T) - u_d(x,T))² dx + β * ∫(∇u(x,0))² dx
    """
    misfit = np.sum((u_final - u_observed)**2) * DX
    
    # Regularization term penalizes large gradients in the initial condition [cite: 127]
    grad_u0 = np.gradient(u0, DX)
    regularization = BETA * np.sum(grad_u0**2) * DX
    
    return misfit + regularization

def adjoint_solve(u_history, u_observed):
    """
    Solves the discrete adjoint problem backwards in time to find the adjoint
    variables (Lagrange multipliers) λ at time step 1.
    This implements "Algorithm 1" (A1) from the report[cite: 125].
    The adjoint equation is (∂N/∂u)ᵀ λ = (∂J/∂u)ᵀ[cite: 65].

    Args:
        u_history: The solution from the forward solve.
        u_observed: The observed data at the final time.

    Returns:
        The adjoint variables λ for all time steps.
    """
    lambda_hist = np.zeros_like(u_history)
    u_final = u_history[-1, :]

    # Start with the gradient of the cost function J w.r.t. the state u
    # at the final time step, m. This is G^m in the report[cite: 88].
    # G_i^m = ∂J/∂u_i^m = 2 * Δx * (u_i^m - u_di) [derived from 80].
    lambda_hist[-1, :] = 2 * DX * (u_final - u_observed)

    # Solve backwards in time from m-1 to 1[cite: 125].
    # The equation is A2*λ^i = G^i - A1^T*λ^(i+1). Since G^i=0 for i<m,
    # this simplifies to λ^i = (A2)^-1 * (-A1^T * λ^(i+1)).
    # (A2)^-1 is a diagonal matrix with DT on the diagonal[cite: 120].
    # A1^T is a bidiagonal matrix [cite: 115-119].
    for k in range(NT - 1, 0, -1):
        for i in range(NX - 1):
            # This implements the operation λ^k = (A2)^-1 * (-A1^T * λ^(k+1))
            term1 = -1/DT + A/DX
            term2 = -A/DX
            # This is the expansion of the matrix-vector product -A1^T * λ^(i+1)
            adjoint_forcing = term1 * lambda_hist[k+1, i] + term2 * lambda_hist[k+1, i+1]
            lambda_hist[k, i] = -DT * adjoint_forcing

    return lambda_hist

# --- 4. Gradient Descent Optimization ---

# Start with a poor initial guess [cite: 185]
u_guess = np.full(NX, 0.5)

print("Starting optimization to recover the initial condition...")
print("-" * 50)

for it in range(ITERATIONS):
    # 1. Forward Solve: Solve the governing equation with the current guess.
    u_history_guess = forward_solve(u_guess)
    u_final_guess = u_history_guess[-1, :]

    # 2. Calculate Cost: Evaluate how far the current solution is from the observation.
    cost = calculate_cost(u_final_guess, U_OBSERVED, u_guess)
    if (it % 50) == 0:
        print(f"Iteration {it:4d}, Cost: {cost:.6e}")

    # 3. Adjoint Solve: Solve the adjoint equation backward in time.
    lambda_history = adjoint_solve(u_history_guess, U_OBSERVED)
    
    # 4. Compute Gradient: The gradient of the cost function with respect to the
    # initial condition (the design variable) is λ at time 0, plus the gradient
    # of the regularization term[cite: 66, 127].
    grad_regularization = -2 * BETA * np.gradient(np.gradient(u_guess, DX), DX)
    gradient = lambda_history[0, :] + grad_regularization
    
    # 5. Update Guess: Update the initial condition using gradient descent.
    u_guess -= LEARNING_RATE * gradient

print("-" * 50)
print("Optimization finished.")

# --- 5. Results ---
final_cost = calculate_cost(forward_solve(u_guess)[-1,:], U_OBSERVED, u_guess)
print(f"Final cost: {final_cost:.6e}")

plt.figure(figsize=(12, 5))
plt.suptitle("Initial Condition Recovery using the Discrete Adjoint Method")

# Plot initial conditions
plt.subplot(1, 2, 1)
plt.title("Initial Conditions (t=0)")
plt.plot(X, U_TRUE_0, 'k-', label="Exact IC", linewidth=2)
plt.plot(X, u_guess, 'r--.', label="Recovered IC", markersize=4)
plt.xlabel("Position (x)")
plt.ylabel("u(x,0)")
plt.legend()
plt.grid(True)

# Plot advected solutions at final time
u_recovered_final = forward_solve(u_guess)[-1, :]
plt.subplot(1, 2, 2)
plt.title("Solutions at Final Time (t=T)")
plt.plot(X, U_OBSERVED, 'k-', label="Observed Data", linewidth=2)
plt.plot(X, u_recovered_final, 'r--.', label="Recovered Solution", markersize=4)
plt.xlabel("Position (x)")
plt.ylabel(f"u(x,T={T})")
plt.legend()
plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()