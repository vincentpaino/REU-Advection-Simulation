import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
"""
This model takes the known u at the last time step T of the 1D linear advection equation
and predicts the initial condition g(x) at t=0 using a neural network.
The loss function includes the difference between the true value of u at the last time step (u[-1]) 
and the u obtained from running FDM with the g_pred(x) (IC that network has predicted)
"""

# Constants
a = 1.0            # Advection speed
c = 1.0            # Boundary condition u(-1, t) = c
T = 1.0            # Final time
nx = 100           # Number of spatial grid points
nt = 100           # Number of time steps
x_start = -1.0     # Leftmost bound
x_end = 1.0        # Rightmost bound

# Grid
x = torch.linspace(x_start, x_end, steps=nx) # tensor
dx = x[1] - x[0]    # spatial step (this is 0.02)
dt = T / nt         # time step (this is 0.01)

# CFL condition to check for instability
assert a * dt / dx <= 1.0, "CFL condition violated! Adjust dt or dx to avoid instability."

def g(x):
    return torch.exp(-50 * (x)**2) # Defines and sets the initial condition g(x) with a gaussian/normal distribution "bump" to see how the wave propagates (can center x at different values)

u = g(x)                       # Initial condition at t = 0
initial_conditions = u.clone() # Store the initial condition for loss function
print(u)                       # Printing the distribution

u_hist = [u.clone()]

for n in range(nt): # Time-stepping loop
    u_new = u.clone()
    u_new[1:] = u[1:] - a * dt / dx * (u[1:] - u[:-1]) # first-order upwind finite difference method 
    u_new[0] = c  # left boundary condition
    u = u_new
    u_hist.append(u.clone()) # used for plotting

    
# Plot initial and final states
plt.plot(x, u_hist[0], label="Initial $g(x)$")
plt.plot(x, u_hist[-1], label="Final $u(x,T)$")
plt.title("1D Linear Advection")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.grid(True)
#plt.show()

# --- ANIMATION SETUP ---
# Convert history to numpy for animation
u_hist_np = [u.numpy() for u in u_hist]
x_np = x.numpy()

fig, ax = plt.subplots()
line, = ax.plot(x_np, u_hist_np[0], lw=2)
ax.set_title("1D Linear Advection (Upwind Scheme)")
ax.set_xlim(x_start, x_end)
ax.set_ylim(0, 1.1)
ax.set_xlabel("x")
ax.set_ylabel("u")

def update(frame):
    line.set_ydata(u_hist_np[frame])
    ax.set_title(f"Time Step: {frame}")
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(u_hist_np), interval=60)
plt.show()


## INTRODUCE NEURAL NETWORK TO PREDICT g(x) BASED OFF GROUND TRUTH u(x,T) ##

class InitialConditionPrediction(nn.Module):
    """
    Defining a PINN that knows the wave equation that predicts 'g' (the initial condition when t=0)
    The input, in theory, should be any shape of tensor (German paper used torch.randn((1, 128, 8, 4))
    We know what the wave equation looks like at the last time-step T, and we need to work backwards to find the initial condition g.
    (For help look at Duffy paper)
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim)  # Output layer to predict initial condition g
        )

    def forward(self, x):
        return self.model(x)
    
def loss_function(outputs, true_g_distribution): # Should not be using the real g(x) here
    L = torch.mean((outputs - true_g_distribution) ** 2)
    return L


def train_model(model, inputs, true_g_distribution):
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=2e-3)
    epochs = 50
    
    for epoch in range(epochs):
        outputs = model(inputs)
        loss = loss_function(outputs, true_g_distribution)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
    
def eval_model(model, x, pred_g_dist, true_g_dist):
    model.eval()
    fig = plt.figure(figsize=(10, 5))
    plt.title("PINN Prediction for Initial Conditions")
    plt.plot(x, pred_g_dist, color="red", linestyle="-", label="Predicted $g(x)$")
    plt.plot(x, true_g_dist, color="green", linestyle="-", label="True $g(x)$")
    plt.xlabel("Grid Points")
    plt.ylabel("$g(x)$")
    plt.legend()
    plt.show()
    
## Init PINN and Train ##
model = InitialConditionPrediction(input_dim=nx, hidden_dim=128, output_dim=nx)
modelInput = u_hist_np[-1]        # Use the final state of u at time step T as input to the model, which is already calculated
modelInput = torch.from_numpy(modelInput) # Convert to tensor


train_model(model, modelInput, true_g_distribution=initial_conditions)  # Train the model with the final state of u and the true initial condition g

print(modelInput)
sampleOutput = model(modelInput)  # Predict the initial condition g using the trained model
print("Sample Output From Model: ", sampleOutput)
print("Initial Conditions g(x): ", initial_conditions)
sampleOutput = sampleOutput.detach().numpy()  # Convert the output to numpy for evaluation
initial_conditions = initial_conditions.detach().numpy()  # Convert the initial conditions to numpy for evaluation

eval_model(model, x=x, pred_g_dist=sampleOutput, true_g_dist=initial_conditions)  # Evaluate the model with the final state of u and the true initial condition g


