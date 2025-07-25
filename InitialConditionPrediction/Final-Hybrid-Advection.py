import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print("\n")


# import time # track how long it takes for network to train
"""
This model takes the known u at the last time step T of the 1D linear advection equation
and predicts the initial condition g(x) at t=0 using a neural network.
The loss function includes the difference between the true value of u at the last time step (u[-1]) 
and the u obtained from running FDM with the g_pred(x) (IC that network has predicted).

This is an inverse problem where g(x) (the initial condition) is unknown, the PINN should be trained to discover g(x) 
by minimizing a loss J, which compares the predicted solution at the final time to the observed data u_d(x, T).
"""
# Constants
a_um = 0.9         # Advection speed for "simulated data"
a_uref = 1.0       # Advection speed for "experimental/measured data"
c = 0.0            # Boundary condition u(-1, t) = c
T = 1.0            # Final time
nx = 100           # Number of spatial grid points
nt = 100           # Number of time steps
x_start = -1.0     # Leftmost bound
x_end = 1.0        # Rightmost bound
hybrid_loss_history = []  # To store loss values for plotting
adjoint_loss_history = [] # To store adjoint loss values for plotting (future work)

# Set same seeds for reproducibility
torch.manual_seed(40) 

x = torch.linspace(x_start, x_end, steps=nx) # Grid tensor
#print("shape of grid:",x.shape)
dx = x[1] - x[0]    # spatial step (this is 0.02)
dx = float(dx)      # Convert to float for calculations
dx = round(dx, 2)   # Round to 2 decimal places for clarity
dt = T / nt         # time step (this is 0.01)

# CFL condition to check for instability
assert a_uref * dt / dx <= 1.0, "CFL condition violated! Adjust dt or dx to avoid instability."

def um_g(x):
    amplitude = 1.0
    return amplitude * torch.exp(-50 * (x + 0.5)**2) # Defines and sets the initial condition g(x) with a gaussian/normal distribution "bump" to see how the wave propagates

def uref_g(x):
    amplitude = 0.5
    return amplitude * torch.exp(-50 * (x + 0.4)**2)

um = torch.zeros((nx, nt), dtype=torch.float64)        # Initializing the solution tensor u with zeros 
um[:, 0] = um_g(x)                   # Initial condition at t = 0
um[0,0] = c                       # Set initial condition equal to c at x=0 , t=0
#print(initial_conditions)         # Printing the distribution

uref = torch.zeros((nx, nt), dtype=torch.float64)
uref[:, 0] = uref_g(x)                   
uref[0,0] = c                      
initial_conditions = uref[:,0]  # Store the initial condition for loss function

for k in range(1, nt): # Time-stepping loop, using Duffy paper FDM implementation to get "simulated ground truth u"
    um[1:, k] = um[1:, k-1] - a_um * dt / dx * (um[1:, k-1] - um[:-1, k-1]) # finite difference method (Duffy)
    um[0, k] = c  # left boundary condition remains the same at each timestep
    
for k in range(1, nt): # Time-stepping loop, using Duffy paper FDM implementation to get our "experimental/measured data"
    uref[1:, k] = uref[1:, k-1] - a_uref * dt / dx * (uref[1:, k-1] - uref[:-1, k-1]) # finite difference method (Duffy)
    uref[0, k] = c  # left boundary condition remains the same at each timestep

um_np = um.detach().numpy()  # Convert um to numpy for plotting and animation... um is "simulated"
uref_np = uref.detach().numpy() # Convert uref to numpy for plotting and animation... uref is our reference data which is "experimental/measured"
print("Last timestep at um printed:", um_np[:,-1])
print("Last timestep at uref printed:", uref_np)

#"""
# Plot initial and final states of um & uref (first/last time step)
plt.plot(x, um_np[:,0], label="Initial $g(x)$ of $um(x,t)$")
plt.plot(x, um_np[:,-1], label="Final $um(x,T)$")
plt.plot(x, uref_np[:,0], label="Initial $g(x)$ of $uref(x,t)$")
plt.plot(x, uref_np[:,-1], label="Final $uref(x,T)$")
plt.title("1D Linear Advection Problem")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.grid(True)
#plt.show()

# --- ANIMATION SETUP --- #
# Convert history to numpy for animation
x_np = x.numpy()

fig, ax = plt.subplots()
line, = ax.plot(x_np, um_np[:,0], lw=2)
ax.set_title("1D Linear Advection (FDM)")
ax.set_xlim(x_start, x_end)
ax.set_ylim(0, 1.1)
ax.set_xlabel("x")
ax.set_ylabel("u")

def update(frame):
    line.set_ydata(um_np[:,frame])
    ax.set_title(f"Time Step: {frame}")
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(um_np), interval=60)
plt.show()
#"""

## INTRODUCE ADJOINT METHOD, FDM & PINN TO PREDICT g(x) BASED OFF GROUND TRUTH u(x,T); T being the final timestep ##

def fdm(outputs): # this FDM is the same as before but gives us the prediction for u at u(x,T) used in train_model
    u_pred = torch.zeros((nx,nt))
    u_pred[:, 0] = outputs
    for k in range(1, nt): # Time-stepping loop, using Duffy paper FDM implementation 
        u_pred[1:, k] = u_pred[1:, k-1] - a_uref * dt / dx * (u_pred[1:, k-1] - u_pred[:-1, k-1]) # finite difference method (Duffy)
        u_pred[0, k] = c  # left boundary condition remains the same at each timestep
    return u_pred[:,-1]   

# This function computes dJdphi, which is the gradient of the loss function J with respect to the initial conditions (network output) phi (g(x)).
def adjoint_method(um, uref): # um = simulated data at final timestep (ground_truth_u) = torch.tensor
                              # uref = "experimental" data = torch.tensor

    a2 = torch.eye(nt, dtype=torch.float64) * (1.0/dt) # A2 matrix  
    
    # Compute delJdelum
    delJdelum = 2 * dx * (um - uref)
    delJdelum = delJdelum.unsqueeze(1) # shape (100, 1) to match the dimensions of lambdam
    
    lambdam = torch.linalg.solve(a2, delJdelum) # Solve lambdam = a2^(-1) @ delJdelum
    
    #a_uref_tensor = torch.tensor(a_uref, dtype=torch.float64) #datatype converting
    #dt_tensor = torch.tensor(dt, dtype=torch.float64) #datatype converting
    
    a1 = torch.zeros((nt, nt), dtype=torch.float64) # Construct a1 matrix
    a1[torch.arange(nt), torch.arange(nt)] = -a_uref / dx
    a1[torch.arange(nt-1), torch.arange(1,nt)] = (-1/dt) + a_uref/dx 
   
    lambdam = torch.linalg.solve(a2, delJdelum)
    
    a1[torch.arange(nt), torch.arange(nt)] = -a_uref / dx
    a1[torch.arange(nt-1), torch.arange(1,nt)] = (-1/dt) + a_uref/dx
    
    for i in range(nt-1, 0, -1):
        lambdai = -torch.linalg.solve(a2, a1 @ lambdam)
        lambdam = lambdai
        
    gradLeft = (lambdam.T @ a2).squeeze()
    dJdphi = gradLeft 
    
    return dJdphi # Loss w.r.t. initial conditions of u_ref (our predicted output)

# Manual gradient based optimization algorithm with the discrete adjoint method
def adjoint_gradient_descent(um, uref, um_g, uref_g): # reminder: "g" is our initial conditions (also called phi)
    copy_of_um_g = um_g.clone() # Copy of um_g to use in the graph
    dJdphi = adjoint_method(um, uref)

    lr = 0.0025 # Learning rate for gradient descent (0.025 is a good lr)
    epochs = 1000
    # Gradient descent loop (normalized steepest descent)
    for epoch in range(epochs):
        norm_grad = torch.norm(dJdphi) # the magnitude of the gradients themselves
        phiNSGDi = um_g - lr * (dJdphi / norm_grad)
        um_g = phiNSGDi.clone()

        newum = fdm(phiNSGDi)
        um = newum.clone()

        dJdphi = adjoint_method(um, uref)
    
        loss = 0
        if epoch % 100 == 0:
            for j in range(nx):
                loss += (phiNSGDi[j] - uref_g[j]) ** 2
            print(f" Loss at loop {epoch}/{epochs} {loss}")
     
    # Create x-axis values from 1 to 100
    x = list(range(1,101))

    # Create the plot
    plt.plot(x, phiNSGDi, '-', label='Current IC')
    plt.plot(x, uref_g, 'o', label='Real IC')
    plt.plot(x, copy_of_um_g, '*', label='Initial guess for IC')

    # Add legend
    plt.legend()

    # Optional: Add axis labels and title
    plt.xlabel('Index')
    plt.ylabel('IC values')
    plt.title('Comparison of IC values')

    # Display the plot
    plt.show()   

    

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
            nn.SiLU(),      # Use Tanh, GELU (good w/ lr=2e-3), SiLU
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),  # Output layer to predict initial condition g
        )
        #Xavier/He initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.model(x)
    

def loss_function(predicted_uref, ground_truth_uref): # taking u at the very last time step T of our "real, experimental data" to use FDM to compute g(x)
    """ Run FDM on the outputs (predicted initial condition g) and compare it to the ground truth u at the last time step T"""
    J = dx * torch.sum((predicted_uref - ground_truth_uref) ** 2)
    return J

def train_model(model, inputs, ground_truth_uref, epochs, lr):
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr) 
    
    for epoch in range(epochs):
        outputs = model(inputs)
        predicted_uref = fdm(outputs)  
        loss = loss_function(predicted_uref, ground_truth_uref) 
        hybrid_loss_history.append(loss.item())
        adjoint_gradients = adjoint_method(um=predicted_uref, uref=ground_truth_uref)
        optim.zero_grad()
        outputs.backward(gradient=adjoint_gradients)
        #loss.backward()
        optim.step()        
        if epoch % 25 == 0:
            #print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')
            print(f"Epoch {epoch}/{epochs}")
            
def smoothen_data(outputs):
    """
    Smoothens the data using a simple moving average filter.
    This is useful for reducing noise in the predictions.
    """
    window_size = 5
    smoothed_outputs = torch.zeros_like(outputs)
    for i in range(len(outputs)):
        start = max(0, i - window_size // 2)
        end = min(len(outputs), i + window_size // 2 + 1)
        smoothed_outputs[i] = torch.mean(outputs[start:end])
    smoothed_outputs = smoothed_outputs.detach().numpy()
    return smoothed_outputs
    
def eval_model(model, x, pred_g_dist, true_g_dist):
    model.eval()
    fig = plt.figure(figsize=(10, 5))
    plt.title("Hybrid Prediction for Initial Conditions")
    plt.plot(x, pred_g_dist, color="red", linestyle="-", label="Predicted $g(x)$")
    plt.plot(x, true_g_dist, color="green", linestyle="-", label="True $g(x)$")
    plt.xlabel("x")
    plt.xlim(-1,1)
    #plt.ylim(-6,1)
    plt.ylabel("$u$")
    plt.figtext(0.5, 0.01, f"epochs: {num_epochs}, lr: {lr}", wrap=True, horizontalalignment='center', fontsize=10)
    plt.legend()
    plt.show()
    
def show_loss(epochs, hybrid_loss_history):
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title("Hybrid PINN Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    x = np.linspace(0, epochs, len(hybrid_loss_history))
    y = hybrid_loss_history[:]
    ax.plot(x, y)
    plt.figtext(0.1, 0.01, f"Epochs: {epochs}", wrap=True, horizontalalignment='left', fontsize=10)
    plt.figtext(0.1, 0.05, f"Learning Rate: {lr}", wrap=True, horizontalalignment='left', fontsize=10)
    plt.show()


# --- Init PINN and Train --- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InitialConditionPrediction(input_dim=nx, hidden_dim=250, output_dim=nx)
ground_truth_uref = uref[:,-1]                          # Use the final state of u at time step T as input to the model, which is already calculated 
ground_truth_um = um[:,-1]

#print("ground truth uref: ", ground_truth_uref)
#print("ground truth um: ", ground_truth_um)

modelInput = torch.randn((100), device=device)     # Input tensor of random values of spatial grid size
print("Model Input:",modelInput)

lr = 2.5e-3 
num_epochs = 1000
train_model(model, modelInput, ground_truth_uref, num_epochs, lr)  # Train the model with the final state of u and the true initial condition g

g_pred = model(modelInput)  # Predict the initial condition g(x) using the trained model
g_pred_tensor = g_pred
print("Sample Output From Model: ", g_pred)
print("Real (unknown) Initial Conditions g(x): ", initial_conditions) # 

g_pred = g_pred.detach().numpy()  # Convert the output to numpy for evaluation
initial_conditions = initial_conditions.detach().numpy()  # Convert the initial conditions to numpy for evaluation

eval_model(model, x=x, pred_g_dist=g_pred, true_g_dist=initial_conditions)  # Evaluate the model with the predicted initial u and the true initial condition g
#eval_model(model, x=x, pred_g_dist=smoothen_data(g_pred_tensor), true_g_dist=initial_conditions) # Evaluate model with a smoothened output 
show_loss(num_epochs, hybrid_loss_history)  # Show the loss curve

# --- Plot Adjoint Method ---
dJdphi = adjoint_method(um=ground_truth_um, uref=ground_truth_uref)
print(f"Printed dJdphi from adjoint method: {dJdphi}")
print(f"Shape of dJdphi: ", {dJdphi.shape})


# Plot
#plt.figure(5)
#plt.plot(range(1,101), dJdphi.cpu().numpy(), '-', label="dJdphi")
#plt.title("dJdphi and test_adjoint comparison")
#plt.show()

#print("dJdphi shape", dJdphi.shape)
#torch.set_printoptions(profile="full")
#torch.set_printoptions(profile="default")

# --- Pure Adjoint Optimization Loop --- #
adjoint_gradient_descent(um=ground_truth_um, uref=ground_truth_uref, um_g=um[:,0], uref_g=uref[:,0])