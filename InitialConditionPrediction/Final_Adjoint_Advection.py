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
    amplitude = 1.0
    return amplitude * torch.exp(-50 * (x + 0.4)**2)

def uref_g2(x):
    amplitude = 0.5
    return amplitude * torch.exp(-50 * (x + 0.4)**2)

def uref_g3(x):
    amplitude = 0.5
    return amplitude * torch.exp(-50 * (x + 0.5)**2)

um = torch.zeros((nx, nt), dtype=torch.float64)        # Initializing the solution tensor u with zeros 
um[:, 0] = um_g(x)                   # Initial condition at t = 0
um[0,0] = c                       # Set initial condition equal to c at x=0 , t=0
#print(initial_conditions)         # Printing the distribution

uref = torch.zeros((nx, nt), dtype=torch.float64)
uref[:, 0] = uref_g3(x) # Change this function to simulate different data                   
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
plt.plot(x, um_np[:,0], label="Initial $g(x)$ of $um(x,t)$", linewidth=5, linestyle='--')
plt.plot(x, um_np[:,-1], label="Final $um(x,T)$", linewidth=5)
plt.plot(x, uref_np[:,0], label="Initial $g(x)$ of $uref(x,t)$", linewidth=5, linestyle='--')
plt.plot(x, uref_np[:,-1], label="Final $uref(x,T)$", linewidth=5)
#plt.title("1D Linear Advection Simulation (Adjoint)")
plt.xlabel("x", fontsize=14)
plt.ylabel("u", fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.tick_params(labelsize=14)
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
def adjoint_gradient_descent(um, uref, um_g, uref_g, epochs, lr): # reminder: "g" is our initial conditions (also called phi)
    copy_of_um_g = um_g.clone() # Copy of um_g to use in the graph
    dJdphi = adjoint_method(um, uref)
    # NOTE: can't use lr scheduler without an optimizer, so we will use a fixed learning rate for now
        
    # Gradient descent loop (normalized steepest descent)
    for epoch in range(epochs + 1):
        norm_grad = torch.norm(dJdphi) # the magnitude of the gradients themselves
        phiNSGDi = um_g - lr * (dJdphi / norm_grad)
        um_g = phiNSGDi.clone()

        newum = fdm(phiNSGDi)
        um = newum.clone()

        dJdphi = adjoint_method(um, uref)
    
        loss = 0
        #for j in range(nx):
        loss = torch.norm(phiNSGDi - uref_g)
        adjoint_loss_history.append(loss.item())
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss}")
     
    # Create x-axis values from 1 to 100
    x = list(range(1,101))

    # Create the plot
    plt.plot(x, phiNSGDi, '-', label='Current IC', linewidth=5, color='red')
    plt.plot(x, uref_g, 'o', label='Real IC', linewidth=7, color='blue')
    plt.plot(x, copy_of_um_g, '--', label='Initial Guess for IC', linewidth=5, color='green')

    # Add legend
    plt.legend(fontsize=25)
    plt.grid(True)
    plt.tick_params(labelsize=25)
    # Add axis labels and title
    plt.xlabel('Grid Points', fontsize=35)
    plt.ylabel('IC Values', fontsize=35)
    #plt.title('Adjoint Prediction of Initial Conditions')

    print(f"initial conditions of uref: {uref_g}")
    print("\n")
    print(f"predicted initial conditions of uref: {phiNSGDi}")
    
    # Display the plot
    plt.show()
       

def show_loss(epochs, lr, adjoint_loss_history): 
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title("Pure Adjoint Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_yscale('log') # Set y-axis to log scale for better visibility of loss values
    x = np.linspace(0, epochs, len(adjoint_loss_history))
    y = adjoint_loss_history[:]
    ax.plot(x, y)
    #plt.figtext(0.1, 0.01, f"Epochs: {epochs}", wrap=True, horizontalalignment='left', fontsize=10)
    #plt.figtext(0.1, 0.05, f"Learning Rate: {lr}", wrap=True, horizontalalignment='left', fontsize=10)
    plt.show()
    # Loss (y) axis convert to log scale

# --- Plot Adjoint Method ---
ground_truth_uref = uref[:,-1]                          # Use the final state of u at time step T as input to the model, which is already calculated 
ground_truth_um = um[:,-1]

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
epochs = 2000 # Compare between 1000 and 2000 epochs
lr = 2.5e-3 #0.0025

adjoint_gradient_descent(um=ground_truth_um, uref=ground_truth_uref, um_g=um[:,0], uref_g=uref[:,0], epochs=epochs, lr=lr)

show_loss(epochs=epochs, lr=lr, adjoint_loss_history=adjoint_loss_history)



