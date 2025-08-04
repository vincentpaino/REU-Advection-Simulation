import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time # track how long it takes for network to train
"""
This model takes the known u at the last time step T of the 1D linear advection equation
and predicts the initial condition g(x) at t=0 using a neural network.
The loss function includes the difference between the true value of u at the last time step (u[-1]) 
and the u obtained from running FDM with the g_pred(x) (IC that network has predicted).

This is an inverse problem where g(x) (the initial condition) is unknown, the PINN should be trained to discover g(x) 
by minimizing a loss J, which compares the predicted solution at the final time to the observed data u_ref(x, T).
Also, this model takes advantage of the finite difference PDE solver, zero-ing out the PDE loss in a traditional PINN 
and optimizes based on measured data loss.
"""
# Constants
a = 1.0            # Advection speed
c = 0.0            # Boundary condition u(-1, t) = c
T = 1.0            # Final time
nx = 100           # Number of spatial grid points
nt = 100           # Number of time steps
x_start = -1.0     # Leftmost bound
x_end = 1.0        # Rightmost bound
pinn_loss_history = []  # To store loss values for plotting

# Set same seeds for reproducibility
torch.manual_seed(40) 

x = torch.linspace(x_start, x_end, steps=nx) # Grid tensor
#print("shape of grid:",x.shape)
dx = x[1] - x[0]    # spatial step (this is 0.02)
dt = T / nt         # time step (this is 0.01)

# CFL condition to check for instability
assert a * dt / dx <= 1.0, "CFL condition violated! Adjust dt or dx to avoid instability."

def g(x):
    amplitude = 1.0
    return amplitude * torch.exp(-50 * (x + 0.4)**2) # Defines and sets the initial condition g(x) with a gaussian/normal distribution "bump" to see how the wave propagates

# Different g(x) values for testings
def g2(x): 
    amplitude = 0.5
    return amplitude * torch.exp(-50 * (x + 0.4)**2)

def g3(x):
    amplitude = 0.5
    return amplitude * torch.exp(-50 * (x + 0.5)**2)

u = torch.zeros((nx, nt))        # Initializing the solution tensor u with zeros 
u[:, 0] = g3(x)                   # Initial condition at t = 0
initial_conditions = u[:,0]      # Store the initial condition for loss function
#print(initial_conditions)                   # Printing the distribution

#u_hist = [u.clone()]

for k in range(1, nt): # Time-stepping loop, using Duffy paper FDM implementation 
    u[1:, k] = u[1:, k-1] - a * dt / dx * (u[1:, k-1] - u[:-1, k-1]) # finite difference method (Duffy)
    u[0, k] = c  # left boundary condition remains the same at each timestep

u_np = u.detach().numpy()  # Convert u to numpy for plotting and animation
print("Last timestep u printed:", u_np[:,-1])

#"""
# Plot initial and final states of u (first/last time step)
plt.plot(x, u_np[:,0], label="Initial $g(x)$")
plt.plot(x, u_np[:,-1], label="Final $u(x,T)$")
plt.title("1D Linear Advection Simulation (PINN)")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.grid(True)
#plt.show()

# --- ANIMATION SETUP ---
# Convert history to numpy for animation
x_np = x.numpy()

fig, ax = plt.subplots()
line, = ax.plot(x_np, u_np[:,0], lw=2)
ax.set_title("1D Linear Advection (FDM)")
ax.set_xlim(x_start, x_end)
ax.set_ylim(0, 1.1)
ax.set_xlabel("x")
ax.set_ylabel("u")

def update(frame):
    line.set_ydata(u_np[:,frame])
    ax.set_title(f"Time Step: {frame}")
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(u_np), interval=60)
plt.show()
#"""

## INTRODUCE ADJOINT METHOD, FDM & PINN TO PREDICT g(x) BASED OFF GROUND TRUTH u(x,T); T being the final timestep ##

def fdm(outputs):
    u_pred = torch.zeros((nx,nt))
    u_pred[:, 0] = outputs
    for k in range(1, nt): # Time-stepping loop, using Duffy paper FDM implementation 
        u_pred[1:, k] = u_pred[1:, k-1] - a * dt / dx * (u_pred[1:, k-1] - u_pred[:-1, k-1]) # finite difference method (Duffy)
        u_pred[0, k] = c  # left boundary condition remains the same at each timestep
    return u_pred[:,-1]   

class InitialConditionPrediction(nn.Module):
    """
    Defining a PINN that knows the advection equation that predicts 'g' (the initial condition when t=0)
    The input, in theory, should be any shape of tensor (German paper used torch.randn((1, 128, 8, 4))
    We know what the wave equation looks like at the last time-step T, and we need to work backwards to find the initial condition g.
    (For help look at Duffy paper)
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # Input layer
            nn.Mish(),      # Use, GELU (good w/ lr=2e-3), SiLU, Pure Mish too
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            #nn.Linear(hidden_dim, hidden_dim),
            #nn.Mish(),
            #nn.Linear(hidden_dim, hidden_dim),
            #nn.Mish(),
            nn.Linear(hidden_dim, output_dim), # Output layer
            nn.Mish()
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
    
def loss_function(predicted_u, ground_truth_u): # taking u at the very last time step T
    """ Run FDM on the outputs (predicted initial condition g) and compare it to the ground truth u at the last time step T"""
    J = dx * torch.sum((predicted_u - ground_truth_u) ** 2)
    return J

def loss_function_2(predicted_uref_g, ground_truth_uref): #norm loss to compare to adjoint and hybrid too
    J = torch.norm(predicted_uref_g - ground_truth_uref)
    return J

def physics_loss(u_pred): # we use nx, nt and c in this function too
    """
    u_pred: [Nx, Nt] predicted solution over the grid
    x: [Nx] spatial grid points (1D tensor)
    t: [Nt] time grid points (1D tensor)
    c: advection speed
    """
    lambda_phys, lambda_bc = 1.0, 1.0  # Weights for physics and boundary condition losses

    #Nx, Nt = u_pred.shape

    # Create meshgrid of x and t (for autograd)
    x, t = torch.meshgrid(nx, nt, indexing='ij')  # Shapes: [Nx, Nt]

    x = x.requires_grad_()
    t = t.requires_grad_()

    # Flatten needed for autograd
    u_flat = u_pred.reshape(-1, 1)
    x_flat = x.reshape(-1, 1)
    t_flat = t.reshape(-1, 1)

    # ∂u/∂t
    u_t = torch.autograd.grad(u_flat, t_flat, grad_outputs=torch.ones_like(u_flat),
                              retain_graph=True, create_graph=True)[0]

    # ∂u/∂x
    u_x = torch.autograd.grad(u_flat, x_flat, grad_outputs=torch.ones_like(u_flat),
                              retain_graph=True, create_graph=True)[0]

    # Physics loss
    residual = u_t + c * u_x
    physics_loss = torch.mean(residual**2)

    # Boundary condition u(0, t) = 0 → corresponds to first row of u_pred
    bc_loss = torch.mean(u_pred[0, :]**2)

    return (lambda_phys * physics_loss) + (lambda_bc * bc_loss)


#def loss_function_physics(predicted_u): takes u[100,100]
# Loss is only taken from boundary condition and advection equation


def train_model(model, inputs, ground_truth_u, epochs, lr):
    # [dx] 1 grid point input
    # [dx,dt] output = 100 by 100
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr) # lowest e-6
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.5, patience=10, min_lr=2.5e-7
    )
    
    for epoch in range(epochs + 1):
        outputs = model(inputs)
        
        # standardize outputs
        #target_min, target_max = 0.0, 1.0  # or your known bounds
        #outputs = (outputs - target_min) / (target_max - target_min)
        
        predicted_u = fdm(outputs)  
        loss = loss_function_2(predicted_u, ground_truth_u) #outputs should be predicted_u, keep to compare to Adjoint/Hybrid
        #physics_loss = physics_loss(outputs)
        pinn_loss_history.append(loss.item())  # Store loss for plotting later
        optim.zero_grad()
        loss.backward()
        optim.step()        # figure out how to propagate adjoint gradients
        if epoch % 50 == 0:
            current_lr = optim.param_groups[0]['lr']
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}, LR: {current_lr:.6e}")
        lr_scheduler.step(loss)  # Adjust learning rate based on loss
            
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
    plt.figure(figsize=(10, 5))
    #plt.title("PINN Prediction of Initial Conditions")
    plt.plot(x, pred_g_dist, color="red", linestyle="-", label="Predicted $g(x)$", linewidth=5)
    plt.plot(x, true_g_dist, color="blue", linestyle="-", label="True $g(x)$",linewidth=5)
    plt.xlabel("Grid Points", fontsize=35)
    plt.ylabel("IC Values", fontsize=35)
    #plt.figtext(0.5, 0.01, f"epochs: {num_epochs}, lr: {lr}", wrap=True, horizontalalignment='center', fontsize=10)
    plt.legend(fontsize=25)
    plt.grid(True)
    plt.tick_params(labelsize=25)
    plt.show()
    
def show_loss(epochs, pinn_loss_history):
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title("Pure PINN Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    x = np.linspace(0, epochs, len(pinn_loss_history))
    y = pinn_loss_history[:]
    ax.plot(x, y)
    #plt.figtext(0.1, 0.01, f"Epochs: {epochs}", wrap=True, horizontalalignment='left', fontsize=10)
    #plt.figtext(0.1, 0.05, f"Learning Rate: {lr}", wrap=True, horizontalalignment='left', fontsize=10)
    plt.show()

### Init PINN and Train ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InitialConditionPrediction(input_dim=nx, hidden_dim=180, output_dim=nx)
#model = InitialConditionPrediction(nx=nx, nt=nt, hidden_dim=180).to(device)
#modelInput = torch.randn((1, nx,nt), device=device)     # Input tensor of random values of spatial grid size (batch size of 1)

ground_truth_u = u[:,-1]                           # Use the final state of u at time step T as input to the model, which is already calculated 
modelInput = torch.randn((nx), device=device)
#modelInput = torch.tensor(u[:,-1]).float()  # This is what we observe, the final state of u at time step T
print("Model Input:", modelInput)


lr = 2.5e-3 #2.5e-3 is solid
num_epochs = 1000
train_model(model, modelInput, ground_truth_u, num_epochs, lr)  # Train the model with the final state of u and the true initial condition g


g_pred = model(modelInput)  # Predict the initial condition g(x) using the trained model
g_pred_tensor = g_pred

#u_pred = model(modelInput)
#g_pred = u_pred[0, :,0]

print("Sample Output From Model: ", g_pred)
print("Real (unknown) Initial Conditions g(x): ", initial_conditions) # 

g_pred = g_pred.detach().numpy()  # Convert the output to numpy for evaluation
initial_conditions = initial_conditions.detach().numpy()  # Convert the initial conditions to numpy for evaluation

eval_model(model, x=x, pred_g_dist=g_pred, true_g_dist=initial_conditions)  # Evaluate the model with the predicted initial u and the true initial condition g
#eval_model(model, x=x, pred_g_dist=smoothen_data(g_pred_tensor), true_g_dist=initial_conditions) #Evaluate model with a smoothened output 

#for i in range(len(loss_history)):
    #print(loss_history[i])
show_loss(epochs=num_epochs, pinn_loss_history=pinn_loss_history)






#####################################
## PREVIOUS ADJOINT IMPLEMENTATION ##
"""
def adjoint_method(ground_truth_u, u):

    #Implemented from MATLAB file to Python
    
    a2 = torch.zeros((nt,nt)) #Making the (∂N^k/∂u^k)T matrix with 1/dt on the diagonal
    for i in range(a2.size(dim=0)):
        a2[i,i] = 1/dt  # Fill the diagonal with 1/dt
        
    delJdelU_measured = torch.zeros((nt,1)) # Column vector
    print(delJdelU_measured.shape)
    k = 0
    for i in range(1, delJdelU_measured.size(dim=0)):
        delJdelU_at_i = 2 * dx * ( ground_truth_u[i] - u[i, 0] ) # u at last time-step T minus g(x)
        delJdelU_measured[i,0] = delJdelU_at_i
        
    lambdam = delJdelU_measured.t() / a2.t()
    
    a1 = torch.zeros((nt, nt)) # Making the (∂N^k/∂u^k-1)T matrix with -a/dt on the diagonal
    for i in range(a1.size(dim=0)):
        a1[i,i] = -a * (1/dx)  # Fill the diagonal with -a/dx 
        
    for i in range(a1.shape[0] - 1):
        a1[i,i+1] = -1/dt + a * (1/dx) # a1 fully is defined now
        
    for i in range(nt-1, 0, -1):       # range(start, stop, step)
        lambdai = (-a1.t() * lambdam.t()).t() / a2
        lambdam = lambdai                # working backwards

    dJdphi = -lambdam * a2
    return dJdphi # need to make this a vector
"""