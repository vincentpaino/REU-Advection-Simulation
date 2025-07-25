import torch
import torch.nn as nn
import numpy as np
import maplotlib.pyplot as plt

# Look at White Notebook for Guidance
# Governing Wave equation
# u_tt = c^2 * u_xx

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
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
    
### FUNCTIONS ###

def ground_truth():
    return  
    
def forward_solver(grid_points, timestep, end_time, c):
    """
    Finite difference method that solves the wave equation at each grid point starting at u(x=0, t) = c (wave velocity)
    """
    u = np.array([])  # Placeholder for the wave solution
    u.zeros(())  # Initialize the wave solution array
    for t in range(end_time):
    
        t += timestep
    u[0] = 50

def loss_function(model, inputs, delta_x, m, i, u_known, T):
    """
    Wave loss function
    m = timesteps
    i = index of the 1D grid
    """
    model_output = model(inputs)
    
    J = delta_x * [() ** 2]
    
    return J

def train_model(model, inputs):
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=2e-3)
    
    epochs = 25
    for epoch in range(epochs):
        optim.zero_grad()
        forward_solver(grid_points=25, timesteps=50, end_time=5, c=350)
        
        loss = loss_function(model, inputs, delta_x=0.1, m=10, i=5, u_known=None, T=1)
        loss.backward()
        optim.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
    
def eval_model(model):
    model.eval()
    
    fig, ax = plt.subplots(size=(10, 6))
    ax.set_title('Model Predictions')
    ax.xlabel('Initial Condition G')
    
    
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = InitialConditionPrediction(input_dim=10, hidden_dim=50, output_dim=1)
    grid1D = torch.linspace(0, 1, nodes=20, device=device)  # shape: (128,)
    modelInput = torch.rand(grid1D, device=device)  # Example input tensor, shape (batch_size, input_dim)
    #modelInput = torch.randn((1, 128, 8, 4), device=device)                     #input tensor taken from German Paper, shape (batch_size, channels, height, width)
    #modelInput = modelInput / (torch.max(modelInput) - torch.min(modelInput)) * 2
    
    print(modelInput)
    train_model(model, modelInput)
    



if __name__ == "__main__":
    main()
    
