import torch
import torch.nn as nn
import numpy as np

class ImprovedInitialConditionPrediction(nn.Module):
    """
    Improved PINN for predicting initial conditions from final state
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6):
        super().__init__()
        
        # Create layers dynamically
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())  # Tanh works well for PINNs
        
        # Hidden layers
        for i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            # Add residual connections for deeper networks
            if i > 0 and i % 2 == 0:
                layers.append(ResidualBlock(hidden_dim))
        
        # Output layer (no activation - let the network learn the range)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # Better initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization works better for Tanh
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.model(x)

class ResidualBlock(nn.Module):
    """Simple residual block to help with gradient flow"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()
        
    def forward(self, x):
        residual = x
        out = self.activation(self.layer1(x))
        out = self.layer2(out)
        out += residual  # Residual connection
        return self.activation(out)

def improved_train_model(model, inputs, ground_truth_uref, ground_truth_g, epochs, lr):
    """
    Improved training function with better optimization strategies
    """
    model.train()
    
    # Use AdamW with weight decay for better generalization
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Less aggressive learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.8, patience=50, verbose=True
    )
    
    loss_history = []
    
    for epoch in range(epochs):
        outputs = model(inputs)
        
        # Physics-informed loss: forward through FDM
        predicted_uref = fdm(outputs)
        physics_loss = loss_function_2(predicted_uref, ground_truth_uref)
        
        # Optional: Add regularization on the initial condition
        # This helps prevent overfitting and encourages smoothness
        smoothness_loss = torch.mean(torch.diff(outputs)**2)
        
        # Combined loss
        total_loss = physics_loss + 0.01 * smoothness_loss
        
        # Adjoint gradients for physics-informed training
        adjoint_gradients = adjoint_method(um=predicted_uref, uref=ground_truth_uref)
        
        # Standard backpropagation
        optim.zero_grad()
        total_loss.backward()
        
        # Add adjoint gradients to the parameter gradients
        with torch.no_grad():
            model.model[0].weight.grad += torch.outer(adjoint_gradients, inputs).T * 0.1
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optim.step()
        lr_scheduler.step(total_loss)
        
        loss_history.append(total_loss.item())
        
        if epoch % 100 == 0:
            current_lr = optim.param_groups[0]['lr']
            print(f"Epoch {epoch}/{epochs}, Physics Loss: {physics_loss.item():.6f}, "
                  f"Total Loss: {total_loss.item():.6f}, LR: {current_lr:.6e}")
    
    return loss_history

# Updated training setup
def setup_improved_training():
    """
    Setup function with corrected input and better hyperparameters
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # CRITICAL FIX: Use the final state as input, not random noise!
    modelInput = uref[:, -1].clone().detach()  # This is what we observe
    
    # Larger, deeper network for this complex inverse problem
    model = ImprovedInitialConditionPrediction(
        input_dim=nx, 
        hidden_dim=128,  # Increased from 50
        output_dim=nx,
        num_layers=8     # Deeper network
    )
    
    ground_truth_uref = uref[:, -1]  # Final state (what we observe)
    ground_truth_g = uref[:, 0]      # Initial condition (what we want to predict)
    
    # Better learning rate schedule
    lr = 1e-3  # Start a bit lower
    num_epochs = 2000  # More epochs for deeper network
    
    print("Model Input (Final State):", modelInput.shape)
    print("Target (Initial Condition):", ground_truth_g.shape)
    
    return model, modelInput, ground_truth_uref, ground_truth_g, lr, num_epochs

# Additional debugging function
def check_gradient_flow(model):
    """
    Check if gradients are flowing through the network
    """
    total_norm = 0
    param_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            print(f"{name}: {param_norm:.6f}")
    total_norm = total_norm ** (1. / 2)
    print(f"Total gradient norm: {total_norm:.6f}")
    return total_norm