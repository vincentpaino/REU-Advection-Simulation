import torch
import numpy as np

# Convert a numpy array to a PyTorch tensor
def numpy_to_tensor(np_array):
    return torch.tensor(np_array, dtype=torch.float32)

# Convert a PyTorch tensor to a numpy array
def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.numpy()

# Example usage

if __name__ == "__main__":
    # Create a numpy array
    np_array = np.array([[1, 2, 3], [4, 5, 6]])

    # Convert numpy array to tensor
    tensor = numpy_to_tensor(np_array)
    print("Numpy Array to Tensor:")
    print(tensor)

    # Convert tensor back to numpy array
    converted_np_array = tensor_to_numpy(tensor)
    print("\nTensor back to Numpy Array:")
    print(converted_np_array)
    print("\n")
    print(converted_np_array[0,:])
    
    tensor = torch.tensor(3,4,5)
    print(tensor)