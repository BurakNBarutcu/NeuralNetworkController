"""
A series of helper functions used throughout the course.

If a function gets defined once and could be used over and over, it'll go in here.
"""
import torch
import matplotlib.pyplot as plt
from pathlib import Path



def print_train_time(start, end, device=None):
    """Prints difference between start and end time.
    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.
    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


# Plot loss curves of a model
def plot_loss_curves(results,fig_iden: int=1):
    """Plots training curves of a results dictionary.
    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
        fig_iden (int, optional): Identifier for figure. Defaults to 1. 
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    epochs = range(len(results["train_loss"]))
    plt.close()
    plt.figure(num=fig_iden,figsize=(15, 7))
    # plt.clf()

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo-', label="train_loss")
    plt.plot(epochs, test_loss, 'bo-', label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    # plt.legend()
    plt.show()

    if "train_acc" in results:
        accuracy = results["train_acc"]
        test_accuracy = results["test_acc"]

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy, 'bo-', label="train_accuracy")
        plt.plot(epochs, test_accuracy, 'bo-', label="test_accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        # plt.legend()



def set_seeds(seed: int=42):
    """Sets random sets for torch operations.
    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
    print(f"Manual seed is set as {seed}.")

def save_model(model: torch.nn.Module,
    model_path: str = "Saved_Models",
    model_name: str = "Saved_Model"
):
    """Save the trained model for future use.
    Args:
        model (torch.nn.Module): Trained model to be saved.
        model_path (str, optinal): Path that model is saved. Defaults is "Saved_Models".
        model_name (str, optinal): Model name which is used while saving. Default is "Saved_Model".
        """


    # 1. Create models directory 
    MODEL_PATH = Path(model_path)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # 2. Create model save path 
    MODEL_SAVE_PATH = MODEL_PATH / model_name

    # 3. Save the model state dict 
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the models learned parameters
            f=MODEL_SAVE_PATH)

def load_model_parameters(
    model: torch.nn.Module,
    model_path: str = "Saved_Models",
    model_name: str = "Saved_Model",
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Loads the parameters of a saved model for future use.
    Args:
        model (torch.nn.Module): Model to load the parameters.
        model_path (str, optinal): Path that model is saved. Defaults is "Saved_Models".
        model_name (str, optinal): Model name which is used while saving. Default is "Saved_Model".
        device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".
    
    Returns:
        Loaded model for future use.
    """
    # 1. Get models directory 
    MODEL_PATH = Path(model_path)
    MODEL_SAVE_PATH = MODEL_PATH / model_name

    # Load model state dict 
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # Put model to target device (if your data is on GPU, model will have to be on GPU to make predictions)
    model.to(device)

    print(f"Loaded model:\n{model}")
    print(f"Model on device:\n{next(model.parameters()).device}")

    return model