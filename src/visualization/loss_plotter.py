import os
import matplotlib.pyplot as plt

def plot_loss(history, model_name: str, output_dir: str):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure()
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    filename = f"{model_name}_loss.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)

