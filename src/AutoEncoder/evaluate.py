import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from model import DenoiseAutoencoder
from utils.load_data import load_mnist


def evaluate_denoiser(
    model_path: str,
    x_test_clean,
    noise_factor: float = 0.2,
    n_show: int = 8,
    plot: bool = True
):
    model = tf.keras.models.load_model(model_path)

    x_clean = x_test_clean.numpy() if hasattr(
        x_test_clean, "numpy") else np.asarray(x_test_clean)
    # (N,28,28,1)
    x_clean = x_clean[..., np.newaxis] if x_clean.ndim == 3 else x_clean

    # make noisy input exactly like training
    x_noisy = x_clean + noise_factor * tf.random.normal(shape=x_clean.shape)
    x_noisy = tf.clip_by_value(x_noisy, 0., 1.).numpy()

    # predict denoised
    x_pred = model.predict(x_noisy, verbose=0)
    x_pred = np.clip(x_pred, 0., 1.)

    # metrics
    mse = float(np.mean((x_clean - x_pred) ** 2))
    mae = float(np.mean(np.abs(x_clean - x_pred)))

    # PSNR / SSIM (per-image then mean)
    psnr_vals = tf.image.psnr(x_clean, x_pred, max_val=1.0).numpy()
    ssim_vals = tf.image.ssim(x_clean, x_pred, max_val=1.0).numpy()

    results = {
        "mse": mse,
        "mae": mae,
        "psnr_mean": float(np.mean(psnr_vals)),
        "ssim_mean": float(np.mean(ssim_vals)),
    }

    print(f"MSE      : {results['mse']:.6f}")
    print(f"MAE      : {results['mae']:.6f}")
    print(f"PSNR mean: {results['psnr_mean']:.4f}")
    print(f"SSIM mean: {results['ssim_mean']:.4f}")

    if plot:
        idx = np.random.choice(len(x_clean), size=min(
            n_show, len(x_clean)), replace=False)

        plt.figure(figsize=(3*n_show, 3))
        for i, j in enumerate(idx):
            plt.subplot(3, n_show, i+1)
            plt.imshow(x_clean[j].squeeze(), cmap="gray")
            plt.axis("off")
            if i == 0:
                plt.ylabel("Clean")

            plt.subplot(3, n_show, n_show + i+1)
            plt.imshow(x_noisy[j].squeeze(), cmap="gray")
            plt.axis("off")
            if i == 0:
                plt.ylabel("Noisy")

            plt.subplot(3, n_show, 2*n_show + i+1)
            plt.imshow(x_pred[j].squeeze(), cmap="gray")
            plt.axis("off")
            if i == 0:
                plt.ylabel("Denoised")

        plt.tight_layout()
        plt.show()

    return results


if __name__ == "__main__":
    _, x_test = load_mnist()
    evaluate_denoiser(
        model_path=r"models/denoiser_epoch_10_valLoss_0.0075.keras",
        x_test_clean=x_test,
        noise_factor=0.2,
        plot=True
    )
