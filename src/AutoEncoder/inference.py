import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append(".")
from model import DenoiseAutoencoder


def load_and_preprocess_image(image_path: str, target_size=(28, 28)) -> np.ndarray:
    """
    Loads an image, converts to grayscale, resizes to 28x28, and normalizes to [0, 1].
    Returns: (28, 28, 1) float32 numpy array.
    """
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_image(raw, channels=1, expand_animations=False)  # grayscale
    img = tf.image.convert_image_dtype(img, tf.float32)  # -> [0,1]
    img = tf.image.resize(img, target_size, method="bilinear")

    # Ensure shape is exactly (28, 28, 1)
    img = tf.ensure_shape(img, (target_size[0], target_size[1], 1))
    return img.numpy()


def add_gaussian_noise(x: np.ndarray, noise_factor: float = 0.2, seed: int | None = None) -> np.ndarray:
    """
    x: (28, 28, 1) or (1, 28, 28, 1) in [0, 1]
    Returns same shape, clipped to [0, 1].
    """
    if seed is not None:
        tf.random.set_seed(seed)

    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    noise = noise_factor * tf.random.normal(shape=tf.shape(x_tf))
    x_noisy = tf.clip_by_value(x_tf + noise, 0.0, 1.0)
    return x_noisy.numpy()


def inference(
    model_path: str,
    image_path: str,
    noise_factor: float = 0.2,
    seed: int | None = 123,
    show: bool = True,
    save_dir: str | None = None,
):
    """
    Loads model, reads image, adds noise, reconstructs it, and optionally shows/saves results.
    Returns dict with clean/noisy/reconstructed images (all in [0,1]).
    """
    # Load model (custom class)
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"DenoiseAutoencoder": DenoiseAutoencoder},
    )


    x_clean = load_and_preprocess_image(image_path)  # (28,28,1)

    # Add noise (match training style)
    x_noisy = add_gaussian_noise(x_clean, noise_factor=noise_factor, seed=seed)  # (28,28,1)

    # Predict expects batch
    x_noisy_b = np.expand_dims(x_noisy, axis=0)  # (1,28,28,1)
    x_pred = model.predict(x_noisy_b, verbose=0)[0]  # back to (28,28,1)
    x_pred = np.clip(x_pred, 0.0, 1.0)

    mse = float(np.mean((x_clean - x_pred) ** 2))
    mae = float(np.mean(np.abs(x_clean - x_pred)))
    psnr = float(tf.image.psnr(x_clean, x_pred, max_val=1.0).numpy())
    ssim = float(tf.image.ssim(x_clean, x_pred, max_val=1.0).numpy())

    print(f"Image: {image_path}")
    print(f"noise_factor={noise_factor}")
    print(f"MSE : {mse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"PSNR: {psnr:.4f}")
    print(f"SSIM: {ssim:.4f}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        def save_img(arr, name):
            path = os.path.join(save_dir, name)
            plt.imsave(path, arr.squeeze(), cmap="gray", vmin=0, vmax=1)
            return path

        clean_path = save_img(x_clean, "clean.png")
        noisy_path = save_img(x_noisy, "noisy.png")
        pred_path = save_img(x_pred, "denoised.png")
        print("Saved:")
        print(" -", clean_path)
        print(" -", noisy_path)
        print(" -", pred_path)

    if show:
        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.title("Clean")
        plt.imshow(x_clean.squeeze(), cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Noisy")
        plt.imshow(x_noisy.squeeze(), cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Denoised")
        plt.imshow(x_pred.squeeze(), cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return {
        "clean": x_clean,
        "noisy": x_noisy,
        "denoised": x_pred,
        "metrics": {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim},
    }


if __name__ == "__main__":

    MODEL_PATH = r"models/denoiser_epoch_10_valLoss_0.0075.keras"
    IMAGE_PATH = r"src/AutoEncoder/test_data.png"

    inference(
        model_path=MODEL_PATH,
        image_path=IMAGE_PATH,
        noise_factor=0.2,
        seed=123,
        show=True,
        save_dir="inference_outputs",  # set None to disable saving
    )
