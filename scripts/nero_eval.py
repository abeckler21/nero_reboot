import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.activations import softmax
import pandas as pd

# =====================================================
#  ORBIT GENERATION (new helper function)
# =====================================================
def generate_orbit(
    image: np.ndarray,
    group: str = "rotation",
    num_steps: int = 36,
    transform_fn=None
):
    """
    Generate the orbit G(x) for a given image under the specified transformation group.

    Parameters
    ----------
    image : np.ndarray
        Input image (H x W or H x W x C).
    group : str
        Type of transformation group ('rotation', 'translation', 'flip', etc.).
    num_steps : int
        Number of discrete group elements (e.g. 36 => 10° increments for rotation).
    transform_fn : callable, optional
        Custom transformation φ(g, x). If provided, it overrides built-in group logic.

    Returns
    -------
    orbit : list[np.ndarray]
        List of transformed images [φ(g₁,x), φ(g₂,x), …].
    group_elements : list
        List of group parameters corresponding to each transformation (e.g. angles or shifts).
    """
    image = np.asarray(image)
    if image.ndim == 1:
        image = image.reshape(28, 28)
    image = image.astype(np.float32)

    orbit = []
    group_elements = []

    if transform_fn is not None:
        for g in range(num_steps):
            orbit.append(transform_fn(image, g))
            group_elements.append(g)
        return orbit, group_elements

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    if group == "rotation":
        for i in range(num_steps):
            angle = 360 * i / num_steps
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            transformed = cv2.warpAffine(image, M, (w, h))
            orbit.append(transformed)
            group_elements.append(angle)

    elif group == "translation":
        max_shift = w // 10  # arbitrary range (10% of image width)
        for i in range(num_steps):
            dx = int(max_shift * np.cos(2 * np.pi * i / num_steps))
            dy = int(max_shift * np.sin(2 * np.pi * i / num_steps))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            transformed = cv2.warpAffine(image, M, (w, h))
            orbit.append(transformed)
            group_elements.append((dx, dy))

    elif group == "flip":
        orbit = [image, cv2.flip(image, 1), cv2.flip(image, 0), cv2.flip(image, -1)]
        group_elements = ["none", "horizontal", "vertical", "both"]

    else:
        raise ValueError(f"Unsupported group type: {group}")

    return orbit, group_elements


# =====================================================
#  NERO EVALUATION EXAMPLE
# =====================================================
def evaluate_orbit(
    model,
    image,
    label_raw,                  # the original integer label from CSV
    group="rotation",
    num_steps=360,
    label_map=None,             # the dict used in training to remap labels
    loss_fn=None
):
    """
    Compute model confidence over the orbit. Assumes model outputs probabilities
    (i.e., final Dense(..., activation='softmax')).
    """
    orbit, group_elements = generate_orbit(image, group, num_steps)
    losses = []

    # Map raw label -> contiguous class index (0..23) exactly as in training
    if label_map is not None:
        correct_class = label_map[int(label_raw)]
    else:
        # If your model was trained without remap (unlikely), use raw label
        correct_class = int(label_raw)

    for transformed in orbit:
        x_prime = transformed.astype("float32") / 255.0            # scale [0,1]
        if x_prime.ndim == 2:                                       # (H,W) -> (H,W,1)
            x_prime = np.expand_dims(x_prime, -1)
        x_prime = np.expand_dims(x_prime, 0)                        # (1,H,W,1)

        y_pred = model.predict(x_prime, verbose=0)[0]               # probabilities shape (C,)

        if loss_fn is not None:
            # If you really want a loss, pass a function expecting (probs, class_index)
            loss = float(loss_fn(y_pred, correct_class))
        else:
            loss = float(y_pred[correct_class])                     # confidence for true class

        losses.append(loss)

    return losses, group_elements


# === Aggregate NERO computation ===
def evaluate_aggregate_orbit(model, test_df, label_map, group="rotation", num_steps=360, n_samples=50):
    """Compute aggregate NERO curve across multiple samples."""
    all_confidences = []

    for i in range(n_samples):
        raw_label = int(test_df.iloc[i]["label"])
        image = test_df.drop("label", axis=1).iloc[i].values.reshape(28, 28)

        losses, angles = evaluate_orbit(
            model,
            image,
            label_raw=raw_label,
            group=group,
            num_steps=num_steps,
            label_map=label_map,
        )
        all_confidences.append(losses)

    all_confidences = np.array(all_confidences)  # shape (n_samples, num_steps)
    mean_conf = np.mean(all_confidences, axis=0)
    std_conf = np.std(all_confidences, axis=0)
    return angles, mean_conf, std_conf


# =====================================================
#  VISUALIZATION: CIRCULAR NERO PLOT
# =====================================================
def plot_individual_nero(losses, group_elements, title="NERO Plot (0–1 scale)"):
    """Plot NERO polar diagram with 0°=right, 90°=top, 180°=left, 270°=bottom,
    and radius fixed from 0–1 with horizontal tick labels."""
    # Convert degrees → radians
    if isinstance(group_elements[0], (int, float)):
        angles = np.deg2rad(group_elements)
    else:
        angles = np.linspace(0, 2 * np.pi, len(losses))

    values = np.array(losses)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, polar=True)

    # --- Plot confidence curve ---
    ax.plot(angles, values, linewidth=2, color="royalblue")

    # --- Fix radius to [0, 1] ---
    ax.set_ylim(0, 1)
    ax.set_rticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 6)])

    # --- Set orientation ---
    ax.set_theta_zero_location("E")   # 0° on the right (East)
    ax.set_theta_direction(1)         # counterclockwise rotation
    ax.set_rlabel_position(0)         # put radius labels along 0° line (horizontal)

    # --- Aesthetic tweaks ---
    ax.set_title(title, va="bottom")
    plt.tight_layout()
    plt.show()


def plot_aggregate_and_individual(losses, angles, mean_conf, std_conf, title="NERO Aggregate + Individual"):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # individual
    ax.plot(np.deg2rad(angles), losses, color="gray", alpha=0.5, label="Individual sample")

    # aggregate mean
    ax.plot(np.deg2rad(angles), mean_conf, color="royalblue", linewidth=2, label="Mean confidence")

    # aggregate std shading
    lower = np.clip(mean_conf - std_conf, 0, 1)
    upper = np.clip(mean_conf + std_conf, 0, 1)
    ax.fill_between(np.deg2rad(angles), lower, upper, color="royalblue", alpha=0.3, label="±1 SD")

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, 1)
    ax.set_title(title, va="bottom")
    ax.legend(loc="lower left", bbox_to_anchor=(1.1, 0.1))
    plt.tight_layout()
    plt.show()


# =====================================================
#  EXAMPLE USAGE
# =====================================================
if __name__ == "__main__":
    test_label = 2

    # --- Load your trained model ---
    model = keras.models.load_model("models/model01.keras")

    # --- Load the Sign Language MNIST data ---
    train_df = pd.read_csv("images/sign_mnist_train.csv")
    test_df = pd.read_csv("images/sign_mnist_test.csv")

    # --- Recreate the SAME label_map used in training ---
    unique_labels = sorted(np.unique(train_df["label"].values))   # from your CSV
    label_map = {old: new for new, old in enumerate(unique_labels)}

    # --- Pick one test sample with a specific label (e.g. 0 = 'A') ---
    i = np.where(test_df["label"] == test_label)[0][0]
    raw_label = int(test_df.iloc[i]["label"])
    image = test_df.drop("label", axis=1).iloc[i].values.reshape(28, 28)

    # --- Evaluate the orbit ---
    image = np.array(image, dtype=np.float32)

    # individual NERO
    losses, angles = evaluate_orbit(model, image, label_raw=raw_label, group="rotation", num_steps=360, label_map=label_map)

    # aggregate NERO
    angles, mean_conf, std_conf = evaluate_aggregate_orbit(model, test_df, label_map, group="rotation", num_steps=360, n_samples=5)

    # plot both together
    plot_aggregate_and_individual(losses, angles, mean_conf, std_conf, title="NERO: Individual vs Aggregate")

    # --- Sanity check: unrotated confidence ---
    x0 = image.astype("float32") / 255.0
    x0 = x0[None, ..., None]
    p0 = model.predict(x0, verbose=0)[0]
    print("Top-5 classes:", np.argsort(p0)[-5:][::-1])
    print("Top-5 probs:", np.sort(p0)[-5:][::-1])
    print("Sum of probs:", np.sum(p0))
    print("Correct-class prob:", p0[label_map[raw_label]])

