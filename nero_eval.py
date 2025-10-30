import numpy as np
import cv2
from tensorflow import keras
from sklearn.decomposition import PCA
import plotly.graph_objects as go


# =====================================================
#  Label to letters
# =====================================================
def label_to_letter(label: int) -> str:
    """
    Convert SignMNIST label (0-24) to its corresponding letter (A-Y, skipping J and Z).
    """
    # Letters A-Z, excluding J (9) and Z (25)
    letters = [chr(c) for c in range(ord('A'), ord('Z') + 1) if c not in (ord('J'), ord('Z'))]
    
    if 0 <= label < 25:
        if label < 9:
            return letters[label]
        else:
            return letters[label - 1]
    else:
        raise ValueError(f"Label {label} out of valid range (0-24)")



# =====================================================
#  Orbit generation to calculate all image transformations
# =====================================================
def generate_orbit(
    image: np.ndarray,
    group: str = "rotation",
    num_steps: int = 36,
    transform_fn=None
):
    image = np.asarray(image)
    if image.ndim == 1:
        image = image.reshape(28, 28)
    image = image.astype(np.float32)

    orbit = []
    group_elements = []

    if transform_fn is not None:
        # if given alternate transform do it
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
#  Evaluate confidences of all images in orbit
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
    orbit, group_elements = generate_orbit(image, group, num_steps)
    confidences = []

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
            conf = float(loss_fn(y_pred, correct_class))
        else:
            conf = float(y_pred[correct_class])                     # confidence for true class

        confidences.append(conf)

    return confidences, group_elements


# =====================================================
#  Create polar plot of NERO evaluated orbit
# =====================================================
def plot_nero(r, theta_deg, title=None, include_indicator=False, label=None, sample_idx=None):
    # Create figure
    fig = go.Figure()

    # Main orbit (dark blue)
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=theta_deg,
        mode="lines+markers",
        line=dict(color="darkblue", width=2),
        marker=dict(size=4, color="darkblue"),
        name="NERO Orbit",
    ))

    # Optional rotation indicator
    if include_indicator:
        fig.add_trace(go.Scatterpolar(
            r=[0, 1],
            theta=[0, 0],
            mode="lines",
            line=dict(color="limegreen", width=3),
            name="Rotation Angle",
            hoverinfo="skip",
        ))

    # Construct title if not provided
    if title is None:
        if sample_idx is not None and label is not None:
            title = f"NERO Orbit (Sample {sample_idx}, Label {label}: {label_to_letter(label)})"
        else:
            title = "NERO Orbit"

    # Layout styling
    fig.update_layout(
        title=title,
        polar=dict(
            bgcolor="white",
            radialaxis=dict(
                range=[0, 1],
                showline=True,
                gridcolor="black",
                linecolor="black",
                tickfont=dict(color="black"),
            ),
            angularaxis=dict(
                rotation=0,  # 0 deg = East
                direction="counterclockwise",
                gridcolor="black",
                linecolor="black",
                tickfont=dict(color="black"),
            ),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


# =====================================================
#  Compute the PCA coordinate stuff
# =====================================================
def compute_pca_embeddings(test_df, model, n_samples=500):
    sample_df = test_df.sample(n_samples, random_state=42).reset_index(drop=True)
    images = sample_df.drop("label", axis=1).values.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    labels = sample_df["label"].values

    feature_model = keras.Sequential(model.layers[:-1])
    feats = feature_model.predict(images, verbose=0)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(feats)
    return coords, labels, sample_df