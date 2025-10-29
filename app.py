from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.decomposition import PCA
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nero_eval import evaluate_orbit, plot_individual_nero

app = Flask(__name__)

# --- Load model and data once ---
print("Loading model and data...")
model = keras.models.load_model("training/models/model01.keras")
train_df = pd.read_csv("training/sign_mnist_train.csv")
test_df = pd.read_csv("training/sign_mnist_test.csv")

unique_labels = sorted(np.unique(train_df["label"].values))
label_map = {old: new for new, old in enumerate(unique_labels)}

# --- Compute PCA coordinates once ---
def compute_pca_embeddings(n_samples=500):
    sample_df = test_df.sample(n_samples, random_state=42).reset_index(drop=True)
    images = sample_df.drop("label", axis=1).values.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    labels = sample_df["label"].values

    feature_model = keras.Sequential(model.layers[:-1])
    feats = feature_model.predict(images, verbose=0)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(feats)
    return coords, labels, sample_df

coords, labels, sample_df = compute_pca_embeddings(500)
print("PCA computed for", len(labels), "samples")

# --- Routes ---

@app.route("/")
def index():
    # Prepare data for Plotly scatter
    data = [
        {"x": float(coords[i, 0]),
         "y": float(coords[i, 1]),
         "label": int(labels[i]),
         "idx": int(i)}
        for i in range(len(labels))
    ]
    return render_template("index.html", data=json.dumps(data))


def plot_nero(r, theta_deg, title=None, include_indicator=False, label=None, sample_idx=None):
    """Return a Plotly figure object for a NERO polar plot.

    Args:
        r (array-like): Radial values (confidences/losses).
        theta_deg (array-like): Angles in degrees (same length as r).
        title (str, optional): Figure title.
        include_indicator (bool): If True, adds the green 0° indicator line.
        label (int, optional): Label/class index (for title).
        sample_idx (int, optional): Sample index (for title).

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    print("\n=== DEBUG: plot_nero() data summary ===")

    def summarize(name, arr):
        arr = np.array(arr)
        print(f"{name}:")
        print(f"  type:        {type(arr)}")
        print(f"  dtype:       {arr.dtype}")
        print(f"  ndim:        {arr.ndim}")
        print(f"  shape:       {arr.shape}")
        print(f"  size:        {arr.size}")
        print(f"  min, max:    {arr.min() if arr.size > 0 else 'N/A'}, {arr.max() if arr.size > 0 else 'N/A'}")
        print(f"  has NaNs?    {np.isnan(arr).any()}")
        if arr.size > 0:
            print(f"  first 5:     {np.round(arr[:5], 4)}")
            print(f"  last 5:      {np.round(arr[-5:], 4)}")
        print("-" * 50)

    summarize("r", r)
    summarize("theta_deg", theta_deg)

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
            title = f"NERO Orbit (Sample {sample_idx}, Label {label})"
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
                rotation=0,  # 0° = East
                direction="clockwise",
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



@app.route("/get_nero", methods=["POST"])
def get_nero():
    """Compute NERO orbit and return original image + styled Plotly JSON for NERO plot."""
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.utils import PlotlyJSONEncoder
    import json

    i = int(request.json.get("index"))
    raw_label = int(sample_df.iloc[i]["label"])
    image = sample_df.drop("label", axis=1).iloc[i].values.reshape(28, 28)

    # --- Compute NERO orbit ---
    losses, angles = evaluate_orbit(
        model,
        image,
        label_raw=raw_label,
        group="rotation",
        num_steps=72,
        label_map=label_map,
    )

    # --- Encode original image ---
    fig_img, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(image, cmap="gray")
    ax.axis("off")
    buf = BytesIO()
    fig_img.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig_img)
    buf.seek(0)
    image_b64 = base64.b64encode(buf.read()).decode("utf-8")

    # --- Create Plotly NERO polar plot ---
    theta_deg = np.degrees(angles)

    fig = plot_nero(
        r=losses,
        theta_deg=theta_deg,
        label=raw_label,
        sample_idx=i,
        include_indicator=True
    )

    nero_json = json.dumps(fig, cls=PlotlyJSONEncoder)

    return jsonify({
        "label": int(raw_label),
        "sample": i,
        "original_image": image_b64,
        "nero_plot": nero_json
    })



@app.route("/get_aggregate_nero", methods=["POST"])
def get_aggregate_nero():
    """
    Compute an aggregate NERO confidence orbit and plot as a histogram
    (confidence vs rotation angle).
    """
    import plotly.graph_objects as go
    from plotly.utils import PlotlyJSONEncoder
    import numpy as np, json

    selected_label = request.json.get("label")
    print(f"\n=== Aggregate NERO for: {selected_label} ===")

    # --- Build label mapping (A–Y, excluding J and Z) ---
    letters = [chr(c) for c in range(ord("A"), ord("Z") + 1) if c not in (ord("J"), ord("Z"))]
    valid_labels = [i if i < 9 else i + 1 for i in range(len(letters))]
    label_map_inv = dict(zip(letters, valid_labels))

    # --- Select subset ---
    if selected_label == "All":
        subset_df = sample_df
    else:
        label_num = label_map_inv[selected_label]
        subset_df = sample_df[sample_df["label"] == label_num]

    n_samples = len(subset_df)
    print(f"Subset size: {n_samples}")
    if n_samples == 0:
        return jsonify({"nero_plot": None, "label": selected_label})

    num_steps = 72
    group = "rotation"
    max_samples = min(n_samples, 50)
    all_conf = []

    print(f"Using {max_samples} samples...")

    # --- Collect individual orbits ---
    for _, row in subset_df.sample(max_samples, random_state=42).iterrows():
        image = row.drop("label").values.reshape(28, 28)
        raw_label = int(row["label"])
        try:
            confidences, angles = evaluate_orbit(
                model,
                image,
                label_raw=raw_label,
                group=group,
                num_steps=num_steps,
                label_map=label_map,
            )
            confidences = np.array(confidences, dtype=np.float32)
            if len(confidences) != num_steps or np.isnan(confidences).any():
                continue
            all_conf.append(confidences)
        except Exception as e:
            print(f"Error computing NERO: {e}")
            continue

    if not all_conf:
        print("No valid NERO curves found.")
        return jsonify({"nero_plot": None, "label": selected_label})

    # --- Compute mean orbit and define angle grid ---
    all_conf = np.stack(all_conf)
    mean_conf = np.mean(all_conf, axis=0)
    theta_deg = np.linspace(0, 360, num_steps, endpoint=False)

    # --- Sanitize arrays ---
    mean_conf = np.nan_to_num(mean_conf, nan=0.0, posinf=0.0, neginf=0.0)
    theta_deg = np.nan_to_num(theta_deg, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Convert to Python lists *after* cleaning ---
    mean_conf = mean_conf.tolist()
    theta_deg = theta_deg.tolist()
    print(type(mean_conf[0]), type(theta_deg[0]))

    # --- Debug print ---
    print("\n--- Mean aggregate points (used for plot) ---")
    for a, r in zip(theta_deg, mean_conf):
        print(f"{a:7.2f}° → {r:.4f}")
    print("---------------------------------------------------\n")

    # call the func?
    fig = plot_nero(
        r=mean_conf,
        theta_deg=theta_deg,
        title=f"Aggregate NERO - {selected_label}",
        include_indicator=False
    )

    nero_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    return jsonify({"nero_plot": nero_json, "label": selected_label})




if __name__ == "__main__":
    app.run(debug=True)
