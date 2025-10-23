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
from io import BytesIO
import base64


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

@app.route("/get_nero", methods=["POST"])
def get_nero():
    """Compute NERO orbit and return a Matplotlib-styled image panel."""
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt

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

    # --- Create Matplotlib figure like in nero_eval.py ---
    fig = plt.figure(figsize=(4.8, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.75, 1.5])
    ax_img = fig.add_subplot(gs[0])
    ax_nero = fig.add_subplot(gs[1], projection="polar")

    # Plot image
    ax_img.imshow(image, cmap="gray")
    ax_img.set_title(f"Sample {i} (Label {raw_label})", fontsize=10)
    ax_img.axis("off")

    # Use your exact NERO plotting function for the polar subplot
    plot_individual_nero(losses, angles, title="NERO Plot (0–1 scale)", ax=ax_nero)

    # Save to base64
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return jsonify({
        "label": int(raw_label),
        "sample": i,
        "image": img_b64
    })


@app.route("/get_ice", methods=["POST"])
def get_ice():
    """Compute ICE curves for selected features (pixels or channels)."""
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt

    i = int(request.json.get("index"))
    raw_label = int(sample_df.iloc[i]["label"])
    image = sample_df.drop("label", axis=1).iloc[i].values.reshape(28, 28, 1).astype("float32") / 255.0

    # Choose a few pixels to vary (e.g., 4 corner pixels, 4 center pixels)
    pixel_coords = [(7,7), (14,14), (7,20), (20,7)]
    x_vals = np.linspace(0, 1, 20)
    preds = {coord: [] for coord in pixel_coords}

    for coord in pixel_coords:
        img_mod = image.copy()
        for val in x_vals:
            img_mod[coord] = val
            p = model.predict(img_mod[None, ...])[0][label_map[raw_label]]
            preds[coord].append(p)

    # --- Plot ICE curves ---
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for coord, y in preds.items():
        ax.plot(x_vals, y, label=f"pixel {coord}")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel(f"Predicted P(class={raw_label})")
    ax.legend(fontsize=6)
    ax.set_title("ICE curves for selected pixels")

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return jsonify({
        "label": int(raw_label),
        "sample": i,
        "image": img_b64
    })


@app.route("/get_pca_dim", methods=["POST"])
def get_pca_dim():
    """Recompute PCA and return coordinates using PC1 vs selected PCN."""
    dims = int(request.json.get("dims", 2))
    if dims < 2:
        dims = 2

    print(f"[Recompute] Updating PCA projection: PC1 vs PC{dims}")

    images = test_df.drop("label", axis=1).values.reshape(-1, 28, 28).astype("float32") / 255.0
    feats = [model.predict(img[None, ..., None], verbose=0).flatten() for img in images]
    coords = PCA(n_components=max(dims, 2)).fit_transform(np.array(feats))

    # Use PC1 (x-axis) and PC_N (y-axis)
    reduced = np.column_stack((coords[:, 0], coords[:, dims - 1]))
    labels = test_df["label"].astype(int).tolist()

    data = [
        {"x": float(reduced[i, 0]), "y": float(reduced[i, 1]),
         "label": labels[i], "idx": i}
        for i in range(len(test_df))
    ]
    return jsonify(data)

@app.route("/get_confidence_hist", methods=["POST"])
def get_confidence_hist():
    """Generate histogram of predicted confidences across all labels for the selected image."""
    idx = int(request.json.get("index", 0))
    row = test_df.iloc[idx]
    label = int(row["label"])
    image = row.drop("label").values.reshape(28, 28).astype("float32") / 255.0

    x = image[None, ..., None]
    probs = model.predict(x, verbose=0)[0]

    n_classes = len(probs)
    split_point = 13  # number of bars per row

    # --- Split the bars into two rows ---
    probs_top = probs[:split_point]
    probs_bottom = probs[split_point:]

    fig, axs = plt.subplots(2, 1, figsize=(6, 5), sharey=True)

    # --- Top row ---
    bars_top = axs[0].bar(range(len(probs_top)), probs_top, color="steelblue", edgecolor="black")
    # --- Bottom row ---
    bars_bottom = axs[1].bar(range(len(probs_bottom)), probs_bottom, color="steelblue", edgecolor="black")

    # --- Highlight true label if in range ---
    if label < split_point:
        bars_top[label].set_color("orange")
    elif label < n_classes:
        bars_bottom[label - split_point].set_color("orange")

    # --- Add numeric labels above bars ---
    for ax, bars in zip(axs, [bars_top, bars_bottom]):
        for p in bars:
            ax.text(
                p.get_x() + p.get_width() / 2,
                p.get_height() + 0.02,
                f"{p.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Prob.")
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels(
            [str(i + (0 if ax is axs[0] else split_point)) for i in range(len(bars))],
            fontsize=8,
        )

    axs[0].set_title(f"Confidence Histogram — Sample {idx} (True label: {label})", fontsize=11)
    axs[1].set_xlabel("Class Label")

    plt.tight_layout()

    # --- Convert to base64 for the frontend ---
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return jsonify({"label": label, "sample": idx, "image": img_b64})



if __name__ == "__main__":
    app.run(debug=True)
