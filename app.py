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
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5])
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

if __name__ == "__main__":
    app.run(debug=True)
