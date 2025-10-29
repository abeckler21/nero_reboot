from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
import os
from tensorflow import keras
from sklearn.decomposition import PCA
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nero_eval import evaluate_orbit, plot_nero

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
    label = request.json.get("label")
    path = f"static/aggregate_nero/aggregate_{label}.json"
    if not os.path.exists(path):
        return jsonify({"nero_plot": None, "label": label})
    with open(path, "r") as f:
        nero_json = f.read()
    return jsonify({"nero_plot": nero_json, "label": label})


if __name__ == "__main__":
    app.run(debug=True)
