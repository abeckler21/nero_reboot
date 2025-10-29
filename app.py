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

    fig = go.Figure()

    # Main orbit (dark blue)
    fig.add_trace(go.Scatterpolar(
        r=losses,
        theta=theta_deg,
        mode='lines+markers',
        line=dict(color='darkblue', width=2),
        marker=dict(size=4, color='darkblue'),
        name="NERO Orbit"
    ))

    # Green indicator line (initialized at 0°)
    fig.add_trace(go.Scatterpolar(
        r=[0, 1],
        theta=[0, 0],
        mode='lines',
        line=dict(color='limegreen', width=3),
        name="Rotation Angle",
        hoverinfo='skip'
    ))

    # --- Layout styling ---
    fig.update_layout(
        title=f"NERO Orbit (Sample {i}, Label {raw_label})",
        polar=dict(
            bgcolor='white',
            radialaxis=dict(
                range=[0, 1],
                showline=True,
                gridcolor='black',
                linecolor='black',
                tickfont=dict(color='black')
            ),
            angularaxis=dict(
                gridcolor='black',
                linecolor='black',
                tickfont=dict(color='black')
            )
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40)
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
    Compute aggregate NERO curves for a given label or across all samples.
    Returns average losses vs rotation angle as Plotly JSON.
    """
    import plotly.graph_objects as go
    from plotly.utils import PlotlyJSONEncoder
    import json
    import numpy as np

    selected_label = request.json.get("label")
    print(f"Computing aggregate NERO for: {selected_label}")

    # Build label list (A–Y, excluding J and Z)
    letters = [chr(c) for c in range(ord("A"), ord("Z") + 1) if c not in (ord("J"), ord("Z"))]
    valid_labels = [i if i < 9 else i + 1 for i in range(len(letters))]  # skip label 9
    label_map_inv = dict(zip(letters, valid_labels))

    # Determine subset
    if selected_label == "All":
        subset_df = sample_df
    else:
        label_num = label_map_inv[selected_label]
        subset_df = sample_df[sample_df["label"] == label_num]

    print(f"Subset size: {len(subset_df)}")

    # Skip if empty
    if len(subset_df) == 0:
        return jsonify({"nero_plot": None, "label": selected_label})

    # Parameters
    num_steps = 72
    group = "rotation"

    all_losses = []
    base_angles = None

    # Iterate over subset (limit to keep compute manageable)
    max_samples = min(len(subset_df), 50)
    print(f"Computing using {max_samples} samples...")

    for _, row in subset_df.sample(max_samples, random_state=42).iterrows():
        image = row.drop("label").values.reshape(28, 28)
        raw_label = int(row["label"])

        try:
            losses, angles = evaluate_orbit(
                model,
                image,
                label_raw=raw_label,
                group=group,
                num_steps=num_steps,
                label_map=label_map,
            )
        except Exception as e:
            print(f"Error computing NERO for sample: {e}")
            continue

        losses = np.array(losses, dtype=np.float32)
        if np.any(np.isnan(losses)):
            continue
        all_losses.append(losses)

        if base_angles is None:
            base_angles = np.array(angles, dtype=np.float32)

    if len(all_losses) == 0:
        print("No valid NERO curves.")
        return jsonify({"nero_plot": None, "label": selected_label})

    # Compute mean losses (and optionally convert to "confidence")
    mean_losses = np.mean(np.stack(all_losses), axis=0)
    mean_losses = np.clip(mean_losses, 0, 1)

    angles_deg = np.degrees(base_angles)

    # --- Build Plotly figure ---
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=mean_losses,
        theta=angles_deg,
        mode="lines+markers",
        line=dict(color="darkblue", width=2),
        marker=dict(size=4, color="darkblue"),
        name=f"Aggregate NERO ({selected_label})"
    ))

    fig.update_layout(
        title=f"Aggregate NERO – {selected_label}",
        polar=dict(
            bgcolor="white",
            radialaxis=dict(
                range=[0, 1],
                showline=True,
                gridcolor="black",
                linecolor="black",
                tickfont=dict(color="black")
            ),
            angularaxis=dict(
                gridcolor="black",
                linecolor="black",
                tickfont=dict(color="black")
            )
        ),
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    nero_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    return jsonify({"nero_plot": nero_json, "label": selected_label})


if __name__ == "__main__":
    app.run(debug=True)
