import os
import json
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from nero_eval import evaluate_orbit, plot_nero, label_to_letter  # reuse existing helpers

# --- Configuration ---
OUTPUT_DIR = "static/aggregate_nero"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load model and data ---
print("Loading model and data...")
model = keras.models.load_model("training/models/model01.keras")
train_df = pd.read_csv("training/sign_mnist_train.csv")
test_df = pd.read_csv("training/sign_mnist_test.csv")

unique_labels = sorted(np.unique(train_df["label"].values))
label_map = {old: new for new, old in enumerate(unique_labels)}

# --- Utility ---
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

# --- Label setup ---
letters = [chr(c) for c in range(ord("A"), ord("Z") + 1) if c not in (ord("J"), ord("Z"))]
valid_labels = [i if i < 9 else i + 1 for i in range(len(letters))]
label_map_inv = dict(zip(letters, valid_labels))

# --- Aggregate computation ---
def compute_aggregate_for_label(label_name, max_samples=50, num_steps=72, group="rotation"):
    if label_name == "All":
        subset_df = sample_df
    else:
        label_num = label_map_inv[label_name]
        subset_df = sample_df[sample_df["label"] == label_num]
    if len(subset_df) == 0:
        print(f"[!] Skipping {label_name}: no samples found.")
        return None

    all_conf = []
    for _, row in subset_df.sample(min(len(subset_df), max_samples), random_state=42).iterrows():
        image = row.drop("label").values.reshape(28, 28)
        raw_label = int(row["label"])
        try:
            conf, angles = evaluate_orbit(
                model, image,
                label_raw=raw_label,
                group=group,
                num_steps=num_steps,
                label_map=label_map
            )
            conf = np.array(conf, dtype=np.float32)
            if len(conf) == num_steps and not np.isnan(conf).any():
                all_conf.append(conf)
        except Exception as e:
            print(f"[!] Error for {label_name}: {e}")
            continue

    if not all_conf:
        print(f"[!] No valid orbits for {label_name}.")
        return None

    mean_conf = np.nan_to_num(np.mean(np.stack(all_conf), axis=0), nan=0.0)
    theta_deg = np.linspace(0, 360, num_steps, endpoint=False)

    fig = plot_nero(
        r=mean_conf.tolist(),
        theta_deg=theta_deg.tolist(),
        title=f"Aggregate NERO - {label_name}",
        include_indicator=False
    )
    return fig

# --- Run for all labels ---
labels_to_run = ["All"] + letters
for lbl in labels_to_run:
    print(f"\n=== Generating aggregate for {lbl} ===")
    fig = compute_aggregate_for_label(lbl)
    if fig is not None:
        outpath = os.path.join(OUTPUT_DIR, f"aggregate_{lbl}.json")
        with open(outpath, "w") as f:
            json.dump(fig, f, cls=PlotlyJSONEncoder)
        print(f"[✓] Saved {lbl} → {outpath}")
    else:
        print(f"[x] Skipped {lbl}")

print("All aggregate plots generated and saved.")
