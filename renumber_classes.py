import json
import pandas as pd

LABEL_MAP_PATH = "models/asl_label_map.json"
CSV_PATH = "data/asl_landmarks.csv"

with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)

df = pd.read_csv(CSV_PATH)

# Sort existing target indices for a stable, deterministic renumbering
old_indices = sorted(int(k) for k in label_map.keys())
old_to_new = {old: new for new, old in enumerate(old_indices)}

# Remap the CSV's target column
df["target"] = df["target"].map(old_to_new)
df.to_csv(CSV_PATH, index=False)

# Rebuild the label map with clean 0..N-1 keys
new_label_map = {str(old_to_new[int(k)]): v for k, v in label_map.items()}
with open(LABEL_MAP_PATH, "w") as f:
    json.dump(new_label_map, f, indent=2)

print(f"Renumbered {len(old_indices)} classes to 0..{len(old_indices)-1}")
print("Final row counts per new target:")
print(df["target"].value_counts().sort_index())