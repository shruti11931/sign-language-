import json
import pandas as pd

LABEL_MAP_PATH = "models/asl_label_map.json"
CSV_PATH = "data/asl_landmarks.csv"

with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)

# Group all target indices by their digit name (only "0"-"9" values)
digit_groups = {}
for idx_str, name in label_map.items():
    if name.isdigit() and len(name) == 1:
        digit_groups.setdefault(name, []).append(int(idx_str))

print("Found duplicate groups:")
for name, indices in digit_groups.items():
    print(f"  digit '{name}': indices {sorted(indices)}")

df = pd.read_csv(CSV_PATH)

# For each digit, keep the index with the MOST samples as canonical,
# remap all other duplicate indices' rows to it, merging the data.
canonical_map = {}
for name, indices in digit_groups.items():
    if len(indices) <= 1:
        canonical_map[indices[0]] = indices[0]
        continue
    counts = {i: (df["target"] == i).sum() for i in indices}
    canonical = max(counts, key=counts.get)
    print(f"  '{name}': merging {indices} -> canonical {canonical} (had {counts})")
    for i in indices:
        canonical_map[i] = canonical

df["target"] = df["target"].apply(lambda t: canonical_map.get(t, t))
df.to_csv(CSV_PATH, index=False)

# Rebuild a clean label map: one entry per unique target that's actually still used
kept_targets = sorted(df["target"].unique())
clean_label_map = {}
for name, indices in digit_groups.items():
    canonical = canonical_map[indices[0]]
    clean_label_map[str(canonical)] = name

# Keep all non-digit (letter) entries untouched
for idx_str, name in label_map.items():
    if not (name.isdigit() and len(name) == 1):
        clean_label_map[idx_str] = name

with open(LABEL_MAP_PATH, "w") as f:
    json.dump(clean_label_map, f, indent=2)

print("\nCleaned. Final class count:", len(clean_label_map))
print("Final row counts per target:")
print(df["target"].value_counts().sort_index())