import json
import pandas as pd

LABEL_MAP_PATH = "models/asl_label_map.json"
CSV_PATH = "data/asl_landmarks.csv"

with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)

# Identify every index whose name is a single digit "0"-"9" (letters/space/del are untouched)
digit_indices = {int(k) for k, v in label_map.items() if v.isdigit() and len(v) == 1}
print("Removing digit indices:", sorted(digit_indices))

df = pd.read_csv(CSV_PATH)
before = len(df)
df = df[~df["target"].isin(digit_indices)]
after = len(df)
print(f"Dropped {before - after} digit rows, {after} rows remain (letters only)")

# Renumber remaining letter indices to a clean contiguous 0..N-1 range
old_indices = sorted(int(k) for k in label_map.keys() if int(k) not in digit_indices)
old_to_new = {old: new for new, old in enumerate(old_indices)}

df["target"] = df["target"].map(old_to_new)
df.to_csv(CSV_PATH, index=False)

new_label_map = {str(old_to_new[int(k)]): v for k, v in label_map.items() if int(k) not in digit_indices}
with open(LABEL_MAP_PATH, "w") as f:
    json.dump(new_label_map, f, indent=2)

print(f"Clean label map now has {len(new_label_map)} classes (letters only)")
print(df["target"].value_counts().sort_index())