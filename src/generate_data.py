import os
import csv

os.makedirs("data", exist_ok=True)
output_file = "data/image_manifest.csv" 

rows = []

for label in ["amd", "cataract"]:
    folder = f"data/raw/AMDNet23_Dataset/train/{label}"

    if not os.path.exists(folder):
        print(f"Warning: {folder} not found, skipping.")
        continue

    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            rows.append([filename, label])

print(f"Generated manifest with {len(rows)} records")

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "label"])
    writer.writerows(rows)

print(f"Created {output_file} with {len(rows)} records")