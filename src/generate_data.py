import os
import csv

output_file = "data/image_manifest.csv"

rows = []

for label in ["amd", "cataract"]:
    folder = f"data/raw/AMDNet23 Dataset/train/{label}"

    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            rows.append([filename, label])

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "label"])
    writer.writerows(rows)

print(f"Created {output_file} with {len(rows)} records")