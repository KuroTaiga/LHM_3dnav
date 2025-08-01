import os
import json
import shutil

# === Settings ===
input_dir = "./input_para"       # Folder with original SMPL-X .json files
output_dir = "./output_inplace"  # Folder to save new in-place files

# === Ensure output folder exists ===
os.makedirs(output_dir, exist_ok=True)

# === Load all JSON file paths, sorted ===
json_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".json"))

# === Extract reference translation from first file ===
with open(os.path.join(input_dir, json_files[0]), 'r') as f:
    first_data = json.load(f)
trans_ref = first_data['trans']

# === Process and save each JSON with fixed translation ===
for fname in json_files:
    input_path = os.path.join(input_dir, fname)
    output_path = os.path.join(output_dir, fname)

    with open(input_path, 'r') as f:
        data = json.load(f)

    data['trans'] = trans_ref  # Replace translation
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

print(f"Processed {len(json_files)} files. Output saved to '{output_dir}'.")
