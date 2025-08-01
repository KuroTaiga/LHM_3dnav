import os
import json
from tqdm import tqdm

def make_motion_in_place(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # List all JSON parameter files
    param_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    print(f"Found {len(param_files)} parameter files in: {input_folder}")

    for file_name in tqdm(param_files, desc="Processing"):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        with open(input_path, 'r') as f:
            data = json.load(f)

        # Set translation to zero
        if "trans" in data:
            data["trans"] = [0.0, 0.0, 0.0]

        # Save modified file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    print(f"\n✅ Output saved in: {output_folder}")

# Example usage
if __name__ == "__main__":
    input_dir = "./outputs/custom_motion/T_frontview/walk_45/smplx_params"  # Replace with your input folder path
    output_dir = "./outputs/custom_motion/T_frontview/walk_45/inplace_smplx_params"
    make_motion_in_place(input_dir, output_dir)
