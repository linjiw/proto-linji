# create_sfu_yaml_relative.py
import os
import yaml
from pathlib import Path
import typer

def main(motion_dir: Path, output_yaml: Path):
    """
    Generates a YAML file listing all .pt files in a directory
    using relative paths from that directory, with a fixed FPS.
    """
    motion_dir = motion_dir.resolve() # Get absolute path of base dir
    output_yaml = output_yaml.resolve()

    motion_dict = {}
    print(f"Scanning directory: {motion_dir}")

    for root, _, files in os.walk(motion_dir):
        for file in files:
            if file.endswith(".pt"):
                full_path = Path(root) / file
                # Get path relative to the input motion_dir
                relative_path = full_path.relative_to(motion_dir)
                # Use '/' for paths in YAML, even on Windows
                yaml_path = str(relative_path).replace(os.path.sep, '/')

                # Assuming retargeting standardized FPS to 30
                motion_dict[yaml_path] = {'fps': 30}
                print(f"  Found: {yaml_path}")

    print(f"\nFound {len(motion_dict)} motion files.")

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, 'w') as f:
        yaml.dump(motion_dict, f, default_flow_style=None, sort_keys=True)

    print(f"YAML file saved to: {output_yaml}")

if __name__ == "__main__":
    typer.run(main)