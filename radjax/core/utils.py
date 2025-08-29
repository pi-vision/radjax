import os, yaml

def dump_yaml(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"✔️ Saved YAML → {path}")

# Helper: pretty preview (first ~80 lines)
def preview_yaml(data, max_lines=80):
    s = yaml.safe_dump(data, sort_keys=False)
    print("\n".join(s.splitlines()[:max_lines]))