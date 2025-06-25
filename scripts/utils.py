import os

def get_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def get_latest_pipeline_dir() -> str:
    project_root = get_project_root()
    latest_file = os.path.join(project_root, "data", "processed", "latest.txt")
    if not os.path.exists(latest_file):
        raise FileNotFoundError("Missing data/processed/latest.txt. Run load_data.py first.")
    with open(latest_file, "r") as f:
        return f.read().strip()
