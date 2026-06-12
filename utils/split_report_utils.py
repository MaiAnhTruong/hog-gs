import json
import os


def json_path(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")


def write_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)


def write_lines(path: str, values):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for value in values:
            f.write(str(value) + "\n")

