import argparse
import json
import os
import re
import sys


DEFAULT_EXCLUDES = {
    "__pycache__",
    ".git",
    "submodules",
}


DANGEROUS_PATTERNS = [
    ("first-K train pool slice", re.compile(r"train_pool\s*\[\s*:\s*[A-Za-z0-9_]+\s*\]")),
    ("first-K sorted image slice", re.compile(r"sorted_images\s*\[\s*:\s*[A-Za-z0-9_]+\s*\]")),
    ("first-K selected image slice", re.compile(r"selected_images\s*\[\s*:\s*[A-Za-z0-9_]+\s*\]")),
    ("legacy train_only_colmap default", re.compile(r"split_init_policy\s*=\s*[\"']train_only_colmap[\"']")),
    ("preset uses train_only_colmap", re.compile(r"args\.split_init_policy\s*=\s*[\"']train_only_colmap[\"']")),
]


STRICT_DANGEROUS_PATTERNS = [
    (
        "image directory list used as split source",
        re.compile(r"os\.listdir\s*\(\s*(?:[^)]*images[^)]*|images_dir|image_dir)\s*\)", re.IGNORECASE),
    ),
    (
        "full points3D copied into train split",
        re.compile(r"(?:copy|copyfile|copy2)\s*\([^)]*points3D\.(?:bin|ply)[^)]*train", re.IGNORECASE | re.DOTALL),
    ),
]


ACTIVE_SPLIT_LOGIC_FILES = [
    "utils/view_split_utils.py",
    "utils/auto_split_3dgs.py",
    "scene/dataset_readers.py",
    "scene/__init__.py",
    "train.py",
]


def _iter_python_files(repo):
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in DEFAULT_EXCLUDES]
        for filename in files:
            if filename.endswith(".py"):
                path = os.path.join(root, filename)
                if os.path.basename(path) == os.path.basename(__file__):
                    continue
                yield path


def _record_match(label, repo, path, text, match):
    return {
        "label": label,
        "file": os.path.relpath(path, repo).replace("\\", "/"),
        "line": text.count("\n", 0, match.start()) + 1,
        "snippet": " ".join(match.group(0).split()),
    }


def _strict_context_checks(repo):
    failures = []

    dataset_readers = os.path.join(repo, "scene", "dataset_readers.py")
    if os.path.isfile(dataset_readers):
        with open(dataset_readers, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        external_branch = "if use_auto_external_test:" in text
        external_branch_disables = (
            "use_auto_external_test = bool(eval and external_test_source_path)" in text
            and "[SPLIT-READER] internal LLFF/every-8 split disabled" in text
        )
        llff_branch_is_separate = "elif eval and not sparse_split_active and not full_test_split_active" in text
        if external_branch and not (external_branch_disables and llff_branch_is_separate):
            failures.append(
                {
                    "label": "external split may still use internal llffhold",
                    "file": "scene/dataset_readers.py",
                    "line": 1,
                    "snippet": "external_test_source_path branch did not prove LLFF split is disabled",
                }
            )

        hardcoded_reader = re.compile(r"\b(?:kitchen|279)\b|train_count\s*=\s*12")
        for match in hardcoded_reader.finditer(text):
            failures.append(_record_match("hardcoded scene/count in reader", repo, dataset_readers, text, match))

    return failures


def main():
    parser = argparse.ArgumentParser(description="Audit that legacy incorrect split logic is disabled.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    failures = []
    for path in _iter_python_files(repo):
        rel = os.path.relpath(path, repo)
        rel_norm = rel.replace("\\", "/")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        for label, pattern in DANGEROUS_PATTERNS:
            for match in pattern.finditer(text):
                failures.append(_record_match(label, repo, path, text, match))
        if args.strict:
            for label, pattern in STRICT_DANGEROUS_PATTERNS:
                for match in pattern.finditer(text):
                    if rel_norm not in ACTIVE_SPLIT_LOGIC_FILES:
                        continue
                    failures.append(_record_match(label, repo, path, text, match))

    if args.strict:
        failures.extend(_strict_context_checks(repo))

    if failures:
        for item in failures:
            print(
                "[LEGACY-SPLIT-AUDIT][FAIL] "
                f"{item['label']}: {item['file']}:{item['line']}: {item['snippet']}"
            )
        print(json.dumps({
            "status": "FAIL",
            "dangerous_patterns_found": failures,
            "deprecated_code_paths": [],
            "active_split_logic_files": ACTIVE_SPLIT_LOGIC_FILES,
        }, indent=2))
        return 1

    print("[LEGACY-SPLIT-AUDIT] status=PASS")
    print(json.dumps({
        "status": "PASS",
        "dangerous_patterns_found": [],
        "deprecated_code_paths": [],
        "active_split_logic_files": ACTIVE_SPLIT_LOGIC_FILES,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
