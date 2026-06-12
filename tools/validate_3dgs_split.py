import argparse
import json
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.auto_split_3dgs import validate_split_report_file


def main():
    parser = argparse.ArgumentParser(description="Validate an auto 3DGS split report.")
    parser.add_argument("--report", required=True, help="Path to reports/split_report.json")
    parser.add_argument("--allow_overlap", action="store_true", default=False)
    args = parser.parse_args()

    validation = validate_split_report_file(
        args.report,
        strict_no_overlap=not args.allow_overlap,
    )

    with open(args.report, "r", encoding="utf-8") as f:
        report = json.load(f)

    print(f"[VALIDATE_3DGS_SPLIT] status={validation['status']}")
    print(f"[VALIDATE_3DGS_SPLIT] report={args.report}")
    print(f"[VALIDATE_3DGS_SPLIT] train_count={report.get('train_count')}")
    print(f"[VALIDATE_3DGS_SPLIT] test_count={report.get('test_count')}")
    print(f"[VALIDATE_3DGS_SPLIT] overlap_count={report.get('overlap_count')}")
    print(f"[VALIDATE_3DGS_SPLIT] duplicate_test_count={report.get('duplicate_test_count')}")

    if validation["status"] != "PASS":
        print("[VALIDATE_3DGS_SPLIT] failed_checks=" + ", ".join(validation.get("failed_checks", [])))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
