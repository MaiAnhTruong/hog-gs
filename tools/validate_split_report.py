import argparse
import json
import sys


def _failures(report, args):
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    status = report.get("status")
    train_count = report.get("train_count")
    effective_hold = report.get("effective_hold")
    full_camera_count = report.get("full_camera_count")

    check(status == "PASS", f"status must be PASS, got {status!r}")
    check(train_count == args.expected_train_count, f"train_count must be {args.expected_train_count}, got {train_count!r}")
    check(effective_hold == args.expected_hold, f"effective_hold must be {args.expected_hold}, got {effective_hold!r}")

    if args.expected_full_camera_count is not None:
        check(
            full_camera_count == args.expected_full_camera_count,
            f"full_camera_count must be {args.expected_full_camera_count}, got {full_camera_count!r}",
        )

    if full_camera_count is not None and effective_hold:
        expected_raw = len(range(0, int(full_camera_count), int(effective_hold)))
        check(
            report.get("raw_candidate_count") == expected_raw,
            f"raw_candidate_count must be {expected_raw}, got {report.get('raw_candidate_count')!r}",
        )
    else:
        check(False, "full_camera_count and effective_hold are required to validate raw_candidate_count")

    if args.require_full_eval:
        check(report.get("mode") == "full_eval_external_test", "mode must be full_eval_external_test")
        check(bool(report.get("full_eval_path")), "full_eval_path is required")

    check(report.get("test_count", 0) > 0, f"test_count must be > 0, got {report.get('test_count')!r}")
    check(report.get("final_overlap_count") == 0, f"final_overlap_count must be 0, got {report.get('final_overlap_count')!r}")
    check(report.get("duplicate_test_count") == 0, f"duplicate_test_count must be 0, got {report.get('duplicate_test_count')!r}")
    check(
        report.get("full_scene_point_cloud_used_for_training") is False,
        "full_scene_point_cloud_used_for_training must be false",
    )
    check(report.get("point_cloud_source") == "source_train_sparse", "point_cloud_source must be source_train_sparse")
    check(
        report.get("nerf_normalization_source") == "train_cameras_only",
        "nerf_normalization_source must be train_cameras_only",
    )
    check(
        report.get("pose_frame_audit", {}).get("status") == "PASS",
        f"pose_frame_audit.status must be PASS, got {report.get('pose_frame_audit', {}).get('status')!r}",
    )

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate baseline full-eval split_report.json")
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected_train_count", type=int, required=True)
    parser.add_argument("--expected_hold", type=int, required=True)
    parser.add_argument("--expected_full_camera_count", type=int, default=None)
    parser.add_argument("--require_full_eval", action="store_true", default=False)
    args = parser.parse_args()

    with open(args.report, "r", encoding="utf-8") as f:
        report = json.load(f)

    errors = _failures(report, args)
    if errors:
        print("[VALIDATE_SPLIT_REPORT] FAIL")
        for error in errors:
            print(f" - {error}")
        return 1

    print("[VALIDATE_SPLIT_REPORT] PASS")
    print(f"report={args.report}")
    print(f"train_count={report.get('train_count')} test_count={report.get('test_count')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
