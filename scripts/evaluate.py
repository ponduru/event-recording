"""Evaluate model detection performance against ground truth labels."""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import click


@dataclass
class EvalMetrics:
    """Evaluation metrics for event detection."""

    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


def load_ground_truth(labels_path: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]], float]:
    """Load ground truth labels and return (positives, false_positives, fps)."""
    with open(labels_path) as f:
        data = json.load(f)

    fps = data["fps"]

    # Convert frame ranges to time ranges
    positives = []
    for d in data.get("deliveries", []):
        start_sec = d["start_frame"] / fps
        end_sec = d["end_frame"] / fps
        positives.append((start_sec, end_sec))

    # Also load marked false positives (these should NOT be detected)
    false_positives = []
    for fp in data.get("false_positives", []):
        start_sec = fp["start_frame"] / fps
        end_sec = fp["end_frame"] / fps
        false_positives.append((start_sec, end_sec))

    return positives, false_positives, fps


def load_detections(detections_path: Path) -> list[tuple[float, float, float]]:
    """Load detection results and return list of (start_time, end_time, confidence)."""
    with open(detections_path) as f:
        data = json.load(f)

    detections = []
    for event in data.get("events", []):
        detections.append((
            event["start_time"],
            event["end_time"],
            event["confidence"]
        ))

    return detections


def compute_overlap(range1: tuple[float, float], range2: tuple[float, float]) -> float:
    """Compute IoU (Intersection over Union) between two time ranges."""
    start1, end1 = range1
    start2, end2 = range2

    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)

    if intersection_start >= intersection_end:
        return 0.0

    intersection = intersection_end - intersection_start
    union = (end1 - start1) + (end2 - start2) - intersection

    return intersection / union if union > 0 else 0.0


def evaluate(
    ground_truth: list[tuple[float, float]],
    marked_fps: list[tuple[float, float]],
    detections: list[tuple[float, float, float]],
    iou_threshold: float = 0.1,
    time_tolerance: float = 5.0,
) -> tuple[EvalMetrics, dict]:
    """
    Evaluate detections against ground truth.

    Args:
        ground_truth: List of (start, end) time ranges for true events
        marked_fps: List of (start, end) time ranges for marked false positives
        detections: List of (start, end, confidence) for detected events
        iou_threshold: Minimum IoU to consider a match
        time_tolerance: Maximum time difference (seconds) for center-based matching

    Returns:
        EvalMetrics and detailed results dict
    """
    # Track which ground truth events are matched
    gt_matched = [False] * len(ground_truth)

    # Track detection results
    det_results = []

    for det_start, det_end, confidence in detections:
        det_center = (det_start + det_end) / 2
        det_range = (det_start, det_end)

        # Check if this detection matches any ground truth
        best_match_idx = None
        best_iou = 0.0

        for i, (gt_start, gt_end) in enumerate(ground_truth):
            gt_center = (gt_start + gt_end) / 2

            # Check center-based tolerance OR IoU threshold
            center_dist = abs(det_center - gt_center)
            iou = compute_overlap(det_range, (gt_start, gt_end))

            if center_dist <= time_tolerance or iou >= iou_threshold:
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = i

        # Check if detection matches a marked false positive
        is_marked_fp = False
        for fp_start, fp_end in marked_fps:
            fp_center = (fp_start + fp_end) / 2
            center_dist = abs(det_center - fp_center)
            iou = compute_overlap(det_range, (fp_start, fp_end))
            if center_dist <= time_tolerance or iou >= iou_threshold:
                is_marked_fp = True
                break

        if best_match_idx is not None and not gt_matched[best_match_idx]:
            # True positive
            gt_matched[best_match_idx] = True
            det_results.append({
                "start": det_start,
                "end": det_end,
                "confidence": confidence,
                "result": "TP",
                "matched_gt": best_match_idx,
                "iou": best_iou,
            })
        elif is_marked_fp:
            # Detected a known false positive - count as FP
            det_results.append({
                "start": det_start,
                "end": det_end,
                "confidence": confidence,
                "result": "FP_KNOWN",
                "matched_gt": None,
            })
        else:
            # False positive (or potential new discovery)
            det_results.append({
                "start": det_start,
                "end": det_end,
                "confidence": confidence,
                "result": "FP",
                "matched_gt": None,
            })

    # Count metrics
    true_positives = sum(1 for d in det_results if d["result"] == "TP")
    false_positives = sum(1 for d in det_results if d["result"] in ("FP", "FP_KNOWN"))
    false_negatives = sum(1 for matched in gt_matched if not matched)

    metrics = EvalMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )

    # Detailed results
    details = {
        "detection_results": det_results,
        "missed_ground_truth": [
            {"index": i, "start": ground_truth[i][0], "end": ground_truth[i][1]}
            for i, matched in enumerate(gt_matched) if not matched
        ],
        "known_fps_detected": sum(1 for d in det_results if d["result"] == "FP_KNOWN"),
    }

    return metrics, details


@click.command()
@click.argument("detections", type=click.Path(exists=True))
@click.option(
    "-l", "--labels",
    type=click.Path(exists=True),
    help="Path to ground truth labels JSON",
)
@click.option(
    "--iou-threshold",
    type=float,
    default=0.1,
    help="Minimum IoU for matching (default: 0.1)",
)
@click.option(
    "--time-tolerance",
    type=float,
    default=5.0,
    help="Time tolerance in seconds for center matching (default: 5.0)",
)
@click.option(
    "-o", "--output",
    type=click.Path(),
    help="Save detailed results to JSON file",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Show detailed results",
)
def main(
    detections: str,
    labels: Optional[str],
    iou_threshold: float,
    time_tolerance: float,
    output: Optional[str],
    verbose: bool,
):
    """
    Evaluate detection results against ground truth labels.

    DETECTIONS is the path to the detection results JSON file.

    Example usage:

        prismata-eval results/detections.json -l data/labels/video.json

        prismata-eval results/detections.json -l data/labels/video.json -v
    """
    detections_path = Path(detections)

    # Auto-find labels if not specified
    if labels is None:
        # Try to find matching labels file
        det_data = json.loads(detections_path.read_text())
        video_path = Path(det_data.get("video_path", ""))
        labels_path = Path("data/labels") / f"{video_path.stem}.json"
        if not labels_path.exists():
            raise click.ClickException(
                f"Could not find labels file. Specify with --labels. "
                f"Tried: {labels_path}"
            )
    else:
        labels_path = Path(labels)

    # Load data
    ground_truth, marked_fps, fps = load_ground_truth(labels_path)
    det_list = load_detections(detections_path)

    click.echo(f"Ground truth: {len(ground_truth)} events")
    click.echo(f"Marked false positives: {len(marked_fps)}")
    click.echo(f"Detections: {len(det_list)} events")
    click.echo()

    # Evaluate
    metrics, details = evaluate(
        ground_truth,
        marked_fps,
        det_list,
        iou_threshold=iou_threshold,
        time_tolerance=time_tolerance,
    )

    # Print results
    click.echo("=" * 50)
    click.echo("EVALUATION RESULTS")
    click.echo("=" * 50)
    click.echo()
    click.echo(f"  True Positives:  {metrics.true_positives}")
    click.echo(f"  False Positives: {metrics.false_positives}")
    click.echo(f"  False Negatives: {metrics.false_negatives}")
    click.echo()
    click.echo(f"  Precision: {metrics.precision:.3f}")
    click.echo(f"  Recall:    {metrics.recall:.3f}")
    click.echo(f"  F1 Score:  {metrics.f1:.3f}")
    click.echo()

    if details["known_fps_detected"] > 0:
        click.echo(f"  Known FPs still detected: {details['known_fps_detected']}")
        click.echo()

    if verbose and details["missed_ground_truth"]:
        click.echo("Missed ground truth events:")
        for item in details["missed_ground_truth"]:
            click.echo(f"  #{item['index']+1}: {item['start']:.1f}s - {item['end']:.1f}s")
        click.echo()

    if verbose:
        # Show false positives
        fps_list = [d for d in details["detection_results"] if d["result"] in ("FP", "FP_KNOWN")]
        if fps_list:
            click.echo("False positive detections:")
            for d in fps_list:
                label = "(known)" if d["result"] == "FP_KNOWN" else "(new)"
                click.echo(f"  {d['start']:.1f}s - {d['end']:.1f}s {label} conf={d['confidence']:.2f}")
            click.echo()

    # Save detailed results
    if output:
        output_path = Path(output)
        results = {
            "labels_file": str(labels_path),
            "detections_file": str(detections_path),
            "settings": {
                "iou_threshold": iou_threshold,
                "time_tolerance": time_tolerance,
            },
            "metrics": {
                "true_positives": metrics.true_positives,
                "false_positives": metrics.false_positives,
                "false_negatives": metrics.false_negatives,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
            },
            "details": details,
        }
        output_path.write_text(json.dumps(results, indent=2))
        click.echo(f"Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
