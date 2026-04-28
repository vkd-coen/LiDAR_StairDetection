"""
testing/run_test.py
Structured test protocol for stair detection evaluation.
Records N scans at a fixed position and reports detection rate.

Usage:
    python testing/run_test.py eno2 --scans 20 --label "corridor_wall"

Results are saved to testing/results.csv for report compilation.
"""
import sys
import time
import argparse
import csv
import os
import numpy as np

sys.path.append('/home/user/Desktop/stair_detection')
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_ as UnitreeCloud
from inference.predict import StairDetector
from models.stair_svm import StairSVM
from preprocessing.feature_extractor import extract_features

CHECKPOINT = "/home/user/Desktop/stair_detection/checkpoints/best_model.pth"
SVM_PATH   = "/home/user/Desktop/stair_detection/checkpoints/stair_svm.pkl"
RESULTS_CSV = "/home/user/Desktop/stair_detection/testing/results.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Stair detection test protocol")
    parser.add_argument("interface", type=str,
                        help="Network interface (e.g. eno2)")
    parser.add_argument("--scans", type=int, default=20,
                        help="Number of scans to evaluate (default: 20)")
    parser.add_argument("--label", type=str, default="unnamed_test",
                        help="Label for this test (e.g. 'stairs_center_1m')")
    parser.add_argument("--expected", type=str, choices=["stairs", "no_stairs"],
                        default="stairs",
                        help="Expected outcome for accuracy calculation")
    return parser.parse_args()


def run_test(args):
    # --- Initialize models ---
    print("\nLoading models...")
    detector = StairDetector(checkpoint=CHECKPOINT)
    svm = StairSVM()
    svm.load(SVM_PATH)
    print("Models loaded.\n")

    # --- Initialize DDS ---
    ChannelFactoryInitialize(0, args.interface)
    sub = ChannelSubscriber("rt/utlidar/cloud", UnitreeCloud)
    sub.Init()

    print(f"Test label    : {args.label}")
    print(f"Expected      : {args.expected}")
    print(f"Scans to record: {args.scans}")
    print(f"\nKeep robot STILL. Starting in 3 seconds...")
    time.sleep(3)
    print("="*50)

    results = []
    svm_probs = []
    cnn_probs = []
    count = 0

    while count < args.scans:
        msg = sub.Read()
        if msg is None:
            time.sleep(0.01)
            continue

        # --- Parse point cloud ---
        raw = bytes(msg.data)
        dt = np.dtype([
            ('x', np.float32), ('y', np.float32),
            ('z', np.float32), ('intensity', np.float32)
        ])
        points = np.frombuffer(raw, dtype=dt)
        xyz = np.stack([points['x'], points['y'], points['z']], axis=-1)
        valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 0] < 250.0)
        xyz = xyz[valid]

        if len(xyz) < 50:
            continue

        # --- Stage 1: SVM ---
        features = extract_features(xyz)
        if features is None:
            detected = False
            svm_prob = 0.0
            cnn_prob = 0.0
        else:
            svm_result = svm.predict(features)
            svm_prob = svm_result['stairs_prob']

            if svm_prob < 0.7:
                detected = False
                cnn_prob = 0.0
            else:
                # --- Stage 2: CNN ---
                cnn_result = detector.predict(xyz)
                cnn_prob = cnn_result['stairs_prob']
                detected = (svm_prob > 0.7) and (cnn_prob > 0.8)

        results.append(detected)
        svm_probs.append(svm_prob)
        cnn_probs.append(cnn_prob)
        count += 1

        status = "STAIRS ✓" if detected else "no stairs ✗"
        print(f"  Scan {count:02d}/{args.scans} | "
              f"SVM: {svm_prob:.0%} | "
              f"CNN: {cnn_prob:.0%} | "
              f"{status}")

    sub.Close()

    # --- Compute metrics ---
    n = len(results)
    detections = sum(results)
    detection_rate = detections / n * 100

    if args.expected == "stairs":
        # True positives = correct detections
        tp = detections
        fn = n - detections
        fp = 0
        tn = 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        # True negatives = correct non-detections
        tn = n - detections
        fp = detections
        tp = 0
        fn = 0
        precision = 0.0
        recall    = tn / n

    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    avg_svm = np.mean(svm_probs)
    avg_cnn = np.mean(cnn_probs)

    # --- Print summary ---
    print("="*50)
    print(f"\nRESULTS — {args.label}")
    print(f"  Total scans      : {n}")
    print(f"  Detections       : {detections}")
    print(f"  Detection rate   : {detection_rate:.1f}%")
    print(f"  Avg SVM prob     : {avg_svm:.1%}")
    print(f"  Avg CNN prob     : {avg_cnn:.1%}")
    if args.expected == "stairs":
        print(f"  True positives   : {tp}")
        print(f"  False negatives  : {fn}")
        print(f"  Recall           : {recall:.1%}")
    else:
        print(f"  True negatives   : {tn}")
        print(f"  False positives  : {fp}")
        print(f"  Specificity      : {recall:.1%}")
    print(f"  F1 score         : {f1:.3f}")
    print("="*50)

    # --- Save to CSV ---
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    file_exists = os.path.isfile(RESULTS_CSV)

    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "label", "expected", "n_scans", "detections",
                "detection_rate", "avg_svm_prob", "avg_cnn_prob",
                "tp", "fn", "fp", "tn", "recall_or_specificity", "f1"
            ])
        writer.writerow([
            args.label, args.expected, n, detections,
            f"{detection_rate:.1f}",
            f"{avg_svm:.3f}", f"{avg_cnn:.3f}",
            tp, fn, fp, tn,
            f"{recall:.3f}", f"{f1:.3f}"
        ])

    print(f"\nResults saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    args = parse_args()
    run_test(args)
