"""Print every saved result (see `common.save_result`) against the @224 baseline.

Doesn't import `common`, so it runs instantly without a GPU or the dataset.
"""
import glob
import json
import os

RESULTS_DIR = "results"

results = {}
for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
    with open(path) as f:
        r = json.load(f)
    results[r["name"]] = r["acc"]

baseline = results.get("baseline_224")

print("=" * 64)
print("RESULTS SUMMARY")
print("=" * 64)
for name, acc in results.items():
    delta = "" if baseline is None else f"  ({(acc - baseline) * 100:+.2f} pp)"
    print(f"  {name:<36s}  {acc:.2%}{delta}")
