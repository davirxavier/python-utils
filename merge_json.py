import json
import sys
from pathlib import Path

def deep_merge(a, b):
    """
    Recursively merge dict b into dict a.
    - If both values are dicts → merge them.
    - If both values are lists → concatenate (no dedup).
    - Otherwise → b overwrites a.
    """
    for key, val in b.items():
        if key in a:
            if isinstance(a[key], dict) and isinstance(val, dict):
                deep_merge(a[key], val)
            elif isinstance(a[key], list) and isinstance(val, list):
                a[key] = a[key] + val
            else:
                a[key] = val  # overwrite
        else:
            a[key] = val
    return a

def merge_json_files(file1, file2, output):
    with open(file1, "r", encoding="utf-8") as f1:
        data1 = json.load(f1)
    with open(file2, "r", encoding="utf-8") as f2:
        data2 = json.load(f2)

    merged = deep_merge(data1, data2)

    with open(output, "w", encoding="utf-8") as out:
        json.dump(merged, out, indent=4, ensure_ascii=False)

    print(f"Merged file saved to {output}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage:")
        print("  python merge_info_labels.py file1.json file2.json output.json")
        sys.exit(1)

    merge_json_files(sys.argv[1], sys.argv[2], sys.argv[3])
