"""
numpy_explorer.py
Project: NumPy Explorer
Author: Sumit Gupta
Purpose: Demonstrates NumPy fundamentals (creation, indexing, ops, reshape/broadcasting),
         saving/loading arrays, and compares NumPy performance vs Python lists.
Usage: python numpy_explorer.py
Optional: python numpy_explorer.py --plot
"""

import numpy as np
import time
import argparse
import csv
import os
import sys

def create_arrays():
    """Create a variety of NumPy arrays demonstrating common constructors."""
    a = np.array([5, 10, 15, 20, 25], dtype=np.int64)
    b = np.arange(12)               # 0..11
    c = b.reshape(3, 4)             # 3x4 matrix
    zeros = np.zeros((2, 3))
    ones = np.ones((4,))
    rand = np.random.default_rng(seed=42).integers(1, 101, size=(3, 3))
    return {"a": a, "b": b, "c": c, "zeros": zeros, "ones": ones, "rand": rand}

def index_and_slice_demo(arrs):
    """Show indexing and slicing examples and return outputs to include in report."""
    a = arrs["a"]
    c = arrs["c"]
    examples = {}
    examples["a_third"] = a[2]             # single element
    examples["a_slice"] = a[1:4]           # slice of 1D array
    examples["c_row1"] = c[0, :]           # first row
    examples["c_col2"] = c[:, 2]           # third column
    examples["c_subblock"] = c[1:3, 1:4]   # block slice
    return examples

def math_stats_demo(arrs):
    """Do arithmetic, axis-wise reductions and basic statistics."""
    a = arrs["a"]
    c = arrs["c"]
    math_results = {}
    math_results["a_plus_7"] = a + 7
    math_results["a_times_3"] = a * 3
    math_results["sqrt_a"] = np.sqrt(a)
    math_results["c_sum_axis0"] = np.sum(c, axis=0).tolist()
    math_results["c_mean"] = float(np.mean(c))
    math_results["c_max_axis1"] = np.max(c, axis=1).tolist()
    math_results["c_std"] = float(np.std(c))
    return math_results

def reshape_and_broadcast_demo():
    """Example that uses reshape and broadcasting with explicit shapes."""
    base = np.linspace(0, 11, 12, dtype=np.int64)   # 0..11
    mat = base.reshape(4, 3)                        # 4x3
    row_to_add = np.array([1, 2, 3])
    added = mat + row_to_add                        # broadcast across rows
    col_to_mul = np.array([1, 10, 100, 1000]).reshape(4, 1)
    scaled = mat * col_to_mul                       # broadcast across columns
    return {"mat": mat, "added": added, "scaled": scaled}

def save_and_load_demo(arrs, folder="saved_arrays"):
    """Save multiple arrays in a single .npz file and load them back."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "explorer_data.npz")
    # Save multiple named arrays
    np.savez(path, **arrs)
    # Load
    loaded = np.load(path)
    loaded_dict = {k: loaded[k] for k in loaded.files}
    return path, loaded_dict

def perf_compare(size=1_000_000):
    """Compare common operations' run time between Python lists and NumPy arrays."""
    rng = list(range(size))
    np_arr = np.arange(size)

    results = {}

    # sum
    t0 = time.perf_counter()
    s_py = sum(rng)
    t1 = time.perf_counter()
    s_np = np.sum(np_arr)
    t2 = time.perf_counter()
    results["sum_list_s"] = t1 - t0
    results["sum_numpy_s"] = t2 - t1

    # elementwise multiply by 2
    t0 = time.perf_counter()
    py_mul = [x * 2 for x in rng]
    t1 = time.perf_counter()
    np_mul = np_arr * 2
    t2 = time.perf_counter()
    results["mul_list_s"] = t1 - t0
    results["mul_numpy_s"] = t2 - t1

    # mean
    t0 = time.perf_counter()
    mean_py = sum(rng) / len(rng)
    t1 = time.perf_counter()
    mean_np = np.mean(np_arr)
    t2 = time.perf_counter()
    results["mean_list_s"] = t1 - t0
    results["mean_numpy_s"] = t2 - t1

    # sanity values (not necessary but useful)
    results["sum_vals"] = (s_py, int(s_np))
    results["mean_vals"] = (mean_py, float(mean_np))
    return results

def csv_demo_write_read(path="sample_dataset.csv", rows=100):
    """Create a small synthetic CSV (like features) and read it using NumPy."""
    header = ["id", "feature_1", "feature_2", "label"]
    rng = np.random.default_rng(seed=123)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(rows):
            f1 = float(rng.normal(loc=50, scale=15))
            f2 = float(rng.uniform(0, 1))
            label = int(f1 + f2 > 50)
            writer.writerow([i, round(f1, 3), round(f2, 4), label])
    # load with numpy (structured)
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    return path, data

def generate_report(arrs, index_examples, math_examples, reshape_examples, save_path, perf, csv_info):
    """Print a tidy summary report of what we did (you can paste into your submission)."""
    print("\n" + "="*60)
    print("NUMPY EXPLORER - EXECUTION REPORT")
    print("="*60)
    print("\nArrays created:")
    for k, v in arrs.items():
        print(f"  - {k}: shape={getattr(v, 'shape', None)}, dtype={getattr(v, 'dtype', None)}")

    print("\nIndexing & slicing examples (showing shapes or values):")
    for k, v in index_examples.items():
        print(f"  - {k}: {v}")

    print("\nMath & stats examples:")
    for k, v in math_examples.items():
        print(f"  - {k}: {v}")

    print("\nReshape & broadcasting snapshots:")
    for k, v in reshape_examples.items():
        print(f"  - {k}: shape={v.shape}")

    print(f"\nSaved arrays to: {save_path}")
    print("\nPerformance comparison (seconds):")
    print("  Operation    | Python list  | NumPy array")
    print("  ----------------------------------------")
    print(f"  sum          | {perf['sum_list_s']:.6f}   | {perf['sum_numpy_s']:.6f}")
    print(f"  multiply     | {perf['mul_list_s']:.6f}   | {perf['mul_numpy_s']:.6f}")
    print(f"  mean         | {perf['mean_list_s']:.6f}   | {perf['mean_numpy_s']:.6f}")

    print(f"\nCSV dataset created at: {csv_info[0]}, loaded rows: {len(csv_info[1])}")
    print("\nReport End.")
    print("="*60 + "\n")

def optional_plot(arrs):
    """Optional: simple visualization (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("Matplotlib not installed; skipping plot. Install with `pip install matplotlib` if you want plots.")
        return

    a = arrs["a"]
    plt.figure(figsize=(6,4))
    plt.plot(a, np.sqrt(a), marker="o")
    plt.title("Square root transformation (example)")
    plt.xlabel("Original value")
    plt.ylabel("Sqrt(value)")
    plt.grid(True)
    plt.tight_layout()
    out = "plot_sqrt_example.png"
    plt.savefig(out)
    print(f"Plot saved to {out}")

def main(do_plot=False):
    arrs = create_arrays()
    idx = index_and_slice_demo(arrs)
    math = math_stats_demo(arrs)
    reshape = reshape_and_broadcast_demo()
    save_path, loaded = save_and_load_demo(arrs)
    perf = perf_compare(size=500_000)   # reduced size so it finishes quickly on small VMs
    csv_path, csv_loaded = csv_demo_write_read(rows=200)
    generate_report(arrs, idx, math, reshape, save_path, perf, (csv_path, csv_loaded))
    if do_plot:
        optional_plot(arrs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NumPy Explorer - demo script")
    parser.add_argument("--plot", action="store_true", help="Generate and save an example plot")
    parser.add_argument("--size", type=int, default=500_000, help="Size for performance tests (default 500k)")
    args = parser.parse_args()
    # pass size override to perf_compare if provided
    # (simpler approach: call main and then run perf_compare separately; keeping main straightforward)
    main(do_plot=args.plot)
