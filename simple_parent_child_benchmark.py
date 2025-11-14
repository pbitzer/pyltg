#!/usr/bin/env python3
"""
Simple benchmark for parent-child relationship methods.
Standalone version that doesn't require package installation.
"""

import numpy as np
import pandas as pd
import time


def create_synthetic_data(n_parents=1000, n_children=10000):
    """Create synthetic lightning data."""
    print(f"Creating synthetic data: {n_parents} parents, {n_children} children...")

    # Create parents (flashes)
    parents = pd.DataFrame({
        'id': np.arange(n_parents),
        'lat': np.random.uniform(-90, 90, n_parents),
        'lon': np.random.uniform(-180, 180, n_parents),
    })

    # Create children (groups) with realistic parent_id distribution
    parent_ids = np.random.choice(n_parents, size=n_children, replace=True)

    children = pd.DataFrame({
        'id': np.arange(n_children),
        'parent_id': parent_ids,
        'lat': np.random.uniform(-90, 90, n_children),
        'lon': np.random.uniform(-180, 180, n_children),
    })

    return parents, children


def method_1_current(children_df, parent_ids):
    """Current implementation: list comprehension with filtering."""
    parent_ids = np.atleast_1d(parent_ids)
    children_list = [
        children_df[children_df.parent_id == _id]
        for _id in parent_ids
    ]
    return children_list


def method_2_groupby(children_df, parent_ids):
    """Method 2: Pre-grouped lookup using pandas groupby."""
    parent_ids = np.atleast_1d(parent_ids)
    grouped = children_df.groupby('parent_id')

    children_list = []
    for _id in parent_ids:
        try:
            children_list.append(grouped.get_group(_id))
        except KeyError:
            children_list.append(pd.DataFrame())
    return children_list


def method_3_dict_lookup(children_df, parent_ids):
    """Method 3: Dictionary-based lookup."""
    parent_ids = np.atleast_1d(parent_ids)

    # Build dictionary
    child_dict = {
        pid: children_df[children_df.parent_id == pid]
        for pid in children_df.parent_id.unique()
    }

    children_list = [child_dict.get(_id, pd.DataFrame()) for _id in parent_ids]
    return children_list


def method_4_isin_split(children_df, parent_ids):
    """Method 4: Filter all at once with isin, then split."""
    parent_ids = np.atleast_1d(parent_ids)

    # Get all children for requested parents at once
    all_children = children_df[children_df.parent_id.isin(parent_ids)]

    # Split by parent
    children_list = [
        all_children[all_children.parent_id == _id]
        for _id in parent_ids
    ]
    return children_list


def method_5_index_based(children_df, parent_ids):
    """Method 5: Set index on parent_id for faster lookup."""
    parent_ids = np.atleast_1d(parent_ids)

    # Set index
    children_indexed = children_df.set_index('parent_id')

    children_list = []
    for _id in parent_ids:
        try:
            subset = children_indexed.loc[[_id]].reset_index()
            children_list.append(subset)
        except KeyError:
            children_list.append(pd.DataFrame())
    return children_list


def method_6_numpy_searchsorted(children_df, parent_ids):
    """Method 6: NumPy searchsorted for sorted data."""
    parent_ids = np.atleast_1d(parent_ids)

    # Sort children by parent_id
    children_sorted = children_df.sort_values('parent_id')
    parent_array = children_sorted.parent_id.values

    children_list = []
    for _id in parent_ids:
        left_idx = np.searchsorted(parent_array, _id, side='left')
        right_idx = np.searchsorted(parent_array, _id, side='right')

        if left_idx < right_idx:
            children_list.append(children_sorted.iloc[left_idx:right_idx])
        else:
            children_list.append(pd.DataFrame())
    return children_list


def method_7_merge(children_df, parent_ids):
    """Method 7: Merge-based approach."""
    parent_ids = np.atleast_1d(parent_ids)

    # Create DataFrame with requested parent IDs
    requested = pd.DataFrame({'req_id': parent_ids})

    # Merge
    merged = requested.merge(
        children_df,
        left_on='req_id',
        right_on='parent_id',
        how='left'
    )

    # Group by parent ID
    children_list = [
        merged[merged.req_id == _id].drop(columns=['req_id'])
        for _id in parent_ids
    ]
    return children_list


def method_8_categorical(children_df, parent_ids):
    """Method 8: Using categorical dtype for faster filtering."""
    parent_ids = np.atleast_1d(parent_ids)

    # Convert to categorical
    children_cat = children_df.copy()
    children_cat['parent_id'] = children_cat['parent_id'].astype('category')

    children_list = [
        children_cat[children_cat.parent_id == _id]
        for _id in parent_ids
    ]
    return children_list


def benchmark_method(method_func, method_name, children_df, test_parent_ids, n_runs=10):
    """Benchmark a single method."""
    times = []

    # Warm-up
    _ = method_func(children_df, test_parent_ids)

    # Timed runs
    for _ in range(n_runs):
        start = time.perf_counter()
        result = method_func(children_df, test_parent_ids)
        end = time.perf_counter()
        times.append(end - start)

    return {
        'method': method_name,
        'mean_time_ms': np.mean(times) * 1000,
        'std_time_ms': np.std(times) * 1000,
        'min_time_ms': np.min(times) * 1000,
        'max_time_ms': np.max(times) * 1000,
    }


def run_benchmarks(parents, children, n_test_parents=100, n_runs=10):
    """Run all benchmarks."""
    test_parent_ids = parents.id.iloc[:n_test_parents].values

    methods = [
        (method_1_current, "1. Current (List Comprehension)"),
        (method_2_groupby, "2. GroupBy Pre-Index"),
        (method_3_dict_lookup, "3. Dictionary Lookup"),
        (method_4_isin_split, "4. isin + Split"),
        (method_5_index_based, "5. Index-Based Lookup"),
        (method_6_numpy_searchsorted, "6. NumPy searchsorted"),
        (method_7_merge, "7. Merge-Based"),
        (method_8_categorical, "8. Categorical Dtype"),
    ]

    print(f"\nTesting with {n_test_parents} parent IDs out of {len(parents)} total")
    print(f"Total children: {len(children)}, Runs per method: {n_runs}\n")

    results = []
    for method_func, method_name in methods:
        print(f"Testing {method_name}...", end='', flush=True)
        try:
            result = benchmark_method(method_func, method_name, children, test_parent_ids, n_runs)
            results.append(result)
            print(f" {result['mean_time_ms']:.2f} ms")
        except Exception as e:
            print(f" ERROR: {e}")

    return pd.DataFrame(results)


def main():
    """Main function."""
    print("=" * 80)
    print("Parent-Child Relationship Benchmark (Standalone)")
    print("=" * 80)

    # Small dataset test
    print("\n### SMALL DATASET (1K parents, 10K children) ###")
    parents_small, children_small = create_synthetic_data(1000, 10000)
    results_small = run_benchmarks(parents_small, children_small, n_test_parents=100, n_runs=10)

    print("\n" + "=" * 80)
    print("RESULTS - SMALL DATASET")
    print("=" * 80)

    results_sorted = results_small.sort_values('mean_time_ms')
    baseline = results_small[results_small.method.str.contains('Current')]['mean_time_ms'].values[0]

    for _, row in results_sorted.iterrows():
        speedup = baseline / row['mean_time_ms']
        print(f"{row['method']:40s} {row['mean_time_ms']:8.2f} ms ({speedup:5.1f}x)")

    # Medium dataset test
    print("\n### MEDIUM DATASET (5K parents, 50K children) ###")
    parents_med, children_med = create_synthetic_data(5000, 50000)
    results_med = run_benchmarks(parents_med, children_med, n_test_parents=100, n_runs=5)

    print("\n" + "=" * 80)
    print("RESULTS - MEDIUM DATASET")
    print("=" * 80)

    results_sorted = results_med.sort_values('mean_time_ms')
    baseline = results_med[results_med.method.str.contains('Current')]['mean_time_ms'].values[0]

    for _, row in results_sorted.iterrows():
        speedup = baseline / row['mean_time_ms']
        print(f"{row['method']:40s} {row['mean_time_ms']:8.2f} ms ({speedup:5.1f}x)")

    # Large dataset test
    print("\n### LARGE DATASET (10K parents, 100K children) ###")
    parents_large, children_large = create_synthetic_data(10000, 100000)
    results_large = run_benchmarks(parents_large, children_large, n_test_parents=100, n_runs=3)

    print("\n" + "=" * 80)
    print("RESULTS - LARGE DATASET")
    print("=" * 80)

    results_sorted = results_large.sort_values('mean_time_ms')
    baseline = results_large[results_large.method.str.contains('Current')]['mean_time_ms'].values[0]

    for _, row in results_sorted.iterrows():
        speedup = baseline / row['mean_time_ms']
        print(f"{row['method']:40s} {row['mean_time_ms']:8.2f} ms ({speedup:5.1f}x)")

    # Save results
    print("\n" + "=" * 80)
    results_small.to_csv('benchmark_small.csv', index=False)
    results_med.to_csv('benchmark_medium.csv', index=False)
    results_large.to_csv('benchmark_large.csv', index=False)
    print("Results saved to benchmark_*.csv files")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nTop 3 methods (Large dataset):")
    for i, (_, row) in enumerate(results_sorted.head(3).iterrows(), 1):
        speedup = baseline / row['mean_time_ms']
        print(f"{i}. {row['method']:40s} {speedup:5.1f}x faster")


if __name__ == '__main__':
    # Check if required packages exist
    try:
        import numpy
        import pandas
        main()
    except ImportError as e:
        print(f"Error: Required package not found: {e}")
        print("Please install numpy and pandas")
