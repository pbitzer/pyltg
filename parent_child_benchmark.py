#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark different methods for handling parent-child relationships in lightning data.

This script tests various approaches to efficiently retrieve child data (groups, events)
for a given set of parent IDs (flashes, groups).

The current implementation uses list comprehension with filtering, which is O(n*m).
We test several alternatives to find the best performing solution.
"""

import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path
import tracemalloc

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

from pyltg.core.glm import GLM
from pyltg.core.lis import LIS


class ParentChildBenchmark:
    """Benchmark different parent-child lookup strategies."""

    def __init__(self, parents_df, children_df, parent_col='id', child_parent_col='parent_id'):
        """
        Initialize the benchmark.

        Parameters
        ----------
        parents_df : pd.DataFrame
            DataFrame containing parent records (e.g., flashes)
        children_df : pd.DataFrame
            DataFrame containing child records (e.g., groups)
        parent_col : str
            Column name for parent ID in parents_df
        child_parent_col : str
            Column name for parent ID reference in children_df
        """
        self.parents = parents_df
        self.children = children_df
        self.parent_col = parent_col
        self.child_parent_col = child_parent_col

        # For testing, we'll request children for a subset of parents
        n_test_parents = min(100, len(parents_df))
        self.test_parent_ids = parents_df[parent_col].iloc[:n_test_parents].values

    def method_1_current(self, parent_ids):
        """Current implementation: list comprehension with filtering."""
        parent_ids = np.atleast_1d(parent_ids)
        children_list = [
            self.children[self.children[self.child_parent_col] == _id]
            for _id in parent_ids
        ]
        return children_list

    def method_2_groupby(self, parent_ids):
        """Method 2: Pre-grouped lookup using pandas groupby."""
        parent_ids = np.atleast_1d(parent_ids)

        # Create grouped object (only once in real usage)
        grouped = self.children.groupby(self.child_parent_col)

        children_list = []
        for _id in parent_ids:
            try:
                children_list.append(grouped.get_group(_id))
            except KeyError:
                # Parent has no children
                children_list.append(pd.DataFrame())

        return children_list

    def method_3_dict_lookup(self, parent_ids):
        """Method 3: Dictionary-based lookup."""
        parent_ids = np.atleast_1d(parent_ids)

        # Build dictionary (only once in real usage)
        child_dict = {}
        for pid in self.children[self.child_parent_col].unique():
            child_dict[pid] = self.children[self.children[self.child_parent_col] == pid]

        children_list = []
        for _id in parent_ids:
            children_list.append(child_dict.get(_id, pd.DataFrame()))

        return children_list

    def method_4_merge(self, parent_ids):
        """Method 4: Merge-based approach."""
        parent_ids = np.atleast_1d(parent_ids)

        # Create a DataFrame with just the parent IDs we want
        requested = pd.DataFrame({self.parent_col: parent_ids})

        # Merge to get children
        merged = requested.merge(
            self.children,
            left_on=self.parent_col,
            right_on=self.child_parent_col,
            how='left'
        )

        # Group by parent ID to create list
        children_list = []
        for _id in parent_ids:
            subset = merged[merged[self.parent_col] == _id]
            # Drop the extra parent_col column
            subset = subset.drop(columns=[self.parent_col])
            children_list.append(subset)

        return children_list

    def method_5_categorical(self, parent_ids):
        """Method 5: Using categorical dtype for faster filtering."""
        parent_ids = np.atleast_1d(parent_ids)

        # Convert to categorical (only once in real usage)
        children_cat = self.children.copy()
        children_cat[self.child_parent_col] = children_cat[self.child_parent_col].astype('category')

        children_list = []
        for _id in parent_ids:
            children_list.append(children_cat[children_cat[self.child_parent_col] == _id])

        return children_list

    def method_6_isin_split(self, parent_ids):
        """Method 6: Filter all at once with isin, then split."""
        parent_ids = np.atleast_1d(parent_ids)

        # Get all children for requested parents at once
        all_children = self.children[self.children[self.child_parent_col].isin(parent_ids)]

        # Now split by parent
        children_list = []
        for _id in parent_ids:
            children_list.append(all_children[all_children[self.child_parent_col] == _id])

        return children_list

    def method_7_index_based(self, parent_ids):
        """Method 7: Set index on child_parent_col for faster lookup."""
        parent_ids = np.atleast_1d(parent_ids)

        # Set index (only once in real usage)
        children_indexed = self.children.set_index(self.child_parent_col)

        children_list = []
        for _id in parent_ids:
            try:
                # loc with index is much faster
                subset = children_indexed.loc[[_id]]
                subset = subset.reset_index()
                children_list.append(subset)
            except KeyError:
                children_list.append(pd.DataFrame())

        return children_list

    def method_8_numpy_searchsorted(self, parent_ids):
        """Method 8: NumPy searchsorted for sorted data."""
        parent_ids = np.atleast_1d(parent_ids)

        # Sort children by parent_id (only once in real usage)
        children_sorted = self.children.sort_values(self.child_parent_col)
        parent_array = children_sorted[self.child_parent_col].values

        children_list = []
        for _id in parent_ids:
            # Find indices where this parent_id appears
            left_idx = np.searchsorted(parent_array, _id, side='left')
            right_idx = np.searchsorted(parent_array, _id, side='right')

            if left_idx < right_idx:
                children_list.append(children_sorted.iloc[left_idx:right_idx])
            else:
                children_list.append(pd.DataFrame())

        return children_list

    def benchmark_method(self, method_func, method_name, n_runs=10):
        """
        Benchmark a single method.

        Parameters
        ----------
        method_func : callable
            The method function to test
        method_name : str
            Name of the method for reporting
        n_runs : int
            Number of times to run the test

        Returns
        -------
        dict
            Dictionary with timing and memory statistics
        """
        times = []

        # Warm-up run
        _ = method_func(self.test_parent_ids)

        # Timed runs
        for _ in range(n_runs):
            start = time.perf_counter()
            result = method_func(self.test_parent_ids)
            end = time.perf_counter()
            times.append(end - start)

        # Memory measurement
        tracemalloc.start()
        result = method_func(self.test_parent_ids)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Verify we got same number of results
        n_results = len(result)

        return {
            'method': method_name,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'peak_memory_mb': peak / 1024 / 1024,
            'n_results': n_results,
        }

    def run_all_benchmarks(self, n_runs=10):
        """Run all benchmarking methods."""
        methods = [
            (self.method_1_current, "1. Current (List Comprehension)"),
            (self.method_2_groupby, "2. GroupBy Pre-Index"),
            (self.method_3_dict_lookup, "3. Dictionary Lookup"),
            (self.method_4_merge, "4. Merge-Based"),
            (self.method_5_categorical, "5. Categorical Dtype"),
            (self.method_6_isin_split, "6. isin + Split"),
            (self.method_7_index_based, "7. Index-Based Lookup"),
            (self.method_8_numpy_searchsorted, "8. NumPy searchsorted"),
        ]

        results = []

        print(f"\nBenchmarking with {len(self.test_parent_ids)} parent IDs")
        print(f"Total parents: {len(self.parents)}, Total children: {len(self.children)}")
        print(f"Runs per method: {n_runs}\n")

        for method_func, method_name in methods:
            print(f"Testing {method_name}...", end='', flush=True)
            try:
                result = self.benchmark_method(method_func, method_name, n_runs)
                results.append(result)
                print(f" {result['mean_time']*1000:.2f} ms")
            except Exception as e:
                print(f" ERROR: {e}")
                traceback.print_exc()

        return pd.DataFrame(results)


def load_test_data():
    """Load actual test data files."""
    test_files_dir = Path(__file__).parent / 'pyltg' / 'examples' / 'test_files'

    # Try GLM first
    glm_files = list(test_files_dir.glob('OR_GLM*.nc'))
    if glm_files:
        print(f"Loading GLM data from {glm_files[0].name}...")
        glm = GLM(str(glm_files[0]))
        return glm.flashes._data, glm.groups._data, 'flash -> group'

    # Try LIS
    lis_files = list(test_files_dir.glob('ISS_LIS_SC*.nc'))
    if lis_files:
        print(f"Loading LIS data from {lis_files[0].name}...")
        lis = LIS(str(lis_files[0]))
        return lis.flashes._data, lis.groups._data, 'flash -> group'

    raise FileNotFoundError("No test data files found")


def create_synthetic_data(n_parents=1000, n_children=10000, children_per_parent_mean=10):
    """
    Create synthetic data for testing.

    Parameters
    ----------
    n_parents : int
        Number of parent records
    n_children : int
        Number of child records
    children_per_parent_mean : int
        Average number of children per parent
    """
    print(f"Creating synthetic data: {n_parents} parents, {n_children} children...")

    # Create parents
    parents = pd.DataFrame({
        'id': np.arange(n_parents),
        'lat': np.random.uniform(-90, 90, n_parents),
        'lon': np.random.uniform(-180, 180, n_parents),
        'energy': np.random.exponential(1e-14, n_parents),
        'time': pd.date_range('2020-01-01', periods=n_parents, freq='1s'),
        'active': True,
        'alt': 0.0,
    })

    # Create children with parent_id following a realistic distribution
    # Some parents have many children, some have few
    parent_ids = np.random.choice(n_parents, size=n_children, replace=True)

    children = pd.DataFrame({
        'id': np.arange(n_children),
        'parent_id': parent_ids,
        'lat': np.random.uniform(-90, 90, n_children),
        'lon': np.random.uniform(-180, 180, n_children),
        'energy': np.random.exponential(1e-15, n_children),
        'time': pd.date_range('2020-01-01', periods=n_children, freq='100ms'),
        'active': True,
        'alt': 0.0,
    })

    return parents, children, 'synthetic'


def main():
    """Main benchmarking function."""
    print("=" * 80)
    print("Parent-Child Relationship Benchmark")
    print("=" * 80)

    # Try to load real data, fall back to synthetic
    try:
        parents, children, data_type = load_test_data()
        print(f"Using real data: {data_type}")
    except Exception as e:
        print(f"Could not load real data: {e}")
        print("Falling back to synthetic data")
        parents, children, data_type = create_synthetic_data(
            n_parents=1000,
            n_children=10000
        )

    # Create benchmark instance
    benchmark = ParentChildBenchmark(parents, children)

    # Run benchmarks
    results = benchmark.run_all_benchmarks(n_runs=10)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(results.to_string(index=False))

    # Sort by time and show ranking
    results_sorted = results.sort_values('mean_time')
    print("\n" + "=" * 80)
    print("RANKING (by mean time)")
    print("=" * 80)

    baseline_time = results[results['method'].str.contains('Current')]['mean_time'].values[0]

    for idx, row in results_sorted.iterrows():
        speedup = baseline_time / row['mean_time']
        print(f"{row['method']:40s} "
              f"{row['mean_time']*1000:8.2f} ms "
              f"({speedup:5.1f}x speedup) "
              f"[{row['peak_memory_mb']:.2f} MB peak]")

    # Save results
    results.to_csv('parent_child_benchmark_results.csv', index=False)
    print("\nResults saved to parent_child_benchmark_results.csv")

    # Test with larger synthetic dataset
    print("\n" + "=" * 80)
    print("LARGE DATASET TEST (10x scale)")
    print("=" * 80)

    parents_large, children_large, _ = create_synthetic_data(
        n_parents=10000,
        n_children=100000
    )

    benchmark_large = ParentChildBenchmark(parents_large, children_large)
    results_large = benchmark_large.run_all_benchmarks(n_runs=5)

    print("\n" + "=" * 80)
    print("LARGE DATASET RESULTS")
    print("=" * 80)
    results_large_sorted = results_large.sort_values('mean_time')

    baseline_time_large = results_large[results_large['method'].str.contains('Current')]['mean_time'].values[0]

    for idx, row in results_large_sorted.iterrows():
        speedup = baseline_time_large / row['mean_time']
        print(f"{row['method']:40s} "
              f"{row['mean_time']*1000:8.2f} ms "
              f"({speedup:5.1f}x speedup) "
              f"[{row['peak_memory_mb']:.2f} MB peak]")

    results_large.to_csv('parent_child_benchmark_results_large.csv', index=False)


if __name__ == '__main__':
    main()
