import subprocess
import json
import os

# List of test files
test_files = [
    # "Ex1",
    # "Ex2",
    "Ex3",
]


def run_futhark_benchmarks(filename):
    # Ensure output directory exists
    os.makedirs("Test_Results", exist_ok=True)

    # Commands to run
    backends = ["c", "multicore", "opencl"]
    results = {}
    for backend in backends:
        output_file = f"Test_Results/{filename}_{backend}.json"
        cmd = [
            "futhark",
            "bench",
            f"{filename}.fut",
            f"--backend={backend}",
            f"--json={output_file}",
        ]
        try:
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            with open(output_file, "r") as f:
                results[backend] = json.load(f)
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark for backend '{backend}': {e}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            results[backend] = None

    return results


def reverse_benchmark_data(results):
    reversed_results = {}
    for backend, result in results.items():
        if result:
            reversed_results[backend] = {}
            for entry, entry_data in result.items():
                reversed_entry_data = {"datasets": {}}
                datasets = entry_data.get("datasets", {})

                # Reverse the order of the datasets
                for dataset_label, dataset_info in reversed(datasets.items()):
                    # Reverse the order of runtimes
                    reversed_entry_data["datasets"][dataset_label] = {
                        "bytes": dataset_info.get("bytes", {}),
                        "runtimes": dataset_info.get("runtimes", [])[
                            ::-1
                        ],  # Reverse runtimes list
                    }

                reversed_results[backend][entry] = reversed_entry_data
    return reversed_results


def main():
    for filename in test_files:
        results = run_futhark_benchmarks(filename)
        rev_results = reverse_benchmark_data(results)
        # Save reversed results for further processing
        with open(f"Test_Results/{filename}_reversed.json", "w") as f:
            json.dump(rev_results, f, indent=4)


if __name__ == "__main__":
    main()
