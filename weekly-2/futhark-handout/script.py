import subprocess
import json
import matplotlib.pyplot as plt
import os

# "Exercise1", "Exercise13",
test_files = [
    "Ex1",
    "Ex2",
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


def extract_benchmark_data(results):
    plot_data = {}
    for backend, result in results.items():
        if result:
            entry_points = list(
                result.keys()
            )  # Typically, only one entry point like "Exercise1.fut:test_process"
            avg_times = []
            dataset_labels = []

            for entry in entry_points:
                datasets = result[entry].get("datasets", {})
                for dataset_label, dataset_info in datasets.items():
                    runtimes = dataset_info.get("runtimes", [])
                    avg_time = sum(runtimes) / len(runtimes) if runtimes else 0
                    avg_times.append(avg_time)
                    dataset_labels.append(dataset_label)

            plot_data[backend] = {
                "dataset_labels": dataset_labels,
                "avg_times": avg_times,
            }

    return plot_data


def plot_benchmark_results(plot_data, filename):
    plt.figure(figsize=(10, 6))

    for backend, data in plot_data.items():
        plt.plot(
            data["dataset_labels"],
            data["avg_times"],
            marker="o",
            label=f"Backend: {backend}",
        )

    plt.xlabel("Dataset")
    plt.ylabel("Average Runtime (μs)")
    plt.title(f"Futhark Benchmark {filename}")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.yscale("log")
    plt.savefig(f"latex/{filename}benchmark_results.png")
    plt.show()


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
        plot_data = extract_benchmark_data(rev_results)
        plot_benchmark_results(plot_data, filename)


if __name__ == "__main__":
    main()
