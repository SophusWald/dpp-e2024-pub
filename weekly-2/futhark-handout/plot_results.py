import json
import matplotlib.pyplot as plt


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
    plt.savefig(f"tex/{filename}benchmark_results.png")
    plt.show()


def main():
    test_files = [
        "Ex3",
    ]
    for filename in test_files:
        # Load reversed results
        with open(f"Test_Results/{filename}_reversed.json", "r") as f:
            results = json.load(f)
        plot_data = extract_benchmark_data(results)
        plot_benchmark_results(plot_data, filename)


if __name__ == "__main__":
    main()
