import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def main():
    results = []
    with open(f"results.csv") as f:
        reader = csv.DictReader(f)
        for line in reader:
            results.append({
                "x": int(line["NumEntryPoints"]),
                "y": int(line["NCandidates"]),
                "z": int(line["MaxIter"]),
                "time": float(line["time"]),
                "num_queries": float(line["num_queries"]),
                "score": float(line["nDCG@10"]),
            })

    # Derived metric
    for r in results:
        r["time_per_query"] = r["time"] / r["num_queries"]
        r["combined"] = r["score"] / r["time_per_query"]   # example metric


    plt.scatter(
        [r["time_per_query"] for r in results],
        [r["score"] for r in results],
    )

    plt.xlabel("time per query (lower is better)")
    plt.ylabel("score (higher is better)")
    plt.title("Pareto analysis: score vs. time_per_query")
    plt.show()




if __name__ == "__main__":
    main()
