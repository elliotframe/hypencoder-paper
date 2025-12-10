import csv
import matplotlib.pyplot as plt
import mplcursors

results = []
with open("results.csv") as f:
    r = csv.DictReader(f)
    for line in r:
        time = float(line["time"])
        q = float(line["num_queries"])
        tpq = time/q
        results.append({
            "x": int(line["NumEntryPoints"]),
            "y": int(line["NCandidates"]),
            "z": int(line["MaxIter"]),
            "time_per_query": tpq,
            "score": float(line["nDCG@10"]),
        })

fig, ax = plt.subplots()
scatter = ax.scatter(
    [r["time_per_query"] for r in results],
    [r["score"] for r in results],
)

ax.set_xlabel("time_per_query")
ax.set_ylabel("score")
ax.set_title("Pareto plot (hover labels via mplcursors)")

cursor = mplcursors.cursor(scatter, hover=True)
@cursor.connect("add")
def on_add(sel):
    idx = sel.index
    r = results[idx]
    sel.annotation.set(text=f"x={r['x']}, y={r['y']}, z={r['z']}\n"
                             f"time/query={r['time_per_query']:.4f}\n"
                             f"score={r['score']:.4f}")

plt.show()