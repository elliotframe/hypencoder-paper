import fire
import csv
import json
from pathlib import Path


def main(ret_name: str):
    results_path = Path(f"metrics/{ret_name}/results.csv")

    rows = []
    with results_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ["num_queries", "time"]

        for line in reader:
            w = line["Seed"]
            x = line["NumEntryPoints"]
            y = line["NCandidates"]
            z = line["MaxIter"]

            timing_path = Path(f"metrics/{ret_name}/entries/{w}-{x}-{y}-{z}/timing.json")
            with timing_path.open("r") as g:
                timing = json.load(g)

            line["num_queries"] = timing["num_queries"]
            line["time"] = timing["time"]
            rows.append(line)

    with results_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


    print("Done :)")

            
    




if __name__ == "__main__":
    fire.Fire(main)

    