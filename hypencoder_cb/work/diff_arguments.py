from hypencoder_cb.inference.approx_retrieve import do_retrieval
from typing import Dict, List, Optional, Union
from itertools import product
import json
import csv
import fire

def main(
    model_name_or_path: str,
    encoded_item_path: str,
    item_neighbors_path: str,
    output_dir: str,
    ir_dataset_name: Optional[str] = None,
    dtype: str = "fp16",
    graph: Optional[str] = None,
) -> None:

    if not graph:
        graph=item_neighbors_path.split("/")[-1]
    cache_file=f"cache/{graph}"
    ret_name=output_dir.split("/")[-1]

    nep = [5_000, 10_000, 50_000, 100_000, 500_000]
    nc = [24, 64, 150, 328, 600]
    mi = [6, 12, 16, 20, 24]


    for x, y, z in product(nep, nc, mi):
        num_entry_points = x
        ncandidates = y
        max_iter = z
        metric_dir=f"metrics/{ret_name}/entries/{num_entry_points}-{ncandidates}-{max_iter}"

        print(f"Starting retrieval: num_entry_points={num_entry_points}, ncandidates={ncandidates}, max_iter={max_iter}")
        
        do_retrieval(
            model_name_or_path=model_name_or_path,
            encoded_item_path=encoded_item_path,
            item_neighbors_path=item_neighbors_path,
            output_dir=output_dir,
            ir_dataset_name=ir_dataset_name,
            dtype=dtype,
            num_entry_points=num_entry_points,
            ncandidates=ncandidates,
            max_iter=max_iter,
            cache_file=cache_file,
            metric_dir=metric_dir
        )


    print("Combining metrics.")
    fieldnames = ["NumEntryPoints", "NCandidates", "MaxIter",
              "P@10", "P@5", "R@10", "R@1000",
              "RR", "RR@10", "nDCG@10", "nDCG@5"]
    with open(f"metrics/{ret_name}/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for x, y, z in product(nep, nc, mi):
            metric_dir=f"metrics/{ret_name}/entries/{x}-{y}-{z}/aggregated_metrics.json"

            with open(metric_dir, "r") as g:
                metrics = json.load(g)

            row = {"NumEntryPoints": x, "NCandidates": y, "MaxIter": z}
            row.update(metrics)

            writer.writerow(row)
    print("Done :)")

    

        



    

if __name__ == "__main__":
    fire.Fire(main)