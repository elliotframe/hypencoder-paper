from hypencoder_cb.inference.approx_retrieve import do_retrieval, HypecoderGraphRetriever, HypecoderGraphRetrieverBM25
from typing import Dict, List, Optional, Union
from itertools import product
import json
import csv
import fire
import os

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
    index_path = os.path.abspath(f"BM25index/trecdl2019judged")

    nep = [5_000, 10_000, 50_000, 100_000, 500_000]
    nc = [24, 64, 150, 328, 600]
    mi = [6, 12, 16, 20, 24]

    retriever_kwargs=dict(
            model_name_or_path=model_name_or_path,
            encoded_item_path=encoded_item_path,
            dtype=dtype,
            batch_size=100_000,
            query_max_length=64,
            item_neighbors_path=item_neighbors_path,
            num_entry_points=5_000,
            ncandidates=24,
            max_iter=6,
            early_stop=True,
            device="cuda",
            cache_file=cache_file,
            ir_dataset=ir_dataset_name,
            index_path=index_path,
        )


    retriever = HypecoderGraphRetrieverBM25(
            **retriever_kwargs
        )
    

    print(
        "Enter parameters as: <num_entry_points> <ncandidates> <max_iter> <alpha> \n"
        "Type 'end' to stop."
    )

    while True:
        user_input = input(">>> ").strip()

        if user_input.lower() == "end":
            print("Ending retrieval loop.")
            break

        try:
            x, y, z, a = map(int, user_input.split())
        except ValueError:
            print("Invalid input. Expected four integers or 'end'.")
            continue

        num_entry_points = x
        ncandidates = y
        max_iter = z
        alpha = a / 100.0

        metric_dir = (
            f"metrics/{ret_name}/entries/"
            f"{num_entry_points}-{ncandidates}-{max_iter}-{alpha}"
        )

        retriever.set_parameters(num_entry_points, ncandidates, max_iter, alpha)

        print(
            f"Starting retrieval: "
            f"num_entry_points={num_entry_points}, "
            f"ncandidates={ncandidates}, "
            f"max_iter={max_iter}"
            f"alpha={alpha}"
        )

        do_retrieval(
            retriever=retriever,
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
            metric_dir=metric_dir,
        )


    # print("Combining metrics.")
    # fieldnames = ["NumEntryPoints", "NCandidates", "MaxIter",
    #           "P@10", "P@5", "R@10", "R@1000",
    #           "RR", "RR@10", "nDCG@10", "nDCG@5"]
    # with open(f"metrics/{ret_name}/results.csv", "w", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for x, y, z in product(nep, nc, mi):
    #         metric_dir=f"metrics/{ret_name}/entries/{x}-{y}-{z}/aggregated_metrics.json"

    #         with open(metric_dir, "r") as g:
    #             metrics = json.load(g)

    #         row = {"NumEntryPoints": x, "NCandidates": y, "MaxIter": z}
    #         row.update(metrics)

    #         writer.writerow(row)
    # print("Done :)")

    

        



    

if __name__ == "__main__":
    fire.Fire(main)