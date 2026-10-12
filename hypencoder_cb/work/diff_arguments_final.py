from hypencoder_cb.inference.approx_retrieve import do_retrieval, HypecoderGraphRetriever, HypecoderGraphRetrieverNew
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
    index_path = os.path.abspath(f"BM25index/trecdl2019judged")
    

    
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


    retriever = HypecoderGraphRetrieverNew(
            **retriever_kwargs
        )
    


    # -----------
    # Speed
    # -----------
    
    output_dir = "retrievals/efficient/preset/speed/2019"
    ret_name="efficient/preset/speed/2019"
    metric_dir=f"metrics/{ret_name}/"
    nep = 10_000
    nc = 64
    mi = 16



    retriever.set_parameters(
        num_entry_points=nep,
        ncandidates=nc,
        max_iter=mi,
        seed_bm25=False,
        rrf_bm25=False,
    )

    print(f"Starting retrieval: num_entry_points={nep}, ncandidates={nc}, max_iter={mi}")
    
    do_retrieval(
        retriever=retriever,
        model_name_or_path=model_name_or_path,
        encoded_item_path=encoded_item_path,
        item_neighbors_path=item_neighbors_path,
        output_dir=output_dir,
        ir_dataset_name=ir_dataset_name,
        dtype=dtype,
        num_entry_points=nep,
        ncandidates=nc,
        max_iter=mi,
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
        metric_dir=f"metrics/{ret_name}/aggregated_metrics.json"

        with open(metric_dir, "r") as g:
            metrics = json.load(g)

        row = {"NumEntryPoints": nep, "NCandidates": nc, "MaxIter": mi}
        row.update(metrics)

        writer.writerow(row)
    print("Done :)")


    ir_dataset_name="msmarco-passage/trec-dl-2020/judged"



      # -----------
    # Speed
    # -----------
    output_dir = "retrievals/efficient/preset/speed/2020"
    ret_name="efficient/preset/speed/2020"
    metric_dir=f"metrics/{ret_name}/"
    nep = 10_000
    nc = 64
    mi = 16



    retriever.set_parameters(
        num_entry_points=nep,
        ncandidates=nc,
        max_iter=mi,
        seed_bm25=False,
        rrf_bm25=False,
    )

    print(f"Starting retrieval: num_entry_points={nep}, ncandidates={nc}, max_iter={mi}")
    
    do_retrieval(
        retriever=retriever,
        model_name_or_path=model_name_or_path,
        encoded_item_path=encoded_item_path,
        item_neighbors_path=item_neighbors_path,
        output_dir=output_dir,
        ir_dataset_name=ir_dataset_name,
        dtype=dtype,
        num_entry_points=nep,
        ncandidates=nc,
        max_iter=mi,
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
        metric_dir=f"metrics/{ret_name}/aggregated_metrics.json"

        with open(metric_dir, "r") as g:
            metrics = json.load(g)

        row = {"NumEntryPoints": nep, "NCandidates": nc, "MaxIter": mi}
        row.update(metrics)

        writer.writerow(row)
        print("Done :)")



    ir_dataset_name="msmarco-passage/trec-dl-2019/judged"

    # -----------
    # Speed
    # -----------
    output_dir = "retrievals/efficient/preset/quality/2019"
    ret_name="efficient/preset/quality/2019"
    metric_dir=f"metrics/{ret_name}/"
    nep = 100_000
    nc = 328
    mi = 20



    retriever.set_parameters(
        num_entry_points=nep,
        ncandidates=nc,
        max_iter=mi,
        seed_bm25=False,
        rrf_bm25=False,
    )

    print(f"Starting retrieval: num_entry_points={nep}, ncandidates={nc}, max_iter={mi}")
    
    do_retrieval(
        retriever=retriever,
        model_name_or_path=model_name_or_path,
        encoded_item_path=encoded_item_path,
        item_neighbors_path=item_neighbors_path,
        output_dir=output_dir,
        ir_dataset_name=ir_dataset_name,
        dtype=dtype,
        num_entry_points=nep,
        ncandidates=nc,
        max_iter=mi,
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
        metric_dir=f"metrics/{ret_name}/aggregated_metrics.json"

        with open(metric_dir, "r") as g:
            metrics = json.load(g)

        row = {"NumEntryPoints": nep, "NCandidates": nc, "MaxIter": mi}
        row.update(metrics)

        writer.writerow(row)
    print("Done :)")


    ir_dataset_name="msmarco-passage/trec-dl-2020/judged"

    # -----------
    # Speed
    # -----------
    output_dir = "retrievals/efficient/preset/quality/2020"
    ret_name="efficient/preset/quality/2020"
    metric_dir=f"metrics/{ret_name}/"
    nep = 100_000
    nc = 328
    mi = 20



    retriever.set_parameters(
        num_entry_points=nep,
        ncandidates=nc,
        max_iter=mi,
        seed_bm25=False,
        rrf_bm25=False,
    )

    print(f"Starting retrieval: num_entry_points={nep}, ncandidates={nc}, max_iter={mi}")
    
    do_retrieval(
        retriever=retriever,
        model_name_or_path=model_name_or_path,
        encoded_item_path=encoded_item_path,
        item_neighbors_path=item_neighbors_path,
        output_dir=output_dir,
        ir_dataset_name=ir_dataset_name,
        dtype=dtype,
        num_entry_points=nep,
        ncandidates=nc,
        max_iter=mi,
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
        metric_dir=f"metrics/{ret_name}/aggregated_metrics.json"

        with open(metric_dir, "r") as g:
            metrics = json.load(g)

        row = {"NumEntryPoints": nep, "NCandidates": nc, "MaxIter": mi}
        row.update(metrics)

        writer.writerow(row)
    print("Done :)")






if __name__ == "__main__":
    fire.Fire(main)