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
    

    # # -----------
    # # test9: Different values of k1, b for BM25 seeding
    # # -----------
    # output_dir = "retrievals/test9-3090"
    # ret_name=output_dir.split("/")[-1]
    # k1_values = [0.5, 0.9, 1.2, 1.5, 2.0, 2.5]
    # b_values = [0.3, 0.5, 0.75, 0.8, 1.0] 
    # num_entry_points = 1000
    # ncandidates = 6
    # max_iter = 4


    # for k1, b in product(k1_values, b_values):

    #     metric_dir=f"metrics/{ret_name}/entries/{k1}-{b}"

    #     retriever.set_parameters(
    #         num_entry_points=num_entry_points,
    #         ncandidates=ncandidates,
    #         max_iter=max_iter,
    #         seed_bm25=True,
    #         seed_dph=False,
    #         rrf_bm25=False,
    #         rrf_dph=False,
    #         k1=k1,
    #         b=b,
    #     )

    #     print(f"Starting retrieval: num_entry_points={num_entry_points}, ncandidates={ncandidates}, max_iter={max_iter}, k1={k1}, b={b}")
        
    #     do_retrieval(
    #         retriever=retriever,
    #         model_name_or_path=model_name_or_path,
    #         encoded_item_path=encoded_item_path,
    #         item_neighbors_path=item_neighbors_path,
    #         output_dir=output_dir,
    #         ir_dataset_name=ir_dataset_name,
    #         dtype=dtype,
    #         num_entry_points=num_entry_points,
    #         ncandidates=ncandidates,
    #         max_iter=max_iter,
    #         cache_file=cache_file,
    #         metric_dir=metric_dir
    #     )


    # print("Combining metrics.")
    # fieldnames = ["k1", "b", "NumEntryPoints", "NCandidates", "MaxIter",
    #           "P@10", "P@5", "R@10", "R@1000",
    #           "RR", "RR@10", "nDCG@10", "nDCG@5"]
    # with open(f"metrics/{ret_name}/results.csv", "w", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for k1, b in product(k1_values, b_values):
    #         metric_dir=f"metrics/{ret_name}/entries/{k1}-{b}/aggregated_metrics.json"

    #         with open(metric_dir, "r") as g:
    #             metrics = json.load(g)

    #         row = {"k1": k1, "b":b, "NumEntryPoints": num_entry_points, "NCandidates": ncandidates, "MaxIter": max_iter}
    #         row.update(metrics)

    #         writer.writerow(row)
    # print("Done :)")




    # # -----------
    # # test10: Test dph seeding
    # # -----------
    # output_dir = "retrievals/test10-3090"
    # ret_name=output_dir.split("/")[-1]
    # nep = [1_000, 2_048, 5_000, 10_000, 50_000, 100_000]
    # nc = [6, 12, 18, 24, 64, 150]
    # mi = [3,4,5,6,12,16]



    # for num_entry_points, ncandidates, max_iter in product(nep, nc, mi):

    #     metric_dir=f"metrics/{ret_name}/entries/{num_entry_points}-{ncandidates}-{max_iter}"

    #     retriever.set_parameters(
    #         num_entry_points=num_entry_points,
    #         ncandidates=ncandidates,
    #         max_iter=max_iter,
    #         seed_bm25=False,
    #         seed_dph=True,
    #         rrf_bm25=False,
    #         rrf_dph=False,
    #     )

    #     print(f"Starting retrieval: num_entry_points={num_entry_points}, ncandidates={ncandidates}, max_iter={max_iter}")
        
    #     do_retrieval(
    #         retriever=retriever,
    #         model_name_or_path=model_name_or_path,
    #         encoded_item_path=encoded_item_path,
    #         item_neighbors_path=item_neighbors_path,
    #         output_dir=output_dir,
    #         ir_dataset_name=ir_dataset_name,
    #         dtype=dtype,
    #         num_entry_points=num_entry_points,
    #         ncandidates=ncandidates,
    #         max_iter=max_iter,
    #         cache_file=cache_file,
    #         metric_dir=metric_dir
    #     )


    # print("Combining metrics.")
    # fieldnames = ["NumEntryPoints", "NCandidates", "MaxIter",
    #           "P@10", "P@5", "R@10", "R@1000",
    #           "RR", "RR@10", "nDCG@10", "nDCG@5"]
    # with open(f"metrics/{ret_name}/results.csv", "w", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for num_entry_points, ncandidates, max_iter in product(nep, nc, mi):
    #         metric_dir=f"metrics/{ret_name}/entries/{num_entry_points}-{ncandidates}-{max_iter}/aggregated_metrics.json"

    #         with open(metric_dir, "r") as g:
    #             metrics = json.load(g)

    #         row = {"NumEntryPoints": num_entry_points, "NCandidates": ncandidates, "MaxIter": max_iter}
    #         row.update(metrics)

    #         writer.writerow(row)
    # print("Done :)")



    # # -----------
    # # test11: Test different k1, b values for seeding and rrf with BM25
    # # -----------
    # output_dir = "retrievals/test11-3090"
    # ret_name=output_dir.split("/")[-1]
    # k1_values = [0.5, 0.9, 1.2, 1.5, 2.0, 2.5]
    # b_values = [0.3, 0.5, 0.75, 0.8, 1.0]
    # alpha_values = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    # num_entry_points = 1000
    # ncandidates = 6
    # max_iter = 4


    # for k1, b, alpha in product(k1_values, b_values, alpha_values):

    #     metric_dir=f"metrics/{ret_name}/entries/{k1}-{b}-{alpha}"

    #     retriever.set_parameters(
    #         num_entry_points=num_entry_points,
    #         ncandidates=ncandidates,
    #         max_iter=max_iter,
    #         seed_bm25=True,
    #         seed_dph=False,
    #         rrf_bm25=True,
    #         rrf_dph=False,
    #         k1=k1,
    #         b=b,
    #         alpha=alpha,
    #     )

    #     print(f"Starting retrieval: num_entry_points={num_entry_points}, ncandidates={ncandidates}, max_iter={max_iter}, k1={k1}, b={b}, alpha={alpha}")
        
    #     do_retrieval(
    #         retriever=retriever,
    #         model_name_or_path=model_name_or_path,
    #         encoded_item_path=encoded_item_path,
    #         item_neighbors_path=item_neighbors_path,
    #         output_dir=output_dir,
    #         ir_dataset_name=ir_dataset_name,
    #         dtype=dtype,
    #         num_entry_points=num_entry_points,
    #         ncandidates=ncandidates,
    #         max_iter=max_iter,
    #         cache_file=cache_file,
    #         metric_dir=metric_dir
    #     )


    # print("Combining metrics.")
    # fieldnames = ["k1", "b", "alpha", "NumEntryPoints", "NCandidates", "MaxIter",
    #           "P@10", "P@5", "R@10", "R@1000",
    #           "RR", "RR@10", "nDCG@10", "nDCG@5"]
    # with open(f"metrics/{ret_name}/results.csv", "w", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for k1, b, alpha in product(k1_values, b_values, alpha_values):
    #         metric_dir=f"metrics/{ret_name}/entries/{k1}-{b}-{alpha}/aggregated_metrics.json"

    #         with open(metric_dir, "r") as g:
    #             metrics = json.load(g)

    #         row = {"k1": k1, "b":b, "alpha":alpha, "NumEntryPoints": num_entry_points, "NCandidates": ncandidates, "MaxIter": max_iter}
    #         row.update(metrics)

    #         writer.writerow(row)
    # print("Done :)")




    # # -----------
    # # test12: Test different k1, b values for seeding W/ BM25 and rrh with dph
    # # -----------
    # output_dir = "retrievals/test12-3090"
    # ret_name=output_dir.split("/")[-1]
    # k1_values = [0.5, 0.9, 1.2, 1.5, 2.0, 2.5]
    # b_values = [0.3, 0.5, 0.75, 0.8, 1.0]
    # alpha_values = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    
    # num_entry_points = 1000
    # ncandidates = 6
    # max_iter = 4


    # for k1, b, alpha in product(k1_values, b_values, alpha_values):

    #     metric_dir=f"metrics/{ret_name}/entries/{k1}-{b}-{alpha}"

    #     retriever.set_parameters(
    #         num_entry_points=num_entry_points,
    #         ncandidates=ncandidates,
    #         max_iter=max_iter,
    #         seed_bm25=True,
    #         seed_dph=False,
    #         rrf_bm25=False,
    #         rrf_dph=True,
    #         k1=k1,
    #         b=b,
    #         alpha=alpha,
    #     )

    #     print(f"Starting retrieval: num_entry_points={num_entry_points}, ncandidates={ncandidates}, max_iter={max_iter}, k1={k1}, b={b}, alpha={alpha}")
        
    #     do_retrieval(
    #         retriever=retriever,
    #         model_name_or_path=model_name_or_path,
    #         encoded_item_path=encoded_item_path,
    #         item_neighbors_path=item_neighbors_path,
    #         output_dir=output_dir,
    #         ir_dataset_name=ir_dataset_name,
    #         dtype=dtype,
    #         num_entry_points=num_entry_points,
    #         ncandidates=ncandidates,
    #         max_iter=max_iter,
    #         cache_file=cache_file,
    #         metric_dir=metric_dir
    #     )


    # print("Combining metrics.")
    # fieldnames = ["k1", "b", "alpha", "NumEntryPoints", "NCandidates", "MaxIter",
    #           "P@10", "P@5", "R@10", "R@1000",
    #           "RR", "RR@10", "nDCG@10", "nDCG@5"]
    # with open(f"metrics/{ret_name}/results.csv", "w", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for k1, b, alpha in product(k1_values, b_values, alpha_values):
    #         metric_dir=f"metrics/{ret_name}/entries/{k1}-{b}-{alpha}/aggregated_metrics.json"

    #         with open(metric_dir, "r") as g:
    #             metrics = json.load(g)

    #         row = {"k1": k1, "b":b, "alpha":alpha, "NumEntryPoints": num_entry_points, "NCandidates": ncandidates, "MaxIter": max_iter}
    #         row.update(metrics)

    #         writer.writerow(row)
    # print("Done :)")




    # # -----------
    # # test13: Test different k1, b values for seeding W/ BM25 and rrh with dph and BM25
    # # -----------
    # output_dir = "retrievals/test13-3090"
    # ret_name=output_dir.split("/")[-1]
    # k1_values = [0.5, 0.9, 1.2, 1.5, 2.0, 2.5]
    # b_values = [0.3, 0.5, 0.75, 0.8, 1.0]
    # alpha_values = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    # num_entry_points = 1000
    # ncandidates = 6
    # max_iter = 4


    # for k1, b, alpha in product(k1_values, b_values, alpha_values):

    #     metric_dir=f"metrics/{ret_name}/entries/{k1}-{b}-{alpha}"

    #     retriever.set_parameters(
    #         num_entry_points=num_entry_points,
    #         ncandidates=ncandidates,
    #         max_iter=max_iter,
    #         seed_bm25=True,
    #         seed_dph=False,
    #         rrf_bm25=True,
    #         rrf_dph=True,
    #         k1=k1,
    #         b=b,
    #         alpha=alpha
    #     )

    #     print(f"Starting retrieval: num_entry_points={num_entry_points}, ncandidates={ncandidates}, max_iter={max_iter}, k1={k1}, b={b}, alpha={alpha}")
        
    #     do_retrieval(
    #         retriever=retriever,
    #         model_name_or_path=model_name_or_path,
    #         encoded_item_path=encoded_item_path,
    #         item_neighbors_path=item_neighbors_path,
    #         output_dir=output_dir,
    #         ir_dataset_name=ir_dataset_name,
    #         dtype=dtype,
    #         num_entry_points=num_entry_points,
    #         ncandidates=ncandidates,
    #         max_iter=max_iter,
    #         cache_file=cache_file,
    #         metric_dir=metric_dir
    #     )


    # print("Combining metrics.")
    # fieldnames = ["k1", "b", "alpha", "NumEntryPoints", "NCandidates", "MaxIter",
    #           "P@10", "P@5", "R@10", "R@1000",
    #           "RR", "RR@10", "nDCG@10", "nDCG@5"]
    # with open(f"metrics/{ret_name}/results.csv", "w", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for k1, b, alpha in product(k1_values, b_values, alpha_values):
    #         metric_dir=f"metrics/{ret_name}/entries/{k1}-{b}-{alpha}/aggregated_metrics.json"

    #         with open(metric_dir, "r") as g:
    #             metrics = json.load(g)

    #         row = {"k1": k1, "b":b, "alpha":alpha, "NumEntryPoints": num_entry_points, "NCandidates": ncandidates, "MaxIter": max_iter}
    #         row.update(metrics)

    #         writer.writerow(row)
    # print("Done :)")



    # # -----------
    # # test14: Test different random seeds compared to original
    # # -----------
    # output_dir = "retrievals/test14-3090"
    # ret_name=output_dir.split("/")[-1]
    # params = [[100_000, 328, 20], [10_000, 64, 16]]
    # seeds = [x for x in range(30,60)]


    # for seed in seeds:
    #     for num_entry_points, ncandidates, max_iter in params:
    #         metric_dir=f"metrics/{ret_name}/entries/{seed}-{num_entry_points}-{ncandidates}-{max_iter}"
    #         retriever.set_parameters(
    #             num_entry_points=num_entry_points,
    #             ncandidates=ncandidates,
    #             max_iter=max_iter,
    #             seed_bm25=False,
    #             seed_dph=False,
    #             rrf_bm25=False,
    #             rrf_dph=False,
    #             random_seed=seed
    #         )

    #         print(f"Starting retrieval: seed={seed}, num_entry_points={num_entry_points}, ncandidates={ncandidates}, max_iter={max_iter}")
            
    #         do_retrieval(
    #             retriever=retriever,
    #             model_name_or_path=model_name_or_path,
    #             encoded_item_path=encoded_item_path,
    #             item_neighbors_path=item_neighbors_path,
    #             output_dir=output_dir,
    #             ir_dataset_name=ir_dataset_name,
    #             dtype=dtype,
    #             num_entry_points=num_entry_points,
    #             ncandidates=ncandidates,
    #             max_iter=max_iter,
    #             cache_file=cache_file,
    #             metric_dir=metric_dir
    #         )


    # print("Combining metrics.")
    # fieldnames = ["Seed", "NumEntryPoints", "NCandidates", "MaxIter",
    #           "P@10", "P@5", "R@10", "R@1000",
    #           "RR", "RR@10", "nDCG@10", "nDCG@5"]
    # with open(f"metrics/{ret_name}/results.csv", "w", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for seed in seeds:
    #         for num_entry_points, ncandidates, max_iter in params:
    #             metric_dir=f"metrics/{ret_name}/entries/{seed}-{num_entry_points}-{ncandidates}-{max_iter}/aggregated_metrics.json"

    #             with open(metric_dir, "r") as g:
    #                 metrics = json.load(g)

    #             row = {"Seed": seed, "NumEntryPoints": num_entry_points, "NCandidates": ncandidates, "MaxIter": max_iter}
    #             row.update(metrics)

    #             writer.writerow(row)
    # print("Done :)")


    # # -----------
    # # test15: Test dph + bm25 combined seeding
    # # -----------
    # output_dir = "retrievals/test15-3090"
    # ret_name=output_dir.split("/")[-1]
    # nep = [750, 1_000, 2_048, 5_000, 10_000, 50_000]
    # nc = [6, 12, 18, 24, 64, 150]
    # mi = [3,4,5,6,12,16]
    # k1, b = 0.5, 0.75




    # for num_entry_points, ncandidates, max_iter in product(nep, nc, mi):

    #     metric_dir=f"metrics/{ret_name}/entries/{num_entry_points}-{ncandidates}-{max_iter}"

    #     retriever.set_parameters(
    #         num_entry_points=num_entry_points,
    #         ncandidates=ncandidates,
    #         max_iter=max_iter,
    #         seed_bm25=True,
    #         seed_dph=True,
    #         rrf_bm25=False,
    #         rrf_dph=False,
    #         k1=k1,
    #         b=b
    #     )

    #     print(f"Starting retrieval: num_entry_points={num_entry_points}, ncandidates={ncandidates}, max_iter={max_iter}, k1={k1}, b={b}")
        
    #     do_retrieval(
    #         retriever=retriever,
    #         model_name_or_path=model_name_or_path,
    #         encoded_item_path=encoded_item_path,
    #         item_neighbors_path=item_neighbors_path,
    #         output_dir=output_dir,
    #         ir_dataset_name=ir_dataset_name,
    #         dtype=dtype,
    #         num_entry_points=num_entry_points,
    #         ncandidates=ncandidates,
    #         max_iter=max_iter,
    #         cache_file=cache_file,
    #         metric_dir=metric_dir
    #     )


    # print("Combining metrics.")
    # fieldnames = ["NumEntryPoints", "NCandidates", "MaxIter",
    #           "P@10", "P@5", "R@10", "R@1000",
    #           "RR", "RR@10", "nDCG@10", "nDCG@5"]
    # with open(f"metrics/{ret_name}/results.csv", "w", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for num_entry_points, ncandidates, max_iter in product(nep, nc, mi):
    #         metric_dir=f"metrics/{ret_name}/entries/{num_entry_points}-{ncandidates}-{max_iter}/aggregated_metrics.json"

    #         with open(metric_dir, "r") as g:
    #             metrics = json.load(g)

    #         row = {"NumEntryPoints": num_entry_points, "NCandidates": ncandidates, "MaxIter": max_iter}
    #         row.update(metrics)

    #         writer.writerow(row)
    # print("Done :)")



    #  # -----------
    # # test16: Test dph + bm25 combined seeding, + rrf w/ bm25 and dph
    # # -----------
    # output_dir = "retrievals/test16"
    # ret_name=output_dir.split("/")[-1]
    # nep = [750, 1_000, 2_048, 5_000]
    # nc = [6, 12]
    # mi = [3,4,5,6]
    # k1, b = 0.5, 0.75
    # alpha_values = [0.91, 0.92, 0.93, 0.94]




    # for num_entry_points, ncandidates, max_iter, alpha in product(nep, nc, mi, alpha_values):

    #     metric_dir=f"metrics/{ret_name}/entries/{num_entry_points}-{ncandidates}-{max_iter}-{alpha}"

    #     retriever.set_parameters(
    #         num_entry_points=num_entry_points,
    #         ncandidates=ncandidates,
    #         max_iter=max_iter,
    #         seed_bm25=True,
    #         seed_dph=True,
    #         rrf_bm25=True,
    #         rrf_dph=True,
    #         k1=k1,
    #         b=b,
    #         alpha=alpha
    #     )

    #     print(f"Starting retrieval: num_entry_points={num_entry_points}, ncandidates={ncandidates}, max_iter={max_iter}, k1={k1}, b={b}, alpha={alpha}")
        
    #     do_retrieval(
    #         retriever=retriever,
    #         model_name_or_path=model_name_or_path,
    #         encoded_item_path=encoded_item_path,
    #         item_neighbors_path=item_neighbors_path,
    #         output_dir=output_dir,
    #         ir_dataset_name=ir_dataset_name,
    #         dtype=dtype,
    #         num_entry_points=num_entry_points,
    #         ncandidates=ncandidates,
    #         max_iter=max_iter,
    #         cache_file=cache_file,
    #         metric_dir=metric_dir
    #     )


    # print("Combining metrics.")
    # fieldnames = ["NumEntryPoints", "NCandidates", "MaxIter",
    #           "P@10", "P@5", "R@10", "R@1000",
    #           "RR", "RR@10", "nDCG@10", "nDCG@5"]
    # with open(f"metrics/{ret_name}/results.csv", "w", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for num_entry_points, ncandidates, max_iter, alpha in product(nep, nc, mi, alpha_values):
    #         metric_dir=f"metrics/{ret_name}/entries/{num_entry_points}-{ncandidates}-{max_iter}-{alpha}/aggregated_metrics.json"

    #         with open(metric_dir, "r") as g:
    #             metrics = json.load(g)

    #         row = {"NumEntryPoints": num_entry_points, "NCandidates": ncandidates, "MaxIter": max_iter, "alpha": alpha}
    #         row.update(metrics)

    #         writer.writerow(row)
    # print("Done :)")


    # -----------
    # test17: Test different k1, b values for rrf with BM25
    # -----------
    output_dir = "retrievals/test17"
    ret_name=output_dir.split("/")[-1]
    k1_values = [0.5, 0.9, 1.2, 1.5, 2.0, 2.5]
    b_values = [0.3, 0.5, 0.75, 0.8, 1.0]
    alpha_values = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    num_entry_points = 1000
    ncandidates = 6
    max_iter = 4


    for k1, b, alpha in product(k1_values, b_values, alpha_values):

        metric_dir=f"metrics/{ret_name}/entries/{k1}-{b}-{alpha}"

        retriever.set_parameters(
            num_entry_points=num_entry_points,
            ncandidates=ncandidates,
            max_iter=max_iter,
            seed_bm25=False,
            seed_dph=False,
            rrf_bm25=True,
            rrf_dph=False,
            k1=k1,
            b=b,
            alpha=alpha,
        )

        print(f"Starting retrieval: num_entry_points={num_entry_points}, ncandidates={ncandidates}, max_iter={max_iter}, k1={k1}, b={b}, alpha={alpha}")
        
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
            metric_dir=metric_dir
        )


    print("Combining metrics.")
    fieldnames = ["k1", "b", "alpha", "NumEntryPoints", "NCandidates", "MaxIter",
              "P@10", "P@5", "R@10", "R@1000",
              "RR", "RR@10", "nDCG@10", "nDCG@5"]
    with open(f"metrics/{ret_name}/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k1, b, alpha in product(k1_values, b_values, alpha_values):
            metric_dir=f"metrics/{ret_name}/entries/{k1}-{b}-{alpha}/aggregated_metrics.json"

            with open(metric_dir, "r") as g:
                metrics = json.load(g)

            row = {"k1": k1, "b":b, "alpha":alpha, "NumEntryPoints": num_entry_points, "NCandidates": ncandidates, "MaxIter": max_iter}
            row.update(metrics)

            writer.writerow(row)
    print("Done :)")




      # -----------
    # test18: Test different alpha values for rrf with dph
    # -----------
    output_dir = "retrievals/test18"
    ret_name=output_dir.split("/")[-1]
    alpha_values = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    num_entry_points = 1000
    ncandidates = 6
    max_iter = 4


    for alpha in alpha_values:

        metric_dir=f"metrics/{ret_name}/entries/{alpha}"

        retriever.set_parameters(
            num_entry_points=num_entry_points,
            ncandidates=ncandidates,
            max_iter=max_iter,
            seed_bm25=False,
            seed_dph=False,
            rrf_bm25=False,
            rrf_dph=True,
            k1=k1,
            b=b,
            alpha=alpha,
        )

        print(f"Starting retrieval: num_entry_points={num_entry_points}, ncandidates={ncandidates}, max_iter={max_iter}, alpha={alpha}")
        
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
            metric_dir=metric_dir
        )


    print("Combining metrics.")
    fieldnames = ["alpha", "NumEntryPoints", "NCandidates", "MaxIter",
              "P@10", "P@5", "R@10", "R@1000",
              "RR", "RR@10", "nDCG@10", "nDCG@5"]
    with open(f"metrics/{ret_name}/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for alpha in alpha_values:
            metric_dir=f"metrics/{ret_name}/entries/{k1}-{b}-{alpha}/aggregated_metrics.json"

            with open(metric_dir, "r") as g:
                metrics = json.load(g)

            row = {"alpha":alpha, "NumEntryPoints": num_entry_points, "NCandidates": ncandidates, "MaxIter": max_iter}
            row.update(metrics)

            writer.writerow(row)
    print("Done :)")



    # -----------
    # test19: Test different k1, b values for rrf with BM25 and DPH
    # -----------
    output_dir = "retrievals/test19"
    ret_name=output_dir.split("/")[-1]
    k1_values = [0.5, 0.9, 1.2, 1.5, 2.0, 2.5]
    b_values = [0.3, 0.5, 0.75, 0.8, 1.0]
    alpha_values = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    num_entry_points = 1000
    ncandidates = 6
    max_iter = 4


    for k1, b, alpha in product(k1_values, b_values, alpha_values):

        metric_dir=f"metrics/{ret_name}/entries/{k1}-{b}-{alpha}"

        retriever.set_parameters(
            num_entry_points=num_entry_points,
            ncandidates=ncandidates,
            max_iter=max_iter,
            seed_bm25=False,
            seed_dph=False,
            rrf_bm25=True,
            rrf_dph=True,
            k1=k1,
            b=b,
            alpha=alpha,
        )

        print(f"Starting retrieval: num_entry_points={num_entry_points}, ncandidates={ncandidates}, max_iter={max_iter}, k1={k1}, b={b}, alpha={alpha}")
        
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
            metric_dir=metric_dir
        )


    print("Combining metrics.")
    fieldnames = ["k1", "b", "alpha", "NumEntryPoints", "NCandidates", "MaxIter",
              "P@10", "P@5", "R@10", "R@1000",
              "RR", "RR@10", "nDCG@10", "nDCG@5"]
    with open(f"metrics/{ret_name}/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k1, b, alpha in product(k1_values, b_values, alpha_values):
            metric_dir=f"metrics/{ret_name}/entries/{k1}-{b}-{alpha}/aggregated_metrics.json"

            with open(metric_dir, "r") as g:
                metrics = json.load(g)

            row = {"k1": k1, "b":b, "alpha":alpha, "NumEntryPoints": num_entry_points, "NCandidates": ncandidates, "MaxIter": max_iter}
            row.update(metrics)

            writer.writerow(row)
    print("Done :)")

    

    
    

    

        



    

if __name__ == "__main__":
    fire.Fire(main)