import os
import pickle
import random
from collections import defaultdict
from queue import PriorityQueue
from typing import Dict, List, Optional, Union, Iterable

import fire
import torch
import copy
from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path
import faiss
import numpy as np
import pyterrier as pt
from pyterrier_pisa import PisaIndex
import pandas as pd
import shutil

from hypencoder_cb.inference.retrieve import do_retrieval_shared
from hypencoder_cb.inference.shared import (
    BaseRetriever,
    Item,
    TextQuery,
    load_encoded_items_from_disk,
    items_from_ir_dataset,
)
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder
from hypencoder_cb.utils.jsonl_utils import JsonlReader
from hypencoder_cb.utils.torch_utils import dtype_lookup

import subprocess

def print_gpu_memory():
    """Prints current used and total GPU memory in GB."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE,
        encoding='utf-8'
    )
    used, total = result.stdout.strip().split(',')
    used_gb = int(used.strip()) / 1024
    total_gb = int(total.strip()) / 1024
    print(f"GPU Memory: {used_gb:.2f} GB / {total_gb:.2f} GB used")


class HypecoderGraphRetriever(BaseRetriever):

    def __init__(
        self,
        model_name_or_path: str,
        encoded_item_path: str,
        item_neighbors_path: str,
        batch_size: int = 100_000,
        device: str = "cuda",
        query_max_length: int = 32,
        cache_file: Optional[str] = None,
        num_entry_points: int = 10_000,
        ncandidates: int = 50,
        max_iter: int = 16,
        early_stop: bool = True,
        dtype: Union[torch.dtype, str] = "float32",
    ) -> None:
        """

        Args:
            model_name_or_path (str): The HypencoderDualEncoder model to use,
                this should match the model used for encoding the items.
            encoded_item_path (str): The path to the encoded items.
            item_neighbors_path (str): The path to the item neighbors JSONL.
                Should have the keys "item_id" and "neighbors".
            batch_size (int, optional): The batch size to use for inference.
                Defaults to 100_000.
            device (str, optional): The device to use to store the embeddings
                and to run the model. Defaults to "cuda".
            query_max_length (int, optional): The maximum length of the query.
                Defaults to 32.
            cache_file (Optional[str], optional): If provided, the cache file
                to use for loading the encoded items and item neighbors. If
                the file does not exist, it will be created and the data will
                stored in it. Defaults to None.
            num_entry_points (int, optional): The number of randomly selected
                initial entry points. This is equal to len(initial_candidates).
                Defaults to 10_000.
            ncandidates (int, optional): The number of candidates to explore
                as each step. Defaults to 50.
            max_iter (int, optional): The maximum number of candidate expansion
                steps to do. Defaults to 16.
            early_stop (bool, optional): Whether to stop early if no new
                candidates are added to the queue at a given step. Defaults
                to True.
            dtype (Union[torch.dtype, str], optional): The dtype to use for
                the model and embeddings. Defaults to "float32".
        """

        if isinstance(dtype, str):
            dtype = dtype_lookup(dtype)

        self.dtype = dtype
        self.device = device
        self.batch_size = batch_size
        self.encoded_item_path = encoded_item_path
        self.num_entry_points = num_entry_points
        self.ncandidates = ncandidates
        self.max_iter = max_iter
        self.query_max_length = query_max_length
        self.early_stop = early_stop

        print(model_name_or_path)
        self.model = (
            HypencoderDualEncoder.from_pretrained(model_name_or_path)
            .to(device, dtype=self.dtype)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if cache_file is not None and os.path.exists(cache_file):
            print(f"Loading from cache {cache_file}")
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)

                self.ids = cache["item_ids"]
                self.encoded_item_embeddings = cache["encoded_item_embeddings"]
                self.item_id_to_index = cache["item_id_to_index"]
                self.item_id_to_content = cache["item_id_to_content"]
                self.item_neighbor_ids = cache["item_neighbor_ids"]
                self.item_id_to_neighbor_indices = cache[
                    "item_id_to_neighbor_indices"
                ]

        else:
            self.encoded_items = load_encoded_items_from_disk(
                encoded_item_path
            )

            self.encoded_item_embeddings = torch.stack(
                [
                    torch.tensor(x.representation)
                    for x in tqdm(
                        self.encoded_items, desc="Item Embeddings to Tensor"
                    )
                ]
            ).to(self.device, dtype=self.dtype)

            self.item_id_to_index = {
                item.id: idx
                for idx, item in tqdm(
                    enumerate(self.encoded_items), desc="Item ID to Index"
                )
            }

            self.ids = [item.id for item in self.encoded_items]

            self.item_id_to_content = {
                item.id: item.text
                for item in tqdm(self.encoded_items, desc="Item ID to Content")
            }

            with JsonlReader(item_neighbors_path) as reader:
                self.item_neighbor_ids = {
                    line["item_id"]: line["neighbors"]
                    for line in tqdm(reader, desc="Loading Item Graph")
                }

            self.item_id_to_neighbor_indices = defaultdict(list)

            for item_id, neighbors in tqdm(
                self.item_neighbor_ids.items(),
                desc="Building Neighbor Indices",
            ):
                self.item_id_to_neighbor_indices[item_id] = [
                    self.item_id_to_index[neighbor] for neighbor in neighbors
                ]


            if cache_file is not None:
                cache_file = Path(cache_file)
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                print(f"Caching to {cache_file}")
                cache_values = {
                    "item_ids": self.ids,
                    "encoded_item_embeddings": self.encoded_item_embeddings,
                    "item_id_to_index": self.item_id_to_index,
                    "item_id_to_content": self.item_id_to_content,
                    "item_neighbor_ids": self.item_neighbor_ids,
                    "item_id_to_neighbor_indices": self.item_id_to_neighbor_indices,
                }

                with open(cache_file, "wb") as f:
                    pickle.dump(cache_values, f)

        # Originally uncommented, set once
        # self._set_entry_points()
                    
        nlist = 100
        nprobe = 10
        # self.item_matrix_np = self.encoded_item_embeddings.cpu().numpy().astype('float32')
        # N, D = self.item_matrix_np.shape

        # # --- 3. Create approximate FAISS index ---
        # quantizer = faiss.IndexFlatIP(D)          # the coarse quantizer
        # self.index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
        # self.index.train(self.item_matrix_np)               # train the IVF clusters
        # self.index.add(self.item_matrix_np) 

        # --------------
        # GPU
        # --------------

        # # --- 2. Prepare item embeddings ---
        
        self.item_matrix = self.encoded_item_embeddings  # [N_items, 768]
        self.item_matrix_np = self.item_matrix.cpu().numpy()
        N, D = self.item_matrix_np.shape

        # --- 3. Initialize GPU resources ---
        breakpoint()
        res = faiss.StandardGpuResources()
        res.setTempMemory(4 * 1024**3)

        cfg = faiss.GpuIndexIVFFlatConfig()
        cfg.useFloat16 = True

        # --- 4. Create GPU IVF index ---
        quantizer = faiss.GpuIndexFlatIP(res, D)
        self.gpu_index = faiss.GpuIndexIVFFlat(res, quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT, cfg)

        # print(self.gpu_index)
        # assert self.gpu_index.useFloat16
        # breakpoint()

        # --- 5. Train IVF on GPU ---
        self.gpu_index.train(self.item_matrix_np)  # training happens fully on GPU

        # --- 6. Add embeddings to GPU index ---
        self.gpu_index.add(self.item_matrix_np)
        # batch_size = 250_000  # adjust for your GPU
        # for start in range(0, N, batch_size):
        #     end = min(start + batch_size, N)

        #     print(f"[Batch {start}-{end}] Before add:")
        #     print_gpu_memory()

        #     self.gpu_index.add(self.item_matrix_np[start:end])

        #     print(f"[Batch {start}-{end}] After add:")
        #     print_gpu_memory()


        # --- 7. Configure search ---
        self.gpu_index.nprobe = min(nprobe, nlist)

    def set_parameters(self, num_entry_points, ncandidates, max_iter):
        self.num_entry_points = num_entry_points
        self.ncandidates = ncandidates
        self.max_iter = max_iter
        # self._set_entry_points()

    def _set_entry_points(self):
        random.seed(43)
        self.entry_point_indices = torch.Tensor(
            random.sample(
                range(self.encoded_item_embeddings.shape[0]),
                self.num_entry_points,
            ),
        ).to(self.device, dtype=torch.long)

        self.entry_point_embeddings = self.encoded_item_embeddings[
            self.entry_point_indices
        ]
        self.entry_point_ids = [
            self.ids[idx] for idx in self.entry_point_indices
        ]



    def _set_entry_points_similar(self, query_model):
        """
        Selects initial entry points based on similarity between
        encoded items and the first-layer weights of the query-specific q-net,
        using a fully GPU FAISS approximate index (IVF) for fast similarity search.

        Args:
            nlist (int): number of IVF clusters. More clusters = more accurate.
            nprobe (int): number of clusters to probe during search.
            gpu_id (int): which GPU to use (0-based).
        """


        # --- 1. Extract first-layer weight vectors ---
        W = query_model.layers[0].linear.weight.detach().squeeze(0)
        avg_vec = W.mean(dim=0)  # shape [768]

        # --- 8. Prepare query vector ---
        query_np = avg_vec.cpu().numpy().reshape(1, -1)

        # --- 9. Perform GPU search ---
        D, I = self.gpu_index.search(query_np, self.num_entry_points)

        # --- 10. Convert results back to torch tensors ---
        self.entry_point_indices = torch.tensor(I[0], device=self.device, dtype=torch.long)
        self.entry_point_embeddings = self.item_matrix[self.entry_point_indices]
        self.entry_point_ids = [self.ids[i] for i in self.entry_point_indices]

        print(f"Selected {len(self.entry_point_ids)} query-conditioned entry points (fully GPU approximate search).")
        




        

    # def _set_entry_points_similar(self, query_model):
    #     """
    #     Selects initial entry points based on similarity between
    #     encoded items and the first-layer weights of the query-specific q-net,
    #     using a CPU FAISS approximate index (IVF) for faster similarity search.

    #     Args:
    #         nlist (int): number of clusters for IVF. More clusters = more accurate, slower indexing.
    #     """

    #     # --- 1. Extract first-layer weight vectors ---
    #     W = query_model.layers[0].linear.weight.detach().squeeze(0)
    #     avg_vec = W.mean(dim=0)  # shape [768]

    #     # --- 4. Prepare query vector ---
    #     query_np = avg_vec.cpu().numpy().astype('float32').reshape(1, -1)

    #     # --- 5. Perform approximate search ---
    #     self.index.nprobe = 10  # number of clusters to search; higher = more accurate
    #     D, I = self.index.search(query_np, self.num_entry_points)

    #     # --- 6. Convert results back to torch tensors on your device ---
    #     self.entry_point_indices = torch.tensor(I[0], device=self.device, dtype=torch.long)
    #     self.entry_point_embeddings = self.encoded_item_embeddings[self.entry_point_indices]
    #     self.entry_point_ids = [self.ids[i] for i in self.entry_point_indices]

    #     print(f"Selected {len(self.entry_point_ids)} query-conditioned entry points (approximate search).")



    def retrieve(self, query: TextQuery, top_k: int) -> List[Item]:
        tokenized_query = self.tokenizer(
            query.text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.query_max_length,
        ).to(self.device)

        with torch.no_grad():
            query_output = self.model.query_encoder(
                input_ids=tokenized_query["input_ids"],
                attention_mask=tokenized_query["attention_mask"],
            )
            query_model = query_output.representation

        final_queue = PriorityQueue(maxsize=top_k)

        # Not in original code, entry points set once for Graph Retriever usually
        self._set_entry_points_similar(query_model)

        candidates = [x for x in self.entry_point_ids]
        explored = set(candidates)

        curr_iter = 0
        while curr_iter < self.max_iter:
            candidate_embeddings = self.encoded_item_embeddings[
                [self.item_id_to_index[x] for x in candidates]
            ]
            candidate_embeddings = candidate_embeddings.unsqueeze(0)

            similarity_matrix = query_model(candidate_embeddings).view(-1)

            ncandidates = min(
                max(self.ncandidates, top_k), similarity_matrix.shape[0]
            )
            values, indices = torch.topk(similarity_matrix, ncandidates, dim=0)

            indices = indices.view(-1).cpu()
            values  = values.view(-1).cpu()

            prev_candidates = copy.deepcopy(candidates)
            candidates = []

            added_candidates = 0
            added_to_queue = 0
            for i, idx in enumerate(indices):
                idx = idx.item()

                item_id = prev_candidates[idx]
                score = similarity_matrix[idx].item()

                if final_queue.full():
                    # If queue is full, only add item to queue if score is
                    # greater than the minimum score in the queue
                    current_min = final_queue.get()
                    if score > current_min[0]:
                        added_to_queue += 1
                        final_queue.put((score, item_id))
                    else:
                        final_queue.put(current_min)
                else:
                    # If queue is not full add item to queue regardless of
                    # score
                    added_to_queue += 1
                    final_queue.put((score, item_id))

                if i < self.ncandidates:
                    for neighbor in self.item_neighbor_ids[item_id]:
                        if neighbor in explored:
                            continue

                        candidates.append(neighbor)
                        explored.add(neighbor)
                        added_candidates += 1

            if added_to_queue == 0 and self.early_stop:
                break

            curr_iter += 1

        items = []
        while not final_queue.empty():
            score, item_id = final_queue.get()
            items.append(
                Item(
                    text=self.item_id_to_content[item_id],
                    id=item_id,
                    score=score,
                    type="hypecoder_graph_retriever",
                )
            )

        return sorted(items, key=lambda x: x.score, reverse=True)





def do_retrieval(
    model_name_or_path: str,
    encoded_item_path: str,
    item_neighbors_path: str,
    output_dir: str,
    num_entry_points: int = 10_000,
    ncandidates: int = 50,
    max_iter: int = 16,
    early_stop: bool = True,
    device: str = "cuda",
    cache_file: Optional[str] = None,
    ir_dataset_name: Optional[str] = None,
    query_jsonl: Optional[str] = None,
    qrel_json: Optional[str] = None,
    query_id_key: str = "id",
    query_text_key: str = "text",
    dtype: str = "fp32",
    top_k: int = 1000,
    batch_size: int = 100_000,
    retriever_kwargs: Optional[Dict] = None,
    query_max_length: int = 64,
    include_content: bool = True,
    do_eval: bool = True,
    metric_names: Optional[List[str]] = None,
    metric_dir: Optional[str] = None,
    ignore_same_id: bool = False,
    retriever = None,
) -> None:
    """Does retrieval and optionally evaluation.

    Args:
        model_name_or_path (str): Name or path to a HypencoderDualEncoder
            checkpoint.
        encoded_item_path (str): Path to the encoded items.
        item_neighbors_path (str): Path to the item neighbors JSONL.
            Should have the keys "item_id" and "neighbors".
        output_dir (str): Path to the output directory which will contain the
            retrieval results and optionally the evaluation results.
        num_entry_points (int, optional): The number of randomly selected
            initial entry points. This is equal to len(initial_candidates).
            Defaults to 10_000.
        ncandidates (int, optional): The number of candidates to explore
            as each step. Defaults to 50.
        max_iter (int, optional): The maximum number of candidate expansion
            steps to do. Defaults to 16.
        early_stop (bool, optional): Whether to stop early if no new
            candidates are added to the queue at a given step. Defaults
            to True.
        device (str, optional): The device to use to store the embeddings
            and to run the model. Defaults to "cuda".
        cache_file (Optional[str], optional): If provided, the cache file
            to use for loading the encoded items and item neighbors. If
            the file does not exist, it will be created and the data will
            stored in it. Defaults to None.
        ir_dataset_name (Optional[str], optional): If provided is used to
            get the queries used for retrieval and qrels used for evaluation.
            If None, then `query_jsonl` must be provided and `qrel_json` must
            be provided if `do_eval` is True. Defaults to None.
        query_jsonl (Optional[str], optional): If provided is used as the
            queries for retrieval. If None, then `ir_dataset_name` must
            be provided. Defaults to None.
        qrel_json (Optional[str], optional): If provided is used as the qrels
            for evaluation. If None, then `ir_dataset_name` must
            be provided. Defaults to None.
        query_id_key (str, optional): The key in `query_jsonl` for the
            query ID. Not used if `ir_dataset_name` is provided. Defaults to
            "id".
        query_text_key (str, optional): The key in `query_jsonl` for the
            query text. Not used if `ir_dataset_name` is provided. Defaults to
            "text".
        dtype (str, optional): The dtype to use for the model and embedded
            items. Options are "fp16", "fp32", and "bf16". Defaults to "fp32".
        top_k (int, optional): The number of top items to retrieve. Defaults to
            1000.
        batch_size (int, optional): The batch size to use for retrieval.
            Defaults to 100,000.
        retriever_kwargs (Optional[Dict], optional): Additional keyword
            arguments to pass to the retriever. Defaults to None.
        query_max_length (int, optional): Maximum length of the query.
            Defaults to 64.
        include_content (bool, optional): Whether to include the content of the
            retrieved items in the output. Defaults to True.
        do_eval (bool, optional): Whether to do evaluation. Defaults to True.
        metric_names (Optional[List[str]], optional): A list of metrics to
            compute. These are passed to IR-Measures so should be compatible.
            If None, a default set of metrics is found. Defaults to None.
        metric_dir (Optional[str]): File path of directory where metrics should
            be saved.
        ignore_same_id (bool, optional): Whether to ignore retrievals with the
            same ID as the query. This is only relevant for certain datasets.
            Defaults to False.

    Raises:
        ValueError: If both `query_jsonl` and `ir_dataset_name` are provided.
        ValueError: If `do_eval` is True and `ir_dataset_name` is None and
            `qrel_json` is None.
    """

    retriever_kwargs = retriever_kwargs if retriever_kwargs is not None else {}

    do_retrieval_shared(
        retriever=retriever,
        retriever_cls=HypecoderGraphRetriever,
        retriever_kwargs=dict(
            model_name_or_path=model_name_or_path,
            encoded_item_path=encoded_item_path,
            dtype=dtype,
            batch_size=batch_size,
            query_max_length=query_max_length,
            # ignore_same_id=ignore_same_id,
            item_neighbors_path=item_neighbors_path,
            num_entry_points=num_entry_points,
            ncandidates=ncandidates,
            max_iter=max_iter,
            early_stop=early_stop,
            device=device,
            cache_file=cache_file,
            **retriever_kwargs,
        ),
        output_dir=output_dir,
        ir_dataset_name=ir_dataset_name,
        query_jsonl=query_jsonl,
        qrel_json=qrel_json,
        query_id_key=query_id_key,
        query_text_key=query_text_key,
        top_k=top_k,
        include_content=include_content,
        do_eval=do_eval,
        metric_names=metric_names,
        metric_dir=metric_dir,
    )





class HypecoderGraphRetrieverBM25(BaseRetriever):

    def __init__(
        self,
        model_name_or_path: str,
        encoded_item_path: str,
        item_neighbors_path: str,
        index_path: str,
        ir_dataset: str,
        batch_size: int = 100_000,
        device: str = "cuda",
        query_max_length: int = 32,
        cache_file: Optional[str] = None,
        num_entry_points: int = 10_000,
        ncandidates: int = 50,
        max_iter: int = 16,
        early_stop: bool = True,
        dtype: Union[torch.dtype, str] = "float32",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """

        Args:
            model_name_or_path (str): The HypencoderDualEncoder model to use,
                this should match the model used for encoding the items.
            encoded_item_path (str): The path to the encoded items.
            item_neighbors_path (str): The path to the item neighbors JSONL.
                Should have the keys "item_id" and "neighbors".
            index_path (str): The path to store/load BM25 Index on dataset
            ir_dataset (str): IR dataset to be used
            batch_size (int, optional): The batch size to use for inference.
                Defaults to 100_000.
            device (str, optional): The device to use to store the embeddings
                and to run the model. Defaults to "cuda".
            query_max_length (int, optional): The maximum length of the query.
                Defaults to 32.
            cache_file (Optional[str], optional): If provided, the cache file
                to use for loading the encoded items and item neighbors. If
                the file does not exist, it will be created and the data will
                stored in it. Defaults to None.
            num_entry_points (int, optional): The number of randomly selected
                initial entry points. This is equal to len(initial_candidates).
                Defaults to 10_000.
            ncandidates (int, optional): The number of candidates to explore
                as each step. Defaults to 50.
            max_iter (int, optional): The maximum number of candidate expansion
                steps to do. Defaults to 16.
            early_stop (bool, optional): Whether to stop early if no new
                candidates are added to the queue at a given step. Defaults
                to True.
            dtype (Union[torch.dtype, str], optional): The dtype to use for
                the model and embeddings. Defaults to "float32".
            k1 (float, optional): Term frequency saturation parameter (default: 1.5)
            b: (float, optional): Length normalisation parameter (default: 0.75)
        """

        if isinstance(dtype, str):
            dtype = dtype_lookup(dtype)

        self.dtype = dtype
        self.device = device
        self.batch_size = batch_size
        self.encoded_item_path = encoded_item_path
        self.num_entry_points = num_entry_points
        self.ncandidates = ncandidates
        self.max_iter = max_iter
        self.query_max_length = query_max_length
        self.early_stop = early_stop
        self.k1 = k1
        self.b = b
        self.index_path = index_path
        self.ir_dataset = ir_dataset
        self.index_ref = None
        self.retriever = None

        print(model_name_or_path)
        self.model = (
            HypencoderDualEncoder.from_pretrained(model_name_or_path)
            .to(device, dtype=self.dtype)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if cache_file is not None and os.path.exists(cache_file):
            print(f"Loading from cache {cache_file}")
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)

                self.ids = cache["item_ids"]
                self.encoded_item_embeddings = cache["encoded_item_embeddings"]
                self.item_id_to_index = cache["item_id_to_index"]
                self.item_id_to_content = cache["item_id_to_content"]
                self.item_neighbor_ids = cache["item_neighbor_ids"]
                self.item_id_to_neighbor_indices = cache[
                    "item_id_to_neighbor_indices"
                ]

        else:
            self.encoded_items = load_encoded_items_from_disk(
                encoded_item_path
            )

            self.encoded_item_embeddings = torch.stack(
                [
                    torch.tensor(x.representation)
                    for x in tqdm(
                        self.encoded_items, desc="Item Embeddings to Tensor"
                    )
                ]
            ).to(self.device, dtype=self.dtype)

            self.item_id_to_index = {
                item.id: idx
                for idx, item in tqdm(
                    enumerate(self.encoded_items), desc="Item ID to Index"
                )
            }

            self.ids = [item.id for item in self.encoded_items]

            self.item_id_to_content = {
                item.id: item.text
                for item in tqdm(self.encoded_items, desc="Item ID to Content")
            }

            with JsonlReader(item_neighbors_path) as reader:
                self.item_neighbor_ids = {
                    line["item_id"]: line["neighbors"]
                    for line in tqdm(reader, desc="Loading Item Graph")
                }

            self.item_id_to_neighbor_indices = defaultdict(list)

            for item_id, neighbors in tqdm(
                self.item_neighbor_ids.items(),
                desc="Building Neighbor Indices",
            ):
                self.item_id_to_neighbor_indices[item_id] = [
                    self.item_id_to_index[neighbor] for neighbor in neighbors
                ]


            if cache_file is not None:
                cache_file = Path(cache_file)
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                print(f"Caching to {cache_file}")
                cache_values = {
                    "item_ids": self.ids,
                    "encoded_item_embeddings": self.encoded_item_embeddings,
                    "item_id_to_index": self.item_id_to_index,
                    "item_id_to_content": self.item_id_to_content,
                    "item_neighbor_ids": self.item_neighbor_ids,
                    "item_id_to_neighbor_indices": self.item_id_to_neighbor_indices,
                }

                with open(cache_file, "wb") as f:
                    pickle.dump(cache_values, f)

        # Originally uncommented, set once
        # self._set_entry_points()
                    
        if self._index_exists():
            self._load_index()
        else:
            self.fit(items_from_ir_dataset(ir_dataset))

    def _index_exists(self) -> bool:
        """Check if a PISA index exists at the given path."""
        # PISA indices have these files
        return os.path.exists(os.path.join(self.index_path, "index.pisa"))
      
    def _load_index(self) -> None:
        """Load an existing PISA index from disk."""
        self.index_ref = self.index_path
        self.retriever = pt.pisa.PisaIndex(self.index_path).bm25(
            k1=self.k1,
            b=self.b
        )

    def fit(self, documents: Iterable, overwrite: bool = False) -> 'BM25':
        """
        Preprocess documents and build PISA index.
        
        Args:
            documents: Iterable of Item objects with 'text' and 'id' attributes
            overwrite: If True, rebuild index even if one exists (default: False)
            
        Returns:
            self for method chaining
        """
        # If index exists and we're not overwriting, skip building
        if self._index_exists() and not overwrite:
            print(f"PISA index already exists at {self.index_path}. Loading existing index.")
            if self.retriever is None:
                self._load_index()
            return self
        
        # Convert documents to DataFrame format required by PyTerrier
        docs_data = []
        for item in documents:
            docs_data.append({
                'docno': item.id,
                'text': item.text
            })
        
        df = pd.DataFrame(docs_data)
        
        if len(df) == 0:
            return self
        
        print(f"Building PISA index at {self.index_path}...")

        os.makedirs(self.index_path, exist_ok=True)
        
        # First create a standard PyTerrier index
        temp_index_path = self.index_path + "_temp"
        os.makedirs(temp_index_path, exist_ok=True)
        iter_indexer = pt.IterDictIndexer(
            temp_index_path,
            overwrite=True,
            meta={'docno': 100}
        )
        temp_index_ref = iter_indexer.index(df.to_dict('records'))
        
        # Convert to PISA format
        PisaIndex.from_terrier(
            temp_index_ref,
            self.index_path,
            overwrite=True
        )
        
        # Clean up temporary index (optional)
        try:
            shutil.rmtree(temp_index_path)
        except:
            pass  # If cleanup fails, it's not critical

        # Load the PISA index
        self._load_index()
        
        print(f"PISA index built successfully with {len(df)} documents.")
        
        return self
    

    def query(self, query: str, n: int) -> List[str]:
        """
        Find the n most relevant documents to the query.
        
        Args:
            query: Search query string
            n: Number of top documents to return
            
        Returns:
            List of document IDs for the n most relevant documents
        """
        if self.retriever is None:
            raise ValueError("No index loaded. Call fit() first or provide a valid index_path.")
        
        # Create query DataFrame
        query_df = pd.DataFrame([{'qid': '1', 'query': query}])
        
        # Retrieve with custom num_results
        retriever_n = self.retriever % n  # Set number of results
        results = retriever_n.transform(query_df)
        
        if len(results) == 0:
            return []
        
        # Return document IDs in order
        return results['docno'].tolist()


    def set_parameters(self, num_entry_points, ncandidates, max_iter):
        self.num_entry_points = num_entry_points
        self.ncandidates = ncandidates
        self.max_iter = max_iter
        # self._set_entry_points()

    def _set_entry_points(self):
        random.seed(43)
        self.entry_point_indices = torch.Tensor(
            random.sample(
                range(self.encoded_item_embeddings.shape[0]),
                self.num_entry_points,
            ),
        ).to(self.device, dtype=torch.long)

        self.entry_point_embeddings = self.encoded_item_embeddings[
            self.entry_point_indices
        ]
        self.entry_point_ids = [
            self.ids[idx] for idx in self.entry_point_indices
        ]



    def _set_entry_points_similar(self, query):

        top_item_ids = self.query(query, self.num_entry_points)
        self.entry_point_indices = torch.tensor(
            [self.item_id_to_index[item_id] for item_id in top_item_ids],
            dtype=torch.long,
            device=self.device
        )
        self.entry_point_embeddings = self.encoded_item_embeddings[self.entry_point_indices]
        self.entry_point_ids = top_item_ids

        print(f"Selected {len(self.entry_point_ids)} query-conditioned entry points (BM25)")


    def retrieve(self, query: TextQuery, top_k: int) -> List[Item]:
        tokenized_query = self.tokenizer(
            query.text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.query_max_length,
        ).to(self.device)

        with torch.no_grad():
            query_output = self.model.query_encoder(
                input_ids=tokenized_query["input_ids"],
                attention_mask=tokenized_query["attention_mask"],
            )
            query_model = query_output.representation

        final_queue = PriorityQueue(maxsize=top_k)

        # Not in original code, entry points set once for Graph Retriever usually
        self._set_entry_points_similar(query.text)

        candidates = [x for x in self.entry_point_ids]
        explored = set(candidates)

        curr_iter = 0
        while curr_iter < self.max_iter:
            candidate_embeddings = self.encoded_item_embeddings[
                [self.item_id_to_index[x] for x in candidates]
            ]
            candidate_embeddings = candidate_embeddings.unsqueeze(0)

            similarity_matrix = query_model(candidate_embeddings).view(-1)

            ncandidates = min(
                max(self.ncandidates, top_k), similarity_matrix.shape[0]
            )
            values, indices = torch.topk(similarity_matrix, ncandidates, dim=0)

            indices = indices.view(-1).cpu()
            values  = values.view(-1).cpu()

            prev_candidates = copy.deepcopy(candidates)
            candidates = []

            added_candidates = 0
            added_to_queue = 0
            for i, idx in enumerate(indices):
                idx = idx.item()

                item_id = prev_candidates[idx]
                score = similarity_matrix[idx].item()

                if final_queue.full():
                    # If queue is full, only add item to queue if score is
                    # greater than the minimum score in the queue
                    current_min = final_queue.get()
                    if score > current_min[0]:
                        added_to_queue += 1
                        final_queue.put((score, item_id))
                    else:
                        final_queue.put(current_min)
                else:
                    # If queue is not full add item to queue regardless of
                    # score
                    added_to_queue += 1
                    final_queue.put((score, item_id))

                if i < self.ncandidates:
                    for neighbor in self.item_neighbor_ids[item_id]:
                        if neighbor in explored:
                            continue

                        candidates.append(neighbor)
                        explored.add(neighbor)
                        added_candidates += 1

            if added_to_queue == 0 and self.early_stop:
                break

            curr_iter += 1

        items = []
        while not final_queue.empty():
            score, item_id = final_queue.get()
            items.append(
                Item(
                    text=self.item_id_to_content[item_id],
                    id=item_id,
                    score=score,
                    type="hypecoder_graph_retriever",
                )
            )

        return sorted(items, key=lambda x: x.score, reverse=True)



if __name__ == "__main__":
    fire.Fire(do_retrieval)