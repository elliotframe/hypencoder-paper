from hypencoder_cb.inference.retrieve import do_retrieval_shared
from hypencoder_cb.inference.shared import (
    BaseRetriever,
    Item,
    TextQuery,
    load_encoded_items_from_disk,
)
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder
from hypencoder_cb.utils.torch_utils import dtype_lookup
from transformers import AutoTokenizer
from typing import List, Optional, Union
from pyterrier_pisa import PisaIndex
from tqdm import tqdm
import pyterrier as pt
import pickle
import torch
import json
import csv
import fire
import gc
import os


# (name, encoding dir, ir_dataset, index/cache name)
DATASETS = [
    ("trecdl2019", "msmarco", "msmarco-passage/trec-dl-2019/judged", "msmarco"),
    ("trecdl2020", "msmarco", "msmarco-passage/trec-dl-2020/judged", "msmarco"),
    ("msmarcodev", "msmarco", "msmarco-passage/dev/small", "msmarco"),
    ("fiqaTest", "fiqaTest", "beir/fiqa/test", "fiqaTest"),
    ("treccovid", "treccovid", "beir/trec-covid", "treccovid"),
    ("nfcorpus", "nfcorpusTest", "beir/nfcorpus/test", "nfcorpusTest"),
    ("dbpedia", "dpedia-entityTest", "beir/dbpedia-entity/test", "dbpedia"),
    ("touche", "touche2020v2", "beir/webis-touche2020/v2", "touche"),
]


class HypencoderBM25Reranker(BaseRetriever):
    """Retrieves the top `num_bm25` items with BM25 and reranks them with
    Hypencoder. No graph exploration is done."""

    def __init__(
        self,
        model_name_or_path: str,
        encoded_item_path: str,
        index_path: str,
        ir_dataset: str,
        num_bm25: int = 1000,
        device: str = "cuda",
        query_max_length: int = 64,
        cache_file: Optional[str] = None,
        dtype: Union[torch.dtype, str] = "float32",
        k1: float = 1.5,
        b: float = 0.75,
        model=None,
        tokenizer=None,
    ) -> None:
        if isinstance(dtype, str):
            dtype = dtype_lookup(dtype)

        self.dtype = dtype
        self.device = device
        self.num_bm25 = num_bm25
        self.query_max_length = query_max_length

        self.model = model if model is not None else (
            HypencoderDualEncoder.from_pretrained(model_name_or_path)
            .to(device, dtype=self.dtype)
            .eval()
        )
        self.tokenizer = tokenizer if tokenizer is not None else (
            AutoTokenizer.from_pretrained(model_name_or_path)
        )

        if cache_file is not None and os.path.exists(cache_file):
            # Reuse the graph retriever's cache, only the embeddings are used
            print(f"Loading from cache {cache_file}")
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            self.encoded_item_embeddings = cache["encoded_item_embeddings"].to(
                self.device, dtype=self.dtype
            )
            self.item_id_to_index = cache["item_id_to_index"]
            self.item_id_to_content = cache["item_id_to_content"]
            del cache
        else:
            encoded_items = load_encoded_items_from_disk(encoded_item_path)
            self.encoded_item_embeddings = torch.stack(
                [
                    torch.tensor(x.representation)
                    for x in tqdm(encoded_items, desc="Item Embeddings to Tensor")
                ]
            ).to(self.device, dtype=self.dtype)
            self.item_id_to_index = {
                item.id: idx for idx, item in enumerate(encoded_items)
            }
            self.item_id_to_content = {
                item.id: item.text for item in encoded_items
            }
            del encoded_items

        if not pt.started():
            pt.init()
        self.index = PisaIndex(index_path)
        if not os.path.exists(os.path.join(index_path, "fwd.documents")):
            dataset = pt.get_dataset(f"irds:{ir_dataset}")
            self.index.index(dataset.get_corpus_iter())
        else:
            print(f"PISA index already exists at {index_path}. Loading existing index.")

        self.bm25_retriever = self.index.bm25(k1=k1, b=b, num_results=self.num_bm25)

    def retrieve(self, query: TextQuery, top_k: int) -> List[Item]:
        bm25_ids = self.bm25_retriever.transform(
            pt.new.queries(query.text)
        )["docno"].tolist()
        bm25_ids = [x for x in bm25_ids if x in self.item_id_to_index]

        if len(bm25_ids) == 0:
            return []

        tokenized_query = self.tokenizer(
            query.text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.query_max_length,
        ).to(self.device)

        with torch.no_grad():
            query_model = self.model.query_encoder(
                input_ids=tokenized_query["input_ids"],
                attention_mask=tokenized_query["attention_mask"],
            ).representation

            candidate_embeddings = self.encoded_item_embeddings[
                [self.item_id_to_index[x] for x in bm25_ids]
            ].unsqueeze(0)
            scores = query_model(candidate_embeddings).view(-1)

        values, indices = torch.topk(scores, min(top_k, scores.shape[0]))

        return [
            Item(
                text=self.item_id_to_content[bm25_ids[idx]],
                id=bm25_ids[idx],
                score=score,
                type="hypencoder_bm25_reranker",
            )
            for score, idx in zip(values.float().cpu().tolist(), indices.cpu().tolist())
        ]


def main(
    model_name_or_path: str = "jfkback/hypencoder.6_layer",
    encoding_root: str = "~/nfs/hypencoder-paper/encodings",
    ret_name: str = "bm25_rerank",
    num_bm25: int = 1000,
    query_max_length: int = 64,
    dtype: str = "fp16",
    datasets: Optional[List[str]] = None,
) -> None:
    """Runs BM25 top-`num_bm25` + Hypencoder reranking on each dataset.

    Args:
        datasets: Optional subset of dataset names from DATASETS to run.
    """
    selected = [d for d in DATASETS if datasets is None or d[0] in datasets]

    model = (
        HypencoderDualEncoder.from_pretrained(model_name_or_path)
        .to("cuda", dtype=dtype_lookup(dtype))
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    retriever = None
    loaded_encoding = None
    for name, encoding, ir_dataset_name, index_name in selected:
        # Datasets sharing a corpus (msmarco) reuse the loaded retriever
        if encoding != loaded_encoding:
            retriever = None
            gc.collect()
            torch.cuda.empty_cache()
            retriever = HypencoderBM25Reranker(
                model_name_or_path=model_name_or_path,
                encoded_item_path=os.path.expanduser(f"{encoding_root}/{encoding}"),
                index_path=os.path.abspath(f"BM25index/{index_name}"),
                ir_dataset=ir_dataset_name,
                num_bm25=num_bm25,
                query_max_length=query_max_length,
                cache_file=f"cache/{index_name}",
                dtype=dtype,
                model=model,
                tokenizer=tokenizer,
            )
            loaded_encoding = encoding

        print(f"Starting retrieval: dataset={name}, num_bm25={num_bm25}")

        do_retrieval_shared(
            retriever=retriever,
            retriever_cls=HypencoderBM25Reranker,
            retriever_kwargs={},
            output_dir=f"retrievals/{ret_name}/{name}",
            ir_dataset_name=ir_dataset_name,
            top_k=num_bm25,
            metric_dir=f"metrics/{ret_name}/{name}",
        )

    print("Combining metrics.")
    rows = []
    for name, _, _, _ in selected:
        with open(f"metrics/{ret_name}/{name}/aggregated_metrics.json", "r") as g:
            metrics = json.load(g)
        with open(f"metrics/{ret_name}/{name}/timing.json", "r") as g:
            timing = json.load(g)

        row = {"Dataset": name, "NumBM25": num_bm25}
        row.update(metrics)
        row["num_queries"] = timing["num_queries"]
        row["time"] = timing["time"]
        rows.append(row)

    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with open(f"metrics/{ret_name}/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("Done :)")


if __name__ == "__main__":
    fire.Fire(main)
