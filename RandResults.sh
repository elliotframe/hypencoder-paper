# trecdl19
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
# export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/msmarco"
# export RETRIEVAL_DIR="retrievals/September/3090/original/trecdl2019"
# export ITEM_NEIGHBOR_GRAPH="graphs/msmarco"
# export IR_DATASET_NAME="msmarco-passage/trec-dl-2019/judged"

# python hypencoder_cb/work/random_seeds.py \
# --model_name_or_path=$MODEL_NAME_OR_PATH \
# --encoded_item_path=$ENCODING_PATH \
# --item_neighbors_path=$ITEM_NEIGHBOR_GRAPH \
# --output_dir=$RETRIEVAL_DIR \
# --ir_dataset_name=$IR_DATASET_NAME \
# --dtype=fp16 \

# # trecdl20
# export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/msmarco"
# export RETRIEVAL_DIR="retrievals/September/3090/original/trecdl2020"
# export ITEM_NEIGHBOR_GRAPH="graphs/msmarco"
# export IR_DATASET_NAME="msmarco-passage/trec-dl-2020/judged"

# python hypencoder_cb/work/random_seeds.py \
# --model_name_or_path=$MODEL_NAME_OR_PATH \
# --encoded_item_path=$ENCODING_PATH \
# --item_neighbors_path=$ITEM_NEIGHBOR_GRAPH \
# --output_dir=$RETRIEVAL_DIR \
# --ir_dataset_name=$IR_DATASET_NAME \
# --dtype=fp16 \


# # msmarco dev
# export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/msmarco"
# export RETRIEVAL_DIR="retrievals/September/3090/original/msmarcodev"
# export ITEM_NEIGHBOR_GRAPH="graphs/msmarco"
# export IR_DATASET_NAME="msmarco-passage/dev/small"

# python hypencoder_cb/work/random_seeds.py \
# --model_name_or_path=$MODEL_NAME_OR_PATH \
# --encoded_item_path=$ENCODING_PATH \
# --item_neighbors_path=$ITEM_NEIGHBOR_GRAPH \
# --output_dir=$RETRIEVAL_DIR \
# --ir_dataset_name=$IR_DATASET_NAME \
# --dtype=fp16 \


# fiqaTest
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/fiqaTest"
export RETRIEVAL_DIR="retrievals/September/3090/original/fiqaTest"
export IR_DATASET_NAME="beir/fiqa/test"
export ITEM_NEIGHBOR_GRAPH="graphs/fiqaTest"

python hypencoder_cb/work/random_seeds.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--item_neighbors_path=$ITEM_NEIGHBOR_GRAPH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--dtype=fp16 \

# treccovid
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/treccovid"
export RETRIEVAL_DIR="retrievals/September/3090/original/treccovid"
export IR_DATASET_NAME="beir/trec-covid"
export ITEM_NEIGHBOR_GRAPH="graphs/treccovid"

python hypencoder_cb/work/random_seeds.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--item_neighbors_path=$ITEM_NEIGHBOR_GRAPH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--dtype=fp16 \

# nfcorpusTest
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/nfcorpusTest"
export RETRIEVAL_DIR="retrievals/September/3090/original/nfcorpus"
export IR_DATASET_NAME="beir/nfcorpus/test"
export ITEM_NEIGHBOR_GRAPH="graphs/nfcorpusTest"

python hypencoder_cb/work/random_seeds.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--item_neighbors_path=$ITEM_NEIGHBOR_GRAPH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--dtype=fp16 \

# dbpedia
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/dpedia-entityTest"
export RETRIEVAL_DIR="retrievals/September/3090/original/dbpedia"
export IR_DATASET_NAME="beir/dbpedia-entity/test"
export ITEM_NEIGHBOR_GRAPH="graphs/dbpedia"

python hypencoder_cb/work/random_seeds.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--item_neighbors_path=$ITEM_NEIGHBOR_GRAPH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--dtype=fp16 \

# touche
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/touche2020v2"
export RETRIEVAL_DIR="retrievals/September/3090/original/touche"
export IR_DATASET_NAME="beir/webis-touche2020/v2"
export ITEM_NEIGHBOR_GRAPH="graphs/touche"

python hypencoder_cb/work/random_seeds.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--item_neighbors_path=$ITEM_NEIGHBOR_GRAPH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--dtype=fp16 \





