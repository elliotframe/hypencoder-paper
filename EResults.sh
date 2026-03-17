export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/msmarco"
export RETRIEVAL_DIR="retrievals/exhaustive/in-domain/2019"
export IR_DATASET_NAME="msmarco-passage/trec-dl-2019/judged"
export ITEM_NEIGHBOR_GRAPH="graphs/msmarco"

python hypencoder_cb/work/diff_arguments_final.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--item_neighbors_path=$ITEM_NEIGHBOR_GRAPH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--dtype=bf16

