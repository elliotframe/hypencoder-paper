
# Exhaustive 2019
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/msmarco"
export RETRIEVAL_DIR="retrievals/exhaustive/in-domain/2019"
export IR_DATASET_NAME="msmarco-passage/trec-dl-2019/judged"
export METRIC_NAME="metrics/exhaustive/in-domain/2019"

python hypencoder_cb/inference/retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--query_max_length=64 \
--metric_dir=$METRIC_NAME

python hypencoder_cb/work/add_timings_exhaustive.py \
--ret_name="exhaustive/in-domain/2019"


# Exhaustive 2020
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/msmarco"
export RETRIEVAL_DIR="retrievals/exhaustive/in-domain/2020"
export IR_DATASET_NAME="msmarco-passage/trec-dl-2020/judged"
export METRIC_NAME="metrics/exhaustive/in-domain/2020"

python hypencoder_cb/inference/retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--query_max_length=64 \
--metric_dir=$METRIC_NAME

python hypencoder_cb/work/add_timings_exhaustive.py \
--ret_name="exhaustive/in-domain/2020"

# Exhaustive dev
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/msmarco"
export RETRIEVAL_DIR="retrievals/exhaustive/in-domain/dev"
export IR_DATASET_NAME="msmarco-passage/dev/small"
export METRIC_NAME="metrics/exhaustive/in-domain/dev"

python hypencoder_cb/inference/retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--query_max_length=64 \
--metric_dir=$METRIC_NAME

python hypencoder_cb/work/add_timings_exhaustive.py \
--ret_name="exhuastive/in-domain/dev"



# Exhaustive fiqaTest
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/fiqaTest"
export RETRIEVAL_DIR="retrievals/exhaustive/out-domain/fiqaTest"
export IR_DATASET_NAME="beir/fiqa/test"
export METRIC_NAME="metrics/exhaustive/out-domain/fiqaTest"

python hypencoder_cb/inference/retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--query_max_length=512 \
--metric_dir=$METRIC_NAME

python hypencoder_cb/work/add_timings_exhaustive.py \
--ret_name="exhaustive/out-domain/fiqaTest"


# Exhaustive covid
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/treccovid"
export RETRIEVAL_DIR="retrievals/exhaustive/out-domain/treccovid"
export IR_DATASET_NAME="beir/trec-covid"
export METRIC_NAME="metrics/exhaustive/out-domain/treccovid"

python hypencoder_cb/inference/retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--query_max_length=512 \
--metric_dir=$METRIC_NAME

python hypencoder_cb/work/add_timings_exhaustive.py \
--ret_name="exhaustive/out-domain/treccovid"

# Exhaustive nfcorpus
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/nfcorpusTest"
export RETRIEVAL_DIR="retrievals/exhaustive/out-domain/nfcorpus"
export IR_DATASET_NAME="beir/nfcorpus/test"
export METRIC_NAME="metrics/exhaustive/out-domain/nfcorpus"

python hypencoder_cb/inference/retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--query_max_length=512 \
--metric_dir=$METRIC_NAME

python hypencoder_cb/work/add_timings_exhaustive.py \
--ret_name="exhaustive/out-domain/nfcorpus"

# Exhaustive dpedia
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/dpedia-entityTest"
export RETRIEVAL_DIR="retrievals/exhaustive/out-domain/dpedia"
export IR_DATASET_NAME="beir/dbpedia-entity/test"
export METRIC_NAME="metrics/exhaustive/out-domain/dpedia"

python hypencoder_cb/inference/retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--query_max_length=512 \
--metric_dir=$METRIC_NAME

python hypencoder_cb/work/add_timings_exhaustive.py \
--ret_name="exhaustive/out-domain/dpedia"

# Exhaustive touche
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/touche2020v2"
export RETRIEVAL_DIR="retrievals/exhaustive/out-domain/touche"
export IR_DATASET_NAME="beir/webis-touche2020/v2"
export METRIC_NAME="metrics/exhaustive/out-domain/touche"

python hypencoder_cb/inference/retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--query_max_length=512 \
--metric_dir=$METRIC_NAME

python hypencoder_cb/work/add_timings_exhaustive.py \
--ret_name="exhaustive/out-domain/touche"


# Exhaustive 2019 2
export MODEL_NAME_OR_PATH="jfkback/hypencoder.2_layer"
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/msmarco"
export RETRIEVAL_DIR="retrievals/exhaustive/layers/2/2019"
export IR_DATASET_NAME="msmarco-passage/trec-dl-2019/judged"
export METRIC_NAME="metrics/exhaustive/layers/2/2019"

python hypencoder_cb/inference/retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--query_max_length=64 \
--metric_dir=$METRIC_NAME

python hypencoder_cb/work/add_timings_exhaustive.py \
--ret_name="exhaustive/layers/2/2019"

# Exhaustive 2019 4
export MODEL_NAME_OR_PATH="jfkback/hypencoder.4_layer"
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/msmarco"
export RETRIEVAL_DIR="retrievals/exhaustive/layers/4/2019"
export IR_DATASET_NAME="msmarco-passage/trec-dl-2019/judged"
export METRIC_NAME="metrics/exhaustive/layers/4/2019"

python hypencoder_cb/inference/retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--query_max_length=64 \
--metric_dir=$METRIC_NAME

python hypencoder_cb/work/add_timings_exhaustive.py \
--ret_name="exhaustive/layers/4/2019"

# Exhaustive 2019 8
export MODEL_NAME_OR_PATH="jfkback/hypencoder.8_layer"
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/msmarco"
export RETRIEVAL_DIR="retrievals/exhaustive/layers/8/2019"
export IR_DATASET_NAME="msmarco-passage/trec-dl-2019/judged"
export METRIC_NAME="metrics/exhaustive/layers/8/2019"

python hypencoder_cb/inference/retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--query_max_length=64 \
--metric_dir=$METRIC_NAME

python hypencoder_cb/work/add_timings_exhaustive.py \
--ret_name="exhaustive/layers/8/2019"

# Exhaustive 2020 2
export MODEL_NAME_OR_PATH="jfkback/hypencoder.2_layer"
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/msmarco"
export RETRIEVAL_DIR="retrievals/exhaustive/layers/2/2020"
export IR_DATASET_NAME="msmarco-passage/trec-dl-2020/judged"
export METRIC_NAME="metrics/exhaustive/layers/2/2020"

python hypencoder_cb/inference/retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--query_max_length=64 \
--metric_dir=$METRIC_NAME

python hypencoder_cb/work/add_timings_exhaustive.py \
--ret_name="exhaustive/layers/2/2020"

# Exhaustive 2020 4
export MODEL_NAME_OR_PATH="jfkback/hypencoder.4_layer"
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/msmarco"
export RETRIEVAL_DIR="retrievals/exhaustive/layers/4/2020"
export IR_DATASET_NAME="msmarco-passage/trec-dl-2020/judged"
export METRIC_NAME="metrics/exhaustive/layers/4/2020"

python hypencoder_cb/inference/retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--query_max_length=64 \
--metric_dir=$METRIC_NAME

python hypencoder_cb/work/add_timings_exhaustive.py \
--ret_name="exhaustive/layers/4/2020"

# Exhaustive 2020 8
export MODEL_NAME_OR_PATH="jfkback/hypencoder.8_layer"
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/msmarco"
export RETRIEVAL_DIR="retrievals/exhaustive/layers/8/2020"
export IR_DATASET_NAME="msmarco-passage/trec-dl-2020/judged"
export METRIC_NAME="metrics/exhaustive/layers/8/2020"

python hypencoder_cb/inference/retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--query_max_length=64 \
--metric_dir=$METRIC_NAME

python hypencoder_cb/work/add_timings_exhaustive.py \
--ret_name="exhaustive/layers/8/2020"




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