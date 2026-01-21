# Must be sourced via:
# source set_env.bash <retrieval_name>

[[ "${BASH_SOURCE[0]}" != "$0" ]] || {
  echo "ERROR: source this file with [source set_env.bash], do not execute it" >&2
  return 1
}

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "ERROR: retrieval name required" >&2
  return 1
fi

export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/msmarco"
export RETRIEVAL_DIR="retrievals/$1"
export ITEM_NEIGHBOR_GRAPH="graphs/msmarco"

# export IR_DATASET_NAME="msmarco-passage/trec-dl-2020/judged"
export IR_DATASET_NAME="msmarco-passage/trec-dl-2019/judged"
# export IR_DATASET_NAME="msmarco-passage/dev/small"

echo "Retrieval environment set:"
echo "  MODEL_NAME_OR_PATH=$MODEL_NAME_OR_PATH"
echo "  ENCODING_PATH=$ENCODING_PATH"
echo "  RETRIEVAL_DIR=$RETRIEVAL_DIR"
echo "  ITEM_NEIGHBOR_GRAPH=$ITEM_NEIGHBOR_GRAPH"