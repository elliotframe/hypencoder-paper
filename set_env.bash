# Bash-only environment setup
# Must be sourced via:
# source set_env.bash

[[ "${BASH_SOURCE[0]}" != "$0" ]] || {
  echo "ERROR: source this file with [source set_env.bash], do not execute it" >&2
  return 1
}

set -euo pipefail

export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export ENCODING_PATH="~/nfs/hypencoder-paper/encodings/msmarco"
export RETRIEVAL_DIR="retrievals/[name of new file]"
export ITEM_NEIGHBOR_GRAPH="graphs/msmarco"

# export IR_DATASET_NAME="msmarco-passage/trec-dl-2020/judged"
export IR_DATASET_NAME="msmarco-passage/trec-dl-2019/judged"
# export IR_DATASET_NAME="msmarco-passage/dev/small"

echo "Arguments for retrieval set as environment variables."