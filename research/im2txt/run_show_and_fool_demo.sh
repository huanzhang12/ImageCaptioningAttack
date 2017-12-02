#!/bin/bash
## Copyright (C) 2017, Hongge Chen <chenhg@mit.edu>
## Copyright (C) 2017, Huan Zhang <ecezhang@ucdavis.edu>
# Use this bash code to do a quick run of show-and-fool demo on a single image 
if [ "$#" -le 2 ]; then
    echo "Usage:" $0 "<offset> <number_attacks> <output_directory> [option 1] [option 2] ..."
    echo "Example:" $0 "0 100 debug_dir --use_keywords --input_feed=\"dog cat\""
    exit 1
fi

ATTACK_FILEPATH=$1
TARGET_FILEPATH=$2
OUTPUT_DIR=$3
GPU_number=$4

shift
shift
shift

PARAMS=()

for PARAM in "$@"
do
    PARAMS+=("${PARAM}")
done

echo Additional parameters: ${PARAMS[@]}

mkdir -p ${OUTPUT_DIR}

# if [ "$(arch)" = "x86_64" ]; then
#    source activate tensorflow
# fi
# if [ "$(arch)" = "ppc64le" ]; then
#    source ${HOME}/.bashrc
#    source activate huanpy36
# fi
source activate huanpy36
VOCAB_FILE="im2txt/pretrained/word_counts.txt"
CHECKPOINT_PATH="im2txt/pretrained/model2.ckpt-2000000"
CAPTION_FILE="/home/hongge/im2txt/EasyCocoEval/coco-caption/annotations/captions_val2014.json"
bazel build -c opt im2txt/show_and_fool_demo
set -x
CUDA_VISIBLE_DEVICES=${GPU_number} bazel-bin/im2txt/show_and_fool_demo --checkpoint_path=${CHECKPOINT_PATH} --vocab_file=${VOCAB_FILE}   --attack_filepath=${ATTACK_FILEPATH} --target_filepath=${TARGET_FILEPATH} --use_logits=True --result_directory="${OUTPUT_DIR}" --norm="l2" -seed=8 --C=1 "${PARAMS[@]}" 2>&1|tee ${OUTPUT_DIR}/log.txt
set +x
rm -r bazel-*
source deactivate

