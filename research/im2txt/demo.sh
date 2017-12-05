#!/bin/bash
## Copyright (C) 2017, Hongge Chen <chenhg@mit.edu>
## Copyright (C) 2017, Huan Zhang <ecezhang@ucdavis.edu>
# Use this bash code to do a quick run of show-and-fool demo on a single image 
if [ "$#" -le 2 ]; then
    echo "Usage:" $0 "<image_to_attack> <image_of_target_sentence> <result_output_directory> [additional option 1] [additional option 2] ..."
    echo "Example:" $0 "examples/image1.png examples/image2.png result_dir"
    echo "Example:" $0 "examples/image1.png examples/image2.png result_dir --use_keywords --input_feed=\"dog frisbee\""
    exit 1
fi

ATTACK_FILEPATH=$1
TARGET_FILEPATH=$2
OUTPUT_DIR=$3

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

VOCAB_FILE="im2txt/pretrained/word_counts.txt"
CHECKPOINT_PATH="im2txt/pretrained/model2.ckpt-2000000"
bazel build -c opt im2txt/show_and_fool_demo
set -x
bazel-bin/im2txt/show_and_fool_demo --checkpoint_path=${CHECKPOINT_PATH} --vocab_file=${VOCAB_FILE}   --attack_filepath=${ATTACK_FILEPATH} --target_filepath=${TARGET_FILEPATH} --use_logits=True --result_directory="${OUTPUT_DIR}" --norm="l2" -seed=8 --C=1 "${PARAMS[@]}" 2>&1|tee ${OUTPUT_DIR}/log.txt
set +x

