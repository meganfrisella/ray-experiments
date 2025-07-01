#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

debug=false
while getopts "d" opt; do
	case $opt in
	d) debug=true ;;
	*)
		echo "Usage: $0 [-d]" >&2
		echo "  -d    Enable debug mode"
		exit 1
		;;
	esac
done

export TZ="America/Los_Angeles"
timestamp=$(date '+%Y%m%d_%H%M%S')

export RAY_DEDUP_LOGS=0
export CUDA_VISIBLE_DEVICES=2,3

output_path=results/mfris/
mkdir -p $output_path
rm -r ${output_path}/*.csv
rm -f ${output_path}/*.json
echo "Running $output_path..."

rm my_trace.nsys-rep

nproc_per_node=2
batch_size=256
seq_len=32
num_microbatches=4
num_iters=10
model=LLAMA_3B

# RAY_CGRAPH_VISUALIZE_SCHEDULE=1 \
torchrun --nnodes 1 --nproc-per-node $nproc_per_node train.py \
  --num-iters $num_iters \
	--seq-len $seq_len \
	--batch-size $batch_size \
  --num-microbatches $num_microbatches \
  --model $model \
	--output-path $output_path \
	--timestamp $timestamp
	#>$log_file 2>&1
# --save-model \
status=$?

if $debug; then
	code $output_path/${timestamp}.log
fi

if [ $status -ne 0 ]; then
	echo -e "${RED}ER${NC}"
	exit 1
fi

compare_files() {
	local file1="$1"
	local file2="$2"

	if [ ! -f "$file1" ]; then
		echo -e "${RED}Error: File '$file1' does not exist${NC}"
		exit 1
	fi
	if [ ! -f "$file2" ]; then
		echo -e "${RED}Error: File '$file2' does not exist${NC}"
		exit 1
	fi

	if ! diff "$file1" "$file2"; then
		echo -e "${RED}ER${NC}"
		if $debug; then
			code "$file1"
			code "$file2"
		fi
		exit 1
	fi
}

# file1="${output_path}/${timestamp}_model_0.log"
# file2="${output_path}/${timestamp}_model_1.log"
# compare_files "$file1" "$file2"

echo -e "${GREEN}AC${NC}"
