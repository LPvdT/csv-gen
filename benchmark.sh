#!/usr/bin/env bash
set -eu -o pipefail

if [ -z "$(command -v hyperfine)" ]; then
	echo "hyperfine not found"
	echo "Please install hyperfine: https://github.com/sharkdp/hyperfine"
fi

##################
# Config csv-gen #
##################
# FILE_SIZE_BYTES=26843545600            # 25 * 1024**3 (25 GB)
# FILE_SIZE_BYTES=1073741824             # 1 * 1024**3 (1 GB)
FILE_SIZE_BYTES=104857600              # 100 * 1024**2 (100 MB)
NUM_CPUS="${NUM_CPUS:-$(nproc --all)}" # Use all available CPUs by default
ALGORITHM="${ALGORITHM:-numpy}"        # Use numpy by default

####################
# Config hyperfine #
####################
WARMUP_RUNS="${WARMUP_RUNS:=0}" # No warmup runs by default
RUNS="${RUNS:=0}"               # Force the amount of runs, by default hyperfine will run at least 10 runs and measure for at least 3 seconds
if [ "$RUNS" -gt 0 ]; then
	runs="--runs $RUNS"
else
	runs=""
fi
USE_PARAMETER_LIST="${USE_PARAMETER_LIST:-false}" # Do not use parameter list by default
ALGORITHM_LIST="numpy,faker"                      # Algorithm list for --parameter-list
if ! $USE_PARAMETER_LIST; then
	ALGORITHM_LIST=""
fi

# Temporary file for CSV data
tmpfile="$(mktemp).csv"

# Create benchmarks directory
output_dir="$(dirname "$(realpath "$0")")/src/csv_gen/logs/benchmarks"
mkdir -p "$output_dir"

# Generate a time-based UUID
uuid=$(uuidgen --time)

# Run benchmark
if $USE_PARAMETER_LIST; then
	echo "CPUs: $NUM_CPUS, $FILE_SIZE_BYTES bytes, algorithms: $ALGORITHM_LIST, generating: $tmpfile"
	hyperfine --command-name "Algorithms: $ALGORITHM_LIST - CPUs: $NUM_CPUS - Bytes: $FILE_SIZE_BYTES" \
		--warmup "$WARMUP_RUNS" \
		$runs \
		--cleanup "rm $tmpfile" \
		--style full \
		--shell "bash" \
		--export-json "${output_dir}/${ALGORITHM}_${NUM_CPUS}_${FILE_SIZE_BYTES}_${uuid}.json" \
		--parameter-list algorithm "$ALGORITHM_LIST" \
		"csv-gen generate --file-size-bytes $FILE_SIZE_BYTES --cpus $NUM_CPUS --algorithm {algorithm} $tmpfile"
else
	echo "CPUs: $NUM_CPUS, $FILE_SIZE_BYTES bytes, algorithm: $ALGORITHM, generating: $tmpfile"
	hyperfine --command-name "Algorithm: $ALGORITHM - CPUs: $NUM_CPUS - Bytes: $FILE_SIZE_BYTES" \
		--warmup "$WARMUP_RUNS" \
		$runs \
		--cleanup "rm $tmpfile" \
		--style full \
		--shell "bash" \
		--export-json "${output_dir}/${ALGORITHM}_${NUM_CPUS}_${FILE_SIZE_BYTES}_${uuid}.json" \
		"csv-gen generate --file-size-bytes $FILE_SIZE_BYTES --cpus $NUM_CPUS --algorithm $ALGORITHM $tmpfile"
fi
