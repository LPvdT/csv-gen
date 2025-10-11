#!/usr/bin/env bash
set -eu -o pipefail

if ! [ command -v hyperfine ]; then
	echo "hyperfine not found"
	echo "Please install hyperfine: https://github.com/sharkdp/hyperfine"
fi

##################
# Config csv-gen #
##################
# FILE_SIZE_BYTES=26843545600            # 25 * 1024**3 (25 GB)
FILE_SIZE_BYTES=5368709120             # 5 * 1024**3 (5 GB)
NUM_CPUS="${NUM_CPUS:-$(nproc --all)}" # Use all available CPUs by default
ALGORITHM="${ALGORITHM:-numpy}"        # Use numpy by default

####################
# Config hyperfine #
####################
WARMUP_RUNS="${WARMUP_RUNS:0}"                    # No warmup runs by default
RUNS="${RUNS:5}"                                  # 5 runs by default
USE_PARAMETER_LIST="${USE_PARAMETER_LIST:-false}" # Do not use parameter list by default
ALGORITHM_LIST="numpy,faker"                      # Algorithm list for --parameter-list
if ! $USE_PARAMETER_LIST; then
	ALGORITHM_LIST=""
fi

# Temporary file for CSV data
tmpfile="$(mktemp).csv"

# Run benchmark
if $USE_PARAMETER_LIST; then
    echo "CPUs: $NUM_CPUS, $FILE_SIZE_BYTES bytes, algorithms: $ALGORITHM_LIST, generating: $tmpfile"
	hyperfine --name "Benchmark $ALGORITHM" \
		--warmup "$WARMUP_RUNS" \
		--min-runs "$RUNS" \
		--max-runs "$RUNS" \
		--cleanup "rm $tmpfile" \
		--style full \
		--shell "bash" \
		--export-json "$ALGORITHM_$NUM_CPUS_$FILE_SIZE_BYTES.json" \
		--parameter-list algorithm "$ALGORITHM_LIST" \
		"./csv-gen generate --file-size-bytes $FILE_SIZE_BYTES --cpus $NUM_CPUS --algorithm {algorithm} $tmpfile"
else
    echo "CPUs: $NUM_CPUS, $FILE_SIZE_BYTES bytes, algorithm: $ALGORITHM, generating: $tmpfile"
	hyperfine --name "Benchmark $ALGORITHM" \
		--warmup "$WARMUP_RUNS" \
		--min-runs "$RUNS" \
		--max-runs "$RUNS" \
		--cleanup "rm $tmpfile" \
		--style full \
		--shell "bash" \
		--export-json "$ALGORITHM_$NUM_CPUS_$FILE_SIZE_BYTES.json" \
		"./csv-gen generate --file-size-bytes $FILE_SIZE_BYTES --cpus $NUM_CPUS --algorithm $ALGORITHM $tmpfile"
fi
