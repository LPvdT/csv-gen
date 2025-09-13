# CSV Gen

CSV Gen is a Python project that provides a command-line interface (`CLI`) to generate large CSV files using two different algorithms:

1. ✅ NumPy based
2. ❌ Pure Python _(deprecated)_

## Installation

To install the project, you can use `uv`:

```shell
uv sync --managed-python --all-groups --compile-bytecode
```

## Usage

You can use the following command to generate a large CSV file:

```shell
csv-gen generate -s SIZE_BYTES -w NUM_CPUS [FILENAME]
```

Where:

- `SIZE_BYTES`: The size of the generated file in bytes _(default is a gigabyte: `1 * 1024**3`)_
- `NUM_CPUS`: The number of CPU cores to use for generation _(uses **all** cores when not provided)_
- `[FILENAME]`: The name of the file to generate _(default is `bigfile.csv`)_

Or simply run the following to display the **help menu**:

```shell
csv-gen generate --help
```

### Example

For example, to generate a 25 GB CSV file (`25 * 1024**3 = 26843545600`) called `output.csv`, using 8 CPU cores, you can use:

```shell
# The verbose version
csv-gen generate --file-size-gb 26843545600 --cpus 8 output.csv

# Or the short version
csv-gen generate -s 26843545600 -w 8 output.csv
```

To generate a 1 GB CSV file, called `bigfile.csv`, utilising all available CPU cores, you can simply call the default command:

```shell
csv-gen generate
```

## License

This project is licensed under the terms of the MIT license.
