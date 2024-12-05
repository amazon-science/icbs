Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0

# CSA - Cardinality Simulated Annealing

## Compiling

### Instructions

Create a local build folder and compile the solver:

```
mkdir build
cd build
cmake ..
cmake --build . -j8
```

The compiled solver `sa-card` should then be in `build/src/`. To use `sa-card` with
iCBS, copy it to `src/icbs`.

### Testing

You can test the build like so:

```
make test
```

or by running the compiled tests at `build/src/sa-card-test`.

## Running the solver

Running the solver with no arguments:

```
./sa-card
```

provides a list of the arguments.

A toy test instance `test_instance.ubqp` with five variables is provided in the same
directory as this README. To solve this instance with a cardinality (number of ones) of 3, execute
the following from the `build/src/` directory:

```
./sa-card ubqp ../../test_instance.ubqp 3
```

Example output (from the above):

```
# Reading an instance from ../../test_instance.ubqp
# Energy offset to consider: 8.15998
n: 5 t: 0.00008663 bestsol: -10.97219342 offset: 8.15998399
sol: [1, 1, 0, 1, 0]
```

The rest of the arguments can be specified optionally, for example these are the
defaults and should result in the same result as above:

```
./sa-card ubqp ../../test_instance.ubqp 3 0.001 100 1.1 1 1
```
