"""The memory setting the benchmarks compare, in one place.

`acs_benchmark.py` runs it, `plot_benchmarks.py` prints its definition under
every figure that draws it, and the documentation can quote it. Imports
nothing, so any of them can import this.
"""

# "low memory": smaller blocks of the design on disk and smaller chunks of a
# block in memory, so less of the data is in flight at once. These are explicit
# settings of the two options; the defaults are 64 and 64.
LOW_MEMORY = {"block_mb": 8, "chunk_mb": 8}

LOW_MEMORY_NOTE = (
    f"Low memory: block_mb={LOW_MEMORY['block_mb']} and chunk_mb={LOW_MEMORY['chunk_mb']} "
    "(the defaults are 64 and 64).")
