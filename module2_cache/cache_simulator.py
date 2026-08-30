# Module 2 - LRU Cache Simulator

from collections import OrderedDict
import pandas as pd
import os


print("======================================")
print("       MODULE 2 - CACHE SIMULATOR")
print("======================================")


# Cache capacity
CACHE_SIZE = 16


# Find project folder
project_folder = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# Input file
input_file = os.path.join(
    project_folder,
    "data",
    "memory_trace.csv"
)


# Output file
output_file = os.path.join(
    project_folder,
    "data",
    "cache_results.csv"
)


# Load memory trace
print("\nLoading memory trace...")

data = pd.read_csv(input_file)

print(
    f"Loaded {len(data)} memory accesses."
)


# Create LRU cache
cache = OrderedDict()


# Performance counters
cache_hits = 0
cache_misses = 0


# Store results
results = []


print("\nStarting cache simulation...")


# Process every memory access
for _, row in data.iterrows():

    access_id = int(row["access_id"])

    address = int(
        row["memory_address"]
    )

    operation = row["operation"]


    # Check cache
    if address in cache:

        result = "HIT"

        cache_hits += 1

        # Mark as recently used
        cache.move_to_end(address)


    else:

        result = "MISS"

        cache_misses += 1

        # Add address
        cache[address] = True

        # Remove least recently used
        if len(cache) > CACHE_SIZE:

            cache.popitem(
                last=False
            )


    # Save individual result
    results.append({

        "access_id": access_id,

        "memory_address": address,

        "operation": operation,

        "cache_result": result

    })


# Convert results into DataFrame
results_df = pd.DataFrame(results)


# Save results
results_df.to_csv(
    output_file,
    index=False
)


# Calculate statistics
total_accesses = (
    cache_hits + cache_misses
)

hit_rate = (
    cache_hits / total_accesses
) * 100

miss_rate = (
    cache_misses / total_accesses
) * 100


# Display results
print("\n======================================")
print("       CACHE SIMULATION RESULTS")
print("======================================")

print(
    f"Total Accesses : {total_accesses}"
)

print(
    f"Cache Hits     : {cache_hits}"
)

print(
    f"Cache Misses   : {cache_misses}"
)

print(
    f"Hit Rate       : {hit_rate:.2f}%"
)

print(
    f"Miss Rate      : {miss_rate:.2f}%"
)


print("\nCache results saved to:")

print(output_file)


print("\nFirst 10 cache results:")

print(
    results_df.head(10)
)


print("\nCache simulation completed!")