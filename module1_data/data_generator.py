# Module 1 - Realistic Memory Trace Generator

import random
import pandas as pd
import os


print("======================================")
print("   MODULE 1 - MEMORY TRACE GENERATOR")
print("======================================")


# Number of CPU memory accesses
NUM_ACCESSES = 5000

# Simulated memory address range
ADDRESS_RANGE = 256


# Store generated accesses
memory_addresses = []
operations = []


# Start with a random memory address
current_address = random.randint(
    0,
    ADDRESS_RANGE - 1
)


# Generate memory accesses
for i in range(NUM_ACCESSES):

    # Sometimes stay near the current address
    if random.random() < 0.80:

        change = random.choice([
            -2,
            -1,
             0,
             1,
             2
        ])

        current_address += change

    else:

        # Occasionally jump to a new memory region
        current_address = random.randint(
            0,
            ADDRESS_RANGE - 1
        )


    # Keep address inside memory range
    current_address = max(
        0,
        min(
            current_address,
            ADDRESS_RANGE - 1
        )
    )


    # Generate READ or WRITE
    operation = random.choice([
        "READ",
        "WRITE"
    ])


    # Store values
    memory_addresses.append(
        current_address
    )

    operations.append(
        operation
    )


# Create DataFrame
memory_trace = pd.DataFrame({

    "access_id": range(
        1,
        NUM_ACCESSES + 1
    ),

    "memory_address": memory_addresses,

    "operation": operations

})


# Find project folder
project_folder = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# Data folder
data_folder = os.path.join(
    project_folder,
    "data"
)


os.makedirs(
    data_folder,
    exist_ok=True
)


# Output file
file_path = os.path.join(
    data_folder,
    "memory_trace.csv"
)


# Save CSV
memory_trace.to_csv(
    file_path,
    index=False
)


# Display results
print("\nMemory trace generated successfully!")

print(
    f"Total accesses: {NUM_ACCESSES}"
)

print(
    f"CSV saved to: {file_path}"
)


print("\nFirst 20 memory accesses:")
print(
    memory_trace.head(20)
)