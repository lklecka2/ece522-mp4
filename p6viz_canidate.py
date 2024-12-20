import os
import matplotlib.pyplot as plt

# Constants for transfer latency calculations
GPU_FREQUENCY_GHZ = 1.2             # GPU frequency in GHz
SYSTEM_LATENCY_US = 45              # System latency in microseconds
CPU_PCIE_BANDWIDTH_GBPS = 15.754    # CPU <-> GPU PCIe bandwidth in GB/s
SSD_PCIE_BANDWIDTH_GBPS = 3.2       # CPU <-> SSD PCIe bandwidth in GB/s
SSD_READ_LATENCY_US = 12            # SSD read latency in microseconds
SSD_WRITE_LATENCY_US = 16           # SSD write latency in microseconds

BYTES_PER_GB = 1024**3               # Bytes in a gigabyte

DIR = 'results/Inceptionv3/sim_input/'  # Directory containing the info files

def parse_line(line):
    tokens = []
    i = 0
    n = len(line)
    while i < n:
        if line[i].isspace():
            i += 1
            continue
        elif line[i] == '[':
            j = i
            while j < n and line[j] != ']':
                j += 1
            if j < n:
                j += 1  # Include ']'
                tokens.append(line[i:j])
                i = j
            else:
                raise ValueError('Unmatched [')
        else:
            j = i
            while j < n and not line[j].isspace():
                j += 1
            tokens.append(line[i:j])
            i = j
    return tokens

def parse_array(s):
    s = s.strip('[]')
    if not s:
        return []
    else:
        return [int(x) for x in s.split(',') if x]

# Helper functions to calculate transfer latencies in milliseconds
def cpu_transfer_time_ms(size_bytes):
    transfer_time_sec = size_bytes / (CPU_PCIE_BANDWIDTH_GBPS * 1e9)  # seconds
    total_time_sec = (SYSTEM_LATENCY_US * 1e-6) + transfer_time_sec  # seconds
    total_time_ms = total_time_sec * 1e3  # milliseconds
    return total_time_ms

def ssd_transfer_time_ms(size_bytes, transfer_type='read'):
    if transfer_type.lower() == 'read':
        ssd_latency_us = SSD_READ_LATENCY_US
    elif transfer_type.lower() == 'write':
        ssd_latency_us = SSD_WRITE_LATENCY_US
    else:
        raise ValueError("transfer_type must be 'read' or 'write'")
    transfer_time_sec = size_bytes / (SSD_PCIE_BANDWIDTH_GBPS * 1e9)  # seconds
    total_time_sec = ((SYSTEM_LATENCY_US + ssd_latency_us) * 1e-6) + transfer_time_sec  # seconds
    total_time_ms = total_time_sec * 1e3  # milliseconds
    return total_time_ms

# Parse tensor info
tensor_info = {}
tensor_info_path = os.path.join(DIR, '1024Tensor.info')
with open(tensor_info_path) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        tid = int(parts[0])
        size = int(parts[1])
        is_global = parts[2].lower() == 'true'
        tensor_info[tid] = {
            'size': size,
            'is_global': is_global,
            'usage_timeframes': []  # Initialize list to store usage kernels
        }

# Parse kernel events
events = []
kernel_info_path = os.path.join(DIR, '1024Kernel.info')
with open(kernel_info_path) as f:
    for line in f:
        tokens = parse_line(line.strip())
        if len(tokens) < 5:
            continue
        timeframe_id = int(tokens[0])
        time_amount = float(tokens[2])  # Assume this is in ms
        input_tensors = parse_array(tokens[3])
        output_tensors = parse_array(tokens[4])
        workspace_tensor = None
        if len(tokens) > 5:
            workspace_tensor = int(tokens[5])
        events.append({
            'timeframe_id': timeframe_id,
            'time_amount': time_amount,
            'input_tensors': input_tensors,
            'output_tensors': output_tensors,
            'workspace_tensor': workspace_tensor
        })

if not events:
    print("No events found.")
    exit(0)

max_timeframe_id = max(event['timeframe_id'] for event in events)

# Compute lifetimes and usage counts
tensor_lifetimes = {tid: {'first_used': None, 'last_used': None} for tid in tensor_info}
usage_count = {tid: 0 for tid in tensor_info}

for event in events:
    timeframe_id = event['timeframe_id']
    tensors_involved = event['input_tensors'] + event['output_tensors']
    if event['workspace_tensor'] is not None:
        tensors_involved.append(event['workspace_tensor'])
    for tid in tensors_involved:
        if tid in tensor_lifetimes:
            if tensor_lifetimes[tid]['first_used'] is None:
                tensor_lifetimes[tid]['first_used'] = timeframe_id
            tensor_lifetimes[tid]['last_used'] = timeframe_id
            usage_count[tid] += 1
            tensor_info[tid]['usage_timeframes'].append(timeframe_id)  # Record usage

# Handle global tensors
for tid in tensor_info:
    if tensor_info[tid]['is_global']:
        tensor_lifetimes[tid]['first_used'] = 0
        tensor_lifetimes[tid]['last_used'] = max_timeframe_id
        # Assuming global tensors are used in all timeframes
        tensor_info[tid]['usage_timeframes'] = list(range(0, max_timeframe_id + 1))

# Compute lifetime (in number of timeframes)
for tid in tensor_info:
    if (tensor_lifetimes[tid]['first_used'] is not None and 
        tensor_lifetimes[tid]['last_used'] is not None):
        # Lifetime = (last_used - first_used + 1)
        lifetime = tensor_lifetimes[tid]['last_used'] - tensor_lifetimes[tid]['first_used'] + 1
        tensor_info[tid]['lifetime'] = lifetime
    else:
        tensor_info[tid]['lifetime'] = 0  # Not used or not found

# Define selection criteria
# Adjust these thresholds as needed
LIFETIME_MIN = 20         # Minimum lifetime in timeframes
LIFETIME_MAX = 999         # Minimum lifetime in timeframes
USAGE_COUNT_MIN = 2              # Maximum usage count
USAGE_COUNT_MAX = 999             # Maximum usage count
TOP_SIZE_PERCENTILE = 90         # Top 10% by size

# Sort tensors by size to determine top 10%
all_sizes = [tensor_info[tid]['size'] for tid in tensor_info]
all_sizes_sorted = sorted(all_sizes)
if all_sizes_sorted:
    index_90 = int(0.9 * len(all_sizes_sorted))
    size_threshold = all_sizes_sorted[index_90] if index_90 < len(all_sizes_sorted) else all_sizes_sorted[-1]
else:
    size_threshold = 0

# Identify candidate tensors
candidate_tensors = []
for tid, info in tensor_info.items():
    if (info.get('lifetime', 0) > LIFETIME_MIN and
        info.get('lifetime', 0) < LIFETIME_MAX and
        info['size'] >= size_threshold and
        usage_count.get(tid, 0) < USAGE_COUNT_MAX and
        usage_count.get(tid, 0) > USAGE_COUNT_MIN):
        candidate_tensors.append((tid, info['size'], info['lifetime'], usage_count[tid], info['usage_timeframes']))

# Sort candidates by lifetime then size descending
candidate_tensors.sort(key=lambda x: (x[2], x[1]), reverse=True)

if candidate_tensors:
    print("Candidate tensors (tid, size, lifetime, usage_count, usage_timeframes):")
    for c in candidate_tensors:
        tid, size, lifetime, count, usage_tfs = c
        size_gb = size / BYTES_PER_GB
        print(f"Tensor ID: {tid}, Size: {size_gb:.2f} GB, Lifetime: {lifetime} timeframes, Usage Count: {count}, Used in Kernels: {usage_tfs}")
    # Pick the first one as our chosen tensor
    chosen_tid = candidate_tensors[0][0]
    chosen_size_bytes = candidate_tensors[0][1]
    chosen_size_gb = chosen_size_bytes / BYTES_PER_GB
    chosen_lifetime = candidate_tensors[0][2]
    chosen_usage = candidate_tensors[0][3]
    chosen_usage_timeframes = candidate_tensors[0][4]
    print(f"\nChosen tensor: {chosen_tid} with size {chosen_size_gb:.2f} GB, lifetime {chosen_lifetime} timeframes, used {chosen_usage} times.")
else:
    print("No tensor meets the criteria for large, long-lived, and infrequently used.")
    exit(0)

# Extract the list of kernels where the chosen tensor is used
used_timeframes_sorted = sorted(chosen_usage_timeframes)

print(f"Kernels where tensor {chosen_tid} is used: {used_timeframes_sorted}")