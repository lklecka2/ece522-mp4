import os
import matplotlib.pyplot as plt
import math

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

def cpu_transfer_cycles(size_bytes):
    """
    Calculate the total CPU transfer latency in cycles for offloading (write) or retrieving (read).
    For simplicity, we assume read and write cost the same.
    """
    transfer_time_sec = (size_bytes / (CPU_PCIE_BANDWIDTH_GBPS * 1e9)) + (SYSTEM_LATENCY_US * 1e-6)
    transfer_cycles = transfer_time_sec * (GPU_FREQUENCY_GHZ * 1e9)
    return math.ceil(transfer_cycles)

def ssd_transfer_cycles(size_bytes, transfer_type='write'):
    """
    Calculate the SSD transfer latency in cycles for writing or reading.
    """
    if transfer_type.lower() == 'write':
        ssd_latency_us = SSD_WRITE_LATENCY_US
    elif transfer_type.lower() == 'read':
        ssd_latency_us = SSD_READ_LATENCY_US
    else:
        raise ValueError("transfer_type must be 'write' or 'read'")
    
    transfer_time_sec = (size_bytes / (SSD_PCIE_BANDWIDTH_GBPS * 1e9)) + ((SYSTEM_LATENCY_US + ssd_latency_us) * 1e-6)
    transfer_cycles = transfer_time_sec * (GPU_FREQUENCY_GHZ * 1e9)
    return math.ceil(transfer_cycles)

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
            'usage_timeframes': []
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
        time_amount = float(tokens[2])  # ms
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
            tensor_info[tid]['usage_timeframes'].append(timeframe_id)

# Handle global tensors
for tid in tensor_info:
    if tensor_info[tid]['is_global']:
        tensor_lifetimes[tid]['first_used'] = 0
        tensor_lifetimes[tid]['last_used'] = max_timeframe_id
        tensor_info[tid]['usage_timeframes'] = list(range(0, max_timeframe_id + 1))

# Compute lifetime (in number of timeframes)
for tid in tensor_info:
    if (tensor_lifetimes[tid]['first_used'] is not None and 
        tensor_lifetimes[tid]['last_used'] is not None):
        lifetime = tensor_lifetimes[tid]['last_used'] - tensor_lifetimes[tid]['first_used'] + 1
        tensor_info[tid]['lifetime'] = lifetime
    else:
        tensor_info[tid]['lifetime'] = 0

# Focus on tensor 995
chosen_tid = 995
if chosen_tid not in tensor_info:
    print(f"Tensor {chosen_tid} not found.")
    exit(0)

chosen_size_bytes = tensor_info[chosen_tid]['size']
chosen_size_gb = chosen_size_bytes / BYTES_PER_GB
chosen_lifetime = tensor_info[chosen_tid].get('lifetime', 0)
chosen_usage = usage_count.get(chosen_tid, 0)
chosen_usage_timeframes = sorted(tensor_info[chosen_tid]['usage_timeframes'])

if chosen_lifetime == 0:
    print(f"Tensor {chosen_tid} has no valid lifetime.")
    exit(0)

print(f"Chosen tensor: {chosen_tid} with size {chosen_size_gb:.2f} GB, lifetime {chosen_lifetime} timeframes, used {chosen_usage} times.")
print(f"Kernels where tensor {chosen_tid} is used: {chosen_usage_timeframes}")

# Calculate CPU and SSD cycles
cpu_read_cycles = cpu_transfer_cycles(chosen_size_bytes)
cpu_write_cycles = cpu_transfer_cycles(chosen_size_bytes)
cpu_total_cycles = cpu_read_cycles + cpu_write_cycles

ssd_write_cycles = ssd_transfer_cycles(chosen_size_bytes, transfer_type='write')
ssd_read_cycles = ssd_transfer_cycles(chosen_size_bytes, transfer_type='read')
ssd_total_cycles = ssd_write_cycles + ssd_read_cycles

print(f"Size: {chosen_size_bytes} CPU total:{cpu_total_cycles} (read:{cpu_read_cycles} write:{cpu_write_cycles}) SSD total:{ssd_total_cycles} (read:{ssd_read_cycles} write:{ssd_write_cycles})")

# -----------------------------------------------------
# Compute start cycles for each kernel (timeframe)
# -----------------------------------------------------
events_sorted = sorted(events, key=lambda x: x['timeframe_id'])

start_cycle_of_kernel = {}
cumulative_cycles = 0
for e in events_sorted:
    kernel = e['timeframe_id']
    time_amount_ms = e['time_amount']
    # Convert ms to cycles: ms -> s: ms * 1e-3
    # cycles = seconds * GPU_FREQUENCY_GHZ * 1e9
    kernel_cycles = time_amount_ms * 1e-3 * GPU_FREQUENCY_GHZ * 1e9
    
    start_cycle_of_kernel[kernel] = cumulative_cycles
    cumulative_cycles += kernel_cycles

def first_kernel_after(cycle):
    for tf in sorted(start_cycle_of_kernel.keys()):
        if start_cycle_of_kernel[tf] > cycle:
            return tf
    return None

usage_points = []
cpu_lines = []
ssd_lines = []

plt.figure(figsize=(10, 3))

for i in range(len(chosen_usage_timeframes) - 1):
    current_tf = chosen_usage_timeframes[i]
    next_tf = chosen_usage_timeframes[i + 1]

    cpu_receives_kernel = first_kernel_after(start_cycle_of_kernel[current_tf+1]+cpu_write_cycles)
    ssd_receives_kernel = first_kernel_after(start_cycle_of_kernel[current_tf+1]+ssd_write_cycles)
    cpu_sends_kernel = first_kernel_after(start_cycle_of_kernel[next_tf]-cpu_write_cycles)-1
    ssd_sends_kernel = first_kernel_after(start_cycle_of_kernel[next_tf]-ssd_write_cycles)-1

    offload_type = None
    if ssd_receives_kernel*1.05 < ssd_sends_kernel:
        offload_type = 'SSD'
        receives_cycle = start_cycle_of_kernel[ssd_receives_kernel]
        sends_cycle = start_cycle_of_kernel[ssd_sends_kernel]
    elif cpu_receives_kernel*1.05 < cpu_sends_kernel:
        offload_type = 'CPU'
        receives_cycle = start_cycle_of_kernel[cpu_receives_kernel]
        sends_cycle = start_cycle_of_kernel[cpu_sends_kernel]
    
    start_idle = start_cycle_of_kernel[current_tf+1]
    end_idle = start_cycle_of_kernel[next_tf]
    idle_cycles = math.ceil(end_idle - start_idle)
    KERNEL = False
    if not KERNEL:
        if offload_type:
            print(f"{offload_type} | ({start_idle})\t-({receives_cycle}-{sends_cycle})->\t({end_idle})")
        else:
            print(f"--- | ({start_idle})\t-({idle_cycles})->\t({end_idle})")
    
    # Store usage points
    usage_points.append((start_cycle_of_kernel[current_tf], 0))
    # If offload occurs, add vertical lines
    if offload_type == 'SSD':
        plt.axvline(x=receives_cycle/1e9, color='purple', linestyle=':', label='to/from SSD')
        plt.axvline(x=sends_cycle/1e9, color='purple', linestyle=':')
    elif offload_type == 'CPU':
        plt.axvline(x=receives_cycle/1e9, color='green', linestyle=':', label='to/from CPU')
        plt.axvline(x=sends_cycle/1e9, color='green', linestyle=':')

    if offload_type == 'SSD':
        plt.annotate('', xytext=(start_cycle_of_kernel[current_tf]/1e9,0), xy=(receives_cycle/1e9,0), arrowprops=dict(arrowstyle='->', color='purple'))
        plt.annotate('', xytext=(sends_cycle/1e9,0), xy=(end_idle/1e9,0), arrowprops=dict(arrowstyle='->', color='purple'))
    elif offload_type == 'CPU':
        plt.annotate('', xytext=(start_cycle_of_kernel[current_tf]/1e9,0), xy=(receives_cycle/1e9,0), arrowprops=dict(arrowstyle='->', color='green'))
        plt.annotate('', xytext=(sends_cycle/1e9,0), xy=(end_idle/1e9,0), arrowprops=dict(arrowstyle='->', color='green'))
    else:
        # No offload: already connected with a black line by existing code
        pass
# Add the last usage point after the loop
if chosen_usage_timeframes:
    usage_points.append((start_cycle_of_kernel[chosen_usage_timeframes[-1]], 0))

# -----------------------------
# Plotting Code (Minimal Changes)
# -----------------------------
# We'll do the plotting now that all computations are done.

# Convert usage points
usage_x = [p[0]/ 1e9 for p in usage_points]
usage_y = [0 for _ in usage_points]

plt.plot(usage_x, usage_y, 'o', color='black', label='Usage')

# Connect usage points where no offload occurs
for i in range(len(usage_points)-1):
    start_p = usage_points[i]
    end_p = usage_points[i+1]
    # Re-check offload for this interval
    start_tf = chosen_usage_timeframes[i]
    end_tf = chosen_usage_timeframes[i+1]
    cpu_receives_kernel = first_kernel_after(start_cycle_of_kernel[start_tf+1] + cpu_write_cycles)
    ssd_receives_kernel = first_kernel_after(start_cycle_of_kernel[start_tf+1] + ssd_write_cycles)
    cpu_sends_kernel = None
    ssd_sends_kernel = None
    if first_kernel_after(start_cycle_of_kernel[end_tf] - cpu_write_cycles) is not None:
        cpu_sends_kernel = first_kernel_after(start_cycle_of_kernel[end_tf] - cpu_write_cycles)-1
    if first_kernel_after(start_cycle_of_kernel[end_tf] - ssd_write_cycles) is not None:
        ssd_sends_kernel = first_kernel_after(start_cycle_of_kernel[end_tf] - ssd_write_cycles)-1

    ssd_offload_possible = (ssd_receives_kernel is not None and ssd_sends_kernel is not None)
    cpu_offload_possible = (cpu_receives_kernel is not None and cpu_sends_kernel is not None)
    offload_type = None
    if ssd_offload_possible:
        offload_type = 'SSD'
    elif cpu_offload_possible:
        offload_type = 'CPU'

    if offload_type is None:
        # No offload, connect them with a line
        plt.plot([start_p[0]/1e9, end_p[0]/1e9],
                 [0, 0], color='black', marker='o-')

# Remove duplicate labels in legend
handles, labels = plt.gca().get_legend_handles_labels()
seen = set()
unique_handles_labels = []
for h, l in zip(handles, labels):
    if l not in seen and l != '':
        unique_handles_labels.append((h,l))
        seen.add(l)

plt.xlabel("Cycles (in billions)")
plt.yticks([])
plt.title(f"Optimal Migration for Tensor {chosen_tid} (Size: 0.9GB) in Inceptionv3 with Batch Size 1024")
plt.legend([h for h,l in unique_handles_labels], [l for h,l in unique_handles_labels])


plt.tight_layout()
output_file = f'tensor_{chosen_tid}_usage_offload_time_axis.png'
plt.savefig(output_file)
plt.show()
print(f"Saved tensor usage and offload opportunities plot to {output_file}")