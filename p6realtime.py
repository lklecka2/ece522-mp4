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
    transfer_time_sec = (size_bytes / (CPU_PCIE_BANDWIDTH_GBPS * 1e9)) + (SYSTEM_LATENCY_US * 1e-6)
    transfer_cycles = transfer_time_sec * (GPU_FREQUENCY_GHZ * 1e9)
    total_cycles = transfer_cycles
    return math.ceil(total_cycles)

def ssd_transfer_cycles(size_bytes, transfer_type='write'):
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

# Calculate total cycles required for CPU and SSD offloading
cpu_transfer_cycles_val = cpu_transfer_cycles(chosen_size_bytes)
cpu_total_cycles = cpu_transfer_cycles_val * 2  # Offload + Retrieve

ssd_write_cycles = ssd_transfer_cycles(chosen_size_bytes, transfer_type='write')
ssd_read_cycles = ssd_transfer_cycles(chosen_size_bytes, transfer_type='read')
ssd_total_cycles = ssd_write_cycles + ssd_read_cycles

print(f"Size: {chosen_size_bytes} bytes")
print(f"CPU Offload Total Cycles: {cpu_total_cycles}")
print(f"SSD Offload Total Cycles: {ssd_total_cycles}")

# Build mapping from timeframe_id to cumulative start time in ms
events_sorted = sorted(events, key=lambda x: x['timeframe_id'])
timeframe_start_time = {}
cumulative_time = 0.0
for e in events_sorted:
    timeframe_start_time[e['timeframe_id']] = cumulative_time
    cumulative_time += e['time_amount']  # ms

chosen_usage_times = [timeframe_start_time[tf] for tf in chosen_usage_timeframes]

# Convert cycles back to ms for retrieval times
def cycles_to_ms(cycles):
    # cycles / (GPU_FREQUENCY_GHZ * 1e9 cycles/s) = sec
    # sec * 1000 = ms
    return (cycles / (GPU_FREQUENCY_GHZ * 1e9)) * 1000

cpu_retrieval_ms = cycles_to_ms(cpu_transfer_cycles_val)  # retrieval time for CPU
ssd_retrieval_ms = cycles_to_ms(ssd_read_cycles)          # retrieval time for SSD

offload_opportunities = []

for i in range(len(chosen_usage_timeframes) - 1):
    current_tf = chosen_usage_timeframes[i]
    next_tf = chosen_usage_timeframes[i + 1]
    
    # Calculate idle cycles between current_tf and next_tf
    idle_cycles = 0.0
    for event in events:
        tf_id = event['timeframe_id']
        if current_tf < tf_id < next_tf:
            idle_cycles += event['time_amount'] * (GPU_FREQUENCY_GHZ * 1e6) #ms
    
    can_offload_cpu = idle_cycles > cpu_total_cycles
    can_offload_ssd = idle_cycles > ssd_total_cycles

    offload_type = None
    if can_offload_ssd:
        offload_type = 'SSD'
    elif can_offload_cpu:
        offload_type = 'CPU'
    
    if offload_type:
        arrival_tf = math.ceil(current_tf + 1)
        transfer_back_tf = math.floor(next_tf - 1)
        offload_opportunities.append((arrival_tf, transfer_back_tf, offload_type))
        print(f"Idle Cycles: {idle_cycles:.2f} | Offload Type: {offload_type} | Arrival TF: {arrival_tf} | Transfer Back TF: {transfer_back_tf}")
    else:
        print(f"Idle Cycles: {idle_cycles:.2f} | No Offload")

active_regions = []
offload_opportunities_sorted = sorted(offload_opportunities, key=lambda x: x[0])

prev_transfer_back_tf = None
for op in offload_opportunities_sorted:
    arrival_tf, transfer_back_tf, offload_type = op
    if prev_transfer_back_tf is None:
        active_start = chosen_usage_timeframes[0]
    else:
        active_start = prev_transfer_back_tf + 1
    active_end = arrival_tf - 1
    if active_start <= active_end:
        active_regions.append((active_start, active_end))
    prev_transfer_back_tf = transfer_back_tf

if prev_transfer_back_tf is not None and prev_transfer_back_tf < chosen_usage_timeframes[-1]:
    active_regions.append((prev_transfer_back_tf + 1, chosen_usage_timeframes[-1]))

# Convert active regions and opportunities to times
active_regions_times = [(timeframe_start_time[s], timeframe_start_time[e]) for (s, e) in active_regions]

offload_opportunities_times = []
for (arrival_tf, transfer_back_tf, offload_type) in offload_opportunities:
    arrival_time = timeframe_start_time[arrival_tf]
    # Instead of using transfer_back_time directly, subtract retrieval time so we get the start of the transfer back
    if offload_type == 'CPU':
        transfer_back_start_time = timeframe_start_time[transfer_back_tf] - cpu_retrieval_ms
    else:
        transfer_back_start_time = timeframe_start_time[transfer_back_tf] - ssd_retrieval_ms
    offload_opportunities_times.append((arrival_time, transfer_back_start_time, offload_type))

# Plot with real time on the x-axis
plt.figure(figsize=(20, 4))

for (start_t, end_t) in active_regions_times:
    region_usage_times = [ut for ut in chosen_usage_times if start_t <= ut <= end_t]
    plt.plot(region_usage_times, [1]*len(region_usage_times), 'bo-')

# Add vertical lines for offload arrivals and transfers back (start times)
for (arrival_time, transfer_back_start_time, offload_type) in offload_opportunities_times:
    color = 'green' if offload_type == 'CPU' else 'purple'
    # Offload start (to CPU/SSD)
    plt.axvline(x=arrival_time, color=color, linestyle='--', linewidth=2)
    # Start of transfer back to GPU
    plt.axvline(x=transfer_back_start_time, color=color, linestyle='-.', linewidth=2)

plt.xlabel('Time (ms)', fontsize=14)
plt.yticks([1], ['Used'])
plt.title(f'Tensor {chosen_tid} Usage and Offload Opportunities (Size: {chosen_size_gb:.2f} GB)', fontsize=16)
plt.ylim(0.9, 1.3)
plt.grid(True, linestyle='--', alpha=0.5)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Used Kernels', markerfacecolor='b', markersize=8),
    Line2D([0], [0], color='green', linestyle='--', lw=2, label='Offload to CPU/SSD Start'),
    Line2D([0], [0], color='green', linestyle='-.', lw=2, label='CPU->GPU Transfer Start'),
    Line2D([0], [0], color='purple', linestyle='--', lw=2, label='Offload to SSD Start'),
    Line2D([0], [0], color='purple', linestyle='-.', lw=2, label='SSD->GPU Transfer Start')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
output_file = f'tensor_{chosen_tid}_usage_offload_time_axis.png'
plt.savefig(output_file)
plt.show()
print(f"Saved tensor usage and offload opportunities plot to {output_file}")
