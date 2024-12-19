import os
import matplotlib.pyplot as plt

BYTES_PER_GB = 1024**3

def ensure_dir(directory):
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)

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

# Base directory containing model results
base_dir = 'results'

# We only consider batch sizes of 1024
batch_size_filter = '1024'

# Directory to save the generated graphs
output_folder = 'graphs-lifetimes'
ensure_dir(output_folder)

# List all model directories within the base directory
model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d, 'sim_input'))]

for model_name in model_dirs:
    model_path = os.path.join(base_dir, model_name, 'sim_input')
    
    # Define the Tensor.info and Kernel.info file paths for batch size 1024
    tensor_file = os.path.join(model_path, batch_size_filter + 'Tensor.info')
    kernel_file = os.path.join(model_path, batch_size_filter + 'Kernel.info')
    
    # Check if both Tensor.info and Kernel.info files exist
    if not (os.path.exists(tensor_file) and os.path.exists(kernel_file)):
        print(f"Skipping model '{model_name}' as required info files for batch size {batch_size_filter} are missing.")
        continue
    
    # Parse tensor information
    tensor_info = {}
    with open(tensor_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            tid = int(parts[0])
            size = int(parts[1])
            is_global = parts[2].lower() == 'true'
            tensor_info[tid] = {'size': size, 'is_global': is_global}
    
    # Parse kernel events
    events = []
    with open(kernel_file) as f:
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
        print(f"No events found for model '{model_name}' with batch size {batch_size_filter}.")
        continue
    
    # Determine the maximum timeframe ID
    max_timeframe_id = max(event['timeframe_id'] for event in events)
    
    # Sort events by timeframe_id to compute cumulative time
    events_sorted = sorted(events, key=lambda e: e['timeframe_id'])
    cumulative_time = [0.0] * (max_timeframe_id + 2)
    # cumulative_time[i] will hold the start time (ms) of timeframe i
    for i in range(1, max_timeframe_id + 2):
        if i-1 <= max_timeframe_id:
            cumulative_time[i] = cumulative_time[i-1] + events_sorted[i-1]['time_amount']
        else:
            cumulative_time[i] = cumulative_time[i-1]
    
    # Compute tensor lifetimes
    tensor_lifetimes = {}
    for tid in tensor_info:
        tensor_lifetimes[tid] = {'first_used': None, 'last_used': None}
    
    for event in events:
        timeframe_id = event['timeframe_id']
        tensors_involved = event['input_tensors'] + event['output_tensors']
        if event['workspace_tensor'] is not None:
            tensors_involved.append(event['workspace_tensor'])
        for tid in tensors_involved:
            if tid not in tensor_lifetimes:
                continue
            if tensor_lifetimes[tid]['first_used'] is None:
                tensor_lifetimes[tid]['first_used'] = timeframe_id
            tensor_lifetimes[tid]['last_used'] = timeframe_id
    
    # Global tensors span the entire run
    for tid in tensor_info:
        if tensor_info[tid]['is_global']:
            tensor_lifetimes[tid]['first_used'] = 0
            tensor_lifetimes[tid]['last_used'] = max_timeframe_id
    
    # Compute lifetime in ms and gather sizes/lifetimes
    sizes_gb = []
    lifetimes_ms = []
    total_run_time_ms = cumulative_time[max_timeframe_id + 1]
    
    active_times_ms = []
    
    for tid, info in tensor_info.items():
        first_used = tensor_lifetimes[tid]['first_used']
        last_used = tensor_lifetimes[tid]['last_used']
        if first_used is not None and last_used is not None:
            # Lifetime in ms
            lifetime_ms = cumulative_time[last_used + 1] - cumulative_time[first_used]
            size_gb = info['size'] / BYTES_PER_GB  # Convert size to GB
            sizes_gb.append(size_gb)
            lifetimes_ms.append(lifetime_ms)
            
            # Active time is equal to lifetime
            active_time = lifetime_ms
            active_times_ms.append(active_time)
    
    # Check if there are tensors to plot
    if not sizes_gb or not lifetimes_ms:
        print(f"No valid tensor lifetimes found for model '{model_name}' with batch size {batch_size_filter}.")
        continue
    
    # (2) Plot the distribution of tensor size vs. lifetime in ms
    plt.figure(figsize=(10, 7))
    plt.scatter(lifetimes_ms, sizes_gb, alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.xlabel('Lifetime (ms)', fontsize=12)
    plt.ylabel('Tensor Size (GB)', fontsize=12)
    plt.title(f'Tensor Size vs Lifetime (ms) for {model_name} (Batch Size: {batch_size_filter})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    scatter_plot_path = os.path.join(output_folder, f'{model_name}_1024_size_vs_lifetime_ms.png')
    plt.savefig(scatter_plot_path)
    plt.close()
    print(f"Saved scatter plot: {scatter_plot_path}")
    
    # (3) Plot the distribution of active time of tensors in ms as a histogram
    plt.figure(figsize=(10, 7))
    plt.hist(active_times_ms, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Active Time (ms)', fontsize=12)
    plt.ylabel('Number of Tensors', fontsize=12)
    plt.title(f'Distribution of Tensor Active Times (ms) for {model_name} (Batch Size: {batch_size_filter})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    histogram_plot_path = os.path.join(output_folder, f'{model_name}_1024_active_time_distribution.png')
    plt.savefig(histogram_plot_path)
    plt.close()
    print(f"Saved histogram plot: {histogram_plot_path}")

print(f"All graphs have been saved in the '{output_folder}' directory.")
