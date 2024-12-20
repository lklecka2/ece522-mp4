import os
import matplotlib.pyplot as plt

base_dir = 'results'

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

results_table = []  # Will hold tuples of (model, batch_size, max_active_tensor_mem, max_kernel_mem)

for model_name in os.listdir(base_dir):
    model_path = os.path.join(base_dir, model_name, 'sim_input')
    if not os.path.isdir(model_path):
        continue
    
    # Look for files like {BATCHSIZE}Tensor.info and {BATCHSIZE}Kernel.info
    # Extract batch sizes by listing files in the directory
    tensor_files = [f for f in os.listdir(model_path) if f.endswith('Tensor.info')]
    kernel_files = [f for f in os.listdir(model_path) if f.endswith('Kernel.info')]
    
    # Match Tensor and Kernel files by batch size prefix
    # Assuming a pattern like: <batchsize>Tensor.info and <batchsize>Kernel.info
    batch_sizes = []
    for tf in tensor_files:
        # Remove 'Tensor.info'
        batch_prefix = tf.replace('Tensor.info', '')
        # Check if corresponding Kernel.info file exists
        kf = batch_prefix + 'Kernel.info'
        if kf in kernel_files:
            batch_sizes.append(batch_prefix)
    
    for batch_prefix in batch_sizes:
        tensor_file = os.path.join(model_path, batch_prefix + 'Tensor.info')
        kernel_file = os.path.join(model_path, batch_prefix + 'Kernel.info')
        
        # Parse tensor_info
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
        
        # Parse events
        events = []
        with open(kernel_file) as f:
            for line in f:
                tokens = parse_line(line.strip())
                if len(tokens) < 5:
                    continue
                timeframe_id = int(tokens[0])
                time_amount = float(tokens[2])
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
            continue
        
        max_timeframe_id = max(event['timeframe_id'] for event in events)
        
        # Compute lifetimes
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
        
        # Global tensors used from start to end
        for tid in tensor_info:
            if tensor_info[tid]['is_global']:
                tensor_lifetimes[tid]['first_used'] = 0
                tensor_lifetimes[tid]['last_used'] = max_timeframe_id
        
        # Compute total memory usage per timeframe
        total_memory_usage = [0] * (max_timeframe_id + 1)
        for timeframe_id in range(max_timeframe_id + 1):
            total_mem = 0
            for tid in tensor_info:
                if (tensor_lifetimes[tid]['first_used'] is not None and
                    tensor_lifetimes[tid]['first_used'] <= timeframe_id <= tensor_lifetimes[tid]['last_used']):
                    total_mem += tensor_info[tid]['size']
            total_memory_usage[timeframe_id] = total_mem
        
        max_active_tensor_mem = max(total_memory_usage)
        
        # Compute max kernel memory usage
        max_kernel_mem_usage = 0
        for event in events:
            tensors_involved = event['input_tensors'] + event['output_tensors']
            if event['workspace_tensor'] is not None:
                tensors_involved.append(event['workspace_tensor'])
            kernel_mem = sum(tensor_info[tid]['size'] for tid in tensors_involved if tid in tensor_info)
            if kernel_mem > max_kernel_mem_usage:
                max_kernel_mem_usage = kernel_mem
        
        # Add to results table
        # batch_prefix should be the batch size (convert to int if needed)
        try:
            batch_size = int(batch_prefix)
        except ValueError:
            batch_size = batch_prefix
        
        results_table.append((model_name, batch_size, max_active_tensor_mem, max_kernel_mem_usage))

# Print the table
# Sorting by model name and batch size for neatness
results_table.sort(key=lambda x: (x[0], x[1]))
BYTES_PER_GB = 1024**3

print("Model BatchSize MaxActiveTensorMem(GB) MaxKernelMem(GB)")
for row in results_table:
    model, batch_size, max_active_tensor_mem, max_kernel_mem_usage = row
    max_active_tensor_mem_gb = max_active_tensor_mem / BYTES_PER_GB
    max_kernel_mem_usage_gb = max_kernel_mem_usage / BYTES_PER_GB
    print(f"{model} {batch_size} {max_active_tensor_mem_gb:.3f} {max_kernel_mem_usage_gb:.3f}")

