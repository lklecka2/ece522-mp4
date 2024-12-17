import matplotlib.pyplot as plt

DIR = 'results/Inceptionv3/sim_input/'

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
tensor_info = {}

with open(DIR + '1024Tensor.info') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        tid = int(parts[0])
        size = int(parts[1])
        is_global = parts[2].lower() == 'true'
        tensor_info[tid] = {'size': size, 'is_global': is_global}
events = []

with open(DIR + '1024Kernel.info') as f:
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

max_timeframe_id = max(event['timeframe_id'] for event in events)
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
for tid in tensor_info:
    if tensor_info[tid]['is_global']:
        tensor_lifetimes[tid]['first_used'] = 0
        tensor_lifetimes[tid]['last_used'] = max_timeframe_id
total_memory_usage = [0] * (max_timeframe_id + 1)

for timeframe_id in range(max_timeframe_id + 1):
    total_mem = 0
    for tid in tensor_info:
        if (tensor_lifetimes[tid]['first_used'] is not None and
            tensor_lifetimes[tid]['first_used'] <= timeframe_id <= tensor_lifetimes[tid]['last_used']):
            total_mem += tensor_info[tid]['size']
    total_memory_usage[timeframe_id] = total_mem
plt.figure()
plt.plot(range(max_timeframe_id + 1), total_memory_usage)
plt.xlabel('Timeframe')
plt.ylabel('Total Memory Usage (bytes)')
plt.title('Total Memory Usage over Time')
plt.savefig('total_memory_usage.png')
plt.show()
sizes = []
lifetimes = []

for tid in tensor_info:
    size = tensor_info[tid]['size']
    if tensor_lifetimes[tid]['first_used'] is not None:
        lifetime = (tensor_lifetimes[tid]['last_used'] - 
                    tensor_lifetimes[tid]['first_used'] + 1)
        sizes.append(size)
        lifetimes.append(lifetime)

plt.figure()
plt.scatter(lifetimes, sizes)
plt.xlabel('Lifetime (number of timeframes)')
plt.ylabel('Tensor Size (bytes)')
plt.title('Tensor Size vs Lifetime')
plt.savefig('size_vs_lifetime.png')
plt.show()
plt.figure()
plt.hist(lifetimes, bins=20)
plt.xlabel('Lifetime (number of timeframes)')
plt.ylabel('Number of Tensors')
plt.title('Lifetime Distribution of Tensors')
plt.savefig('lifetime_distribution.png')
plt.show()
