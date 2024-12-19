import os
import matplotlib.pyplot as plt
import numpy as np

# Define the root directory containing the results
ROOT_DIR = 'results'

# Define the experiments to consider
EXPERIMENTS = ['ideal', 'hostdram', 'ssd']

# Define colors for each experiment
EXPERIMENT_COLORS = {
    'ideal': 'blue',
    'hostdram': 'green',
    'ssd': 'darkorange'
}

# Folder to save the graphs
GRAPH_DIR = 'graphs'

# Ensure the graphs folder exists
os.makedirs(GRAPH_DIR, exist_ok=True)

# Initialize a dictionary to hold all execution times
# Structure: { model: { batch_size: { experiment: exec_time } } }
execution_times = {}

# Traverse the directory structure
for model in os.listdir(ROOT_DIR):
    model_path = os.path.join(ROOT_DIR, model)
    if not os.path.isdir(model_path):
        continue  # Skip if not a directory

    execution_times[model] = {}

    for batch_exp in os.listdir(model_path):
        batch_exp_path = os.path.join(model_path, batch_exp)
        if not os.path.isdir(batch_exp_path):
            continue  # Skip if not a directory

        # Split the batch size and experiment
        try:
            batch_size, experiment = batch_exp.split('-')
        except ValueError:
            print(f"Skipping invalid directory name: {batch_exp}")
            continue  # Skip directories that don't match the pattern

        if experiment not in EXPERIMENTS:
            print(f"Skipping unknown experiment: {experiment}")
            continue  # Skip unknown experiments

        # Initialize the batch size entry if not present
        if batch_size not in execution_times[model]:
            execution_times[model][batch_size] = {}

        # Path to sim_result.final
        result_file = os.path.join(batch_exp_path, 'sim_result.final')
        if not os.path.isfile(result_file):
            print(f"sim_result.final not found for {model}, Batch {batch_size}, Experiment {experiment}")
            continue  # Skip if the result file does not exist

        # Read the execution time from the file
        exec_time = None
        with open(result_file, 'r') as f:
            for line in f:
                if 'kernel_stat.total.exe_time' in line:
                    try:
                        exec_time = int(line.strip().split('=')[1])
                        exec_time /= 2  # Divide by 2 as per requirement
                    except (IndexError, ValueError):
                        print(f"Invalid format in {result_file}: {line.strip()}")
                    break  # Stop reading after finding the line

        if exec_time is not None:
            execution_times[model][batch_size][experiment] = exec_time
        else:
            print(f"Execution time not found in {result_file}")

# Plotting the bar graphs for each model
for model, batches in execution_times.items():
    batch_sizes = sorted(batches.keys(), key=lambda x: int(x))  # Sort batch sizes numerically
    x = np.arange(len(batch_sizes))  # Label locations
    width = 0.2  # Width of each bar

    fig, ax = plt.subplots(figsize=(10, 6))

    # Initialize positions for each experiment's bars
    offsets = {
        'ideal': -width,
        'hostdram': 0,
        'ssd': width
    }

    # Plot each experiment
    for experiment in EXPERIMENTS:
        exec_times = []
        for batch_size in batch_sizes:
            exec_time = batches[batch_size].get(experiment, 0)  # Default to 0 if not present
            exec_times.append(exec_time)

        ax.bar(x + offsets[experiment], exec_times, width, label=experiment.capitalize(), color=EXPERIMENT_COLORS[experiment])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Execution Time (cycles)')
    ax.set_title(f'Execution Times for {model}')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend()

    # Optional: Improve layout
    fig.tight_layout()

    # Save the figure to the graphs folder
    graph_path = os.path.join(GRAPH_DIR, f'{model}_execution_times.png')
    plt.savefig(graph_path)
    plt.close()

    print(f"Bar graph saved for model: {model} at {graph_path}")

print("All bar graphs have been generated and saved in the 'graphs' folder.")
