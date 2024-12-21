import os
import matplotlib.pyplot as plt
import numpy as np

# Define the root directory containing the results
ROOT_DIR = 'results'

# Define the experiments to consider
EXPERIMENTS = ['hostdram20', 'hostdram80', 'hostdram20p5', 'hostdram80p5']

# Define colors for each experiment
EXPERIMENT_COLORS = {
    'hostdram20': 'blue',
    'hostdram80': 'green',
    'hostdram20p5': 'darkorange',
    'hostdram80p5': 'red'
}

# Folder to save the graph
GRAPH_DIR = 'graphs-5'

# Ensure the graphs folder exists
os.makedirs(GRAPH_DIR, exist_ok=True)

# Initialize a dictionary to hold all execution times
# Structure: { model: { experiment: [exec_time1, exec_time2, ...] } }
execution_times = {}

# Traverse the directory structure
for model in os.listdir(ROOT_DIR):
    model_path = os.path.join(ROOT_DIR, model)
    if not os.path.isdir(model_path):
        continue  # Skip if not a directory

    execution_times[model] = {experiment: [] for experiment in EXPERIMENTS}

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
                        if "p5" in result_file:
                            exec_time /= 4
                        else:
                            exec_time /= 2
                    except (IndexError, ValueError):
                        print(f"Invalid format in {result_file}: {line.strip()}")
                    break  # Stop reading after finding the line

        if exec_time is not None:
            execution_times[model][experiment].append(exec_time)
        else:
            print(f"Execution time not found in {result_file}")

# Now, compute the average execution time per model and experiment
avg_execution_times = {}
for model, experiments in execution_times.items():
    avg_execution_times[model] = {}
    for experiment, times in experiments.items():
        if times:
            avg_execution_times[model][experiment] = np.mean(times)
        else:
            avg_execution_times[model][experiment] = 0  # or np.nan if preferred

# Plotting all models on the same bar graph
models = sorted(avg_execution_times.keys())
num_models = len(models)
num_experiments = len(EXPERIMENTS)

# Define the positions
x = np.arange(num_models)  # the label locations
total_width = 0.8
bar_width = total_width / num_experiments

fig, ax = plt.subplots(figsize=(12, 8))

# Plot each experiment
for i, experiment in enumerate(EXPERIMENTS):
    # Calculate position offset for each experiment
    offset = (i - num_experiments / 2) * bar_width + bar_width / 2
    exec_times = [avg_execution_times[model][experiment] for model in models]
    ax.bar(x + offset, exec_times, bar_width, label=experiment.capitalize(), color=EXPERIMENT_COLORS[experiment])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Models')
ax.set_ylabel('Average Execution Time (cycles)')
ax.set_title('Execution Times for All Models Across Experiments')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()

# Optional: Improve layout
fig.tight_layout()

# Save the figure to the graphs folder
graph_path = os.path.join(GRAPH_DIR, 'all_models_execution_times.png')
plt.savefig(graph_path)
plt.close()

print(f"Combined bar graph saved at {graph_path}")
print("All bar graphs have been generated and saved in the 'graphs-5' folder.")
