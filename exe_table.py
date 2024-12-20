import os
from pathlib import Path

def extract_avg_execution_times():
    """
    Traverses the results directory, extracts execution times from sim_result.final files,
    computes the average execution time, and compiles the results into a comma-separated list
    with each record on a new line.
    """
    # Define the models
    models = ["BERT", "Inceptionv3", "ResNet152", "SENet154", "VIT"]

    # Define the experiments and their corresponding hostdram percentages
    experiments = {
        'hostdram': 100,       # 100% hostdram
        'hostdram80': 80,      # 80% hostdram
        'hostdram60': 60,      # 60% hostdram
        'hostdram40': 40,      # 40% hostdram
        'hostdram20': 20,      # 20% hostdram
        'ssd': 0               # 0% hostdram
    }

    # List to store the results
    results = []

    # Iterate over each model
    for model in models:
        # Iterate over each experiment
        for exp_suffix, percent in experiments.items():
            # Construct the path to sim_result.final
            sim_result_path = Path('results') / model / f'1024-{exp_suffix}' / 'sim_result.final'

            # Check if the sim_result.final file exists
            if sim_result_path.exists():
                try:
                    with open(sim_result_path, 'r') as file:
                        exec_time = None
                        for line in file:
                            if 'kernel_stat.total.exe_time' in line:
                                # Example line: kernel_stat.total.exe_time = 278938587406
                                parts = line.strip().split('=')
                                if len(parts) == 2:
                                    exec_time_str = parts[1].strip()
                                    if exec_time_str.isdigit():
                                        exec_time = int(exec_time_str)
                                        break  # Stop reading after finding the line
                        if exec_time is not None:
                            avg_time = exec_time / 2  # Divide by two iterations
                            # Append the data as a comma-separated string
                            record = f"{model},{percent},{avg_time}"
                            results.append(record)
                        else:
                            print(f"Execution time not found in {sim_result_path}.")
                except Exception as e:
                    print(f"Error reading {sim_result_path}: {e}")
            else:
                # Folder or file does not exist; skip
                print(f"Folder or file does not exist: {sim_result_path}. Skipping.")

    # Create a comma-separated list with each record on a new line
    comma_separated_list = '\n'.join(results)

    # Output the result
    print(comma_separated_list)

if __name__ == "__main__":
    extract_avg_execution_times()
