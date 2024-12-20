import os
from pathlib import Path

# Define your models and batch sizes (only 1024 is relevant now)
models_batchsizes = [
    ("BERT", [1024]),
    ("Inceptionv3", [1024]),
    ("ResNet152", [1024]),
    ("SENet154", [1024]),
    ("VIT", [1024])
]

# GPU memory size is fixed for all models
fixed_gpu_memory_size = 40

# Define the CPU memory percentages
cpu_memory_percentages = [80, 20]

# Define the maximum memory requirements for each model (in GB)
max_memory_requirements = {
    "BERT": 189.823,
    "Inceptionv3": 89.307,
    "ResNet152": 218.811,
    "SENet154": 344.071,
    "VIT": 45.932
}

# Configuration template with placeholders
config_template = """# Simulation output specifications
output_folder           ../results/{MODEL}/1024-hostdram{PERCENT}p5
stat_output_file        sim_result
# Simulation input specifications
tensor_info_file        ../results/{MODEL}/sim_input/1024Tensor.info 
kernel_info_file        ../results/{MODEL}/sim_input/1024Kernel.info 
kernel_aux_time_file    ../results/{MODEL}/sim_input/1024AuxTime.info 
# System specifications
system_latency_us       45
# CPU specifications
CPU_PCIe_bandwidth_GBps 15.754
CPU_memory_line_GB      {CPU_memory_line_GB}
# GPU specifications
GPU_memory_size_GB      {GPU_memory_size_GB}
GPU_frequency_GHz       1.2
GPU_PCIe_bandwidth_GBps 15.754
GPU_malloc_uspB         0.000000814
# SSD specifications
SSD_PCIe_bandwidth_GBps 3.2
SSD_read_latency_us     12
SSD_write_latency_us    16
SSD_latency_us          20
# PCIe specifications
PCIe_batch_size_page    50
# Simulation parameters
is_simulation           1
is_ideal                0
num_iteration           4
eviction_policy         CUSTOM
use_movement_hints      1
"""

# Directory where config files will be saved
output_base_dir = "gen_p5_configs"

def gen_cost_configs():
    """
    Generates configuration files for each model with batch size 1024
    and varying CPU memory percentages.
    """
    # Ensure the base directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # Iterate over each model and its batch sizes
    for model, batch_sizes in models_batchsizes:
        for batch_size in batch_sizes:
            max_memory = max_memory_requirements.get(model)
            if max_memory is None:
                print(f"Max memory requirement for {model} not defined. Skipping.")
                continue  # Skip models without defined max memory

            for percent in cpu_memory_percentages:
                # Calculate CPU memory line in GB
                cpu_memory_line_gb = round((percent / 100) * max_memory, 3)

                # Define the output folder path based on the percentage
                output_folder = f"results/{model}/1024-hostdram{percent}p5"
                output_folder_path = Path(output_folder).resolve()

                # Check if the results directory already exists
                if output_folder_path.exists():
                    print(f"Skipping generation for {model}-1024-hostdram{percent} as results directory already exists.")
                    continue  # Skip generating this config

                # Prepare the config content by replacing placeholders
                config_content = config_template.format(
                    MODEL=model,
                    PERCENT=percent,
                    CPU_memory_line_GB=cpu_memory_line_gb,
                    GPU_memory_size_GB=fixed_gpu_memory_size
                )

                # Define the output file path
                filename = f"{model}-1024-hostdram{percent}p5.conf"  # Added .conf extension for clarity
                file_path = Path(output_base_dir) / filename

                # Create the directory if it doesn't exist
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # Write the config content to the file
                with open(file_path, 'w') as config_file:
                    config_file.write(config_content)

                print(f"Generated config file: {file_path}")

    print("Configuration file generation completed.")

# Call the function to generate configs
if __name__ == "__main__":
    gen_cost_configs()
