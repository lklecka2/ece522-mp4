# reg_configs.py

import os
from pathlib import Path
import argparse

# Define your models and batch sizes
models_batchsizes = [
    ("BERT", [256, 384, 512, 640, 768, 1024]),
    ("Inceptionv3", [512, 768, 1024, 1152, 1280, 1536]),  # 1792 is excluded
    ("ResNet152", [256, 512, 768, 1024, 1280, 1536]),
    ("SENet154", [256, 512, 768, 1024]),
    ("VIT", [256, 512, 768, 1024, 1280])
]

# Optional: Define GPU memory size overrides
# Format: {"MODEL-BATCHSIZE": gpu_mem_size}
default_gpu_memory_size = 40
gpu_mem_overrides = {
    # Example:
    # "BERT-256": 32,
    # "ResNet152-512": 48,
}

# Define the experiments with their specific parameters
experiments = {
    "ideal": {
        "is_ideal": 1,
        "CPU_memory_line_GB": -1,  # Default value; won't be used in 'ideal'
    },
    "hostdram": {
        "is_ideal": 0,
        "CPU_memory_line_GB": 1024,
    },
    "ssd": {
        "is_ideal": 0,
        "CPU_memory_line_GB": 0,
    }
}

# Configuration template with placeholders
config_template = """# Simulation output specifications
output_folder           ../results/{MODEL}/{BATCHSIZE}-{EXPERIMENT} 
stat_output_file        sim_result
# Simulation input specifications
tensor_info_file        ../results/{MODEL}/sim_input/{BATCHSIZE}Tensor.info 
kernel_info_file        ../results/{MODEL}/sim_input/{BATCHSIZE}Kernel.info 
kernel_aux_time_file    ../results/{MODEL}/sim_input/{BATCHSIZE}AuxTime.info 
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
is_ideal                {is_ideal}
num_iteration           2
eviction_policy         LRU
use_movement_hints      1
"""

def generate_configs(output_base_dir: Path):
    """
    Generates configuration files based on the models, batch sizes, and experiments.
    """
    # Ensure the base directory exists
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over each model and its batch sizes
    for model, batch_sizes in models_batchsizes:
        for batch_size in batch_sizes:
            for experiment_name, params in experiments.items():
                # Determine GPU memory size
                key = f"{model}-{batch_size}"
                gpu_memory_size = gpu_mem_overrides.get(key, default_gpu_memory_size)
                
                # Define the output folder path based on the experiment
                output_folder = f"../results/{model}/{batch_size}-{experiment_name}"
                output_folder_path = Path(output_folder).resolve()
                
                # Check if the results directory already exists
                if output_folder_path.exists():
                    print(f"Skipping generation for {model}-{batch_size}-{experiment_name} as results directory already exists.")
                    continue  # Skip generating this config
                
                # Prepare the config content by replacing placeholders
                config_content = config_template.format(
                    MODEL=model,
                    BATCHSIZE=batch_size,
                    EXPERIMENT=experiment_name,
                    is_ideal=params["is_ideal"],
                    CPU_memory_line_GB=params["CPU_memory_line_GB"],
                    GPU_memory_size_GB=gpu_memory_size
                )
                
                # Define the output file path
                filename = f"{model}-{batch_size}-{experiment_name}.conf"  # Added .conf extension for clarity
                file_path = output_base_dir / filename
                
                # Create the directory if it doesn't exist
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write the config content to the file
                with open(file_path, 'w') as config_file:
                    config_file.write(config_content)
                
                print(f"Generated config file: {file_path}")

    print("Configuration file generation completed.")

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate configuration files for simulations."
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="gen_configs",
        help="Directory where config files will be saved. Default is 'gen_configs'."
    )
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    output_base_dir = Path(args.output_dir).resolve()

    # Generate configuration files
    generate_configs(output_base_dir)

if __name__ == "__main__":
    main()
