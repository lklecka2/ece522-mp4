import os
from pathlib import Path

def get_folder_sizes(base_dir='results'):
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Base directory '{base_dir}' does not exist.")
        return
    
    size_list = []
    
    # Iterate over MODEL directories
    for model_dir in base_path.iterdir():
        if model_dir.is_dir():
            sim_input_dir = model_dir / 'sim_input'
            if not sim_input_dir.exists():
                print(f"Warning: 'sim_input' directory not found in '{model_dir.name}'. Skipping.")
                continue
            
            # Iterate over *AuxTime.info files to identify BATCHSIZE
            for aux_time_file in sim_input_dir.glob('*AuxTime.info'):
                # Extract BATCHSIZE by removing 'AuxTime.info' suffix
                batchsize_str = aux_time_file.stem.replace('AuxTime', '')
                if not batchsize_str:
                    print(f"Warning: Could not extract BATCHSIZE from '{aux_time_file.name}'. Skipping.")
                    continue
                
                # Define corresponding Kernel and Tensor files
                kernel_file = sim_input_dir / f"{batchsize_str}Kernel.info"
                tensor_file = sim_input_dir / f"{batchsize_str}Tensor.info"
                
                # Check existence of all three files
                if not kernel_file.exists() or not tensor_file.exists():
                    print(f"Warning: Missing files for BATCHSIZE '{batchsize_str}' in '{model_dir.name}'. Skipping.")
                    continue
                
                # Get file sizes
                try:
                    aux_size = aux_time_file.stat().st_size
                    kernel_size = kernel_file.stat().st_size
                    tensor_size = tensor_file.stat().st_size
                except Exception as e:
                    print(f"Error accessing files for BATCHSIZE '{batchsize_str}' in '{model_dir.name}': {e}")
                    continue
                
                total_size = aux_size + kernel_size + tensor_size
                size_list.append((total_size, model_dir.name, batchsize_str))
    
    # Sort the list from smallest to largest total size
    size_list.sort()
    
    # Print the sorted results
    print("Folder Sizes (from smallest to largest):")
    for total_size, model, batchsize in size_list:
        print(f"Model: {model}, BATCHSIZE: {batchsize}, Total Size: {total_size} bytes")

if __name__ == "__main__":
    get_folder_sizes()
