# run_sims.py

import os
import subprocess
import logging
from pathlib import Path
import argparse
from typing import List
from tqdm import tqdm  # Optional: For progress bar
import concurrent.futures
from threading import Lock

def setup_logging(log_file_path: Path):
    """
    Sets up the logging configuration.
    Logs are written to the specified log file and also output to the console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )

def find_config_files(gen_config_dir: Path, selected_experiments: List[str] = None) -> List[Path]:
    """
    Finds all configuration files in the specified directory, optionally filtering by experiments.
    
    Args:
        gen_config_dir (Path): Path to the directory containing config files.
        selected_experiments (List[str], optional): List of experiments to include. If None, include all.
        
    Returns:
        List[Path]: List of configuration file paths.
    """
    if not gen_config_dir.exists():
        logging.error(f"Configuration directory does not exist: {gen_config_dir}")
        return []
    
    all_config_files = list(gen_config_dir.glob("*.conf"))  # Assuming config files have .conf extension
    if not all_config_files:
        logging.warning(f"No configuration files found in directory: {gen_config_dir}")
        return []
    
    if selected_experiments:
        # Filter files based on the selected experiments
        filtered_files = [f for f in all_config_files if any(exp in f.name for exp in selected_experiments)]
        logging.info(f"Filtering configurations for experiments: {', '.join(selected_experiments)}")
    else:
        filtered_files = all_config_files
    
    logging.info(f"Found {len(filtered_files)} configuration file(s) after filtering.")
    return sorted(filtered_files)

def run_simulation(sim_command: str, config_file: Path, cwd: Path) -> subprocess.CompletedProcess:
    """
    Runs the simulation command with the given configuration file.
    
    Args:
        sim_command (str): The simulation executable command.
        config_file (Path): Path to the configuration file.
        cwd (Path): Directory from which to run the command.
        
    Returns:
        subprocess.CompletedProcess: The result of the subprocess.run call.
    """
    cmd = [sim_command, str(config_file)]
    logging.info(f"Starting simulation: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False  # We handle errors manually
        )
        return result
    except Exception as e:
        logging.error(f"Exception occurred while running simulation for {config_file}: {e}")
        return None

def clean_simulation_result(model: str, batch_size: str, experiment: str, base_dir: Path):
    """
    Cleans up the results directory for a specific simulation by deleting all files
    except 'sim_result.final'.
    
    Args:
        model (str): The model name.
        batch_size (str): The batch size.
        experiment (str): The experiment name.
        base_dir (Path): The base directory of the project.
    """
    results_dir = (base_dir / ".." / "results" / model / f"{batch_size}-{experiment}").resolve()
    if not results_dir.exists():
        logging.warning(f"Results directory does not exist: {results_dir}")
        return
    
    for file in results_dir.iterdir():
        if file.is_file() and file.name != "sim_result.final":
            try:
                file.unlink()
                logging.info(f"Deleted file: {file}")
            except Exception as e:
                logging.error(f"Failed to delete file {file}: {e}")

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run simulations based on configuration files."
    )
    parser.add_argument(
        '--config-dir',
        type=str,
        required=True,
        help="Directory containing configuration (.conf) files."
    )
    parser.add_argument(
        '--experiments',
        nargs='+',
        help="Specify which experiments to run. Choices: ideal, hostdram, ssd. If not specified, all experiments are run."
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help="Number of parallel simulations to run. Default is 1 (sequential)."
    )
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    config_dir = Path(args.config_dir).resolve()
    selected_experiments = args.experiments
    parallel_jobs = args.parallel

    # Define other configurations
    BASE_DIR = Path('.').resolve()  # Assuming the script is placed appropriately
    SIM_COMMAND = "./sim"  # The simulation command; ensure it's executable and in PATH or provide full path
    LOG_FILE = BASE_DIR / "simulation_run.log"  # Log file path

    # Set up logging
    setup_logging(LOG_FILE)
    logging.info("=== Simulation Run Started ===")
    if selected_experiments:
        logging.info(f"Selected experiments to run: {', '.join(selected_experiments)}")
    else:
        logging.info("No specific experiments selected; running all experiments.")

    # Find all (filtered) configuration files
    config_files = find_config_files(config_dir, selected_experiments)
    if not config_files:
        logging.warning("No configuration files found after filtering. Exiting.")
        return

    # Initialize counters for summary
    total = len(config_files)
    successes = 0
    failures = 0

    # Lock for thread-safe counter updates
    lock = Lock()

    def process_config(config_file: Path):
        nonlocal successes, failures
        # Extract model, batch_size, and experiment from config file name
        try:
            model, batch_size, experiment = config_file.stem.split('-', 2)
        except ValueError:
            logging.error(f"Config file name format invalid: {config_file.name}. Expected format 'MODEL-BATCHSIZE-EXPERIMENT.conf'.")
            with lock:
                failures += 1
            return

        # Define the expected results directory
        results_dir = (BASE_DIR / ".." / "results" / model / f"{batch_size}-{experiment}").resolve()

        # Check if the results directory already exists
        if results_dir.exists():
            logging.info(f"Skipping simulation for {model}-{batch_size}-{experiment} as results directory already exists.")
            with lock:
                successes += 1  # Consider it as already succeeded
            return

        # Run the simulation
        result = run_simulation(SIM_COMMAND, config_file, cwd=BASE_DIR)
        if result is None:
            # An exception occurred; already logged
            with lock:
                failures += 1
            return
        if result.returncode != 0:
            # Simulation failed; log the error
            logging.error(
                f"Simulation failed for {config_file.name} with return code {result.returncode}.\n"
                f"Stdout: {result.stdout}\nStderr: {result.stderr}"
            )
            with lock:
                failures += 1
        else:
            # Simulation succeeded; optionally log success
            logging.info(f"Simulation succeeded for {config_file.name}.")
            with lock:
                successes += 1
        
        # Perform cleanup for this simulation
        clean_simulation_result(model, batch_size, experiment, BASE_DIR)

    if parallel_jobs > 1:
        # Parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            list(tqdm(executor.map(process_config, config_files), total=total, desc="Running Simulations"))
    else:
        # Sequential execution with progress bar
        for config_file in tqdm(config_files, desc="Running Simulations"):
            process_config(config_file)

    # Summary Report
    logging.info("=== Simulation Run Completed ===")
    logging.info(f"Total Simulations Processed: {total}")
    logging.info(f"Successful Simulations: {successes}")
    logging.info(f"Failed Simulations: {failures}")

if __name__ == "__main__":
    main()
