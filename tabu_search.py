import numpy as np
import matplotlib.pyplot as plt
import random
import os
import glob
import pandas as pd
from datetime import datetime
import time  # Add time module for measuring execution time
import signal
from contextlib import contextmanager

def run_dataset_with_timeout(filename, time_limit=60, max_iterations=1000000):
    """Run tabu search on a single dataset for a fixed time limit."""
    print(f"\nProcessing dataset: {filename}")
    print(f"Running for {time_limit} seconds (max {max_iterations} iterations)...")
    
    # Load data
    N, M, L, p1j, p2j, rl, beta_l, gamma_m, djkm, rho_m, s_j = read_data(filename)
    
    # Initialize solution
    initial_sequence = list(range(N))
    random.shuffle(initial_sequence)
    initial_speeds = {(j, m): 1 for j in range(N) for m in range(M)}
    current_solution = Solution(initial_sequence, initial_speeds)
    
    # Calculate initial objective values
    current_proc_time, current_energy, current_makespan, current_idle = calculate_objectives(
        current_solution, N, M, L, p1j, p2j, rl, beta_l, gamma_m, djkm, rho_m, s_j
    )
    
    # Initialize best solution
    best_solution = Solution(current_solution.sequence.copy(), current_solution.speeds.copy())
    best_proc_time = current_proc_time
    best_energy = current_energy
    best_makespan = current_makespan
    best_idle = current_idle
    
    # Initialize tabu list
    tabu_list = []
    tabu_size = 20
    
    # Track iterations and timing
    iteration = 0
    start_time = time.time()
    end_time = start_time + time_limit
    last_print = start_time
    last_iteration = 0
    
    def print_progress():
        nonlocal last_print, last_iteration
        current_time = time.time()
        if current_time - last_print >= 1.0:  # Print at most once per second
            elapsed = current_time - start_time
            remaining = max(0, time_limit - elapsed)
            iter_per_sec = (iteration - last_iteration) / (current_time - last_print)
            print(f"  Progress: {elapsed:.1f}/{time_limit}s | "
                  f"Iterations: {iteration} | "
                  f"Speed: {iter_per_sec:.1f} it/s | "
                  f"Remaining: {remaining:.1f}s", end='\r')
            last_print = current_time
            last_iteration = iteration
    
    # Main loop - run until time limit or max iterations is reached
    while time.time() < end_time and iteration < max_iterations:
        iteration += 1
        
        # Print progress
        if iteration % 100 == 0:  # Update progress every 100 iterations
            print_progress()
        
        # Generate and evaluate multiple neighbors
        best_neighbor = None
        best_neighbor_score = float('inf')
        
        for _ in range(N):
            neighbor = generate_neighbor(current_solution, N, M, L)
            neighbor_key = (tuple(neighbor.sequence), tuple(sorted(neighbor.speeds.items())))
            
            if neighbor_key not in tabu_list:
                try:
                    proc_time, energy, makespan, idle = calculate_objectives(
                        neighbor, N, M, L, p1j, p2j, rl, beta_l, gamma_m, djkm, rho_m, s_j
                    )
                    
                    # Simple sum of objectives (equal weights)
                    score = proc_time + energy + makespan + idle
                    
                    if score < best_neighbor_score:
                        best_neighbor = neighbor
                        best_neighbor_score = score
                        best_neighbor_values = (proc_time, energy, makespan, idle)
                except Exception as e:
                    print(f"\nWarning: Error in objective calculation: {str(e)}")
                    continue  # Skip this neighbor if there's an error
        
        # Update current solution if we found a valid neighbor
        if best_neighbor is not None:
            current_solution = best_neighbor
            current_proc_time, current_energy, current_makespan, current_idle = best_neighbor_values
            
            # Update best solution if better
            current_score = current_proc_time + current_energy + current_makespan + current_idle
            best_score = best_proc_time + best_energy + best_makespan + best_idle
            
            if current_score < best_score:
                best_solution = Solution(
                    current_solution.sequence.copy(), 
                    current_solution.speeds.copy()
                )
                best_proc_time = current_proc_time
                best_energy = current_energy
                best_makespan = current_makespan
                best_idle = current_idle
            
            # Update tabu list
            tabu_list.append((
                tuple(current_solution.sequence), 
                tuple(sorted(current_solution.speeds.items()))
            ))
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)
    
    total_time = time.time() - start_time
    print(f"\nCompleted {iteration} iterations in {total_time:.1f} seconds "
          f"({iteration/max(1, total_time):.1f} it/s)")
    return best_solution, best_proc_time, best_energy, best_makespan, best_idle, iteration

def run_all_datasets(num_runs=15, time_limit=60):
    """Run all datasets for the specified number of runs and time per run."""
    print(f"\nRunning all datasets {num_runs} times, {time_limit} seconds per run")
    print(f"Total time per dataset: {num_runs * time_limit} seconds")
    
    # Get all dataset files
    dataset_files = sorted(glob.glob("ps*j2m-setup*.txt"))
    
    # Initialize results storage
    all_results = []
    
    # First pass: collect all results and find best values
    best_proc_time = float('inf')
    best_energy = float('inf')
    best_makespan = float('inf')
    best_idle_time = float('inf')
    
    # First pass to find best values
    for filename in dataset_files:
        print(f"\nFirst pass - finding best values for: {filename}")
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}")
            try:
                best_solution, proc_time, energy, makespan, idle, _ = run_dataset_with_timeout(
                    filename, time_limit
                )
                best_proc_time = min(best_proc_time, proc_time)
                best_energy = min(best_energy, energy)
                best_makespan = min(best_makespan, makespan)
                best_idle_time = min(best_idle_time, idle)
            except Exception as e:
                print(f"Error in first pass for {filename}, run {run + 1}: {str(e)}")
                continue
    
    print("\nBest values found across all datasets (normalization factors):")
    print(f"Best Processing Time (b₁): {best_proc_time:.2f}")
    print(f"Best Energy (b₂): {best_energy:.2f}")
    print(f"Best Makespan (b₃): {best_makespan:.2f}")
    print(f"Best Idle Time (b₄): {best_idle_time:.2f}")
    
    # Second pass: collect all results with normalized values
    for filename in dataset_files:
        print(f"\nProcessing dataset: {filename}")
        # Extract problem size from filename
        parts = filename.split('-')
        num_jobs = int(parts[0][2])  # Extract number after 'ps'
        setup_size = int(parts[1].split('_')[0].replace('setup', ''))
        
        # Initialize results for this dataset
        total_iterations = 0
        all_run_results = []
        
        # Run the dataset num_runs times
        for run in range(num_runs):
            print(f"\nRun {run + 1}/{num_runs} for {filename}")
            try:
                best_solution, proc_time, energy, makespan, idle, iterations = run_dataset_with_timeout(
                    filename, time_limit
                )
                total_iterations += iterations
                
                # Store results for this run
                all_run_results.append({
                    'proc_time': proc_time,
                    'energy': energy,
                    'makespan': makespan,
                    'idle': idle,
                    'solution': best_solution,
                    'iterations': iterations
                })
            except Exception as e:
                print(f"Error in run {run + 1}: {str(e)}")
                continue
        
        if not all_run_results:
            print(f"Warning: No successful runs for {filename}")
            continue
            
        # Select the best run based on the sum of objectives
        best_run = min(
            all_run_results,
            key=lambda x: x['proc_time'] + x['energy'] + x['makespan'] + x['idle']
        )
        
        # Calculate normalized values using best overall values
        f1_norm = best_run['proc_time'] / best_proc_time
        f2_norm = best_run['energy'] / best_energy
        f3_norm = best_run['makespan'] / best_makespan
        f4_norm = best_run['idle'] / best_idle_time
        
        # Calculate fitness with equal weights (0.25 each)
        fitness = 0.25 * (f1_norm + f2_norm + f3_norm + f4_norm)
        
        # Store the result for this dataset
        result = {
            'Dataset': filename,
            'Number of Jobs': num_jobs,
            'Setup Size': setup_size,
            'Tabu Size': 20,
            # Raw objective values from best run
            'f1_Processing_Time': round(best_run['proc_time'], 2),
            'f2_Energy': round(best_run['energy'], 2),
            'f3_Makespan': round(best_run['makespan'], 2),
            'f4_Idle_Time': round(best_run['idle'], 2),
            # Best values (b values)
            'b1_Best_Processing_Time': round(best_proc_time, 2),
            'b2_Best_Energy': round(best_energy, 2),
            'b3_Best_Makespan': round(best_makespan, 2),
            'b4_Best_Idle_Time': round(best_idle_time, 2),
            # Normalized values
            'f1_norm': round(f1_norm, 4),
            'f2_norm': round(f2_norm, 4),
            'f3_norm': round(f3_norm, 4),
            'f4_norm': round(f4_norm, 4),
            # Fitness value
            'Fitness_Value': round(fitness, 4),
            'Total Execution Time': num_runs * time_limit,
            'Total Iterations': total_iterations,
            'Best Sequence': str(best_run['solution'].sequence),
            'Best Speeds': str(dict(best_run['solution'].speeds))
        }
        all_results.append(result)
    
    # Create DataFrame and sort by number of jobs and setup size
    df = pd.DataFrame(all_results)
    df = df.sort_values(['Number of Jobs', 'Setup Size'])
    
    # Save to CSV with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'tabu_search_results_{timestamp}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\nResults have been saved to {csv_filename}")
    
    # Also save to Excel with better formatting
    excel_filename = f'tabu_search_results_{timestamp}.xlsx'
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Results', index=False)
        # Auto-adjust column widths
        worksheet = writer.sheets['Results']
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).apply(len).max(),
                len(col)
            )
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)
    
    print(f"Results have also been saved to {excel_filename} with better formatting")
    return df
class Solution:
    def __init__(self, sequence, speeds):
        self.sequence = sequence  # Job sequence
        self.speeds = speeds      # Dictionary of (job, machine) -> speed_level
        
def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    N = int(lines[0].split(':')[1])  # number of jobs
    M = int(lines[1].split(':')[1])  # number of machines
    L = int(lines[2].split(':')[1])  # number of speeds
    
    # Processing times for each machine
    p1j = np.array([int(x) for x in lines[3].split(':')[1].split()])  # Machine 1
    p2j = np.array([int(x) for x in lines[4].split(':')[1].split()])  # Machine 2
    
    rl = np.array([float(x) for x in lines[5].split(':')[1].split()])  # speed ratios
    beta_l = np.array([float(x) for x in lines[6].split(':')[1].split()])  # power consumption factors
    gamma_m = np.array([float(x) for x in lines[7].split(':')[1].split()])  # idle power consumption
    
    # Setup times for each machine (djkm)
    djkm = np.zeros((M, N, N))
    
    # Read d1jk (setup times for machine 1)
    for i in range(9, 9+N):
        djkm[0][i-9] = np.array([int(x) for x in lines[i].split()])
    
    # Read d2jk (setup times for machine 2)
    for i in range(10+N, 10+2*N):
        djkm[1][i-(10+N)] = np.array([int(x) for x in lines[i].split()])
    
    # Power cost rates for each machine
    rho_m = np.array([1.0, 1.0])
    
    # Delay between machines for each job (s_j)
    s_j = np.zeros(N)  # Initialize with zero delays
    
    return N, M, L, p1j, p2j, rl, beta_l, gamma_m, djkm, rho_m, s_j

def calculate_objectives(solution, N, M, L, p1j, p2j, rl, beta_l, gamma_m, djkm, rho_m, s_j):
    sequence = solution.sequence
    speeds = solution.speeds
    
    # Initialize variables
    completion_times = np.zeros((N, M))
    machine_times = np.zeros(M)
    total_processing_time = 0
    total_energy = 0
    
    # Calculate completion times and objectives
    for idx, job in enumerate(sequence):
        for m in range(M):
            # Get processing time and speed level
            p_time = p1j[job] if m == 0 else p2j[job]
            speed_level = speeds[(job, m)]
            
            # Calculate actual processing time with speed factor
            actual_p_time = p_time / rl[speed_level]
            
            # Add setup time if not first job
            setup_time = 0
            if idx > 0:
                prev_job = sequence[idx-1]
                setup_time = djkm[m][job][prev_job]
            
            # Calculate start and completion times
            if m == 0:  # First machine
                start_time = machine_times[m]
            else:  # Second machine
                # Add delay between machines (s_j)
                start_time = max(machine_times[m], completion_times[job][m-1] + s_j[job])
            
            completion_times[job][m] = start_time + setup_time + actual_p_time
            machine_times[m] = completion_times[job][m]
            
            # Update total processing time
            total_processing_time += actual_p_time
            
            # Calculate energy consumption for processing
            energy = (actual_p_time / 60) * beta_l[speed_level] * rho_m[m]
            total_energy += energy
    
    # Calculate makespan
    makespan = np.max(completion_times[:, -1])
    
    # Calculate idle times
    idle_times = np.zeros(M)
    for m in range(M):
        idle_times[m] = makespan - machine_times[m]
        # Add idle energy consumption
        total_energy += (idle_times[m] / 60) * gamma_m[m] * rho_m[m]
    
    total_idle_time = np.sum(idle_times)
    
    return total_processing_time, total_energy, makespan, total_idle_time

def generate_neighbor(solution, N, M, L):
    new_sequence = solution.sequence.copy()
    new_speeds = solution.speeds.copy()
    
    # Randomly choose between sequence swap or speed change
    if random.random() < 0.7:  # 70% chance of sequence swap
        i, j = random.sample(range(len(new_sequence)), 2)
        new_sequence[i], new_sequence[j] = new_sequence[j], new_sequence[i]
    else:  # 30% chance of speed change
        job = random.randint(0, N-1)
        machine = random.randint(0, M-1)
        new_speed = random.randint(0, L-1)
        new_speeds[(job, machine)] = new_speed
    
    return Solution(new_sequence, new_speeds)

def tabu_search(filename, max_iterations=10, tabu_size=5):
    print(f"\nProcessing file: {filename}")
    
    # Load data
    N, M, L, p1j, p2j, rl, beta_l, gamma_m, djkm, rho_m, s_j = read_data(filename)
    
    # Initialize solution
    initial_sequence = list(range(N))
    random.shuffle(initial_sequence)
    
    # Initialize speeds (start with normal speed)
    initial_speeds = {(j, m): 1 for j in range(N) for m in range(M)}
    current_solution = Solution(initial_sequence, initial_speeds)
    
    # Calculate initial objective values
    current_proc_time, current_energy, current_makespan, current_idle = calculate_objectives(
        current_solution, N, M, L, p1j, p2j, rl, beta_l, gamma_m, djkm, rho_m, s_j
    )
    
    # Initialize best solution
    best_solution = Solution(current_solution.sequence.copy(), current_solution.speeds.copy())
    best_proc_time = current_proc_time
    best_energy = current_energy
    best_makespan = current_makespan
    best_idle = current_idle
    
    # Initialize tabu list and history
    tabu_list = []
    history = {
        'processing_time': [],
        'energy': [],
        'makespan': [],
        'idle_time': []
    }
    
    # Main loop
    for iteration in range(max_iterations):
        best_neighbor = None
        best_neighbor_score = float('inf')
        
        # Generate and evaluate multiple neighbors
        for _ in range(N):
            neighbor = generate_neighbor(current_solution, N, M, L)
            neighbor_key = (tuple(neighbor.sequence), tuple(sorted(neighbor.speeds.items())))
            
            if neighbor_key not in tabu_list:
                proc_time, energy, makespan, idle = calculate_objectives(
                    neighbor, N, M, L, p1j, p2j, rl, beta_l, gamma_m, djkm, rho_m, s_j
                )
                
                # Simple sum of objectives (equal weights)
                score = proc_time + energy + makespan + idle
                
                if score < best_neighbor_score:
                    best_neighbor = neighbor
                    best_neighbor_score = score
                    best_neighbor_values = (proc_time, energy, makespan, idle)
        
        # Update current solution
        if best_neighbor is not None:
            current_solution = best_neighbor
            current_proc_time, current_energy, current_makespan, current_idle = best_neighbor_values
            
            # Update best solution if better
            current_score = current_proc_time + current_energy + current_makespan + current_idle
            best_score = best_proc_time + best_energy + best_makespan + best_idle
            
            if current_score < best_score:
                best_solution = Solution(current_solution.sequence.copy(), 
                                      current_solution.speeds.copy())
                best_proc_time = current_proc_time
                best_energy = current_energy
                best_makespan = current_makespan
                best_idle = current_idle
            
            # Update tabu list
            tabu_list.append((tuple(current_solution.sequence), 
                            tuple(sorted(current_solution.speeds.items()))))
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)
        
        # Record history
        history['processing_time'].append(current_proc_time)
        history['energy'].append(current_energy)
        history['makespan'].append(current_makespan)
        history['idle_time'].append(current_idle)
        
        print(f"Iteration {iteration + 1}:")
        print(f"  Processing Time = {current_proc_time:.2f}")
        print(f"  Energy = {current_energy:.2f}")
        print(f"  Makespan = {current_makespan:.2f}")
        print(f"  Idle Time = {current_idle:.2f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['processing_time'])
    plt.title(f'Processing Time History\n{os.path.basename(filename)}')
    plt.xlabel('Iteration')
    plt.ylabel('Processing Time')
    
    plt.subplot(2, 2, 2)
    plt.plot(history['energy'])
    plt.title('Energy Consumption History')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    
    plt.subplot(2, 2, 3)
    plt.plot(history['makespan'])
    plt.title('Makespan History')
    plt.xlabel('Iteration')
    plt.ylabel('Makespan')
    
    plt.subplot(2, 2, 4)
    plt.plot(history['idle_time'])
    plt.title('Idle Time History')
    plt.xlabel('Iteration')
    plt.ylabel('Idle Time')
    
    plt.tight_layout()
    plot_filename = f'tabu_search_results_{os.path.basename(filename).replace(".txt", "")}.png'
    plt.savefig(plot_filename)
    plt.close()
    
    return best_solution, best_proc_time, best_energy, best_makespan, best_idle, history

def process_all_datasets():
    # Use fixed tabu size of 20 for all datasets
    TABU_SIZE = 20
    timing_results = []
    
    print("\nTesting with fixed tabu size of 20 for all datasets")
    
    # Get all dataset files
    dataset_files = glob.glob("ps*j2m-setup*.txt")
    
    # First pass: collect all results and find best values
    initial_results = {}
    best_proc_time = float('inf')
    best_energy = float('inf')
    best_makespan = float('inf')
    best_idle_time = float('inf')
    
    # First pass to collect all results and find best values
    for filename in dataset_files:
        print(f"\nProcessing dataset {filename} with tabu size {TABU_SIZE}")
        
        # Add per-dataset timing
        dataset_start_time = time.time()  # Start timing for this dataset
        
        results = tabu_search(filename, max_iterations=10, tabu_size=TABU_SIZE)
        best_solution, proc_time, energy, makespan, idle, _ = results
        
        # Calculate execution time for this dataset
        dataset_end_time = time.time()
        dataset_execution_time = dataset_end_time - dataset_start_time
        
        initial_results[filename] = {
            'solution': best_solution,
            'proc_time': proc_time,
            'energy': energy,
            'makespan': makespan,
            'idle_time': idle,
            'execution_time': dataset_execution_time,
            'tabu_size': TABU_SIZE  # Store the fixed tabu size
        }
        
        # Update best values
        best_proc_time = min(best_proc_time, proc_time)
        best_energy = min(best_energy, energy)
        best_makespan = min(best_makespan, makespan)
        best_idle_time = min(best_idle_time, idle)
    
    print("\nTesting completed with fixed tabu size")
    
    # Calculate total execution time
    total_execution_time = sum(result['execution_time'] for result in initial_results.values())
    
    timing_results.append({
        'Description': 'Fixed Tabu Size (20)',
        'Total Execution Time (s)': round(total_execution_time, 2)
    })
    
    print("\nBest values found across all datasets (normalization factors):")
    print(f"Best Processing Time (b₁): {best_proc_time:.2f}")
    print(f"Best Energy (b₂): {best_energy:.2f}")
    print(f"Best Makespan (b₃): {best_makespan:.2f}")
    print(f"Best Idle Time (b₄): {best_idle_time:.2f}")
    
    # Prepare data for Excel
    excel_data = []
    all_results = {}
    
    # Second pass: calculate fitness using best values as normalization factors
    for filename, result in initial_results.items():
        # Extract problem size from filename
        parts = filename.split('-')
        num_jobs = int(parts[0][2])  # Extract number after 'ps'
        setup_size = int(parts[1].split('_')[0].replace('setup', ''))
        
        # Calculate fitness using best values as normalization factors
        f1_norm = result['proc_time'] / best_proc_time
        f2_norm = result['energy'] / best_energy
        f3_norm = result['makespan'] / best_makespan
        f4_norm = result['idle_time'] / best_idle_time
        
        # Calculate fitness with equal weights (0.25 each)
        fitness = 0.25 * (f1_norm + f2_norm + f3_norm + f4_norm)
        
        # Prepare row for Excel
        excel_row = {
            'Dataset': filename,
            'Number of Jobs': num_jobs,
            'Setup Size': setup_size,
            'Tabu Size Used': TABU_SIZE,  # Add the fixed tabu size
            'Best Sequence': str(result['solution'].sequence),
            'Processing Time (f₁)': round(result['proc_time'], 2),
            'Energy Consumption (f₂)': round(result['energy'], 2),
            'Makespan (f₃)': round(result['makespan'], 2),
            'Idle Time (f₄)': round(result['idle_time'], 2),
            'Normalized f₁': round(f1_norm, 4),
            'Normalized f₂': round(f2_norm, 4),
            'Normalized f₃': round(f3_norm, 4),
            'Normalized f₄': round(f4_norm, 4),
            'Fitness Value': round(fitness, 4),
            'Speed Settings': str(dict(result['solution'].speeds)),
            'Execution Time (s)': round(result['execution_time'], 2)
        }
        excel_data.append(excel_row)
        
        all_results[filename] = {
            'sequence': result['solution'].sequence,
            'speeds': result['solution'].speeds,
            'processing_time': result['proc_time'],
            'energy': result['energy'],
            'makespan': result['makespan'],
            'idle_time': result['idle_time'],
            'fitness': fitness,
            'tabu_size': TABU_SIZE  # Store the fixed tabu size
        }
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(excel_data)
    
    # Sort by number of jobs and setup size
    df = df.sort_values(['Number of Jobs', 'Setup Size'])
    
    # Add a row with normalization factors
    normalization_row = pd.DataFrame([{
        'Dataset': 'Normalization Factors (b values)',
        'Processing Time (f₁)': round(best_proc_time, 2),
        'Energy Consumption (f₂)': round(best_energy, 2),
        'Makespan (f₃)': round(best_makespan, 2),
        'Idle Time (f₄)': round(best_idle_time, 2),
        'Execution Time (s)': round(total_execution_time, 2)
    }])
    
    # Concatenate the normalization row at the top
    df = pd.concat([normalization_row, df], ignore_index=True)
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f'fixed_tabu_size_{timestamp}.xlsx'
    
    # Create Excel writer with xlsxwriter engine
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # Write the main results
        df.to_excel(writer, sheet_name='Results', index=False)
        
        # Auto-adjust column widths
        worksheet = writer.sheets['Results']
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).apply(len).max(),
                len(col)
            )
            worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2
    
    print(f"\nResults have been saved to {excel_filename}")
    
    # Create a summary Excel file for timing results
    timing_df = pd.DataFrame(timing_results)
    timing_filename = f'fixed_tabu_timing_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    
    with pd.ExcelWriter(timing_filename, engine='openpyxl') as writer:
        timing_df.to_excel(writer, sheet_name='Timing Results', index=False)
        
        # Auto-adjust column widths
        worksheet = writer.sheets['Timing Results']
        for idx, col in enumerate(timing_df.columns):
            max_length = max(
                timing_df[col].astype(str).apply(len).max(),
                len(col)
            )
            worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2
    
    print(f"\nTiming summary has been saved to {timing_filename}")
    return all_results

# Add timeout context manager
@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def run_dataset_with_timeout(filename, time_limit=60):
    print(f"\nProcessing dataset: {filename}")
    print(f"Running for {time_limit} seconds...")
    
    # Initialize best solution tracking
    best_solution = None
    best_proc_time = float('inf')
    best_energy = float('inf')
    best_makespan = float('inf')
    best_idle = float('inf')
    
    # Load data
    N, M, L, p1j, p2j, rl, beta_l, gamma_m, djkm, rho_m, s_j = read_data(filename)
    
    # Initialize solution
    initial_sequence = list(range(N))
    random.shuffle(initial_sequence)
    initial_speeds = {(j, m): 1 for j in range(N) for m in range(M)}
    current_solution = Solution(initial_sequence, initial_speeds)
    
    # Calculate initial objective values
    current_proc_time, current_energy, current_makespan, current_idle = calculate_objectives(
        current_solution, N, M, L, p1j, p2j, rl, beta_l, gamma_m, djkm, rho_m, s_j
    )
    
    # Initialize best solution
    best_solution = Solution(current_solution.sequence.copy(), current_solution.speeds.copy())
    best_proc_time = current_proc_time
    best_energy = current_energy
    best_makespan = current_makespan
    best_idle = current_idle
    
    # Initialize tabu list
    tabu_list = []
    tabu_size = 20
    
    # Track iterations
    iteration = 0
    start_time = time.time()
    
    # Main loop - run until time limit is reached
    while time.time() - start_time < time_limit:
        best_neighbor = None
        best_neighbor_score = float('inf')
        
        # Generate and evaluate multiple neighbors
        for _ in range(N):
            neighbor = generate_neighbor(current_solution, N, M, L)
            neighbor_key = (tuple(neighbor.sequence), tuple(sorted(neighbor.speeds.items())))
            
            if neighbor_key not in tabu_list:
                proc_time, energy, makespan, idle = calculate_objectives(
                    neighbor, N, M, L, p1j, p2j, rl, beta_l, gamma_m, djkm, rho_m, s_j
                )
                
                # Simple sum of objectives (equal weights)
                score = proc_time + energy + makespan + idle
                
                if score < best_neighbor_score:
                    best_neighbor = neighbor
                    best_neighbor_score = score
                    best_neighbor_values = (proc_time, energy, makespan, idle)
        
        # Update current solution
        if best_neighbor is not None:
            current_solution = best_neighbor
            current_proc_time, current_energy, current_makespan, current_idle = best_neighbor_values
            
            # Update best solution if better
            current_score = current_proc_time + current_energy + current_makespan + current_idle
            best_score = best_proc_time + best_energy + best_makespan + best_idle
            
            if current_score < best_score:
                best_solution = Solution(current_solution.sequence.copy(), 
                                      current_solution.speeds.copy())
                best_proc_time = current_proc_time
                best_energy = current_energy
                best_makespan = current_makespan
                best_idle = current_idle
            
            # Update tabu list
            tabu_list.append((tuple(current_solution.sequence), 
                            tuple(sorted(current_solution.speeds.items()))))
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)
        
        iteration += 1
        
        # Print progress every 10 seconds
        elapsed_time = time.time() - start_time
        if iteration % 100 == 0:
            print(f"Time elapsed: {elapsed_time:.1f}s, Iteration: {iteration}")
    
    print(f"Completed {iteration} iterations in {time_limit} seconds")
    return best_solution, best_proc_time, best_energy, best_makespan, best_idle, iteration

def run_all_datasets(num_runs=15, time_limit=60):
    print(f"\nRunning all datasets {num_runs} times, {time_limit} seconds per run")
    print(f"Total time per dataset: {num_runs * time_limit} seconds")
    
    # Get all dataset files
    dataset_files = glob.glob("ps*j2m-setup*.txt")
    
    # Initialize results storage
    all_results = []
    
    # First pass: collect all results and find best values for normalization
    best_proc_time = float('inf')
    best_energy = float('inf')
    best_makespan = float('inf')
    best_idle_time = float('inf')
    
    # First pass to find best values
    for filename in dataset_files:
        print(f"\nProcessing dataset: {filename}")
        for run in range(num_runs):
            print(f"\nRun {run + 1}/{num_runs} for {filename}")
            best_solution, proc_time, energy, makespan, idle, _ = run_dataset_with_timeout(filename, time_limit)
            best_proc_time = min(best_proc_time, proc_time)
            best_energy = min(best_energy, energy)
            best_makespan = min(best_makespan, makespan)
            best_idle_time = min(best_idle_time, idle)
    
    print("\nBest values found across all datasets (normalization factors):")
    print(f"Best Processing Time (b₁): {best_proc_time:.2f}")
    print(f"Best Energy (b₂): {best_energy:.2f}")
    print(f"Best Makespan (b₃): {best_makespan:.2f}")
    print(f"Best Idle Time (b₄): {best_idle_time:.2f}")
    
    # Second pass: collect all results with normalized values
    for filename in dataset_files:
        print(f"\nProcessing dataset: {filename}")
        # Extract problem size from filename
        parts = filename.split('-')
        num_jobs = int(parts[0][2])  # Extract number after 'ps'
        setup_size = int(parts[1].split('_')[0].replace('setup', ''))
        
        # Initialize results for this dataset
        total_iterations = 0
        all_run_results = []
        
        # Run the dataset num_runs times
        for run in range(num_runs):
            print(f"\nRun {run + 1}/{num_runs} for {filename}")
            best_solution, proc_time, energy, makespan, idle, iterations = run_dataset_with_timeout(filename, time_limit)
            total_iterations += iterations
            
            # Store results for this run
            run_result = {
                'proc_time': proc_time,
                'energy': energy,
                'makespan': makespan,
                'idle': idle,
                'solution': best_solution,
                'iterations': iterations
            }
            all_run_results.append(run_result)
        
        # Randomly select one run for this dataset
        random_run = random.choice(all_run_results)
        
        # Calculate normalized values using best overall values
        f1_norm = random_run['proc_time'] / best_proc_time
        f2_norm = random_run['energy'] / best_energy
        f3_norm = random_run['makespan'] / best_makespan
        f4_norm = random_run['idle'] / best_idle_time
        
        # Calculate fitness with equal weights (0.25 each)
        fitness = 0.25 * (f1_norm + f2_norm + f3_norm + f4_norm)
        
        # Store the result for this dataset
        result = {
            'Dataset': filename,
            'Number of Jobs': num_jobs,
            'Setup Size': setup_size,
            'Tabu Size': 20,  # Fixed tabu size
            # Raw objective values from random run
            'f1_Processing_Time': round(random_run['proc_time'], 2),
            'f2_Energy': round(random_run['energy'], 2),
            'f3_Makespan': round(random_run['makespan'], 2),
            'f4_Idle_Time': round(random_run['idle'], 2),
            # Best values (b values)
            'b1_Best_Processing_Time': round(best_proc_time, 2),
            'b2_Best_Energy': round(best_energy, 2),
            'b3_Best_Makespan': round(best_makespan, 2),
            'b4_Best_Idle_Time': round(best_idle_time, 2),
            # Normalized values
            'f1_norm': round(f1_norm, 4),
            'f2_norm': round(f2_norm, 4),
            'f3_norm': round(f3_norm, 4),
            'f4_norm': round(f4_norm, 4),
            # Fitness value
            'Fitness_Value': round(fitness, 4),
            'Total Execution Time': 900,  # Fixed at 900 seconds (15 runs × 60 seconds)
            'Total Iterations': total_iterations,
            'Best Sequence': str(random_run['solution'].sequence),
            'Best Speeds': str(dict(random_run['solution'].speeds))
        }
        all_results.append(result)
    
    # Create DataFrame and sort by number of jobs and setup size
    df = pd.DataFrame(all_results)
    df = df.sort_values(['Number of Jobs', 'Setup Size'])
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'random_results_{timestamp}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\nResults have been saved to {csv_filename}")
    
    return df

if __name__ == "__main__":
    # Run all datasets 15 times, 60 seconds per run
    results = run_all_datasets(num_runs=15, time_limit=60)
    
    # Print summary of results
    print("\nSummary of Results:")
    print("-" * 50)
    for _, row in results.iterrows():
        print(f"\nDataset: {row['Dataset']}")
        print(f"Tabu Size: {row['Tabu Size']}")
        print(f"f1 (Processing Time): {row['f1_Processing_Time']:.2f}")
        print(f"f2 (Energy): {row['f2_Energy']:.2f}")
        print(f"f3 (Makespan): {row['f3_Makespan']:.2f}")
        print(f"f4 (Idle Time): {row['f4_Idle_Time']:.2f}")
        print(f"b1 (Best Processing Time): {row['b1_Best_Processing_Time']:.2f}")
        print(f"b2 (Best Energy): {row['b2_Best_Energy']:.2f}")
        print(f"b3 (Best Makespan): {row['b3_Best_Makespan']:.2f}")
        print(f"b4 (Best Idle Time): {row['b4_Best_Idle_Time']:.2f}")
        print(f"f1_norm: {row['f1_norm']:.4f}")
        print(f"f2_norm: {row['f2_norm']:.4f}")
        print(f"f3_norm: {row['f3_norm']:.4f}")
        print(f"f4_norm: {row['f4_norm']:.4f}")
        print(f"Fitness Value: {row['Fitness_Value']:.4f}")
        print(f"Total Execution Time: {row['Total Execution Time']}s")
        print(f"Total Iterations: {row['Total Iterations']}") 