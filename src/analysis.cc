
#include "analysis.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include <algorithm>
#include <fstream>
#include <iostream>

#include "ast.h"
#include "simulationUtils.h"
#include "simulator.h"
#include "printUtils.h"

using Simulator::TensorMovementHint;
using Simulator::TensorLocation;

extern std::string migration_policy_str;
extern std::string eviction_policy_str;

extern double GPU_frequency_GHz;
extern double GPU_memory_size_GB;
extern double CPU_PCIe_bandwidth_GBps;
extern double SSD_PCIe_bandwidth_GBps;
extern double GPU_malloc_uspB;
// extern double GPU_free_uspB;
extern int prefetch_degree;
extern int borden;
extern int is_transformer;
double CPU_memory_line_GB = -1;
double SSD_latency_us = -1;
double system_latency_us = -1;
double delta_parameter = -1;
double loosen_parameter = 1;

long long memory_offset_intermediate = 0;
long long memory_offset_weights = 0;
int kernel_index = 0;
int prefetch_optimize = 1;
std::vector<Tensor *> tensor_list;
std::vector<CUDAKernel> kernel_list;

// TODO: Important: The global list for all inactive tensor periods
std::vector<InactivePeriod *> inactive_periods_list;

std::vector<double> kernel_time_table;
std::vector<EvictionGuideEntry> EvictionGuideTable;
std::vector<long> GPU_resident_memory_estimation;
std::vector<long> CPU_resident_memory_estimation;

std::vector<TensorMovementHint> movement_hints;
std::vector<InactivePeriod *> offloaded_local_intervals;

string Tensor::name() const { return "tensor" + std::to_string(tensor_id); }

bool Tensor::is_alive(int current_kernel) const {
  return is_global_weight || (live_interval.second == -1 ? current_kernel == live_interval.first
                                                         : current_kernel >= live_interval.first &&
                                                               current_kernel < live_interval.second);
}

void Tensor::print() const {
  std::cout << "tensor" << tensor_id << " Is weight (global)?: " << this->is_global_weight << ", "
            << "Size in byte: " << size_in_byte << std::endl;
}

Tensor::Tensor(long long size, bool glob) {
  static int tensor_count = 0;
  tensor_id = tensor_count++;
  size_in_byte = size;
  raw_size_byte = size;
  is_global_weight = glob;
  if (glob) {
    address_offset = memory_offset_weights;
    // page-level alignment
    long N_pages = (size % 4096 == 0) ? (size / 4096) : ((size / 4096) + 1);
    memory_offset_weights += N_pages * 4096;
    size_in_byte = N_pages * 4096;
  } else {
    address_offset = memory_offset_intermediate;
    // page-level alignment
    long N_pages = (size % 4096 == 0) ? (size / 4096) : ((size / 4096) + 1);
    memory_offset_intermediate += N_pages * 4096;
    size_in_byte = N_pages * 4096;
  }
}

unsigned long Tensor::getGlobalOffset() {
  return address_offset + (is_global_weight ? 0 : memory_offset_weights);
}

CUDAKernel::CUDAKernel(int kernel_id, CUDAKernelType t, std::vector<Tensor *> input_tensor_list,
                       std::vector<Tensor *> output_tensor_list, Tensor *workspace_tensor) {
  this->kernel_id = kernel_id;
  this->type = t;
  this->inputs.insert(input_tensor_list.begin(), input_tensor_list.end());
  this->outputs.insert(output_tensor_list.begin(), output_tensor_list.end());
  this->workspace = workspace;
}

void CUDAKernel::print() {
  std::cout << "---------------------------------------------------------------"
               "---------------"
            << std::endl;
  std::cout << "Kernel ID: " << kernel_id << ", "
            << "Name: " << print_kerneltype_array[type] << std::endl;
  std::cout << "Execution Time:            " << execution_cycles << std::endl;
  std::cout << "Input Tensors:" << std::endl;
  for (auto it = inputs.begin(); it != inputs.end(); it++) {
    (*it)->print();
  }
  std::cout << "Output Tensors:" << std::endl;
  for (auto it = outputs.begin(); it != outputs.end(); it++) {
    (*it)->print();
  }
}

void CUDAKernel::getRequiredTensors(std::vector<Tensor *> &required_tensors) const {
  std::unordered_set<Tensor *> set;
  getRequiredTensors(set);
  for (Tensor *tensor : set) required_tensors.push_back(tensor);
}

void CUDAKernel::getRequiredTensors(std::unordered_set<Tensor *> &required_tensors) const {
  for (Tensor *tensor : inputs) required_tensors.insert(tensor);
  for (Tensor *tensor : outputs) required_tensors.insert(tensor);
}

void CUDAKernel::getRequiredTensors(std::vector<Tensor *> &required_tensors,
                                    std::vector<Tensor *> &required_input_tensors,
                                    std::vector<Tensor *> &required_output_tensors) const {
  std::unordered_set<Tensor *> set;
  for (Tensor *tensor : inputs) {
    set.insert(tensor);
    required_tensors.push_back(tensor);
    required_input_tensors.push_back(tensor);
  }
  for (Tensor *tensor : outputs) {
    if (set.find(tensor) == set.end()) {
      required_tensors.push_back(tensor);
      required_output_tensors.push_back(tensor);
    }
  }
}

/**
 * @brief this function is used to fill the liveness information of every tensor
 * @todo you should fill the field live_interval for each tensor in the tensor_list
 *       see descriptions of live_interval in Tensor::live_interval
 */
void tensor_first_pass_liveness_analysis() {
  const int tensor_num = tensor_list.size();
  const int kernel_num = kernel_list.size();

  for (int i = 0; i < kernel_num; i++) {
    CUDAKernel current_kernel = kernel_list[i];
    for (const auto& input : current_kernel.inputs)  {
      if (!input->is_global_weight) {
        if (input->live_interval.first == -1 || input->live_interval.first > current_kernel.kernel_id) {
          input->live_interval.first = current_kernel.kernel_id;
        }
        if (input->live_interval.second == -1 || input->live_interval.second < current_kernel.kernel_id) {
          input->live_interval.second = current_kernel.kernel_id;
        }
      }
    }
    for (const auto& output : current_kernel.outputs)  {
      if (!output->is_global_weight) {
        if (output->live_interval.first == -1 || output->live_interval.first > current_kernel.kernel_id) {
          output->live_interval.first = current_kernel.kernel_id;
        }
        if (output->live_interval.second == -1 || output->live_interval.second < current_kernel.kernel_id) {
          output->live_interval.second = current_kernel.kernel_id;
        }
      }
    }
  }
  // for (int i = 0; i < tensor_num; i++) {
  //   Tensor *current_tensor = tensor_list[i];
  //   // TODO: complete liveness analysis
  //   if (!current_tensor->is_global_weight) {
  //     // This tensor is intermediate

  //   }
  //   // global tensors do not need this info
  // }
}

void Tensor::print_liveness() {
  this->print();
  if (!this->is_global_weight) {
    std::cout << "Liveness: Birth: " << this->live_interval.first << ", Death: " << this->live_interval.second
              << "." << std::endl;
  } else {
    std::cout << "Liveness: Global" << std::endl;
  }
}

/**
 * @brief this function is used to fill the inactive period information of every tensor
 * @todo you should fill the field inactive_periods for each tensor in the tensor_list
 *       see descriptions of inactive_periods in Tensor::inactive_periods
 */
void tensor_second_pass_interval_formation() {
  const int tensor_num = tensor_list.size();
  const int kernel_num = kernel_list.size();

  for (int i = 0; i < tensor_num; i++) {
    Tensor *current_tensor = tensor_list[i];
    if (!current_tensor->is_global_weight) {
      if (current_tensor->live_interval.first != 0) {
        InactivePeriod before(current_tensor);
        before.kernelLevel_interval.first = 0;
        before.kernelLevel_interval.second = current_tensor->live_interval.first-1;
        current_tensor->inactive_periods.push_back(&before);
      }
      if (current_tensor->live_interval.second != kernel_num-1) {
        InactivePeriod after(current_tensor);
        after.kernelLevel_interval.first = current_tensor->live_interval.second+1;
        after.kernelLevel_interval.second = kernel_num-1;
        current_tensor->inactive_periods.push_back(&after);
      }
    } else {
      // Am I supposed to push an empty inactive period with is_looped = true?
      // current_tensor->inactive_periods.push()
    }
  }
}

void Tensor::print_inactive_periods() {
  // print();
  std::cout << "Inactive Periods:" << std::endl;
  for (int i = 0; i < inactive_periods.size(); i++) {
    std::cout << "interval " << i << ": " << inactive_periods[i]->kernelLevel_interval.first << "--------"
              << inactive_periods[i]->kernelLevel_interval.second << std::endl;
    std::cout << "Estimated Time:" << inactive_periods[i]->time_estimated << std::endl;
  }
  std::cout << "_______________________________________________________________" << std::endl;
}

// A provided compiler pass to calculate the estimated execution time for every
// tensors' inactive period length(time)
void get_inactive_periods_time() {
  int kernel_num = kernel_list.size();

  // Setup a cumulative time list;
  double time = 0;
  kernel_time_table.push_back(0);
  for (int i = 0; i < kernel_num; i++) {
    time += (double)kernel_list[i].execution_cycles / (double)(GPU_frequency_GHz * 1000);
    kernel_time_table.push_back(time);
  }

  // Fill the looped extend kernel time table      0 - 2 * kernel_num
  std::vector<double> kernel_time_table_extended;
  kernel_time_table_extended.resize(kernel_num);
  for (int j = 0; j < kernel_num; j++) {
    kernel_time_table_extended[j] = kernel_time_table[j];
  }
  double last_time = kernel_time_table[kernel_num];
  kernel_time_table_extended.push_back(last_time);
  for (int j = 0; j < kernel_num; j++) {
    last_time += (double)kernel_list[j].execution_cycles / (double)(GPU_frequency_GHz * 1000);
    kernel_time_table_extended.push_back(last_time);
  }

  for (int i = 0; i < inactive_periods_list.size(); i++) {
    if (!inactive_periods_list[i]->is_looped) {
      assert(inactive_periods_list[i]->kernelLevel_interval.second >
             inactive_periods_list[i]->kernelLevel_interval.first);
      inactive_periods_list[i]->time_estimated =
          kernel_time_table[inactive_periods_list[i]->kernelLevel_interval.second] -
          kernel_time_table[inactive_periods_list[i]->kernelLevel_interval.first];
    } else {
      assert(inactive_periods_list[i]->kernelLevel_interval.second <
             inactive_periods_list[i]->kernelLevel_interval.first);
      int end = inactive_periods_list[i]->kernelLevel_interval.second;
      int start = inactive_periods_list[i]->kernelLevel_interval.first;
      end += kernel_num;
      inactive_periods_list[i]->time_estimated =
          kernel_time_table_extended[end] - kernel_time_table_extended[start];
    }
  }
}

void InactivePeriod::print() {
  std::cout << "interval " << ": " << kernelLevel_interval.first << "--------" << kernelLevel_interval.second
            << std::endl;
  std::cout << "Estimated Time:" << time_estimated << std::endl;
  std::cout << "Tensor: ";
  this->tensor_back_ptr->print();
  std::cout << "_______________________________________________________________" << std::endl;
}

void print_GPU_mem_really_in_use() {
  for (int i = 0; i < kernel_list.size(); i++) {
    std::vector<Tensor *> r;
    kernel_list[i].getRequiredTensors(r);
    long size_bite = 0;
    for (int j = 0; j < r.size(); j++) {
      size_bite += r[j]->size_in_byte;
    }
    std::cout << "Kernel " << i << ": " << size_bite << std::endl;
  }
}

// Constants (adapt as needed)
static constexpr double GPU_FREQUENCY_GHZ = 1.2;
static constexpr double SYSTEM_LATENCY_US = 45;
static constexpr double CPU_PCIE_BW_GBPS = 15.754;
static constexpr double SSD_PCIE_BW_GBPS = 3.2;
static constexpr double SSD_READ_LATENCY_US = 12;
static constexpr double SSD_WRITE_LATENCY_US = 16;
static constexpr long long BYTES_PER_GB = (1024LL * 1024LL * 1024LL);

// Multiplier to execution time:
static double execution_time_multiplier = 1.5;  // Example: 10% overhead

// Optional size threshold (in bytes). Set to 0 or negative to disable.
static long long size_threshold_bytes = 4 * BYTES_PER_GB;

// Function to set the size threshold (can be called externally)
void set_size_threshold(long long threshold_bytes) {
    size_threshold_bytes = threshold_bytes;
}

// Function to calculate CPU transfer cycles
static long long cpu_transfer_cycles(long long size_bytes) {
    double transfer_time_sec = (double)size_bytes / (CPU_PCIE_BW_GBPS * 1e9) + (SYSTEM_LATENCY_US * 1e-6);
    return static_cast<long long>(std::ceil(transfer_time_sec * GPU_FREQUENCY_GHZ * 1e9));
}

// Function to calculate SSD transfer cycles
static long long ssd_transfer_cycles(long long size_bytes, bool write) {
    double ssd_latency = write ? SSD_WRITE_LATENCY_US : SSD_READ_LATENCY_US;
    double transfer_time_sec = (double)size_bytes / (SSD_PCIE_BW_GBPS * 1e9) + ((SYSTEM_LATENCY_US + ssd_latency) * 1e-6);
    return static_cast<long long>(std::ceil(transfer_time_sec * GPU_FREQUENCY_GHZ * 1e9));
}

// Compute kernel start times in cycles, applying the execution_time_multiplier
static std::vector<long long> compute_kernel_start_cycles() {
    std::vector<long long> start_cycles(kernel_list.size(), 0);
    long long cumulative = 0;
    for (size_t i = 0; i < kernel_list.size(); i++) {
        start_cycles[i] = cumulative;
        // Apply multiplier
        long long adjusted_cycles = static_cast<long long>(kernel_list[i].execution_cycles * execution_time_multiplier);
        cumulative += adjusted_cycles;
    }
    return start_cycles;
}

// Binary search to find the first kernel after a given cycle
static int first_kernel_after(const std::vector<long long>& start_cycle_of_kernel, long long cycle) {
    int left = 0;
    int right = static_cast<int>(start_cycle_of_kernel.size()) - 1;
    int ans = -1;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (start_cycle_of_kernel[mid] > cycle) {
            ans = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return ans;
}

// Helper function to schedule prefetch just-in-time
static void schedule_prefetch(TensorLocation src_loc, Tensor* chosen_tensor, int next_k, long long prefetch_cycles, 
                              const std::vector<long long>& start_cycle_of_kernel) 
{
    long long target_time = start_cycle_of_kernel[next_k] - prefetch_cycles;
    int prefetch_kernel_id;
    if (target_time <= 0) {
        // Prefetch at the very beginning
        prefetch_kernel_id = 0;
    } else {
        // Find the first kernel that starts after target_time
        int k_after = first_kernel_after(start_cycle_of_kernel, target_time);
        if (k_after == -1) {
            // No kernel starts after target_time, use last kernel
            prefetch_kernel_id = static_cast<int>(start_cycle_of_kernel.size()) - 1;
        } else {
            // Schedule prefetch at k_after
            prefetch_kernel_id = k_after;
        }
    }

    movement_hints.emplace_back(src_loc, TensorLocation::IN_GPU, kernel_list[prefetch_kernel_id].kernel_id, chosen_tensor);
}

// Function to add movement hints for a single tensor
void add_movement_hints_for_tensor(Tensor* chosen_tensor, 
                                   const std::vector<int>& chosen_usage_kernels,
                                   const std::vector<long long>& start_cycle_of_kernel) 
{
    // If no usage or just one usage, no offloading/prefetching needed.
    if (chosen_usage_kernels.empty()) {
        return;
    }

    // Apply size threshold
    if (size_threshold_bytes > 0 && chosen_tensor->size_in_byte < size_threshold_bytes) {
        // Skip tensors below the threshold
        return;
    }

    long long size_bytes = chosen_tensor->size_in_byte;
    long long cpu_write = cpu_transfer_cycles(size_bytes);
    long long cpu_read  = cpu_transfer_cycles(size_bytes);
    long long ssd_write = ssd_transfer_cycles(size_bytes, true);
    long long ssd_read  = ssd_transfer_cycles(size_bytes, false);

    // Check if the first usage is as an output. If yes, preallocate at first usage kernel.
    int first_usage_kernel = chosen_usage_kernels[0];
    bool first_usage_input_or_global = false;
    {
        // Check if at first_usage_kernel, tensor is input or workspace or global
        const CUDAKernel &k = kernel_list[first_usage_kernel];
        if (chosen_tensor->is_global_weight) {
            first_usage_input_or_global = true;
        } else if (k.inputs.find(chosen_tensor) != k.inputs.end()) {
            first_usage_input_or_global = true;
        } else if (k.workspace == chosen_tensor) {
            first_usage_input_or_global = true;
        }
    }

    // if (!first_usage_input_or_global) {
    //     // Appears first as output -> preallocate on GPU
    //     movement_hints.emplace_back(TensorLocation::NOT_PRESENT, TensorLocation::IN_GPU, 
    //                                 kernel_list[first_usage_kernel].kernel_id, chosen_tensor);
    // }

    // Handle offloading between usages
    for (size_t i = 0; i + 1 < chosen_usage_kernels.size(); i++) {
        int current_k = chosen_usage_kernels[i];
        int next_k = chosen_usage_kernels[i + 1];

        // Idle interval between these usages:
        long long start_idle = start_cycle_of_kernel[current_k] + kernel_list[current_k].execution_cycles * execution_time_multiplier; // after current kernel finishes
        long long end_idle = start_cycle_of_kernel[next_k];          // start of next usage kernel
        long long idle_cycles = end_idle - start_idle;

        if (idle_cycles <= 0) {
            // No idle gap, skip
            continue;
        }

        // Decide offload location (SSD or CPU)
        bool use_ssd = (idle_cycles > (ssd_write + ssd_read) * 2);

        // Evict after current_k finishes
        if (use_ssd) {
            movement_hints.emplace_back(TensorLocation::IN_GPU, TensorLocation::IN_SSD, 
                                        kernel_list[current_k].kernel_id + 1, chosen_tensor);
        } else {
            movement_hints.emplace_back(TensorLocation::IN_GPU, TensorLocation::IN_CPU, 
                                        kernel_list[current_k].kernel_id + 1, chosen_tensor);
        }

        // Prefetch before next_k starts
        long long prefetch_cost = use_ssd ? ssd_read : cpu_read;
        TensorLocation src_loc = use_ssd ? TensorLocation::IN_SSD : TensorLocation::IN_CPU;
        schedule_prefetch(src_loc, chosen_tensor, next_k, prefetch_cost, start_cycle_of_kernel);
    }
}

// Function to compute the offloading policy for all tensors
void compute_offloading_policy() {
    // Map each tensor to all kernels that use it
    std::unordered_map<Tensor*, std::vector<int>> tensor_usage_map;

    int max_kernel_id = -1;
    for (const auto &k : kernel_list) {
        if (k.kernel_id > max_kernel_id) {
            max_kernel_id = k.kernel_id;
        }
        // Inputs
        for (auto* t : k.inputs) {
            tensor_usage_map[t].push_back(k.kernel_id);
        }
        // Outputs
        for (auto* t : k.outputs) {
            tensor_usage_map[t].push_back(k.kernel_id);
        }
        // Workspace
        if (k.workspace) {
            tensor_usage_map[k.workspace].push_back(k.kernel_id);
        }
    }

    // For global tensors, extend usage from 0 to max_kernel_id if needed
    for (auto* t : tensor_list) {
        if (t->is_global_weight) {
            if (tensor_usage_map.find(t) == tensor_usage_map.end()) {
                tensor_usage_map[t] = {};
            }
            if (tensor_usage_map[t].empty()) {
                // Not used explicitly, but global
                for (int kid = 0; kid <= max_kernel_id; kid++) {
                    tensor_usage_map[t].push_back(kid);
                }
            } else {
                // Ensure full coverage
                std::sort(tensor_usage_map[t].begin(), tensor_usage_map[t].end());
                int first_used = tensor_usage_map[t].front();
                int last_used = tensor_usage_map[t].back();
                if (first_used > 0) {
                    tensor_usage_map[t].insert(tensor_usage_map[t].begin(), 0);
                }
                if (last_used < max_kernel_id) {
                    if (tensor_usage_map[t].back() != max_kernel_id) {
                        tensor_usage_map[t].push_back(max_kernel_id);
                    }
                }
            }
        }
    }

    // Sort usage kernels and remove duplicates
    for (auto &kv : tensor_usage_map) {
        std::sort(kv.second.begin(), kv.second.end());
        kv.second.erase(std::unique(kv.second.begin(), kv.second.end()), kv.second.end());
    }

    // Compute start cycles with multiplier
    std::vector<long long> start_cycle_of_kernel = compute_kernel_start_cycles();

    // Add movement hints for each tensor
    for (auto &kv : tensor_usage_map) {
        Tensor* t = kv.first;
        const std::vector<int>& usage_kernels = kv.second;
        add_movement_hints_for_tensor(t, usage_kernels, start_cycle_of_kernel);
    }
}

/**
 * @brief fill this function to schedule your movement hints
 */
void scheduling_movement_hints() {
  /* std::vector<Tensor *> r;
  kernel_list[getCurrentIteration()+1].getRequiredTensors(r);
  for(int i = 0; i < r.size(); i++) {
    current_kernel = r[i];
    for(const auto& input : current_kernel.inputs) {
      movement_hints.push_back(TensorMovementHint(TensorMovementHint::NOT_KNOWN, TensorMovementHint::IN_GPU, input));
    }
    for(const auto& output: current_kernel.outputs) {
      movement_hints.push_back(TensorMovementHint(TensorMovementHint::NOT_PRESENT, TensorMovementHint::IN_GPU, output))
    }
  } */
  movement_hints.clear();
  int part_6 = 1;
  if (part_6 !=0) {
    for (int i = 0; i < (int)kernel_list.size(); i++) {
      int next_kernel_id = (i + 1) % kernel_list.size();
      std::vector<Tensor*> required_tensors;
      for (auto t : kernel_list[next_kernel_id].inputs) {
          movement_hints.emplace_back(TensorLocation::NOT_KNOWN, TensorLocation::IN_GPU, kernel_list[i].kernel_id, t);
      }
      for (auto t: kernel_list[next_kernel_id].outputs) {
          movement_hints.emplace_back(TensorLocation::NOT_PRESENT, TensorLocation::IN_GPU, kernel_list[i].kernel_id, t);
      }
    }
    long long mem = 0;
    int quit = 0;
    for (int i = 0; i < (int)kernel_list.size(); i++) {
      std::vector<Tensor*> required_tensors;
      for (auto t : kernel_list[i].inputs) {
          mem += t->size_in_byte;
          if (mem > 4 * BYTES_PER_GB) {
            quit = 1;
            break;
          }
          movement_hints.emplace_back(TensorLocation::NOT_KNOWN, TensorLocation::IN_GPU, 0, t);
      }
      if (quit == 1) break;
      for (auto t: kernel_list[i].outputs) {
          mem += t->size_in_byte;
          if (mem > 4 * BYTES_PER_GB) {
            quit = 1;
            break;
          }
          movement_hints.emplace_back(TensorLocation::NOT_PRESENT, TensorLocation::IN_GPU, 0, t);
      }
      if (quit == 1) break;
    }
    compute_offloading_policy();
  }
  else {
  std::unordered_set<Tensor *> ssdItems;
  for (int i = 0; i < (int)kernel_list.size(); i++) {
      int next_kernel_id = (i + 1) % kernel_list.size();
      std::vector<Tensor*> required_tensors;
      for (auto t : kernel_list[next_kernel_id].inputs) {
          movement_hints.emplace_back(TensorLocation::NOT_KNOWN, TensorLocation::IN_GPU, kernel_list[i].kernel_id, t);
      }
      for (auto t: kernel_list[next_kernel_id].outputs) {
          movement_hints.emplace_back(TensorLocation::NOT_PRESENT, TensorLocation::IN_GPU, kernel_list[i].kernel_id, t);
      }

      
      if (part_6 != 0 && i > 0) {
        for (auto t : kernel_list[i-1].inputs) {
          int next = -1;
          for(int j = i; j < (int)kernel_list.size(); j++) {
            for(auto s : kernel_list[j].inputs) {
              if (s == t) {
                next = j-i;
                break;
              }
            }
            if (next!=-1) { //break out of seeking loop if we have found a spot
              break;
            }
          }

          if (next > 20) {
            ssdItems.insert(t);
            movement_hints.emplace_back(TensorLocation::IN_GPU, TensorLocation::IN_SSD, kernel_list[i-1].kernel_id, t);
          }  else if (next > 10) {
            movement_hints.emplace_back(TensorLocation::IN_GPU, TensorLocation::IN_CPU, kernel_list[i-1].kernel_id, t);
          }
        }
        for (auto t: kernel_list[i-1].outputs) {
          int next = -1;
          for(int j = i; j < (int)kernel_list.size(); j++) {
            for(auto s : kernel_list[j].inputs) {
              if (s == t) {
                next = j-i;
                break;
              }
            }
            if (next!=-1) { //break out of seeking loop if we have found a spot
              break;
            }
          }

          if (next > 20) {
            ssdItems.insert(t);
            movement_hints.emplace_back(TensorLocation::IN_GPU, TensorLocation::IN_SSD, kernel_list[i-1].kernel_id, t);
          }  else if (next > 10) {
            movement_hints.emplace_back(TensorLocation::IN_GPU, TensorLocation::IN_CPU, kernel_list[i-1].kernel_id, t);
          }
        }

        int future_kid = (i + 5) % kernel_list.size();
        for (auto t : kernel_list[future_kid].inputs) {
          if (ssdItems.find(t) != ssdItems.end()) {
            ssdItems.erase(t);
            movement_hints.emplace_back(TensorLocation::IN_SSD, TensorLocation::IN_CPU, kernel_list[i].kernel_id, t);
          }
        }
        for (auto t: kernel_list[next_kernel_id].outputs) {
          if (ssdItems.find(t) != ssdItems.end()) {
            ssdItems.erase(t);
            movement_hints.emplace_back(TensorLocation::IN_SSD, TensorLocation::IN_CPU, kernel_list[i].kernel_id, t);
          }
        }

      }
    }
  }

  // make sure the movement hints are sorted, the simulator depends on this
  std::sort(movement_hints.begin(), movement_hints.end());
}
