import pyomo.environ as pyo
import numpy as np # For pre-calculating A_k values if needed
import Solar  # Import the Solar energy module
import NetworkBasic  # Import the NetworkBasic module for channel gain calculations

# --- 0. Helper function to precompute A_k values ---
def precompute_A_k(A_max_is_data, beta_accuracy_data, theta_k_vals_data, theta_max_val_data, V_S_pairs_data, K_breakpoints_data):
    """
    Precomputes A_k values: A_k = A_max_is * (1 - exp(-beta * theta_k / theta_max))
    """
    A_k_param_data = {}
    for i, s in V_S_pairs_data:
        for k_idx, theta_k in enumerate(theta_k_vals_data): # Assuming K_breakpoints_data is just a range, and theta_k_vals_data is the list of values
            k = K_breakpoints_data[k_idx] # Or however K_breakpoints_data is structured
            if (i,s) in A_max_is_data: # Ensure A_max is defined for this pair
                 A_k_param_data[i,s,k] = A_max_is_data[i,s] * (1 - np.exp(-beta_accuracy_data * theta_k / theta_max_val_data))
            else: # Fallback or error
                 A_k_param_data[i,s,k] = 0 # Or handle error appropriately
    return A_k_param_data

# --- 1. Create the Pyomo Model ---
model = pyo.ConcreteModel()

# --- 2. Define Sets ---
# These would be populated with actual data (e.g., lists of strings or numbers)
# For example:
# model.V_set = pyo.Set(initialize=['server1', 'server2'])
# model.M_glob_set = pyo.Set(initialize=['modelA', 'modelB'])
# ... and so on for all sets

# For demonstration, let's use RangeSet for simplicity where appropriate
# You'll need to replace these with actual set initializations based on your input data.

# Example:
num_servers = 5
num_models = 3
num_requests_per_server = {f'server{i}': 3 for i in range(num_servers)} # server_idx: num_requests
num_time_slots = 20
num_breakpoints = 4 # K+1 points, so K segments. K_breakpoints will be 0..K

model.V_set = pyo.Set(initialize=[f'server{i}' for i in range(num_servers)])
model.M_glob_set = pyo.Set(initialize=[f'model{j}' for j in range(num_models)])

# Create a combined set of (server, request) pairs
# This assumes requests are named s0, s1, ... for each server
V_S_pairs_data = []
# S_set_global will store all unique request IDs if needed,
# but V_S_pairs is more direct for variables indexed by (i,s)
S_set_global_data = set()
# This dictionary will map server_idx to a list of its request_ids
requests_S_i_data = {}

for i_idx, i_val in enumerate(model.V_set):
    requests_S_i_data[i_val] = []
    for s_idx in range(num_requests_per_server[i_val]):
        req_id = f'{i_val}_req{s_idx}'
        V_S_pairs_data.append((i_val, req_id))
        S_set_global_data.add(req_id)
        requests_S_i_data[i_val].append(req_id)

model.V_S_pairs = pyo.Set(initialize=V_S_pairs_data, dimen=2)
model.S_set = pyo.Set(initialize=list(S_set_global_data)) # All unique request names

model.T_set = pyo.RangeSet(0, num_time_slots - 1) # Time slots t = 0, ..., N-1
model.T_horizon_set = pyo.RangeSet(0, num_time_slots) # For battery b_i^t up to t=N

# K_breakpoints are typically 0, 1, ..., K. If num_breakpoints = K+1 points.
model.K_breakpoints_set = pyo.RangeSet(0, num_breakpoints - 1)

# Solar energy parameters - scaled based on current system requirements
# Scale solar panel size to match our energy requirements:
# Maximum solar output in original model ~94.6 * V_Solar_Panel * 0.2
# We need ~1000 energy units per time slot, so:
# 94.6 * V_Solar_Panel * 0.2 = 1000
# Therefore V_Solar_Panel = 1000/(94.6*0.2) ≈ 53
V_Solar_Panel = 53  # Solar panel area scaled to match current energy requirements
# Calculate solar energy harvesting for each server
T = list(range(num_time_slots))  # Create time array for Solar.py
E_harvested_solar = Solar.Solar_Value(num_servers, T, V_Solar_Panel)

# Channel gain calculation using NetworkBasic
# Calculate channel gains between edge servers and cloud
# Parameters match those in NetworkBasic.py:
# - Path Loss Exponent (Alpha) = 2.5
# - Reference distance (d_0) = 1m
# - Path loss at reference distance (c_0) = -20dB
T = list(range(num_time_slots))
g_channel_gains = NetworkBasic.Calculate_Channel_Gain(len(T), num_servers, 1)  # 1 cloud server

# --- 3. Define Parameters ---
# These would be initialized with your specific problem data.
# Using dictionaries for data storage is common.

# Example dummy data structure (replace with actual data loading)
dummy_S_cap_data = {f'server{i}': 5000 for i in range(num_servers)}
dummy_C_comp_cap_data = {f'server{i}': 20000 for i in range(num_servers)}
dummy_B_max_data = {f'server{i}': 50000 for i in range(num_servers)}
dummy_W_load_data = {req_id: 0.2 for req_id in model.S_set}
dummy_bits_per_token_data = 8
dummy_sigma_storage_data = {f'model{j}': 30 for j in range(num_models)}
dummy_theta_max_val_data = 2048
dummy_A_max_is_data = {(i,s): 0.9 for i,s in model.V_S_pairs}
dummy_beta_accuracy_data = 2.0

# Adjust breakpoints to be more evenly distributed for accuracy
dummy_theta_k_vals_data = [int(dummy_theta_max_val_data * k / (num_breakpoints - 1)) for k in range(num_breakpoints)]

# Initialize E_harvested_data using Solar.py output
# Convert numpy array to dictionary with proper indexing
dummy_E_harvested_data = {(f'server{i}',t): float(max(0, E_harvested_solar[t][i])) for i in range(num_servers) for t in model.T_set}

# Initialize channel gain data using NetworkBasic.py output
# Convert from mW to linear scale and ensure non-negative values
dummy_g_channel_gain_data = {(f'server{i}',t): float(max(0, g_channel_gains[t][i][0])) for i in range(num_servers) for t in model.T_set}

dummy_nu_gpu_energy_cycle_data = 0.0002  # Further reduced
dummy_alpha_mem_access_bit_data = 0.5  # Further reduced
dummy_e_mem_energy_access_data = 0.00002  # Further reduced
dummy_eta_tx_efficiency_data = 0.7  # Slightly reduced for more realistic value
dummy_lambda_weight_data = 0.2  # Further reduced to prioritize completion
dummy_b_initial_data = {f'server{i}': 25000 for i in range(num_servers)}  # Increased from 10000

# Parameter for linking requests to models: req_needs_model[s,m] = 1 if request s needs model m
# Example: each request needs one specific model
dummy_req_needs_model_data = {}
for s_idx, s_val in enumerate(model.S_set):
    for m_idx, m_val in enumerate(model.M_glob_set):
        # Simple assignment for demo: request k needs model k (cyclically)
        if s_idx % num_models == m_idx:
             dummy_req_needs_model_data[s_val, m_val] = 1
        else:
             dummy_req_needs_model_data[s_val, m_val] = 0


# Precompute A_k values based on the formula for A(theta)
# A_k = A_max_is * (1 - exp(-beta * theta_k / theta_max))
# The resulting A_k_param_data should be indexed by (i,s,k)
dummy_A_k_param_data = precompute_A_k(dummy_A_max_is_data, dummy_beta_accuracy_data, dummy_theta_k_vals_data, dummy_theta_max_val_data, V_S_pairs_data, list(model.K_breakpoints_set))

dummy_Gamma_accuracy_threshold_data = 0.65  # Slightly reduced from 0.7

# Pyomo Parameters
model.S_cap = pyo.Param(model.V_set, initialize=dummy_S_cap_data)
model.C_comp_cap = pyo.Param(model.V_set, initialize=dummy_C_comp_cap_data)
model.B_max = pyo.Param(model.V_set, initialize=dummy_B_max_data)
model.W_load = pyo.Param(model.S_set, initialize=dummy_W_load_data) # W_s
model.bits_per_token = pyo.Param(initialize=dummy_bits_per_token_data)
model.sigma_storage = pyo.Param(model.M_glob_set, initialize=dummy_sigma_storage_data)
model.theta_max_val = pyo.Param(initialize=dummy_theta_max_val_data)
model.A_max_is = pyo.Param(model.V_S_pairs, initialize=dummy_A_max_is_data)
# theta_k_vals are the token values at breakpoints
model.theta_k_vals = pyo.Param(model.K_breakpoints_set, initialize={k: dummy_theta_k_vals_data[k] for k in model.K_breakpoints_set})
# A_k_param are the accuracy values at breakpoints, depends on i,s,k
model.A_k_param = pyo.Param(model.V_S_pairs, model.K_breakpoints_set, initialize=dummy_A_k_param_data)

model.Gamma_accuracy_threshold = pyo.Param(initialize=dummy_Gamma_accuracy_threshold_data)
model.E_harvested = pyo.Param(model.V_set, model.T_set, initialize=dummy_E_harvested_data)
model.g_channel_gain = pyo.Param(model.V_set, model.T_set, initialize=dummy_g_channel_gain_data)
model.nu_gpu_energy_cycle = pyo.Param(initialize=dummy_nu_gpu_energy_cycle_data)
model.alpha_mem_access_bit = pyo.Param(initialize=dummy_alpha_mem_access_bit_data)
model.e_mem_energy_access = pyo.Param(initialize=dummy_e_mem_energy_access_data)
model.eta_tx_efficiency = pyo.Param(initialize=dummy_eta_tx_efficiency_data)
model.lambda_weight_obj = pyo.Param(initialize=dummy_lambda_weight_data) # Renamed to avoid clash
model.b_initial = pyo.Param(model.V_set, initialize=dummy_b_initial_data)
model.req_needs_model = pyo.Param(model.S_set, model.M_glob_set, initialize=dummy_req_needs_model_data, default=0)

# --- 4. Define Variables ---
model.x = pyo.Var(model.V_S_pairs, model.T_set, domain=pyo.Binary) # x_i,s^t
model.y = pyo.Var(model.V_S_pairs, model.T_set, domain=pyo.Binary) # y_i,s^t
model.f = pyo.Var(model.V_set, model.M_glob_set, domain=pyo.Binary) # f_i,m

model.z = pyo.Var(model.V_set, domain=pyo.Binary) # z_i

# SOS2 variables: lambda_sos_i,s^k
model.lambda_sos = pyo.Var(model.V_S_pairs, model.K_breakpoints_set, domain=pyo.NonNegativeReals, bounds=(0,1))

model.omega_stored_energy = pyo.Var(model.V_set, model.T_set, domain=pyo.NonNegativeReals) # omega_i^t
model.b_battery_level = pyo.Var(model.V_set, model.T_horizon_set, domain=pyo.NonNegativeReals) # b_i^t

# D_is represents D_{i,s} = b * theta_{i,s}
model.D_is = pyo.Var(model.V_S_pairs, domain=pyo.NonNegativeReals, bounds=(0, model.bits_per_token * model.theta_max_val))

# Linearization variables for D_is * x_ist and D_is * y_ist
model.D_is_loc_t = pyo.Var(model.V_S_pairs, model.T_set, domain=pyo.NonNegativeReals) # D_is * x_ist
model.D_is_off_t = pyo.Var(model.V_S_pairs, model.T_set, domain=pyo.NonNegativeReals) # D_is * y_ist

# Energy consumption components
model.lambda_E_local_energy = pyo.Var(model.V_set, model.T_set, domain=pyo.NonNegativeReals) # lambda_E_t,i
model.lambda_U_upload_energy = pyo.Var(model.V_set, model.T_set, domain=pyo.NonNegativeReals) # lambda_U_t,i
model.lambda_T_total_energy_it = pyo.Var(model.V_set, model.T_set, domain=pyo.NonNegativeReals) # lambda_T_t,i

# Objective components
model.Z_total_completed_apps = pyo.Var(domain=pyo.NonNegativeReals)
model.Lambda_total_system_energy = pyo.Var(domain=pyo.NonNegativeReals)

# --- 5. Define Constraints ---

# Eq (1) related: Definition of D_is using SOS2 weights for theta_is
# D_is = b * sum(lambda_k * theta_k)
def D_is_definition_rule(m, i, s):
    return m.D_is[i,s] == m.bits_per_token * sum(m.lambda_sos[i,s,k] * m.theta_k_vals[k] for k in m.K_breakpoints_set)
model.D_is_def = pyo.Constraint(model.V_S_pairs, rule=D_is_definition_rule)

# Linearization constraints for D_is_loc_t = D_is * x_ist
def D_is_loc_t_rule1(m, i, s, t):
    return m.D_is_loc_t[i,s,t] <= m.D_is[i,s]
model.D_is_loc_t_con1 = pyo.Constraint(model.V_S_pairs, model.T_set, rule=D_is_loc_t_rule1)

def D_is_loc_t_rule2(m, i, s, t):
    # M_val = m.bits_per_token * m.theta_max_val (upper bound for D_is)
    return m.D_is_loc_t[i,s,t] <= (m.bits_per_token * m.theta_max_val) * m.x[i,s,t]
model.D_is_loc_t_con2 = pyo.Constraint(model.V_S_pairs, model.T_set, rule=D_is_loc_t_rule2)

def D_is_loc_t_rule3(m, i, s, t):
    # M_val = m.bits_per_token * m.theta_max_val
    return m.D_is_loc_t[i,s,t] >= m.D_is[i,s] - (m.bits_per_token * m.theta_max_val) * (1 - m.x[i,s,t])
model.D_is_loc_t_con3 = pyo.Constraint(model.V_S_pairs, model.T_set, rule=D_is_loc_t_rule3)

# Linearization constraints for D_is_off_t = D_is * y_ist (similar to above)
def D_is_off_t_rule1(m, i, s, t):
    return m.D_is_off_t[i,s,t] <= m.D_is[i,s]
model.D_is_off_t_con1 = pyo.Constraint(model.V_S_pairs, model.T_set, rule=D_is_off_t_rule1)

def D_is_off_t_rule2(m, i, s, t):
    return m.D_is_off_t[i,s,t] <= (m.bits_per_token * m.theta_max_val) * m.y[i,s,t]
model.D_is_off_t_con2 = pyo.Constraint(model.V_S_pairs, model.T_set, rule=D_is_off_t_rule2)

def D_is_off_t_rule3(m, i, s, t):
    return m.D_is_off_t[i,s,t] >= m.D_is[i,s] - (m.bits_per_token * m.theta_max_val) * (1 - m.y[i,s,t])
model.D_is_off_t_con3 = pyo.Constraint(model.V_S_pairs, model.T_set, rule=D_is_off_t_rule3)


# Eq (2): Computation Constraint
# sum_{s in S_i} D_{i,s}^{loc,t} * W_s <= C_i
def computation_constraint_rule(m, i, t):
    # Filter V_S_pairs for server i
    requests_for_server_i = [s_p for i_p, s_p in m.V_S_pairs if i_p == i]
    if not requests_for_server_i:
        return pyo.Constraint.Skip # Or handle as sum over empty set = 0
    return sum(m.D_is_loc_t[i,s,t] * m.W_load[s] for s in requests_for_server_i) <= m.C_comp_cap[i]
model.computation_constraint = pyo.Constraint(model.V_set, model.T_set, rule=computation_constraint_rule)

# Eq (3): Storage Constraint
# sum_{m in M} sigma_m * f_im <= S_i
def storage_constraint_rule(m, i):
    return sum(m.sigma_storage[model_id] * m.f[i,model_id] for model_id in m.M_glob_set) <= m.S_cap[i]
model.storage_constraint = pyo.Constraint(model.V_set, rule=storage_constraint_rule)

# Constraint: Local execution requires model
# x_ist = 1 => f_im = 1 for all m in M_s (models needed by request s)
def local_execution_requires_model_rule(m, i, s, t, model_id):
    if m.req_needs_model[s, model_id] == 1:
        return m.x[i,s,t] <= m.f[i,model_id]
    else:
        return pyo.Constraint.Skip # No constraint if model is not needed
model.local_execution_requires_model = pyo.Constraint(model.V_S_pairs, model.T_set, model.M_glob_set, rule=local_execution_requires_model_rule)

# Eq (4): Exclusive Execution
# x_ist + y_ist <= 1
def exclusive_execution_rule(m, i, s, t):
    return m.x[i,s,t] + m.y[i,s,t] <= 1
model.exclusive_execution = pyo.Constraint(model.V_S_pairs, model.T_set, rule=exclusive_execution_rule)

# Eq (5) / (task-completion-tight): Application Completion (preferred version)
# sum_t (x_ist + y_ist) >= z_i   (for each request s in S_i)
def task_completion_rule(m, i, s): # s here is a request tied to server i via V_S_pairs
    return sum(m.x[i,s,t] + m.y[i,s,t] for t in m.T_set) >= m.z[i]
model.task_completion = pyo.Constraint(model.V_S_pairs, rule=task_completion_rule) # Iterates over valid (i,s)

# SOS2 Constraints (Eq 8, 9, 10)
# sum_k lambda_isk = 1
def sos2_sum_to_one_rule(m, i, s):
    return sum(m.lambda_sos[i,s,k] for k in m.K_breakpoints_set) == 1
model.sos2_sum_to_one = pyo.Constraint(model.V_S_pairs, rule=sos2_sum_to_one_rule)

# SOS2 constraint itself (Pyomo handles this specially)
# This tells the solver that for each (i,s), the set of lambda_sos[i,s,k] variables
# forms an SOS of type 2, ordered by k (implicit by RangeSet).
# The weights for SOS2 are the theta_k_vals.
model.sos2_constraint = pyo.SOSConstraint(model.V_S_pairs, var=model.lambda_sos, sos=2) # Check Pyomo docs for exact syntax if theta_k_vals need to be specified as weights here. Often, the order is enough.


# Eq (11): Accuracy Threshold
# sum_k (lambda_isk * A_k_param[i,s,k]) >= Gamma * A_max_is[i,s]
def accuracy_threshold_rule(m, i, s):
    achieved_accuracy = sum(m.lambda_sos[i,s,k] * m.A_k_param[i,s,k] for k in m.K_breakpoints_set)
    return achieved_accuracy >= m.Gamma_accuracy_threshold * m.A_max_is[i,s]
model.accuracy_threshold = pyo.Constraint(model.V_S_pairs, rule=accuracy_threshold_rule)

# Energy Harvesting Model Constraints
# Eq (12): omega_it <= E_it
def harvest_bound_rule(m, i, t):
    return m.omega_stored_energy[i,t] <= m.E_harvested[i,t]
model.harvest_bound = pyo.Constraint(model.V_set, model.T_set, rule=harvest_bound_rule)

# Eq (13): b_it + omega_it <= B_max_i
def battery_capacity_rule(m, i, t):
    return m.b_battery_level[i,t] + m.omega_stored_energy[i,t] <= m.B_max[i]
model.battery_capacity = pyo.Constraint(model.V_set, model.T_set, rule=battery_capacity_rule)

# Energy Consumption Model Definitions
# Eq (energy-local) / (16): lambda_E_it = sum_s x_ist * D_is * (W_s * nu + alpha_mem * e)
# Using D_is_loc_t = D_is * x_ist
def local_energy_definition_rule(m, i, t):
    requests_for_server_i = [s_p for i_p, s_p in m.V_S_pairs if i_p == i]
    if not requests_for_server_i:
         return m.lambda_E_local_energy[i,t] == 0
    return m.lambda_E_local_energy[i,t] == sum(m.D_is_loc_t[i,s,t] * (m.W_load[s] * m.nu_gpu_energy_cycle + m.alpha_mem_access_bit * m.e_mem_energy_access) for s in requests_for_server_i)
model.local_energy_definition = pyo.Constraint(model.V_set, model.T_set, rule=local_energy_definition_rule)

# Eq (energy-upload) / (17): lambda_U_it = sum_s y_ist * D_is * eta / g_ict
# Using D_is_off_t = D_is * y_ist
def upload_energy_definition_rule(m, i, t):
    requests_for_server_i = [s_p for i_p, s_p in m.V_S_pairs if i_p == i]
    if not requests_for_server_i:
        return m.lambda_U_upload_energy[i,t] == 0
    # Ensure g_channel_gain is not zero to avoid division by zero
    if pyo.value(m.g_channel_gain[i,t]) == 0: # Use pyo.value for parameters during rule construction if needed
        # Handle division by zero: either penalize heavily or make infeasible if y is 1
        # For now, assume if y_ist > 0, g_channel_gain > 0, or set energy to a very large number if y_ist=1 and g=0.
        # A robust way is to ensure y_ist is 0 if g_channel_gain is 0, or reformulate.
        # Simplest for now: if g is 0, upload energy is effectively infinite if y_ist is 1.
        # This should ideally be handled by setting y_ist=0 if g_ict=0 if data can be offloaded.
        # Or, use a small epsilon if g=0 is possible and y can be 1.
        # For this code, we proceed assuming g_channel_gain > 0 if y_ist = 1.
        if any(pyo.value(m.D_is_off_t[i,s,t]) > 0 for s in requests_for_server_i): # If any D_is_off_t > 0
             if pyo.value(m.g_channel_gain[i,t]) <= 1e-9: # Effectively zero
                  return m.lambda_U_upload_energy[i,t] >= 1e9 # Very large energy
    return m.lambda_U_upload_energy[i,t] == sum(m.D_is_off_t[i,s,t] * m.eta_tx_efficiency / (m.g_channel_gain[i,t] + 1e-9) for s in requests_for_server_i) # add epsilon for safety
model.upload_energy_definition = pyo.Constraint(model.V_set, model.T_set, rule=upload_energy_definition_rule)

# Eq (energy-total) / (18): lambda_T_it = lambda_E_it + lambda_U_it
def total_energy_definition_rule(m, i, t):
    return m.lambda_T_total_energy_it[i,t] == m.lambda_E_local_energy[i,t] + m.lambda_U_upload_energy[i,t]
model.total_energy_definition = pyo.Constraint(model.V_set, model.T_set, rule=total_energy_definition_rule)

# Battery Dynamics
# Eq (battery-update) / (14) / (19): b_i^(t+1) = b_it + omega_it - lambda_T_it
def initial_battery_rule(m, i):
    return m.b_battery_level[i,0] == m.b_initial[i]
model.initial_battery_constraint = pyo.Constraint(model.V_set, rule=initial_battery_rule)

def battery_update_rule(m, i, t): # t from 0 to N-1
    # Ensure battery does not go below zero by NonNegativeReals domain on b_battery_level
    # The constraint b_i^{t+1} >= 0 is implicitly b_i^t + omega_i^t - lambda_T_{t,i} >= 0
    # which is already handled by b_battery_level[i,t+1] being NonNegativeReals.
    return m.b_battery_level[i, t+1] == m.b_battery_level[i,t] + m.omega_stored_energy[i,t] - m.lambda_T_total_energy_it[i,t]
model.battery_update = pyo.Constraint(model.V_set, model.T_set, rule=battery_update_rule)


# Problem Formulation Section - Objective components
# Eq (total-completed) / (20): Z = sum_i z_i
def total_completed_apps_rule(m):
    return m.Z_total_completed_apps == sum(m.z[i] for i in m.V_set)
model.total_completed_apps_def = pyo.Constraint(rule=total_completed_apps_rule)

# Eq (total-energy) / (22): Lambda = sum_i sum_t lambda_T_it
def total_system_energy_rule(m):
    return m.Lambda_total_system_energy == sum(m.lambda_T_total_energy_it[i,t] for i in m.V_set for t in m.T_set)
model.total_system_energy_def = pyo.Constraint(rule=total_system_energy_rule)


# --- 6. Define Objective Function ---
# Eq (objective) / (23): max (1 - lambda_weight_obj) * Z - lambda_weight_obj * Lambda
def objective_rule(m):
    return (1 - m.lambda_weight_obj) * m.Z_total_completed_apps - m.lambda_weight_obj * m.Lambda_total_system_energy
model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)


# --- 7. Solve the Model ---
if __name__ == '__main__':
    # This part is for running the model.
    solver = pyo.SolverFactory('gurobi')
    
    # Add debugging options
    solver.options['LogToConsole'] = 1
    solver.options['IISMethod'] = 1  # Request IIS (Irreducible Infeasible Subsystem) if infeasible
    
    print("\n=== Model Parameters and Constraints Analysis ===")
    
    # 1. Analyze Storage Requirements
    print("\n1. Storage Constraints Analysis:")
    for i in model.V_set:
        total_storage_needed = sum(pyo.value(model.sigma_storage[m]) for m in model.M_glob_set)
        print(f"Server {i}:")
        print(f"  - Storage capacity: {pyo.value(model.S_cap[i])}")
        print(f"  - Total storage if all models stored: {total_storage_needed}")
        print(f"  - Storage constraint feasible: {total_storage_needed <= pyo.value(model.S_cap[i])}")
    
    # 2. Analyze Battery and Energy Constraints
    print("\n2. Battery and Energy Constraints Analysis:")
    for i in model.V_set:
        print(f"\nServer {i}:")
        print(f"  - Initial battery: {pyo.value(model.b_initial[i])}")
        print(f"  - Maximum battery: {pyo.value(model.B_max[i])}")
        print(f"  - Energy harvested per time slot: {pyo.value(model.E_harvested[i,0])}")
        
        # Calculate minimum energy needed for one local execution
        min_data_size = pyo.value(model.bits_per_token)  # Minimum possible D_is (one token)
        min_workload = min(pyo.value(model.W_load[s]) for s in model.S_set)
        min_local_energy = min_data_size * (min_workload * pyo.value(model.nu_gpu_energy_cycle) + 
                                          pyo.value(model.alpha_mem_access_bit) * pyo.value(model.e_mem_energy_access))
        
        # Calculate minimum energy needed for one offload
        min_offload_energy = min_data_size * pyo.value(model.eta_tx_efficiency) / max(pyo.value(model.g_channel_gain[i,t]) for t in model.T_set)
        
        print(f"  - Minimum energy for one local execution: {min_local_energy:.2f}")
        print(f"  - Minimum energy for one offload: {min_offload_energy:.2f}")
        print(f"  - Energy harvested sufficient for local execution: {pyo.value(model.E_harvested[i,0]) >= min_local_energy}")
        print(f"  - Energy harvested sufficient for offload: {pyo.value(model.E_harvested[i,0]) >= min_offload_energy}")
    
    # 3. Analyze Accuracy Constraints
    print("\n3. Accuracy Constraints Analysis:")
    for i, s in model.V_S_pairs:
        max_accuracy = max(pyo.value(model.A_k_param[i,s,k]) for k in model.K_breakpoints_set)
        required_accuracy = pyo.value(model.Gamma_accuracy_threshold) * pyo.value(model.A_max_is[i,s])
        print(f"\nServer {i}, Request {s}:")
        print(f"  - Maximum possible accuracy: {max_accuracy:.3f}")
        print(f"  - Required accuracy: {required_accuracy:.3f}")
        print(f"  - Accuracy constraint feasible: {max_accuracy >= required_accuracy}")
        
        # Show accuracy progression across breakpoints
        print("  - Accuracy at breakpoints:")
        for k in model.K_breakpoints_set:
            print(f"    θ={pyo.value(model.theta_k_vals[k]):4.0f}: A={pyo.value(model.A_k_param[i,s,k]):.3f}")
    
    # 4. Analyze Task Completion Constraints
    print("\n4. Task Completion Constraints Analysis:")
    for i, s in model.V_S_pairs:
        num_time_slots = len(model.T_set)
        print(f"\nServer {i}, Request {s}:")
        print(f"  - Available time slots: {num_time_slots}")
        print(f"  - Maximum possible executions with battery constraints:")
        max_local_execs = pyo.value(model.b_initial[i]) / min_local_energy
        max_offload_execs = pyo.value(model.b_initial[i]) / min_offload_energy
        print(f"    * Max local executions with initial battery: {max_local_execs:.1f}")
        print(f"    * Max offload executions with initial battery: {max_offload_execs:.1f}")
    
    print("\n=== Attempting to solve the model ===")
    results = solver.solve(model, tee=True)
    
    if results.solver.termination_condition == pyo.TerminationCondition.infeasible:
        print("\nModel is infeasible. Analyzing potential causes based on above analysis...")
        try:
            # Try to compute IIS if supported
            solver.options['computeIIS'] = 1
            solver.solve(model, tee=True)
        except:
            print("Could not compute IIS (Irreducible Infeasible Subsystem)")
    elif (results.solver.status == pyo.SolverStatus.ok) and \
         (results.solver.termination_condition == pyo.TerminationCondition.optimal or
          results.solver.termination_condition == pyo.TerminationCondition.feasible):
        print("\n=== Solution Found ===")
        print(f"Objective Value: {pyo.value(model.objective)}")
        print(f"Total Completed Applications (Z): {pyo.value(model.Z_total_completed_apps)}")
        print(f"Total System Energy (Lambda): {pyo.value(model.Lambda_total_system_energy)}")

        print("\nDecision Variables (sample):")
        for i in model.V_set:
            print(f"  Server {i}:")
            print(f"    App Completed (z_{i}): {pyo.value(model.z[i])}")
            for t in model.T_set:
                if pyo.value(model.lambda_T_total_energy_it[i,t]) > 0:
                     print(f"    Time {t}: Battery={pyo.value(model.b_battery_level[i,t]):.2f}, StoredEnergy={pyo.value(model.omega_stored_energy[i,t]):.2f}, TotalConsumed={pyo.value(model.lambda_T_total_energy_it[i,t]):.2f}")

            for s_pair_idx, (i_p, s_p) in enumerate(model.V_S_pairs):
                if i_p == i: # Request s_p belongs to server i
                    print(f"    Request {s_p}:")
                    # D_is is decided once per (i,s)
                    print(f"      Chosen D_{i},{s_p} (Data Size): {pyo.value(model.D_is[i,s_p]):.2f}")
                    # Show lambda_sos values
                    # for k_sos in model.K_breakpoints_set:
                    #     if pyo.value(model.lambda_sos[i,s_p,k_sos]) > 1e-4:
                    #         print(f"        lambda_sos[{k_sos}]: {pyo.value(model.lambda_sos[i,s_p,k_sos]):.4f} (theta_k = {model.theta_k_vals[k_sos]})")

                    # for t in model.T_set:
                    #     if pyo.value(model.x[i,s_p,t]) > 0.5:
                    #         print(f"        Time {t}: Processed Locally (x=1)")
                    #     if pyo.value(model.y[i,s_p,t]) > 0.5:
                    #         print(f"        Time {t}: Offloaded (y=1)")
        # Add more print statements for other variables as needed
        # e.g., model.f, model.x, model.y, model.lambda_sos
        # Example for f_im:
        # for i in model.V_set:
        #     for m_id in model.M_glob_set:
        #         if pyo.value(model.f[i,m_id]) > 0.5:
        #             print(f"Server {i} stores Model {m_id}")

    else:
        print(f"Solver Status: {results.solver.status}")
        print(f"Termination Condition: {results.solver.termination_condition}")