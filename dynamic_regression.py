import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 

def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
    
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 10  # seconds
    
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = 1000
    kd = 100

    # Initialize data storage
    tau_mes_all = []
    regressor_all = []

    # Data collection loop
    while current_time < max_time:
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)
        
        # Compute sinusoidal reference trajectory
        q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity
        
        # Control command
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)  # Set the torque command
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)

        if dyn_model.visualizer: 
            for index in range(len(sim.bot)):  # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        
        # TODO Compute regressor and store it
        regressor = dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes)
        regressor_all.append(regressor)
        tau_mes_all.append(tau_mes)
        
        current_time += time_step
        # Optional: print current time
        print(f"Current time in seconds: {current_time:.2f}")

    # TODO After data collection, stack all the regressor and all the torque and compute the parameters 'a' using pseudoinverse for all the joint
    print("\nData collection done")
    Y_ALL = np.concatenate(regressor_all, axis=0)
    U_ALL = np.concatenate(tau_mes_all, axis=0)
    
    # TODO reshape the regressor and the torque vector to isolate the last joint and find its dynamical parameters
    # Known parameters for links 1-6 from URDF file
    # m, x, y, z, Ixx, Ixy, Ixz, Iyy, Iyz, Izz
    a_1 = np.array([2.34, 0, 0, 0, 0.3, 0, 0, 0.3, 0, 0.3])
    a_2 = np.array([2.36, 0, 0, 0, 0.3, 0, 0, 0.3, 0, 0.3])
    a_3 = np.array([2.38, 0, 0, 0, 0.3, 0, 0, 0.3, 0, 0.3])
    a_4 = np.array([2.43, 0, 0, 0, 0.3, 0, 0, 0.3, 0, 0.3])
    a_5 = np.array([3.50, 0, 0, 0, 0.3, 0, 0, 0.3, 0, 0.3])
    a_6 = np.array([1.47, 0, 0, 0, 0.3, 0, 0, 0.3, 0, 0.3])
    a_known = np.concatenate([a_1, a_2, a_3, a_4, a_5, a_6])
    
    # Split regressor: first 60 columns for known parameters (links 1-6), last 10 for unknown (link 7)
    num_known_parameter = 60
    Y_known = Y_ALL[:, :num_known_parameter]
    Y_unknown = Y_ALL[:, num_known_parameter:]
    
    # Compute adjusted torque by removing contribution from known parameters
    U_adjusted = U_ALL - Y_known @ a_known
    
    # Estimate unknown parameters using pseudoinverse: a_unknown = pinv(Y_unknown) * U_adjusted
    a_unknown_estimated = np.linalg.pinv(Y_unknown) @ U_adjusted
    
    # Display results
    a_actual = np.array([0.45, 0, 0, 0, 0.3, 0, 0, 0.3, 0, 0.3])

    print(f"Parameters for link7 Estimated: {a_unknown_estimated}")
    print(f"Parameters for link7 Actual:    {a_actual}")
    print(f"Parameters for link7 Error:     {a_unknown_estimated - a_actual}")
    
    # TODO compute the metrics (R-squared adjusted etc...) for the linear model on a different file 
    a_predicted = np.concatenate([a_known, a_unknown_estimated])
    U_predicted = Y_ALL @ a_predicted

    residuals = U_ALL - U_predicted
    n_observations = len(U_ALL)
    n_parameter = len(a_unknown_estimated)  # 10 parameters to estimate
    
    # 1. RMSE
    rmse = np.sqrt(np.mean(residuals**2))
    print(f"\nRMSE: {rmse:.2f} Nm")
    
    # 2. R-squared
    SSE = np.sum(residuals**2)
    SST = np.sum((U_ALL - np.mean(U_ALL))**2)
    R_squared = 1 - (SSE / SST)
    print(f"R^2: {R_squared:.2f}")
    
    # 3. Adjusted R-squared
    R_squared_adj = 1 - (1 - R_squared) * (n_observations - 1) / (n_observations - n_parameter - 1)
    print(f"Adjusted R^2: {R_squared_adj:.2f}")
    
    # 4. F-statistic
    F_statistic = (SST - SSE) / (n_parameter) / (SSE / (n_observations - n_parameter - 1))
    print(f"F-statistic: {F_statistic:.2f}")
    
    # 5. Standard error
    std_error = np.sqrt(SSE / (n_observations - n_parameter - 1))
    print(f"Standard Error: {std_error:.2f} Nm")
    # 6. Confidence intervals (95%)
    from scipy import stats
    alpha = 0.05
    t_value = stats.t.ppf(1 - alpha/2, n_observations - n_parameter - 1)
    
    cov_matrix = std_error**2 * np.linalg.inv(Y_unknown.T @ Y_unknown)
    param_std_errors = np.sqrt(np.diag(cov_matrix))
        
    print(f"\n95% Confidence Intervals:")
    param_names = ['Mass', 'CoM_x', 'CoM_y', 'CoM_z', 'Ixx', 'Ixy', 'Ixz', 'Iyy', 'Iyz', 'Izz']
    for i, name in enumerate(param_names):
        ci_lower = a_unknown_estimated[i] - t_value * param_std_errors[i]
        ci_upper = a_unknown_estimated[i] + t_value * param_std_errors[i]
        actual_val = a_actual[i]
        in_ci = "OK" if ci_lower <= actual_val <= ci_upper else "X"
        print(f"  {name:8s}: [{ci_lower:8.4f}, {ci_upper:8.4f}] (actual: {actual_val:.4f}) [{in_ci}]")
    
    # TODO plot the torque prediction error for each joint (optional)
    
  
if __name__ == '__main__':
    main()