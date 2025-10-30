import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
import threading
import pickle

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rollout_loader import load_rollouts

from pathlib import Path

FINAL_DIR = Path(__file__).resolve().parent
DATA_PART1_DIR = FINAL_DIR / "data_part1"
DATA_PART1_DIR.mkdir(parents=True, exist_ok=True)


PRINT_PLOTS = False
RECORDING = True

downsample_rate = 2

def get_downsample_rate():
    try:
        rate = int(input("Enter downsample rate (integer >=1): "))
        if rate < 1:
            print("Invalid downsample rate. Must be >= 1.")
            return None
        return rate
    except ValueError:
        print("Please enter a valid integer.")
        return None


def main():

    conf_file_name = "pandaconfig.json"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext = root_dir)

    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)

    source_names = ["pybullet"]

    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,root_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos,init_R = dyn_model.ComputeFK(init_joint_angles,controlled_frame_name)
    
    print(f"Initial joint angles: {init_joint_angles}")
    
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    print(f"Lower limits: {lower_limits}")
    print(f"Upper limits: {upper_limits}")

    joint_vel_limits = sim.GetBotJointsVelLimit()
    
    print(f"joint vel limits: {joint_vel_limits}")
    
    q_des =  init_joint_angles
    qd_des_clip = np.zeros(num_joints)
    
    current_time = 0
    cmd = MotorCommands()

    kp_pos = 100
    kp_ori = 0

    kp = 1000
    kd = 100

    np.random.seed(42)
    num_points = 50
    list_of_desired_cartesian_positions = []
    for _ in range(num_points):
        x = np.random.uniform(0.1, 0.6) 
        y = np.random.uniform(-0.3, 0.3)
        z = np.random.uniform(-0.15, 0.3)
        list_of_desired_cartesian_positions.append([x, y, z])
    
    list_of_desired_cartesian_orientations = [[0.0, 0.0, 0.0, 1.0]] * num_points
    list_of_type_of_control = ["pos"] * num_points
    list_of_duration_per_desired_cartesian_positions = [5.0] * num_points
    list_of_initialjoint_positions = [init_joint_angles] * num_points

    q_mes_all, qd_mes_all, q_d_all, qd_d_all, tau_mes_all, tau_cmd_all, cart_pos_all, cart_ori_all = [], [], [], [], [], [], [], []

    current_time = 0
    time_step = sim.GetTimeStep()


    for i in range(len(list_of_desired_cartesian_positions)):
        
        q_mes_all, qd_mes_all, q_d_all, qd_d_all, tau_mes_all, tau_cmd_all, cart_pos_all, cart_ori_all = [], [], [], [], [], [], [], []

        desired_cartesian_pos = np.array(list_of_desired_cartesian_positions[i])
        desired_cartesian_ori = np.array(list_of_desired_cartesian_orientations[i])
        duration_per_desired_cartesian_pos = list_of_duration_per_desired_cartesian_positions[i]
        type_of_control = list_of_type_of_control[i]
        
        init_position = list_of_initialjoint_positions[i]
        
        steps = int(duration_per_desired_cartesian_pos/time_step)

        sim.ResetPose()
        if init_position is not None:
            sim.SetjointPosition(init_position)

        for t in range(steps):
            q_mes = sim.GetMotorAngles(0)
            qd_mes = sim.GetMotorVelocities(0)
            tau_mes = np.asarray(sim.GetMotorTorques(0),dtype=float)

            pd_d = [0.0, 0.0, 0.0]
            ori_d_des = [0.0, 0.0, 0.0]
            
            q_des, qd_des_clip = CartesianDiffKin(dyn_model,controlled_frame_name,q_mes, desired_cartesian_pos, pd_d, desired_cartesian_ori, ori_d_des, time_step, "pos",  kp_pos, kp_ori, np.array(joint_vel_limits))
            
            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
            cmd.SetControlCmd(tau_cmd, ["torque"] * 7)
            sim.Step(cmd, "torque")

            keys = sim.GetPyBulletClient().getKeyboardEvents()
            qKey = ord('q')

            if qKey in keys and keys[qKey] & sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
                print("Exiting simulation.")
                break
            
            if RECORDING:
                q_mes_all.append(q_mes)
                qd_mes_all.append(qd_mes)
                q_d_all.append(q_des)
                tau_mes_all.append(tau_mes)
                tau_cmd_all.append(tau_cmd)

            time.sleep(time_step)
            current_time += time_step
    
        current_time = 0

        if len(q_mes_all) > 0:    
            print("Preparing to save data...")
            
            q_mes_all_downsampled = q_mes_all[::downsample_rate]
            q_d_all_downsampled = q_d_all[::downsample_rate]
            tau_mes_all_downsampled = tau_mes_all[::downsample_rate]
            tau_cmd_all_downsampled = tau_cmd_all[::downsample_rate]

            time_array = [time_step * downsample_rate * i for i in range(len(q_mes_all_downsampled))]

            filename = DATA_PART1_DIR / f"data_{i}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump({
                    'time': time_array,
                    'q_mes_all': q_mes_all_downsampled,
                    'q_d_all': q_d_all_downsampled,
                    'tau_mes_all': tau_mes_all_downsampled,
                    'tau_cmd_all': tau_cmd_all_downsampled,
                }, f)
            print(f"Data saved to {filename}")

        if PRINT_PLOTS:
            pass
    
if __name__ == '__main__':
    main()