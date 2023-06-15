from opcua import Client
from client_classes import *
from antony_mpc import *
import ast
import numpy as np
import pandas as pd
import time
from scipy.signal import butter, lfilter

# Define connectivity strings
URL = "opc.tcp://BB0253:4840/freeopcua/server/"
NAMESPACE = "http://examples.freeopcua.github.io"

CSV_PATH = "./Data/bit_data/bit_data_6.csv"
RGS_CSV = "./Data/MATLAB/rgs_signals_6.csv"
IMG_PATH = "./Data/Images/"

CAMERA_URL = 'http://admin:LogDeltav50@142.244.38.73/video/mjpg.cgi'

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff_freq, sampling_freq, filter_order=5):
    b, a = butter_lowpass(cutoff_freq, sampling_freq, order=filter_order)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def main():
    my_deque = deque(maxlen=2)
    img_thread = ImageThread(CAMERA_URL, my_deque)
    img_thread.start()

    level_df = pd.read_csv("./Data/bit_data/bit_data_6_model.csv", header=None)
    my_arr = level_df.to_numpy()
    minmax_arr = [ast.literal_eval(my_arr[i][0]) for i in range(4)]

    client = Client(URL)
    try:
        client.connect()
        ns_idx = client.get_namespace_index(NAMESPACE)
        bit_obj = client.nodes.objects.get_child(f"{ns_idx}:BIT Object")
        bit_reader = Reader(RGS_CSV, ns_idx, bit_obj, my_deque)
        calibrated_matr, calibrated_cam_matrix = bit_reader.calibrate(stall=3.5)
        calibrated_level = np.array(calibrated_matr[2:4])
        calibrated_cam_level = np.array(calibrated_cam_matrix[0:2])
        calibrated_cam_level = calibrated_cam_level[:, 0:3]

        num_tubes, pred_horizon, sample_num = 3, 2, 400

        Q = np.eye(3) * 9.184619941628146
        R1 = np.eye(3) * 0.0033621924720556584
        R2 = np.eye(3) * 1.771779130188831

        fan_spd_factor, experiment_height, height_sp, cam_height_arr = (
        [np.array([0.0, 0.0, 0.0]) for i in range(sample_num+1)] for y in range(4))

        # level_minmax = np.array(minmax_arr[0:2])
        gain_minmax = np.array(minmax_arr[2:])

        mpc = MPC(ns_idx, bit_obj, num_tubes, pred_horizon)

        # Coefficients in cost function J (choose how much to penalize high error or high controller /
        # change-in controller output)
        P = 'none'

        # Lower and upper limits of controller (used in optimizer)
        umax = 1.0
        umin = 0.0

        # Create set point array
        sps = np.array([0.7 , 0.4, 0.6 , 0.3])
        sp_change = int(sample_num / sps.size)
        sp_index = 0
        height_sp[0] = np.tile(sps[sp_index], num_tubes)
        experiment_height[0] = np.array([0.0, 0.0, 0.0])

        # Initial state of system
        uk, _ = mpc.nl_mpc(Q, R1, R2, experiment_height[0], pred_horizon, umin, umax, xk_sp=height_sp[0])
        fan_spd_factor[0] = np.round(uk, 0)
        time_array = [0]

        # Low-pass filter parameters
        cutoff_freq = 10  # Cutoff frequency (Hz)
        sampling_freq = 100  # Sampling frequency (Hz)
        filter_order = 4  # Filter order

        # Run MPC algorithm for "sample number" of iterations
        for i in range(1, sample_num + 1):
            # print(f"===== Sample: {i}/{sample_num} =====")
            # time_array[i] = round(i * sample_time, 1)
            sample_start = time.time()

            # Get current fan factors
            uk = mpc.denormalize(gain_minmax, fan_spd_factor[i-1])
            print(uk)
            # speed, height = sample(ns_idx, bit_obj, uk[0], uk[1], uk[2], 100, normalized=False)
            speed, height, cam_height = bit_reader.fused_sample(uk[0], uk[1], uk[2], 100, normalized=False)
            experiment_height[i] = mpc.normalize(calibrated_level, height)
            cam_height_arr[i] = mpc.normalize(calibrated_cam_level, cam_height[0:3])

            # Apply low-pass filter to experiment_height
            experiment_height[i] = apply_lowpass_filter(experiment_height[i], cutoff_freq, sampling_freq, filter_order)

            # Set point assignment
            height_sp[i] = np.tile(sps[sp_index], num_tubes)

            # Calculate optimal controller output (fan speed factors)
            uk, _ = mpc.nl_mpc(Q, R1, R2, experiment_height[i], pred_horizon, umin, umax, xk_sp=height_sp[i], P=P)
            uk = 1.0 - uk
            fan_spd_factor[i] = np.round(uk, 2)

            # Set point change
            if i > 0 and i % sp_change == 0:  # set point change frequency (units: samples)
                sp_index += 1
                if sp_index > len(sps) - 1:  # loop around set point array if its end is reached
                    sp_index = 0

            delay = 0.5 - (time.time() - sample_start)
            if(delay > 0):
                time.sleep(delay)
            time_array.append(time.time() - sample_start)

        time_array = np.cumsum(np.array(time_array))
        experiment_height = np.array(experiment_height)
        cam_height_arr = np.array(cam_height_arr)
        height_sp = np.array(height_sp)
        fan_spd_factor = np.array(fan_spd_factor)
        ultrasonic_error = np.sum(np.square(experiment_height - height_sp))
        camera_error = np.sum(np.square(cam_height_arr - height_sp))
        reading_diff = np.sum(np.square(experiment_height - cam_height_arr))



        # Plotting code written by Oguzhan Dogru
        curr_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        fig, ax = plt.subplots(3, 3, figsize=(10, 7), sharex='all', constrained_layout=True)

        ax[0,0].plot(time_array, experiment_height[:,0], '-.', color='tab:blue', alpha=0.5, label='Ultrasonic Height')
        ax[0,0].plot(time_array, height_sp[:,0], '-', color='k', alpha=0.5, label='Set Point')
        ax[1,0].plot(time_array, cam_height_arr[:,0], '-.', color='green', alpha=0.5, label='Camera Height')
        ax[1,0].plot(time_array, height_sp[:,0], '-', color='k', alpha=0.5, label='Set Point')
        ax[2,0].plot(time_array, fan_spd_factor[:,0], '.-', color='tab:red', alpha=0.5)
        ax[0,0].legend()
        ax[1,0].legend()

        ax[0,1].plot(time_array, experiment_height[:,1], '-.', color='tab:blue', alpha=0.5, label='Ultrasonic Height')
        ax[0,1].plot(time_array, height_sp[:,1], '-', color='k', alpha=0.5, label='Set Point')
        ax[1,1].plot(time_array, cam_height_arr[:,1], '-.', color='green', alpha=0.5, label='Camera Height')
        ax[1,1].plot(time_array, height_sp[:,1], '-', color='k', alpha=0.5, label='Set Point')
        ax[2,1].plot(time_array, fan_spd_factor[:,1], '.-', color='tab:red', alpha=0.5)
        ax[0,1].legend()
        ax[1,1].legend()

        ax[0,2].plot(time_array, experiment_height[:,2], '-.', color='tab:blue', alpha=0.5, label='Ultrasonic Height')
        ax[0,2].plot(time_array, height_sp[:,2], '-', color='k', alpha=0.5, label='Set Point')
        ax[1,2].plot(time_array, cam_height_arr[:,2], '-.', color='green', alpha=0.5, label='Camera Height')
        ax[1,2].plot(time_array, height_sp[:,2], '-', color='k', alpha=0.5, label='Set Point')
        ax[2,2].plot(time_array, fan_spd_factor[:,2], '.-', color='tab:red', alpha=0.5)
        ax[0,2].legend()
        ax[1,2].legend()
        ax[0,1].set_title(f'Ultrasonic Readings - Error: {ultrasonic_error}')
        ax[1,1].set_title(f'Camera Readings - Error: {camera_error}')
        ax[2,1].set_title(f'Fan Speed Factor')

        # Reverse the y-axis for each subplot
        ax[0,0].invert_yaxis()
        ax[1,0].invert_yaxis()
        ax[0,1].invert_yaxis()
        ax[1,1].invert_yaxis()
        ax[0,2].invert_yaxis()
        ax[1,2].invert_yaxis()

        fig.suptitle(f'MPC Controller - Reading Error: {reading_diff}\n Ultrasonic Range: {calibrated_level}', fontsize=12)

        plt.tight_layout()
        plt.savefig('./Data/bit_data/cam_height_4.pdf', bbox_inches='tight')  # Save the plot as a pdf file
        plt.show()

    finally:
        client.disconnect()
        img_thread.stop()
        img_thread.join()
        del bit_reader

if __name__ == '__main__':
	main()