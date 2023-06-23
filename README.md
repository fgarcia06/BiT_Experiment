# BiT_Experiment (First Work Term)
Francis Garcia's Github Repo

Last Updated: June 23 2023

# Model Creation:
  - mpcmodelsindy.py is an updated version of Antonio's SINDy model for the MPC. This new version utilizes multiprocessing and uses parallel simulation.
  - MPCmodelSINDy.ipynb is the simulation of the code. Make sure to have bit_data_*.csv, rgs_signal_*.csv, and calib_results.csv in the runtime files on Google Colab or any notebooks.

# Simulations:
- PID_Simulation.ipynb is the simulation code for a PID controller using the position and velocity form of a PID. This file is from Antonio's code and it focuses more on implementing position and velocity forms.
  - BasicPID_Simulation.ipynb is a simulation code I created for a basic PID controller. This is used to understand the different effects of tuning the P,I, and D parameters in a PID controller
  - MPC_Simulation.ipynb (Still working on the basic MPC controller simulation)

# Control Implementations:
- low_pass_test.py : A low pass filter I created for a possible implementation in the experiment
