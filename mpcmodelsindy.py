# -*- coding: utf-8 -*-
"""MPCmodelSINDY.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1J9Ksecv8-q-blGl1ij3xXgGdHh293i5u

##**Install all the required modules/dependencies**
"""

!pip install pysindy
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pysindy as ps
from sklearn.metrics import mean_squared_error, r2_score
import multiprocessing
import concurrent.futures
import pandas as pd

"""# **Creating the SINDy model for the MPC**"""

NUM_TUBES = 3
num_cores = multiprocessing.cpu_count()


def simulate_model(Y_test, U_test, model):
    results = model.simulate(Y_test[0, 0:3], len(Y_test), U_test)
    return results


def main():
    # Make the CSV files into pandas dataframes
    minmax_df = pd.read_csv('./drive/MyDrive/bit_data/calib_results.csv', header=None)
    level_df = pd.read_csv('./drive/MyDrive/bit_data/bit_data_5.csv')
    gain_df = pd.read_csv('./drive/MyDrive/MATLAB/rgs_signals_5.csv')

    # Convert dataframes to numpy arrays
    minmax_array = minmax_df.to_numpy()
    data_array = level_df.to_numpy()[3:, :]
    gain_array = gain_df.to_numpy()[3:, :]

    # Scale the data using MinMaxScaler (0-1)
    gain_scaler = preprocessing.MinMaxScaler()
    height_scaler = preprocessing.MinMaxScaler()
    Y = height_scaler.fit_transform(data_array[:, 4:7])
    gain_scaler.fit_transform(gain_array[:, 0:3])

    # Concatenate minmax values and reshape them
    minmax_vals = np.concatenate((height_scaler.data_min_, height_scaler.data_max_,
                                  gain_scaler.data_min_, gain_scaler.data_max_), axis=0)

    num_elements = int(minmax_vals.size / NUM_TUBES)
    minmax_vals = np.reshape(minmax_vals, (num_elements, NUM_TUBES))

    # Print minmax values
    for i in range(num_elements):
        print(list(minmax_vals[i]))

    # Scale the input data for training and testing
    U = gain_scaler.fit_transform(data_array[:, 0:3])
    train_split = round(0.8 * data_array.shape[0])
    test_split = data_array.shape[0] - train_split
    U_train, U_test = U[0:train_split], U[train_split:]
    Y_train, Y_test = Y[0:train_split], Y[train_split:]

    # Set up SINDy model parameters
    feature_library = ps.PolynomialLibrary(degree=3)
    optimizer = ps.STLSQ(
    threshold=0.13,
    alpha=0.05,
    max_iter=20,
    ridge_kw=None,  # Optional keyword arguments for ridge regression
    fit_intercept=False,
    normalize_columns=False,
    copy_X=True,
    initial_guess=None,
    verbose=False
    )
    diff_method = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 5})
    feature_names = ['x4', 'x3', 'x2', 'u4', 'u3', 'u2']

    # Create and fit the SINDy model
    model = ps.SINDy(discrete_time=True,
                     differentiation_method=diff_method,
                     feature_names=feature_names,
                     feature_library=feature_library,
                     optimizer=optimizer)

    model.fit(x=Y_train, u=U_train, t=1.5, unbias=True)
    model.print()


    # Simulate the model for testing data
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(simulate_model, [Y_test] * num_cores, [U_test] * num_cores, [model] * num_cores))

    # Plot the predicted vs actual results
    fig, axs = plt.subplots(3, sharex='all', figsize=(12, 7))
    fig.suptitle("Predicted vs Actual Results")

    count = 4

    for i in range(3):
        for result in results:
          axs[i].plot(result[:, i], color='tab:blue', ls='--', label=f'Tube {i + count} predicted')
        axs[i].plot(Y_test[:, i], color='tab:gray', alpha=0.5, label=f'Tube {i + count} actual')
        axs[i].legend(loc='upper left')
        count -= 2

    # Calculate and set MSE and R-squared values in the title
    mse = round(mean_squared_error(Y_test, results[0]), 5)
    rs = round(r2_score(Y_test, results[0]), 5)
    axs[0].title.set_text(f'RSquare: {rs} MSE: {mse}')
    plt.savefig('./bit_data/sindy_model_5.pdf')
    plt.show()



if __name__ == '__main__':
    main()