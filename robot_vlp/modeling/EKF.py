import numpy as np
import pandas as pd
from scipy.optimize import minimize
import robot_vlp.data_collection.communication as c
import robot_vlp.data_collection.experment_processing as ep
from joblib import Parallel, delayed
from scipy.optimize import differential_evolution
def calc_EKF_pos_err(df):
    x = df['x_hist']
    y = df['y_hist']
    ex = df['EKF_x']
    ey = df['EKF_y']
    return np.sqrt(np.square(x - ex) + np.square(y - ey))
def calc_EKF_heading_err(df):
    heading = df['heading_hist_rad']
    eheading = df['EKF_heading_rad']
    return np.array([normalize_angle_rad(a) for a in (heading - eheading)])

def calc_pos_err(df):
    x = df['x_hist']
    y = df['y_hist']
    ex = df['encoder_x_hist']
    ey = df['encoder_y_hist']
    return np.sqrt(np.square(x - ex) + np.square(y - ey))
def calc_heading_err(df):
    heading = df['heading_hist_rad']
    eheading = df['encoder_heading_hist_rad']
    return np.array([normalize_angle_rad(a)for a in (heading - eheading)])

def calc_vlp_pos_err(df):
    x = df['x_hist']
    y = df['y_hist']
    ex = df['vlp_x_hist']
    ey = df['vlp_y_hist']
    return np.sqrt(np.square(x - ex) + np.square(y - ey))
def calc_vlp_heading_err(df):
    heading = df['heading_hist_rad']
    eheading = df['vlp_heading_hist_rad']
    return np.array([normalize_angle_rad(a) for a in (heading - eheading)])


def degrees_to_radians(deg):
    """Convert degrees to radians."""
    return np.deg2rad(deg)

def normalize_angle_deg(angle):
    """Normalize an angle in degrees to the range [-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0


def normalize_angle_rad(angle):
    """Normalize an angle in radians to the range [-π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def ekf_predict_with_heading(x_prev, P_prev, d, delta_theta, G, Q):
    """
    EKF Prediction Step with dynamically updated G but constant Q.
    """
    # Extract previous state
    x, y, theta = x_prev

    # Update heading
    theta_new = normalize_angle_rad(theta + delta_theta)


    # Predict new position
    x_pred = np.array([
        x + d * np.sin(theta_new),
        y + d * np.cos(theta_new),
        theta_new
    ])

    # Update transformation matrix G based on the new heading
    G = np.array([
        [np.sin(theta_new), 0],
        [np.cos(theta_new), 0],
        [0, 1]
    ])


    # Transform process noise from motion space to state space
    Q_transformed = G @ Q @ G.T

    # Jacobian of motion model
    F_k = np.array([
        [1, 0, d * np.sin(theta_new)],  
        [0, 1, d * np.cos(theta_new)],  
        [0, 0, 1]
    ])


    # Predict covariance
    P_pred = F_k @ P_prev @ F_k.T + Q_transformed

    return x_pred, P_pred



def ekf_update_with_heading(x_pred, P_pred, z_meas, R, last_position):
    """
    EKF Update Step using VLP x, y for position and VLP-derived heading for heading update.
    
    Args:
        x_pred: Predicted state vector [x, y, theta]
        P_pred: Predicted covariance matrix (3x3)
        z_meas: Measurement vector [x_vlp, y_vlp, theta_vlp]
        R: Measurement noise covariance matrix (3x3)
        last_position: Previous position [x_last, y_last]
    
    Returns:
        x_upd: Updated state vector
        P_upd: Updated covariance matrix
    """
    # Compute Heading from VLP Movement Vector
    delta_x = z_meas[0] - last_position[0]
    delta_y = z_meas[1] - last_position[1]
    theta_vlp = np.arctan2(delta_x, delta_y)

    # Update measurement vector to include heading
    z_meas = np.array([z_meas[0], z_meas[1], theta_vlp])

    # Expanded Measurement Model (x, y, and heading)
    H_k = np.array([
        [1, 0, 0],  # x position
        [0, 1, 0],  # y position
        [0, 0, 1]   # heading
    ])

    # Compute Innovation (Residual)
    y_k = z_meas - (H_k @ x_pred)

    # Normalize heading angle to [-180, 180]
    y_k[2] = normalize_angle_rad(y_k[2])

    # Compute Innovation Covariance
    S_k = H_k @ P_pred @ H_k.T + R

    # Compute Kalman Gain
    K_k = P_pred @ H_k.T @ np.linalg.inv(S_k)

    # Update state vector
    x_upd = x_pred + (K_k @ y_k)

    # Normalize updated heading
    x_upd[2] = normalize_angle_rad(x_upd[2])

    # Update covariance matrix
    P_upd = (np.eye(len(P_pred)) - (K_k @ H_k)) @ P_pred

    return x_upd, P_upd


def run_ekf(df, parameters, err_stats):
    """
    Runs the EKF on a dataset.
    """
    # Extract parameters
    Q_scale_d = parameters['Q_scale_d']
    Q_scale_theta = parameters['Q_scale_theta']
    R_scale_pos = parameters['R_scale_pos']


    # Extract error statistics
    # Extract error statistics
    R_pos = err_stats['R_pos']
    step_size = err_stats['step_size']
    Q_theta = err_stats['Q_theta']
    Q_theta_no_turn = err_stats['Q_theta_no_turn']
    Q_dist = err_stats['Q_dist']


    # Define constant process noise covariance matrix Q (motion space)
    step_variance = Q_dist * Q_scale_d
    heading_variance = Q_theta * Q_scale_theta
    Q_constant = np.diag([step_variance, heading_variance])

    # Define measurement noise covariance matrix R
    


    R = np.diag([
        R_pos * R_scale_pos,
        R_pos * R_scale_pos,
        (R_pos/step_size **2) * R_scale_pos
    ])

    # Initialize transformation matrix G (to be updated dynamically)
    G = np.array([
        [0, 0],  # sin(theta) will be updated dynamically
        [0, 0],  # cos(theta) will be updated dynamically
        [0, 1]   # heading change remains static
    ])

    # Initialize state
    x_init = df.loc[0, 'x_hist']
    y_init = df.loc[0, 'y_hist']
    theta_init = df.loc[0, 'heading_hist_rad']
    x = np.array([x_init, y_init, theta_init])
    P = np.diag([Q_dist, Q_dist, Q_theta])

    # Store last position for heading calculation
    last_position = x[:2].copy()

    # List to store results
    res_lst = [(x[0], x[1], x[2])]

    # Main EKF loop
    for i in range(1, len(df)):
        step = df.iloc[i]
        d = step['encoder_location_change']
        delta_theta = step['encoder_heading_change_rad']

        # EKF Prediction
        x_pred, P_pred = ekf_predict_with_heading(x, P, d, delta_theta, G, Q_constant)

        # EKF Update (VLP sensor)
        z_meas = np.array([step['vlp_x_hist'], step['vlp_y_hist']])
        x, P = ekf_update_with_heading(x_pred, P_pred, z_meas, R, last_position)

        # Store results
        res_lst.append((x[0], x[1], x[2]))
        last_position = x[:2].copy()

    df[['EKF_x', 'EKF_y', 'EKF_heading_rad']] = np.array(res_lst)
    return df



def calc_err_stats(df_lst):
    R_pos_lst = []
    step_size_lst = []

    Q_theta_lst = []
    Q_theta_no_turn_lst = []
    Q_dist_lst = []

    for df in df_lst:
        # VLP Error Statistics
        vlp_x_errs = ep.filter_outliers(df['x_hist'] - df['vlp_x_hist'], threshold=2)
        vlp_y_errs = ep.filter_outliers(df['y_hist'] - df['vlp_y_hist'], threshold=2)

        vlp_x_var = vlp_x_errs.var() if len(vlp_x_errs) > 0 else 0
        vlp_y_var = vlp_y_errs.var() if len(vlp_y_errs) > 0 else 0

        R_pos = (vlp_x_var + vlp_y_var) / 2  # Take the average of x and y variances

        # Robust estimation of step size
        step_size = np.median(ep.filter_outliers(df['vive_location_change']) ) # Using median instead of mean for robustness

        # Compute R_theta from positional variance
        R_theta = R_pos / (step_size ** 2) if step_size > 0 else 0

        # Encoder Heading Change Error
        turn_df = df[df['encoder_heading_change_rad'] != 0]
        move_df = df[df['encoder_heading_change_rad'] == 0]

        Q_theta = ep.filter_outliers(turn_df['encoder_heading_change_err_rad'], threshold=2).var() if not turn_df.empty else 0
        Q_theta_no_turn = ep.filter_outliers(move_df['encoder_heading_change_err_rad'], threshold=2).var() if not move_df.empty else 0

        # Encoder Location Change Error
        Q_dist = ep.filter_outliers(df['encoder_location_change_err'], threshold=2).var()

        # Store in lists
        R_pos_lst.append(R_pos)
        step_size_lst.append(step_size)

        Q_theta_lst.append(Q_theta)
        Q_theta_no_turn_lst.append(Q_theta_no_turn)
        Q_dist_lst.append(Q_dist)

    # Compute final statistics as the mean across all datasets
    return {
        'R_pos': np.mean(R_pos_lst),
        'step_size': np.mean(step_size_lst),
        'Q_theta': np.mean(Q_theta_lst),
        'Q_theta_no_turn': np.mean(Q_theta_no_turn_lst),
        'Q_dist': np.mean(Q_dist_lst)
    }





def run_ekf_testing(df_lst, parameters):
    err_stats = calc_err_stats(df_lst)
    err_lst = []
    for df in df_lst:
        output_df = run_ekf(df, parameters,err_stats)
        err = np.abs(calc_EKF_heading_err(output_df)).mean() + calc_EKF_pos_err(output_df).mean()
        err_lst.append(err)

    return np.mean(err_lst)
        

import numpy as np
import pandas as pd
import time
from scipy.optimize import differential_evolution
from joblib import Parallel, delayed

# Store progress history and dataset reference
optimization_progress = []
df_lst_global = None  # Define a global reference for dataset list
param_keys_global = None  # Define a global reference for parameter names

# Global Objective Function (fixes pickling issue)
def objective_function(param_values):
    """ Objective function to minimize for EKF parameter tuning. """
    global df_lst_global, param_keys_global

    params = {key: param_values[i] for i, key in enumerate(param_keys_global)}
    
    # Run EKF testing serially inside multiprocessing
    err = Parallel(n_jobs=1)(delayed(run_ekf_testing)([df], params) for df in df_lst_global)

    return np.mean(err)

# Progress Tracking Callback Function (Now Uses Global Variables)
def progress_callback(xk, convergence):
    """ Callback function to track optimization progress in differential evolution. """
    global optimization_progress, df_lst_global, param_keys_global

    iteration = len(optimization_progress) + 1

    # Compute current best error using the updated xk values
    best_error = objective_function(xk)
    optimization_progress.append((iteration, best_error, xk))

    # Print progress every 5 iterations
    if iteration % 5 == 0:
        print(f"🔄 Iteration {iteration}: Best Error = {best_error:.4f}")
        print(f"   Best Parameters: {dict(zip(param_keys_global, xk))}")

    # Stop early if convergence is low
    if convergence < 1e-7:
        print("⚡ Early stopping triggered due to minimal improvement.")
        return True  # Stops optimization




def tune_ekf(df_lst, prev_result=None, initial_guess=None):
    """ Optimize EKF parameters using Differential Evolution, supporting resumption and initial guesses. """

    global optimization_progress, df_lst_global, param_keys_global
    optimization_progress = []  # Reset tracking
    df_lst_global = df_lst  # Store dataset list globally

    # ==========================
    # Define Parameter Bounds
    # ==========================
    bounds = [
        (0.0001, 500),  # Q_scale_d
        (0.0001, 500),  # Q_scale_theta
        (0.0001, 500),  # R_scale_pos
    ]

    # ==========================
    # Define Parameter Names
    # ==========================
    param_keys_global = ['Q_scale_d', 'Q_scale_theta', 'R_scale_pos']

    # ==========================
    # Set Initialization Mode
    # ==========================
    if prev_result:
        print("\n🔄 Resuming Optimization with Small Perturbations...")
        best_x = np.clip(prev_result.x, [b[0] for b in bounds], [b[1] for b in bounds])

        # Create a varied population by adding small noise
        noise = np.random.uniform(-0.05, 0.05, size=(10, len(best_x))) * best_x
        init_population = np.clip(best_x + noise, [b[0] for b in bounds], [b[1] for b in bounds])
        
        init_mode = init_population  # Use as initial population

    elif initial_guess is not None:
        print("\n📌 Using User-Provided Initial Guess for Optimization...")
        best_x = np.clip(initial_guess, [b[0] for b in bounds], [b[1] for b in bounds])

        # Expand into a population by adding small random perturbations
        noise = np.random.uniform(-0.05, 0.05, size=(10, len(best_x))) * best_x
        init_population = np.clip(best_x + noise, [b[0] for b in bounds], [b[1] for b in bounds])

        init_mode = init_population  # Use initial guess for optimization

    else:
        print("\n🚀 Starting New Optimization Run...")
        init_mode = 'latinhypercube'  # Default randomized initialization

    # ==========================
    # Run Optimization
    # ==========================
    start_time = time.time()
    result = differential_evolution(
        objective_function,
        bounds=bounds,
        strategy='best1bin',
        popsize=20,
        mutation=(0.95, 1.99),
        recombination=0.9,
        maxiter=5,
        tol=1e-5,
        disp=True,
        callback=progress_callback,
        init=init_mode,  # Use precomputed population if available
        polish=False
    )
    end_time = time.time()

    # Convert optimized parameters back from list
    optimized_params = {key: result.x[i] for i, key in enumerate(param_keys_global)}

    print("\n✅ Optimization Complete!")
    print(f"🕒 Total Time Taken: {end_time - start_time:.2f} seconds")
    print(f"🔍 Final Best Error: {result.fun:.4f}")
    print(f"📌 Best Parameters: {optimized_params}")

    return optimized_params, result, optimization_progress
