import numpy as np
import pandas as pd

import robot_vlp.data_collection.communication as c
import robot_vlp.data_collection.experment_processing as ep

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


def ekf_predict_with_heading(x_prev, P_prev, d, delta_theta, Q):
    """
    EKF Prediction Step
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


def run_ekf(df, err_stats):
    """
    Runs the EKF on a dataset.
    """

    # Define measurement noise covariance matrix R

    R = np.diag([
        err_stats['R_x'] ,
        err_stats['R_y'] ,
        err_stats['R_theta'] 
    ])


    # Initialize state
    x_init = df.loc[0, 'x_hist']
    y_init = df.loc[0, 'y_hist']
    theta_init = df.loc[0, 'heading_hist_rad']
    x = np.array([x_init, y_init, theta_init])

    # ── HERE: initialize P₀ with your “vanilla” startup uncertainty ──
    sigma_x0     = 0.01             # 1 cm
    sigma_y0     = 0.01             # 1 cm
    sigma_theta0 = np.deg2rad(3.0)  # 3°
    P = np.diag([sigma_x0**2, sigma_y0**2, sigma_theta0**2])
  


    # Store last position for heading calculation
    last_position = x[:2].copy()

    # List to store results
    res_lst = [(x[0], x[1], x[2])]

    # Main EKF loop
    for i in range(1, len(df)):
        step = df.iloc[i]
        d = step['encoder_location_change']
        delta_theta = step['encoder_heading_change_rad']

# --------------------   dynamic noise adjustment for heading --------
        # Conditionally adjust process noise for heading:
        if abs(delta_theta) < (1 / 180 * np.pi):
            # Use lower heading noise when no significant turn occurs
            heading_variance = err_stats['Q_theta_no_turn'] 

        else:
            heading_variance = err_stats['Q_theta']

        # Compute step variance for distance (remains constant)
        step_variance = err_stats['Q_dist']

        # Create a local process noise covariance matrix for this iteration
        Q_step = np.diag([step_variance, heading_variance])

# ----------------------------------------------------------------------------

        # EKF Prediction
        x_pred, P_pred = ekf_predict_with_heading(x, P, d, delta_theta, Q_step)

        # EKF Update (VLP sensor)
        z_meas = np.array([step['vlp_x_hist'], step['vlp_y_hist']])
        x, P = ekf_update_with_heading(x_pred, P_pred, z_meas, R, last_position)

        # Store results
        res_lst.append((x[0], x[1], x[2]))
        last_position = x[:2].copy()

    df[['EKF_x', 'EKF_y', 'EKF_heading_rad']] = np.vstack(res_lst)
    return df



# def calc_err_stats(df_lst):
#     R_x_lst = []
#     R_y_lst = []
#     R_theta_lst = []


#     Q_theta_lst = []
#     Q_theta_no_turn_lst = []
#     Q_dist_lst = []

#     for df in df_lst:
#         # VLP Error Statistics
#         vlp_x_errs = ep.filter_outliers(df['x_hist'] - df['vlp_x_hist'], threshold=2)
#         vlp_y_errs = ep.filter_outliers(df['y_hist'] - df['vlp_y_hist'], threshold=2)

#         R_x = vlp_x_errs.var() if len(vlp_x_errs) > 0 else 0
#         R_y = vlp_y_errs.var() if len(vlp_y_errs) > 0 else 0

#         # Compute R_theta from vlp readings
#         abs_heading_errs = normalize_angle_rad(df['heading_hist_rad'] - df['vlp_heading_hist_rad'])
#         abs_heading_errs = ep.filter_outliers(abs_heading_errs, threshold=2)
#         R_theta = abs_heading_errs.var(ddof=1) if len(abs_heading_errs) > 0 else 0
        

#         # Encoder Heading Change Error
#         turn_df = df[df['encoder_heading_change_rad'] != 0]
#         move_df = df[df['encoder_heading_change_rad'] == 0]

#         Q_theta = ep.filter_outliers(turn_df['encoder_heading_change_err_rad'], threshold=2).var() if not turn_df.empty else 0
#         Q_theta_no_turn = ep.filter_outliers(move_df['encoder_heading_change_err_rad'], threshold=2).var() if not move_df.empty else 0

#         # Encoder Location Change Error
#         Q_dist = ep.filter_outliers(df['encoder_location_change_err'], threshold=2).var()

#         # Store in lists
#         R_x_lst.append(R_x)
#         R_y_lst.append(R_y)
#         R_theta_lst.append(R_theta)

#         Q_theta_lst.append(Q_theta)
#         Q_theta_no_turn_lst.append(Q_theta_no_turn)
#         Q_dist_lst.append(Q_dist)

#     # Compute final statistics as the mean across all datasets
#     return {
#         'R_x': np.mean(R_x_lst),
#         'R_y': np.mean(R_y_lst),
#         'R_theta': np.mean(R_theta_lst),
#         'Q_theta': np.mean(Q_theta_lst),
#         'Q_theta_no_turn': np.mean(Q_theta_no_turn_lst),
#         'Q_dist': np.mean(Q_dist_lst)
#     }
def calc_err_stats(df_list):
    """
    Compute EKF noise statistics by pooling errors across all runs.
    
    Returns a dict with:
      - R_x, R_y: VLP position variances
      - R_theta: VLP absolute‐heading variance
      - Q_theta: encoder‐turn Δθ variance
      - Q_theta_no_turn: encoder‐straight Δθ variance
      - Q_dist: encoder distance‐change variance
    """
    # 1) Accumulate all error samples
    vlp_x_errs_all          = []
    vlp_y_errs_all          = []
    vlp_heading_errs_all    = []
    enc_turn_errs_all       = []
    enc_no_turn_errs_all    = []
    enc_dist_errs_all       = []

    for df in df_list:
        # VLP position errors
        x_err = ep.filter_outliers(df['x_hist'] - df['vlp_x_hist'], threshold=2)
        y_err = ep.filter_outliers(df['y_hist'] - df['vlp_y_hist'], threshold=2)
        vlp_x_errs_all.append(x_err)
        vlp_y_errs_all.append(y_err)

        # VLP absolute-heading errors
        abs_heading = normalize_angle_rad(df['heading_hist_rad'] - df['vlp_heading_hist_rad'])
        abs_heading = ep.filter_outliers(abs_heading, threshold=2)
        vlp_heading_errs_all.append(abs_heading)

        # Encoder Δθ errors: turns vs. no‑turns
        turn_mask    = df['encoder_heading_change_rad'] != 0
        no_turn_mask = ~turn_mask

        turn_errs    = ep.filter_outliers(
                          df.loc[turn_mask, 'encoder_heading_change_err_rad'], threshold=2
                       )
        no_turn_errs = ep.filter_outliers(
                          df.loc[no_turn_mask, 'encoder_heading_change_err_rad'], threshold=2
                       )
        enc_turn_errs_all.append(turn_errs)
        enc_no_turn_errs_all.append(no_turn_errs)

        # Encoder distance‐change errors
        dist_errs = ep.filter_outliers(df['encoder_location_change_err'], threshold=2)
        enc_dist_errs_all.append(dist_errs)

    # 2) Concatenate into large arrays
    vlp_x_all       = np.concatenate(vlp_x_errs_all)       if vlp_x_errs_all       else np.array([])
    vlp_y_all       = np.concatenate(vlp_y_errs_all)       if vlp_y_errs_all       else np.array([])
    vlp_heading_all = np.concatenate(vlp_heading_errs_all) if vlp_heading_errs_all else np.array([])
    turn_all        = np.concatenate(enc_turn_errs_all)    if enc_turn_errs_all    else np.array([])
    no_turn_all     = np.concatenate(enc_no_turn_errs_all) if enc_no_turn_errs_all else np.array([])
    dist_all        = np.concatenate(enc_dist_errs_all)    if enc_dist_errs_all    else np.array([])

    # 3) Compute pooled variances (ddof=1 for sample variance)
    R_x               = np.var(vlp_x_all,       ddof=1) if vlp_x_all.size       > 1 else 0.0
    R_y               = np.var(vlp_y_all,       ddof=1) if vlp_y_all.size       > 1 else 0.0
    R_theta           = np.var(vlp_heading_all, ddof=1) if vlp_heading_all.size > 1 else 0.0
    Q_theta           = np.var(turn_all,        ddof=1) if turn_all.size        > 1 else 0.0
    Q_theta_no_turn   = np.var(no_turn_all,     ddof=1) if no_turn_all.size     > 1 else 0.0
    Q_dist            = np.var(dist_all,        ddof=1) if dist_all.size        > 1 else 0.0

    return {
        'R_x':               R_x,
        'R_y':               R_y,
        'R_theta':           R_theta,
        'Q_theta':           Q_theta,
        'Q_theta_no_turn':   Q_theta_no_turn,
        'Q_dist':            Q_dist
    }



class LiveEKF:
    """
    A wrapper for the Extended Kalman Filter (EKF) to support live prediction.
    
    The LiveEKF class maintains the current state and covariance and exposes a predict() method.
    The predict() method accepts a new measurement sample, performs the EKF prediction and update steps,
    updates the state, and returns the current state prediction.
    
    Required inputs at initialization:
        - initial_state: Initial state vector [x, y, theta]
        - err_stats: A dict containing error statistics. Expected keys:
            'R_pos', 'Q_theta', 'Q_theta_no_turn', 'Q_dist'
    """
    def __init__(self, initial_state, err_stats):
        self.x = np.array(initial_state)   # state vector [x, y, theta]
        sigma_x0     = 0.05
        sigma_y0     = 0.05
        sigma_theta0 = np.deg2rad(5.0)
        self.P = np.diag([sigma_x0**2, sigma_y0**2, sigma_theta0**2])
        self.err_stats = err_stats         # e.g., {'R_pos': ..., 'step_size': ..., 'Q_theta': ..., 'Q_theta_no_turn': ..., 'Q_dist': ...}
        self.last_position = self.x[:2].copy()

    
    def predict(self, d, delta_theta, z_meas):
        """
        Process a new measurement and update the EKF state.
        
        Arguments:
            d : float
                Measured distance (e.g. encoder_location_change)
            delta_theta : float
                Measured heading change in radians (e.g. encoder_heading_change_rad)
            z_meas : array-like of shape (2,)
                VLP measurement for position [vlp_x, vlp_y]
        
        Returns:
            Updated state vector [x, y, theta]
        """
        # Calculate the process noise covariance for this step.
        if abs(delta_theta) < (1/180 * np.pi):
            heading_variance = self.err_stats['Q_theta_no_turn'] 
        else:
            heading_variance = self.err_stats['Q_theta'] 
        step_variance = self.err_stats['Q_dist'] 
        Q_step = np.diag([step_variance, heading_variance])
        
        # Compute the measurement noise covariance matrix R.
        R = np.diag([
            self.err_stats['R_x'] ,
            self.err_stats['R_y'] ,
            self.err_stats['R_theta'] 
        ])
        
        # --- EKF Prediction Step ---
        x_pred, P_pred = ekf_predict_with_heading(self.x, self.P, d, delta_theta, Q_step)
        
        # --- EKF Update Step ---
        # z_meas is expected to be [vlp_x, vlp_y]. Heading will be computed using the previous position.
        x_upd, P_upd = ekf_update_with_heading(x_pred, P_pred, z_meas, R, self.last_position)
        
        # Update stored state and covariance.
        self.x = x_upd
        self.P = P_upd
        
        # Update the last_position for next prediction.
        self.last_position = self.x[:2].copy()
        
        return self.x