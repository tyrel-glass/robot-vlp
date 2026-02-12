import numpy as np
import pandas as pd

import robot_vlp.data_collection.experment_processing as ep


def calc_EKF_pos_err(df):
    x = df["x_hist"]
    y = df["y_hist"]
    ex = df["EKF_x"]
    ey = df["EKF_y"]
    return np.sqrt(np.square(x - ex) + np.square(y - ey))


def calc_EKF_heading_err(df):
    heading = df["heading_hist_rad"]
    eheading = df["EKF_heading_rad"]
    return normalize_angle_rad(heading - eheading)


def calc_pos_err(df):
    x = df["x_hist"]
    y = df["y_hist"]
    ex = df["encoder_x_hist"]
    ey = df["encoder_y_hist"]
    return np.sqrt(np.square(x - ex) + np.square(y - ey))


def calc_heading_err(df):
    heading = df["heading_hist_rad"]
    eheading = df["encoder_heading_hist_rad"]
    return normalize_angle_rad(heading - eheading)


def calc_vlp_pos_err(df):
    x = df["x_hist"]
    y = df["y_hist"]
    ex = df["vlp_x_hist"]
    ey = df["vlp_y_hist"]
    return np.sqrt(np.square(x - ex) + np.square(y - ey))


def calc_vlp_heading_err(df):
    heading = df["heading_hist_rad"]
    eheading = df["vlp_heading_hist_rad"]
    return normalize_angle_rad(heading - eheading)


def normalize_angle_rad(angle):
    """Normalize an angle in radians to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _vlp_heading_from_positions(vlp_x, vlp_y):
    delta_x = np.diff(vlp_x)
    delta_y = np.diff(vlp_y)
    return np.arctan2(delta_x, delta_y)


def ekf_predict_with_heading_variant(
    x_prev, P_prev, d, delta_theta, Q, fix_jacobian=True, fix_noise_mapping=True
):
    """
    EKF prediction step with optional fixes for Jacobian and noise mapping.
    """
    x, y, theta = x_prev

    theta_new = normalize_angle_rad(theta + delta_theta)
    x_pred = np.array(
        [
            x + d * np.sin(theta_new),
            y + d * np.cos(theta_new),
            theta_new,
        ]
    )

    if fix_noise_mapping:
        G = np.array(
            [
                [np.sin(theta_new), d * np.cos(theta_new)],
                [np.cos(theta_new), -d * np.sin(theta_new)],
                [0.0, 1.0],
            ]
        )
    else:
        G = np.array(
            [
                [np.sin(theta_new), 0.0],
                [np.cos(theta_new), 0.0],
                [0.0, 1.0],
            ]
        )
    Q_transformed = G @ Q @ G.T

    if fix_jacobian:
        F_k = np.array(
            [
                [1.0, 0.0, d * np.cos(theta_new)],
                [0.0, 1.0, -d * np.sin(theta_new)],
                [0.0, 0.0, 1.0],
            ]
        )
    else:
        F_k = np.array(
            [
                [1.0, 0.0, d * np.sin(theta_new)],
                [0.0, 1.0, d * np.cos(theta_new)],
                [0.0, 0.0, 1.0],
            ]
        )

    P_pred = F_k @ P_prev @ F_k.T + Q_transformed
    return x_pred, P_pred


def ekf_predict_with_heading(x_prev, P_prev, d, delta_theta, Q):
    """
    EKF prediction step for state [x, y, theta] (fixed Jacobian/noise mapping).
    """
    return ekf_predict_with_heading_variant(
        x_prev, P_prev, d, delta_theta, Q, fix_jacobian=True, fix_noise_mapping=True
    )


def ekf_update_with_heading(x_pred, P_pred, z_meas, R, last_position):
    """
    EKF update step using position-based heading from a prior position.

    Args:
        x_pred: Predicted state vector [x, y, theta]
        P_pred: Predicted covariance matrix (3x3)
        z_meas: Measurement vector [x_vlp, y_vlp]
        R: Measurement noise covariance matrix (3x3)
        last_position: Previous position [x_last, y_last]

    Returns:
        x_upd: Updated state vector
        P_upd: Updated covariance matrix
    """
    delta_x = z_meas[0] - last_position[0]
    delta_y = z_meas[1] - last_position[1]
    theta_vlp = np.arctan2(delta_x, delta_y)

    z_meas = np.array([z_meas[0], z_meas[1], theta_vlp])

    H_k = np.eye(3)
    y_k = z_meas - (H_k @ x_pred)
    y_k[2] = normalize_angle_rad(y_k[2])

    S_k = H_k @ P_pred @ H_k.T + R
    K_k = P_pred @ H_k.T @ np.linalg.inv(S_k)

    x_upd = x_pred + (K_k @ y_k)
    x_upd[2] = normalize_angle_rad(x_upd[2])

    P_upd = (np.eye(len(P_pred)) - (K_k @ H_k)) @ P_pred
    return x_upd, P_upd


def run_ekf(df, err_stats):
    """
    Run the fixed EKF on a dataset.
    """
    return run_ekf_variant(
        df,
        err_stats,
        fix_jacobian=True,
        fix_noise_mapping=True,
        heading_source="vlp",
    )


def run_ekf_variant(
    df,
    err_stats,
    fix_jacobian=False,
    fix_noise_mapping=False,
    heading_source="ekf",
):
    """
    Run EKF with selectable fixes and heading source.
    """
    R = np.diag([err_stats["R_x"], err_stats["R_y"], err_stats["R_theta"]])

    x_init = df.loc[0, "x_hist"]
    y_init = df.loc[0, "y_hist"]
    theta_init = df.loc[0, "heading_hist_rad"]
    x = np.array([x_init, y_init, theta_init])

    sigma_x0 = 0.01
    sigma_y0 = 0.01
    sigma_theta0 = np.deg2rad(3.0)
    P = np.diag([sigma_x0**2, sigma_y0**2, sigma_theta0**2])

    last_ekf_position = x[:2].copy()
    last_vlp_position = np.array(
        [df.loc[0, "vlp_x_hist"], df.loc[0, "vlp_y_hist"]]
    )

    res_lst = [(x[0], x[1], x[2])]

    for i in range(1, len(df)):
        step = df.iloc[i]
        d = step["encoder_location_change"]
        delta_theta = step["encoder_heading_change_rad"]

        if abs(delta_theta) < (1 / 180 * np.pi):
            heading_variance = err_stats["Q_theta_no_turn"]
        else:
            heading_variance = err_stats["Q_theta"]

        step_variance = err_stats["Q_dist"]
        Q_step = np.diag([step_variance, heading_variance])

        x_pred, P_pred = ekf_predict_with_heading_variant(
            x,
            P,
            d,
            delta_theta,
            Q_step,
            fix_jacobian=fix_jacobian,
            fix_noise_mapping=fix_noise_mapping,
        )

        z_meas = np.array([step["vlp_x_hist"], step["vlp_y_hist"]])
        if heading_source == "vlp":
            x, P = ekf_update_with_heading(
                x_pred, P_pred, z_meas, R, last_vlp_position
            )
            last_vlp_position = z_meas.copy()
        else:
            x, P = ekf_update_with_heading(
                x_pred, P_pred, z_meas, R, last_ekf_position
            )

        res_lst.append((x[0], x[1], x[2]))
        last_ekf_position = x[:2].copy()

    df[["EKF_x", "EKF_y", "EKF_heading_rad"]] = np.vstack(res_lst)
    return df


def calc_err_stats(df_list):
    """
    Compute EKF noise statistics by pooling errors across all runs.

    Returns a dict with:
      - R_x, R_y: VLP position variances
      - R_theta: VLP heading variance from VLP position deltas
      - Q_theta: encoder turn delta-theta variance
      - Q_theta_no_turn: encoder straight delta-theta variance
      - Q_dist: encoder distance-change variance
    """
    vlp_x_errs_all = []
    vlp_y_errs_all = []
    vlp_heading_errs_all = []
    enc_turn_errs_all = []
    enc_no_turn_errs_all = []
    enc_dist_errs_all = []

    for df in df_list:
        x_err = ep.filter_outliers(df["x_hist"] - df["vlp_x_hist"], threshold=2)
        y_err = ep.filter_outliers(df["y_hist"] - df["vlp_y_hist"], threshold=2)
        vlp_x_errs_all.append(x_err)
        vlp_y_errs_all.append(y_err)

        vlp_x = df["vlp_x_hist"].to_numpy()
        vlp_y = df["vlp_y_hist"].to_numpy()
        if len(vlp_x) > 1:
            theta_vlp = _vlp_heading_from_positions(vlp_x, vlp_y)
            heading_truth = df["heading_hist_rad"].to_numpy()[1:]
            abs_heading = normalize_angle_rad(heading_truth - theta_vlp)
            abs_heading = abs_heading[np.isfinite(abs_heading)]
            abs_heading = ep.filter_outliers(abs_heading, threshold=2)
            vlp_heading_errs_all.append(abs_heading)

        turn_mask = df["encoder_heading_change_rad"] != 0
        no_turn_mask = ~turn_mask

        turn_errs = ep.filter_outliers(
            df.loc[turn_mask, "encoder_heading_change_err_rad"], threshold=2
        )
        no_turn_errs = ep.filter_outliers(
            df.loc[no_turn_mask, "encoder_heading_change_err_rad"], threshold=2
        )
        enc_turn_errs_all.append(turn_errs)
        enc_no_turn_errs_all.append(no_turn_errs)

        dist_errs = ep.filter_outliers(df["encoder_location_change_err"], threshold=2)
        enc_dist_errs_all.append(dist_errs)

    vlp_x_all = np.concatenate(vlp_x_errs_all) if vlp_x_errs_all else np.array([])
    vlp_y_all = np.concatenate(vlp_y_errs_all) if vlp_y_errs_all else np.array([])
    vlp_heading_all = (
        np.concatenate(vlp_heading_errs_all) if vlp_heading_errs_all else np.array([])
    )
    turn_all = np.concatenate(enc_turn_errs_all) if enc_turn_errs_all else np.array([])
    no_turn_all = (
        np.concatenate(enc_no_turn_errs_all) if enc_no_turn_errs_all else np.array([])
    )
    dist_all = np.concatenate(enc_dist_errs_all) if enc_dist_errs_all else np.array([])

    R_x = np.var(vlp_x_all, ddof=1) if vlp_x_all.size > 1 else 0.0
    R_y = np.var(vlp_y_all, ddof=1) if vlp_y_all.size > 1 else 0.0
    R_theta = (
        np.var(vlp_heading_all, ddof=1) if vlp_heading_all.size > 1 else 0.0
    )
    Q_theta = np.var(turn_all, ddof=1) if turn_all.size > 1 else 0.0
    Q_theta_no_turn = np.var(no_turn_all, ddof=1) if no_turn_all.size > 1 else 0.0
    Q_dist = np.var(dist_all, ddof=1) if dist_all.size > 1 else 0.0

    return {
        "R_x": R_x,
        "R_y": R_y,
        "R_theta": R_theta,
        "Q_theta": Q_theta,
        "Q_theta_no_turn": Q_theta_no_turn,
        "Q_dist": Q_dist,
    }


class LiveEKF:
    """
    Live EKF wrapper that uses VLP-only heading updates.
    """

    def __init__(self, initial_state, err_stats, initial_vlp_position=None):
        self.x = np.array(initial_state)
        sigma_x0 = 0.05
        sigma_y0 = 0.05
        sigma_theta0 = np.deg2rad(5.0)
        self.P = np.diag([sigma_x0**2, sigma_y0**2, sigma_theta0**2])
        self.err_stats = err_stats
        if initial_vlp_position is None:
            self.last_vlp_position = self.x[:2].copy()
        else:
            self.last_vlp_position = np.array(initial_vlp_position)

    def predict(self, d, delta_theta, z_meas):
        """
        Process a new measurement and update the EKF state.

        Arguments:
            d: Measured distance (encoder_location_change)
            delta_theta: Measured heading change in radians
            z_meas: VLP measurement for position [vlp_x, vlp_y]

        Returns:
            Updated state vector [x, y, theta]
        """
        if abs(delta_theta) < (1 / 180 * np.pi):
            heading_variance = self.err_stats["Q_theta_no_turn"]
        else:
            heading_variance = self.err_stats["Q_theta"]
        step_variance = self.err_stats["Q_dist"]
        Q_step = np.diag([step_variance, heading_variance])

        R = np.diag(
            [self.err_stats["R_x"], self.err_stats["R_y"], self.err_stats["R_theta"]]
        )

        x_pred, P_pred = ekf_predict_with_heading(
            self.x, self.P, d, delta_theta, Q_step
        )
        x_upd, P_upd = ekf_update_with_heading(
            x_pred, P_pred, z_meas, R, self.last_vlp_position
        )

        self.x = x_upd
        self.P = P_upd
        self.last_vlp_position = np.array(z_meas)
        return self.x
