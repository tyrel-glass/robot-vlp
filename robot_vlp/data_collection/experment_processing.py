import numpy as np
import pandas as pd

import matplotlib.pyplot as plt







# Calculate absolute 2D distance traveled
def calculate_distance_2d(df):
    """
    Calculates the 2D distance traveled between consecutive points.
    """
    x = df['vive_x'].diff().values
    y = df['vive_y'].diff().values
    y[0]= 0
    x[0] = 0

    x = x.astype(np.float64)
    y = y.astype(np.float64)

    df['distance_traveled'] = np.sqrt(x**2 + y**2)

    return df


def calculate_turn_angles(df):
    """
    Calculate the angle formed by vectors originating at the current row
    (for rows with 'TURN:-1250' in the 'last_cmd' column) and spanning to the
    x-y locations 2 rows back and 1 row forward.
    """
    # Ensure the DataFrame is sorted by position or time
    df = df.reset_index(drop=True)

    # Initialize a column for the calculated angles
    df['turn_angle'] = np.nan

    for i in range(2, len(df) - 1):
        if 'TURN'  in df.loc[i, 'last_cmd']:
            # Coordinates for the current row
            x_curr, y_curr = df.loc[i, ['x', 'y']]

            # Coordinates 2 rows back
            x_back_2, y_back_2 = df.loc[i - 2, ['x', 'y']]

           # Coordinates 2 rows back
            x_back_1, y_back_1 = df.loc[i - 1, ['x', 'y']]


            # Coordinates 1 row forward
            x_forward, y_forward = df.loc[i + 1, ['x', 'y']]

            # Vectors originating from the current row
            vector_back = np.array([x_back_1 - x_back_2, y_back_1 - y_back_2])
            vector_forward = np.array([x_forward - x_curr, y_forward - y_curr])

            # Normalize the vectors
            norm_back = np.linalg.norm(vector_back)
            norm_forward = np.linalg.norm(vector_forward)

            if norm_back > 0 and norm_forward > 0:
                vector_back = vector_back / norm_back
                vector_forward = vector_forward / norm_forward

                # Calculate the angle using the dot product
                dot_product = np.clip(np.dot(vector_back, vector_forward), -1.0, 1.0)
                angle = np.arccos(dot_product)  # Angle in radians

                # Convert to degrees and store in the DataFrame
                df.loc[i, 'turn_angle'] = np.degrees(angle)

    return df

# Process 'MOVE' and 'TURN' commands into separate columns
def parse_command(cmd):
    if cmd.startswith("MOVE:"):
        return int(float(cmd.split(":")[1])), 0
    elif cmd.startswith("TURN:"):
        return 0, int(cmd.split(":")[1])
    return 0, 0




def assess_move_errs(df):

    move_df = df[df['move_count']> 0 ]
    move_df['counts_per_cm'] = move_df['move_count']/(move_df['distance_traveled']*100)
    
    print('mean: ', move_df['counts_per_cm'][1:].mean())
    return move_df


def assess_turn_errs(df):
    turn_df = df[df['turn_count'].abs()> 0 ]
    turn_df['counts_per_degree'] = turn_df['turn_count'].abs()/turn_df['turn_angle'].abs()

    print('mean: ', turn_df[(turn_df['counts_per_degree'] < 18)]['counts_per_degree'].mean())
    return turn_df



# ------------------------------------------------ Functions to process VLP data -------------------------------------



def plot_surface(df):

    # Creating 4 subplots for each target peak column
    fig = plt.figure(figsize=(16, 12))

    target_peaks = ['peak_1000Hz', 'peak_3000Hz', 'peak_5000Hz', 'peak_7000Hz']

    # Example additional data (add to match the structure of your actual dataframe)


    # Iterate through each target peak column to create a subplot
    for i, peak in enumerate(target_peaks):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')  # 2x2 grid of subplots

        # Prepare the grid for surface plotting
        X, Y = np.meshgrid(df["cnc_x"].unique(), df["cnc_y"].unique())
        Z = df.pivot_table(index='cnc_y', columns='cnc_x', values=peak).values

        # Plot the surface
        ax.plot_surface(X, Y, Z, cmap='viridis')

        # Label the axes
        ax.set_title(f'{peak}')
        ax.set_xlabel('X Location')
        ax.set_ylabel('Y Location')
        ax.set_zlabel(peak)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()