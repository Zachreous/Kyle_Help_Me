import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def calculate_and_plot_member_forces(members_data):
    """
    Calculate and plot the internal shear forces and bending moments for multiple members.
    Provides a summary table of max and min shears and moments.
    """
    summary_data = []  # List to collect summary data

    for index, member in enumerate(members_data, start=1):
        member_length = member['member_length']
        point_loads = member['point_loads']
        distributed_loads = member['distributed_loads']
        end_forces = member['end_forces']

        # Positions along member for plotting
        x_vals = np.linspace(0, member_length, num=int(member_length) * 10 + 1)
        V = np.zeros_like(x_vals)
        M = np.zeros_like(x_vals)

        # Apply end conditions
        V_start, M_start, V_end, M_end = end_forces
        V += V_start
        M += M_start

        # Process point loads
        for P, a in point_loads:
            idx = np.searchsorted(x_vals, a)  # Find the index where the load is applied
            V[idx:] -= P
            M[idx:] -= P * (x_vals[idx:] - a)

        # Process distributed loads
        for w, a, b in distributed_loads:
            idx_start = np.searchsorted(x_vals, a)  # Find the index for start of the load
            idx_end = np.searchsorted(x_vals, b)  # Find the index for end of the load
            x_load = x_vals[idx_start:idx_end + 1] - a
            V[idx_start:idx_end + 1] -= w * x_load
            M[idx_start:idx_end + 1] -= 0.5 * w * x_load ** 2

        # Apply end shear and moment
        V[-1] += V_end
        M[-1] += M_end

        # Find the max and min shear and moment
        max_shear = np.max(np.abs(V))
        max_shear_pos = x_vals[np.argmax(np.abs(V))]
        max_moment = np.max(np.abs(M))
        max_moment_pos = x_vals[np.argmax(np.abs(M))]

        # Data for the table
        cell_text = [
            [f"{max_shear:.2f}", f"{max_shear_pos:.2f}"],
            [f"{max_moment:.2f}", f"{max_moment_pos:.2f}"]
        ]

        # Plotting
        fig, ax = plt.subplots(2, 1, figsize=(14, 8))

        # Shear Force Diagram (SFD)
        ax[0].plot(x_vals, V, label=f'Member {index} Shear Force (V)')
        ax[0].fill_between(x_vals, V, alpha=0.3)
        ax[0].set_title(f'Member {index} Shear Force Diagram (SFD)')
        ax[0].set_ylabel('Shear Force (kips)')
        ax[0].grid(True)
        ax[0].legend()

        # Bending Moment Diagram (BMD)
        ax[1].plot(x_vals, M, label=f'Member {index} Bending Moment (M)')
        ax[1].fill_between(x_vals, M, alpha=0.3)
        ax[1].set_title(f'Member {index} Bending Moment Diagram (BMD)')
        ax[1].set_xlabel('Length along member (inches)')
        ax[1].set_ylabel('Bending Moment (kip-inches)')
        ax[1].grid(True)
        ax[1].legend()

        # Add a table below the plots
        table_ax = plt.table(cellText=cell_text,
                             colLabels=["Max Value", "Position (in)"],
                             rowLabels=["Shear (kip)      ", "Moment (kip-in)           "],
                             cellLoc='center',
                             loc='center',
                             bbox=[0.2, -0.6, 0.8, 0.3])  #[left, bottom, width, height]

        # Adjust font size of the table if necessary
        table_ax.auto_set_font_size(False)
        table_ax.set_fontsize(10)
        table_ax.scale(1.2, 1.2)  # You can adjust the scale to fit your needs

        # Adjust layout to make room for the table
        plt.subplots_adjust(left=0.1, bottom=0.2, top=0.85)
        plt.show()


    # Create the DataFrame from the collected data
    summary_table = pd.DataFrame(summary_data)
    print(summary_table)
    return summary_table


# Example usage
members_data = [
    {
        'member_length': 120,  # inches
        'point_loads': [(10, 30), (15, 80)],  # (magnitude in kips, position in inches)
        'distributed_loads': [(2, 0, 120)],  # (magnitude in kips per inch, start, end)
        'end_forces': (0, 0, 0, 0),  # (start shear, start moment, end shear, end moment)
    },
    # ... Add more member data as needed
]


# Generate summary table and plots for members
summary = calculate_and_plot_member_forces(members_data)
