import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import matplotlib as mpl
import matplotlib.image as mpimg
from matplotlib.ticker import MaxNLocator
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
from moviepy import *
import shutil
import matplotlib.patches as mpatches




def generate_marker_from_svg(svg_path):
    image_path, attributes = svg2paths(svg_path)

    image_marker = parse_path(attributes[0]['d'])

    image_marker.vertices -= image_marker.vertices.mean(axis=0)

    image_marker = image_marker.transformed(mpl.transforms.Affine2D().rotate_deg(180))
    image_marker = image_marker.transformed(mpl.transforms.Affine2D().scale(-1,1))

    return image_marker

def plot_horns_rev_turbine_locations(model, z_data, z_name, filename, expectation=False):
    rotor_diameter = 126.0 # m
    # Define the vertices of the parallelogram as Params
    vertices=[(0.0, 30.88095238095238), (40.0, 30.88095238095238), (43.79365079365079, 0.0), (3.7936507936507935, 0.0)]
    # Computed from data from here (https://gitlab.windenergy.dtu.dk/fair-data/winddata-revamp/winddata-documentation/-/blob/e34f87a286b7307ee88c9c8e0965f3342255eca9/NDAs/HR1.md)
    # Extract the x and y coordinates from the vertices
    x_values = [x for x, y in vertices]
    y_values = [y for x, y in vertices]

    # Find the maximum x and y values
    max_x = max(x_values) * rotor_diameter
    max_y = max(y_values) * rotor_diameter


    fig, ax = plt.subplots(figsize=(10, 10))

    # Extract x and y coordinates from the Pyomo model
    x_coords = [model.x[i] for i in range(model.N_T)]
    y_coords = [model.y[i] for i in range(model.N_T)]

    # Scatter plot for turbine locations
    scatter = ax.scatter(x_coords, y_coords, s=1)#, c=z_data, cmap=plt.cm.jet)
    img = mpimg.imread('plots/turbine.png')
    # Place the image at each data point
    for (i, j) in zip(x_coords, y_coords):
        imagebox = OffsetImage(img, zoom=0.1)  # Adjust zoom as needed
        ab = AnnotationBbox(imagebox, (i, j), frameon=False, box_alignment=(0.5, 0))
        ax.add_artist(ab)

    # Customizing the plot
    ax.set_facecolor('lightblue')  # Ocean-like background
    #ax.set_title('Wind Turbine Locations', fontsize=16
    #, fontweight='bold')
    ax.set_xlabel('X Coordinate (m)', fontsize=16)
    ax.set_ylabel('Y Coordinate (m)', fontsize=16)
    ax.set_xlim([-100, max_x + 100])
    ax.set_ylim([-100, max_y + 100])
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    #fig.colorbar(scatter, ax=ax, label=z_name)

    # Add an arrow for wind direction
    # Position the arrow in the top right of the plot, adjust the dx and dy to represent the direction
    if not expectation:
        wind_angle = model.theta_wind  # Angle in radians where 0 is towards the positive x-axis
        dx = np.cos(wind_angle)  #  calculate dx
        dy = np.sin(wind_angle)  #  calculate dy
        ax.arrow(0.9, 0.9, 0.08*dx, 0.08*dy, transform=ax.transAxes, width=0.01, head_width=0.02,
                head_length=0.01, fc='white', ec='white')

        # Add label for the wind direction arrow
        ax.text(0.85, 0.85, 'Wind Direction', transform=ax.transAxes, color='white',
                fontsize=10, ha='left', va='bottom')
    SIZE_DEFAULT = 14
    SIZE_LARGE = 20
    plt.rc("font", family="Roboto")  # controls default font
    plt.rc("font", weight="normal")  # controls default font
    plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
    plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
    # Add a box for the perimeter
    # Create a Rectangle patch
    vertices=[(0.0, 30.88095238095238*rotor_diameter), (40.0*rotor_diameter, 30.88095238095238*rotor_diameter), (43.79365079365079*rotor_diameter, 0.0), (3.7936507936507935*rotor_diameter, 0.0)]
    # Add the patch to the Axes
    ax.add_patch(patches.Polygon(xy=list(vertices), fill=False, linewidth=2, linestyle="--", edgecolor='r', facecolor='none'))

    # Show the plot
    #plt.show()
    plt.savefig(filename, format="pdf", bbox_inches="tight")


def plot_turbine_locations(model, z_data, z_name, expectation=False):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Extract x and y coordinates from the Pyomo model
    x_coords = [model.x[i] for i in range(model.N_T)]
    y_coords = [model.y[i] for i in range(model.N_T)]

    # Scatter plot for turbine locations
    scatter = ax.scatter(x_coords, y_coords, s=1)#, c=z_data, cmap=plt.cm.jet)
    img = mpimg.imread('plots/turbine.png')
    # Place the image at each data point
    for (i, j) in zip(x_coords, y_coords):
        imagebox = OffsetImage(img, zoom=0.1)  # Adjust zoom as needed
        ab = AnnotationBbox(imagebox, (i, j), frameon=False)
        ax.add_artist(ab)

    # Customizing the plot
    ax.set_facecolor('lightblue')  # Ocean-like background
    #ax.set_title('Wind Turbine Locations', fontsize=16
    #, fontweight='bold')
    ax.set_xlabel('X Coordinate (m)', fontsize=16)
    ax.set_ylabel('Y Coordinate (m)', fontsize=16)
    ax.set_xlim([-100, model.X_dim + 100])
    ax.set_ylim([-100, model.Y_dim + 100])
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    #fig.colorbar(scatter, ax=ax, label=z_name)

    # Add an arrow for wind direction
    # Position the arrow in the top right of the plot, adjust the dx and dy to represent the direction
    if not expectation:
        wind_angle = model.theta_wind  # Angle in degrees where 0 is towards the positive x-axis
        dx = np.cos(wind_angle)  # Convert angle to radians and calculate dx
        dy = np.sin(wind_angle)  # Convert angle to radians and calculate dy
        ax.arrow(0.9, 0.9, 0.08*dx, 0.08*dy, transform=ax.transAxes, width=0.01, head_width=0.02,
                head_length=0.01, fc='white', ec='white')

        # Add label for the wind direction arrow
        ax.text(0.85, 0.85, 'Wind Direction', transform=ax.transAxes, color='white',
                fontsize=10, ha='left', va='bottom')
    SIZE_DEFAULT = 14
    SIZE_LARGE = 20
    plt.rc("font", family="Roboto")  # controls default font
    plt.rc("font", weight="normal")  # controls default font
    plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
    plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
    # Add a box for the perimeter
    # Create a Rectangle patch
    rect = patches.Rectangle((0, 0), model.X_dim.value, model.X_dim.value, linewidth=2, linestyle="--", edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    # Show the plot
    #plt.show()
    plt.savefig("plots/N_" + str(model.N_T.value) + ".pdf", format="pdf", bbox_inches="tight")

def plot_obj_vs_size(obj_dict, rand_objs=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    # Extract x and y coordinates from the Pyomo model
    x_coords = obj_dict.keys()
    y_coords = [val*1e-6 for val in list(obj_dict.values())]

    # Scatter plot for turbine locations
    scatter = ax.plot(x_coords, y_coords, linestyle='--', marker='o', color='b',lw=3,markersize=10)

    # Customizing the plot
    ax.set_xlabel('Number of Turbines', fontsize=16)
    ax.set_ylabel('Power Generated (MW)', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle='--', alpha=0.5)

    if rand_objs != None:
        x_coords = rand_objs.keys()
        y_coords = [val*1e-6 for val in list(rand_objs.values())]

        # Scatter plot for turbine locations
        ax.plot(x_coords, y_coords, linestyle='--', marker='o', color='r',lw=3,markersize=10)


    # Show the plot
    #plt.show()
    plt.savefig("objective_vs_turbines.pdf", format="pdf", bbox_inches="tight")








def analyze_wind_deficit(model, output_csv = "wake_matrix.csv", output_plot = "wind_deficit_heatmap.png"):
    """
    Analyze wind speed deficit for a given turbine layout.
    - Fix turbine positions
    - Solve the model
    - Extract and visualize wind speed deficit
    - Save wake interaction matrix as CSV
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Extract wind speed deficits
    num_turbines = model.N_T
    wind_deficits = [model.V_eff[i] for i in range(num_turbines)]
    # Heat map for wind speed deficits
    x_coords = [model.x[i] for i in model.I]
    y_coords = [model.y[i] for i in model.I]
    sct=ax.scatter(x_coords, y_coords, c=wind_deficits, cmap='coolwarm', s=200, edgecolors='k')
    fig.colorbar(sct, ax=ax, label="Effective wind speed (m/s)")
    ax.set_xlabel("X/D")
    ax.set_ylabel("Y/D")
    ax.set_title("Effective Wind Speed")

    wind_angle = model.theta_wind  # Wind direction in degrees (meteorological convention)
    dx = np.cos(wind_angle)  # Calculate dx
    dy = np.sin(wind_angle)  # Calculate dy

    ax.arrow(0.9, 0.9, 0.08 * dx, 0.08 * dy, transform=ax.transAxes, width=0.01,
            head_width=0.02, head_length=0.01, fc='black', ec='black')

    # Add label for the wind direction arrow
    ax.text(0.85, 0.85, 'Wind Direction', transform=ax.transAxes, color='black',
            fontsize=10, ha='left', va='bottom')

    # Annotate each turbine location with its index
    for i, (xi, yi) in enumerate(zip(x_coords, y_coords)):
        plt.annotate(f"{i+1}", (xi, yi), textcoords="offset points", xytext=(5, 5),
                    ha='center', fontsize=10, color='black', weight='bold')

    plt.savefig(output_plot)

    # Extract wake interaction matrix
    sigma_matrix = np.zeros((num_turbines, num_turbines))
    dij_matrix = np.zeros((num_turbines, num_turbines))
    effective_matrix = np.zeros((num_turbines, num_turbines))
    for i in range(num_turbines):
        for j in range(num_turbines):
            if i != j:
                sigma_matrix[i, j] = model.sigma[i,j] # calculates wake interaction matrix
                dij_matrix[i, j] = model.d[i,j]
                effective_matrix[i, j] = model.V_eff[i]

    df_sigma = pd.DataFrame(sigma_matrix, columns=[f"Turbine {j+1}" for j in range(num_turbines)],
                            index=[f"Turbine {i+1}" for i in range(num_turbines)])
    df_sigma.to_csv("sigma_matrix.csv")
    df_d = pd.DataFrame(dij_matrix, columns=[f"Turbine {j+1}" for j in range(num_turbines)],
                            index=[f"Turbine {i+1}" for i in range(num_turbines)])
    df_d.to_csv("dij_matrix.csv")
    df_d = pd.DataFrame(effective_matrix, columns=[f"Turbine {j+1}" for j in range(num_turbines)],
                            index=[f"Turbine {i+1}" for i in range(num_turbines)])
    df_d.to_csv("V_eff_matrix.csv")

    print(f"Wake matrix saved to {output_csv}")
    print(f"Wind deficit heatmap saved to {output_plot}")

    return wind_deficits

def plot_layout(optimized_model, filename, xmin, xmax, ymin, ymax):

    rotor_diameter = 126.0  # m
    xmin = xmin/rotor_diameter
    xmax = xmax/rotor_diameter
    ymin = ymin/rotor_diameter
    ymax = ymax/rotor_diameter

    farm_x = [xmin, xmax, xmax, xmin, xmin]
    farm_y = [ymin, ymin, ymax, ymax, ymin]

    # Create figure and axis objects with 3 rows and 1 column
    fig, ax = plt.subplots(figsize=(12, 12), sharex=True, sharey=True)

    # Extract x and y coordinates from the Pyomo model (nominal)
    x_coords = [optimized_model.x[i] for i in range(optimized_model.N_T)]
    y_coords = [optimized_model.y[i] for i in range(optimized_model.N_T)]
    print(z for z in zip(x_coords, y_coords))
    ax.scatter([x for x in x_coords], [y for y in y_coords], c="lightcoral", s=150, marker="s", label="Optimal layout")
    ax.plot(farm_x, farm_y, 'k--', linewidth=2, label='Farm Boundary')


    # Annotate each turbine location with its index
    for i, (xi, yi) in enumerate(zip(x_coords, y_coords)):
        plt.annotate(f"{i+1}", (xi, yi), textcoords="offset points", xytext=(5, 5),
                    ha='center', fontsize=10, color='black', weight='bold')

    # Customizing the plot
    ax.set_xlabel('X/D', fontsize=18)
    ax.set_ylabel('Y/D', fontsize=18)
    ax.set_xlim([-1, xmax + 1])
    ax.set_ylim([-1, ymax + 1])
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.grid(True, linestyle='--', alpha=0.5)

    fig.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), frameon=False, fontsize=12)

    # Adjust margins to make space for the legend
    fig.subplots_adjust(right=0.85)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot
    plt.savefig(filename, format="pdf", bbox_inches="tight")






def plot_layout_JK(x,y, filename, xmin, xmax, ymin, ymax):

    xnp = x.detach().numpy()
    ynp = y.detach().numpy()
    N_T = len(x)


    farm_x = [xmin, xmax, xmax, xmin, xmin]
    farm_y = [ymin, ymin, ymax, ymax, ymin]

    # Create figure and axis objects with 3 rows and 1 column
    fig, ax = plt.subplots(figsize=(12, 12), sharex=True, sharey=True)

    # Extract x and y coordinates from the Pyomo model (nominal)
    x_coords = [xnp[i] for i in range(N_T)]
    y_coords = [ynp[i] for i in range(N_T)]
    print(z for z in zip(x_coords, y_coords))
    ax.scatter([x for x in x_coords], [y for y in y_coords], c="lightcoral", s=150, marker="s", label="Optimal layout")
    ax.plot(farm_x, farm_y, 'k--', linewidth=2, label='Farm Boundary')


    # Annotate each turbine location with its index
    for i, (xi, yi) in enumerate(zip(x_coords, y_coords)):
        plt.annotate(f"{i+1}", (xi, yi), textcoords="offset points", xytext=(5, 5),
                    ha='center', fontsize=10, color='black', weight='bold')

    # Customizing the plot
    ax.set_xlabel('X/D', fontsize=18)
    ax.set_ylabel('Y/D', fontsize=18)
    ax.set_xlim([-1, xmax + 1])
    ax.set_ylim([-1, ymax + 1])
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.grid(True, linestyle='--', alpha=0.5)

    fig.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), frameon=False, fontsize=12)

    # Adjust margins to make space for the legend
    fig.subplots_adjust(right=0.85)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot
    plt.savefig(filename, format="pdf", bbox_inches="tight")








def paint_wind_deficit_JK(x,y, V_eff, U_wind, theta_wind,  xmin, xmax, ymin, ymax, Vmin,Vmax, output_plot = "wind_deficit_heatmap_JK.png"):
    """
    Analyze wind speed deficit for a given turbine layout.
    - Fix turbine positions
    - Solve the model
    - Extract and visualize wind speed deficit
    - Save wake interaction matrix as CSV
    """


    xnp = x.detach().numpy()
    ynp = y.detach().numpy()
    V_effnp = V_eff.detach().numpy()
    N_T = len(x)



    fig, ax = plt.subplots(figsize=(13.5, 10))

    # Extract wind speed deficits
    num_turbines = N_T
    wind_deficits = [V_effnp[i] for i in range(num_turbines)]
    # Heat map for wind speed deficits
    x_coords = [xnp[i] for i in range(num_turbines)]
    y_coords = [ynp[i] for i in range(num_turbines)]
    sct=ax.scatter(x_coords, y_coords, c=wind_deficits, cmap='coolwarm', s=200, edgecolors='k', vmin=Vmin, vmax=Vmax)
    farm_x = [xmin, xmax, xmax, xmin, xmin]
    farm_y = [ymin, ymin, ymax, ymax, ymin]
    ax.plot(farm_x, farm_y, 'k--', linewidth=2, label='Farm Boundary')
    fig.colorbar(sct, ax=ax, label="Effective wind speed (m/s)")
    ax.set_xlabel("X/D")
    ax.set_ylabel("Y/D")
    ax.set_xlim([-1, xmax + 1])
    ax.set_ylim([-1, ymax + 1])
    ax.set_title("Effective Wind Speed")

    wind_angle = theta_wind  # Wind direction in degrees (meteorological convention)
    dx = np.cos(wind_angle)  # Calculate dx
    dy = np.sin(wind_angle)  # Calculate dy

    ax.arrow(0.9, 0.9, 0.08 * dx, 0.08 * dy, transform=ax.transAxes, width=0.01,
            head_width=0.02, head_length=0.01, fc='black', ec='black')

    # Add label for the wind direction arrow
    ax.text(0.85, 0.85, 'Wind Direction', transform=ax.transAxes, color='black',
            fontsize=10, ha='left', va='bottom')

    # Annotate each turbine location with its index
    for i, (xi, yi) in enumerate(zip(x_coords, y_coords)):
        plt.annotate(f"{i+1}", (xi, yi), textcoords="offset points", xytext=(5, 5),
                    ha='center', fontsize=10, color='black', weight='bold')

    plt.savefig(output_plot)

    print(f"Wind deficit heatmap saved to {output_plot}")

    return wind_deficits









def paint_wind_deficit_with_control(x,y, yaw, V_eff, U_wind, theta_wind,  xmin, xmax, ymin, ymax, Vmin,Vmax, yawmin, yawmax, d0, output_plot = "wind_deficit_heatmap_JK.png"):
    """
    Analyze wind speed deficit for a given turbine layout.
    - Fix turbine positions
    - Solve the model
    - Extract and visualize wind speed deficit
    - Save wake interaction matrix as CSV
    """



    xnp = x.detach().numpy()
    ynp = y.detach().numpy()
    yawnp = yaw.detach().numpy()
    V_effnp = V_eff.detach().numpy()
    N_T = len(x)



    fig, ax = plt.subplots(figsize=(13.5, 10))

    # Extract wind speed deficits
    num_turbines = N_T
    wind_deficits = [V_effnp[i] for i in range(num_turbines)]
    # Heat map for wind speed deficits
    x_coords = [xnp[i] for i in range(num_turbines)]
    y_coords = [ynp[i] for i in range(num_turbines)]
    yaw_angles = [yawnp[i] for i in range(num_turbines)]
    sct=ax.scatter(x_coords, y_coords, c=wind_deficits, cmap='coolwarm', s=200, edgecolors='k', vmin=Vmin, vmax=Vmax)
    farm_x = [xmin, xmax, xmax, xmin, xmin]
    farm_y = [ymin, ymin, ymax, ymax, ymin]
    ax.plot(farm_x, farm_y, 'k--', linewidth=2, label='Farm Boundary')
    fig.colorbar(sct, ax=ax, label="Effective wind speed (m/s)")
    ax.set_xlabel("X/D")
    ax.set_ylabel("Y/D")
    ax.set_xlim([-1, xmax + 1])
    ax.set_ylim([-1, ymax + 1])
    ax.set_title("Effective Wind Speed")


    # Draw circles showing the near-wake distance around each turbine
    for i in range(len(x_coords)):
        center = (x_coords[i], y_coords[i])
        radius = 1.0
        circle = plt.Circle(center, radius, linestyle = "--", edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(circle)


    arrow_scale = 2.0
    # Draw arrows in the yaw direction of each turbine
    for i in range(len(yaw_angles)):
        x_loc = x_coords[i]      #0.1+0.9*(x_coords[i])/8.95
        y_loc = y_coords[i]      #0.1+0.9*(y_coords[i])/8.95
        angle = yaw_angles[i]
        dx = np.cos( angle )*arrow_scale*1.3
        dy = np.sin( angle )*arrow_scale*1.3
        #ax.arrow(x_loc, y_loc, 0.06 * dx, 0.06 * dy, transform=ax.transAxes, width=0.005,
        #        head_width=0.01, head_length=0.01, fc='grey', ec='grey')
        arrow = mpatches.FancyArrowPatch( (x_loc, y_loc), (x_loc+dx, y_loc+dy), mutation_scale=10)
        ax.add_patch(arrow)


        # Draw upper and lower bounds on yaw
        angle_lb = yawmin
        dx = np.cos( angle_lb )*arrow_scale
        dy = np.sin( angle_lb )*arrow_scale
        #ax.arrow(x_loc, y_loc, 0.04 * dx, 0.04 * dy, transform=ax.transAxes, width=0.001,
        #        head_width=0.01, head_length=0.01, fc='black', ec='black')
        arrow = mpatches.FancyArrowPatch( (x_loc, y_loc), (x_loc+dx, y_loc+dy), mutation_scale=10, color="black", edgecolor="black")
        ax.add_patch(arrow)

        angle_ub = yawmax
        dx = np.cos( angle_ub )*arrow_scale
        dy = np.sin( angle_ub )*arrow_scale
        #ax.arrow(x_loc, y_loc, 0.04 * dx, 0.04 * dy, transform=ax.transAxes, width=0.001,
        #        head_width=0.01, head_length=0.01, fc='black', ec='black')
        arrow = mpatches.FancyArrowPatch( (x_loc, y_loc), (x_loc+dx, y_loc+dy), mutation_scale=10, color="black", edgecolor="black")
        ax.add_patch(arrow)



    wind_angle = theta_wind  # Wind direction in degrees (meteorological convention)
    dx = np.cos( wind_angle )  # Calculate dx
    dy = np.sin( wind_angle )  # Calculate dy

    ax.arrow(0.9, 0.9, 0.08 * dx, 0.08 * dy, transform=ax.transAxes, width=0.01,
            head_width=0.02, head_length=0.01, fc='black', ec='black')

    # Add label for the wind direction arrow
    ax.text(0.85, 0.85, 'Wind Direction', transform=ax.transAxes, color='black',
            fontsize=10, ha='left', va='bottom')

    # Annotate each turbine location with its index
    for i, (xi, yi) in enumerate(zip(x_coords, y_coords)):
        plt.annotate(f"{i+1}", (xi, yi), textcoords="offset points", xytext=(5, 5),
                    ha='center', fontsize=10, color='black', weight='bold')


    plt.savefig(output_plot)

    print(f"Wind deficit heatmap saved to {output_plot}")

    return wind_deficits








def video_wind_deficit_with_control(x_list, y_list, yaw_list, V_eff_list, U_wind, theta_wind,  xmin, xmax, ymin, ymax, Vmin,Vmax, yaw_min, yaw_max, d0, mp4_filename = "video"):

    pngpath = os.path.join( "./", mp4_filename )
    if not os.path.exists(pngpath):
        os.makedirs(pngpath)
    else:
        shutil.rmtree(pngpath)
        os.makedirs(pngpath)

    for i in range(len(x_list)):
        if i%1 == 0:
            paint_wind_deficit_with_control(x_list[i], y_list[i], yaw_list[i], V_eff_list[i], U_wind, theta_wind, xmin, xmax, ymin, ymax, Vmin,Vmax, yaw_min, yaw_max, d0, output_plot = pngpath+"/iter{}.png".format(i))

    files = [join(pngpath, f) for f in listdir(pngpath) if isfile(join(pngpath, f))]
    files = sorted(files, key=lambda x: os.path.getmtime(x))
    print(files)

    clip = ImageSequenceClip(files, fps = 30)
    clip.write_videofile(mp4_filename+".mp4", fps = 30)
    shutil.rmtree(pngpath)













def paint_wind_deficit_with_control_plus_centerline(x,y, yaw, V_eff, U_wind, theta_wind,  xmin, xmax, ymin, ymax, Vmin,Vmax, yawmin, yawmax, d0, wake_x = None, wake_y = None, output_plot = "wind_deficit_heatmap_JK.png"):
    """
    Analyze wind speed deficit for a given turbine layout.
    - Fix turbine positions
    - Solve the model
    - Extract and visualize wind speed deficit
    - Save wake interaction matrix as CSV
    """


    # Yaw is measured from the negative d direction (Figure 1 of Chen)
    # The math computations are already consistent with this; we revise yaw here to make it consistent in plotting
    # TODO: this is only correct if d = x (theta_wind==0)
    #yaw    = np.radians(180.0) - yaw
    #yawmin = np.radians(180.0) - yawmin
    #yawmax = np.radians(180.0) - yawmax




    xnp = x.detach().numpy()
    ynp = y.detach().numpy()
    yawnp = yaw.detach().numpy()
    V_effnp = V_eff.detach().numpy()
    N_T = len(x)



    fig, ax = plt.subplots(figsize=(13.5, 10))

    # Extract wind speed deficits
    num_turbines = N_T
    wind_deficits = [V_effnp[i] for i in range(num_turbines)]
    # Heat map for wind speed deficits
    x_coords = [xnp[i] for i in range(num_turbines)]
    y_coords = [ynp[i] for i in range(num_turbines)]
    yaw_angles = [yawnp[i] for i in range(num_turbines)]


    sct=ax.scatter(x_coords, y_coords, c=wind_deficits, cmap='coolwarm', s=200, edgecolors='k', vmin=Vmin, vmax=Vmax)
    farm_x = [xmin, xmax, xmax, xmin, xmin]
    farm_y = [ymin, ymin, ymax, ymax, ymin]
    ax.plot(farm_x, farm_y, 'k--', linewidth=2, label='Farm Boundary')
    fig.colorbar(sct, ax=ax, label="Effective wind speed (m/s)")
    ax.set_xlabel("X/D")
    ax.set_ylabel("Y/D")
    ax.set_xlim([-1, xmax + 1])
    ax.set_ylim([-1, ymax + 1])
    ax.set_title("Effective Wind Speed")


    # Draw circles showing the near-wake distance around each turbine
    for i in range(len(x_coords)):
        center = (x_coords[i], y_coords[i])
        radius = d0
        circle = plt.Circle(center, radius, linestyle = "--", edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(circle)

    arrow_scale = 2.0
    # Draw arrows in the yaw direction of each turbine
    for i in range(len(yaw_angles)):
        x_loc = x_coords[i]      #0.1+0.9*(x_coords[i])/8.95
        y_loc = y_coords[i]      #0.1+0.9*(y_coords[i])/8.95
        angle = yaw_angles[i]
        dx = np.cos( angle )*arrow_scale*1.3
        dy = np.sin( angle )*arrow_scale*1.3
        #ax.arrow(x_loc, y_loc, 0.06 * dx, 0.06 * dy, transform=ax.transAxes, width=0.005,
        #        head_width=0.01, head_length=0.01, fc='grey', ec='grey')
        arrow = mpatches.FancyArrowPatch( (x_loc, y_loc), (x_loc+dx, y_loc+dy), mutation_scale=10)
        ax.add_patch(arrow)


        # Draw upper and lower bounds on yaw
        angle_lb = yawmin
        dx = np.cos( angle_lb )*arrow_scale
        dy = np.sin( angle_lb )*arrow_scale
        #ax.arrow(x_loc, y_loc, 0.04 * dx, 0.04 * dy, transform=ax.transAxes, width=0.001,
        #        head_width=0.01, head_length=0.01, fc='black', ec='black')
        arrow = mpatches.FancyArrowPatch( (x_loc, y_loc), (x_loc+dx, y_loc+dy), mutation_scale=10, color="black", edgecolor="black")
        ax.add_patch(arrow)

        angle_ub = yawmax
        dx = np.cos( angle_ub )*arrow_scale
        dy = np.sin( angle_ub )*arrow_scale
        #ax.arrow(x_loc, y_loc, 0.04 * dx, 0.04 * dy, transform=ax.transAxes, width=0.001,
        #        head_width=0.01, head_length=0.01, fc='black', ec='black')
        arrow = mpatches.FancyArrowPatch( (x_loc, y_loc), (x_loc+dx, y_loc+dy), mutation_scale=10, color="black", edgecolor="black")
        ax.add_patch(arrow)



    wind_angle = theta_wind  # Wind direction in degrees (meteorological convention)
    dx = np.cos( wind_angle )  # Calculate dx
    dy = np.sin( wind_angle )  # Calculate dy

    ax.arrow(0.9, 0.9, 0.08 * dx, 0.08 * dy, transform=ax.transAxes, width=0.01,
            head_width=0.02, head_length=0.01, fc='black', ec='black')

    # Add label for the wind direction arrow
    ax.text(0.85, 0.85, 'Wind Direction', transform=ax.transAxes, color='black',
            fontsize=10, ha='left', va='bottom')

    # Annotate each turbine location with its index
    for i, (xi, yi) in enumerate(zip(x_coords, y_coords)):
        plt.annotate(f"{i+1}", (xi, yi), textcoords="offset points", xytext=(5, 5),
                    ha='center', fontsize=10, color='black', weight='bold')



    # Plot the precomputed wake centerline trajectory given by deltaf
    # IMPORTANT! This starts by assuming the wind is at 0 deg. This makes x and d equivalent.
    if (wake_x != None) and (wake_y != None):
        x_coords = [wake_x[i].detach().item() for i in range(len(wake_x))     ]
        y_coords = [wake_y[i].detach().item() for i in range(len(wake_y))]
        ax.scatter(x_coords, y_coords)




    plt.savefig(output_plot)

    print(f"Wind deficit heatmap saved to {output_plot}")

    return wind_deficits
