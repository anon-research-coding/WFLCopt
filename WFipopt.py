import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyomo.environ import *
from pyomo.environ import (ConcreteModel, Var, Constraint)
from pyomo.environ import units as u
from pyomo.environ import SolverFactory, value
from scipy.stats import weibull_min, multivariate_normal
from pyomo.util.infeasible import log_close_to_bounds, log_infeasible_bounds, log_infeasible_constraints
import logging
#logging.basicConfig(level=logging.INFO)

import time
import torch

# Suppress verbose outputs
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

def multi_joint_position_control_model(N_T=7, k=50.0, rho=1.225, d_min=3,  xmax=1000.0/63.0, ymax=1000.0/63.0, yawmin=np.radians(180.0-30.0), yawmax=np.radians(180.0+30.0), xi=0.033, U_wind=8.0, theta_wind_list = [0.0], x0=[], y0=[], a0=[], yaw0=[], fix_pos = False, fix_ctrl = False):
    model = ConcreteModel()
    myrandom = random.Random(6548)
    e = np.exp(1)

    N_W = len(theta_wind_list)

    if (len(x0)==0) or (len(y0)==0):
        print("No initial x and y provided to solver - randomizing x0,y0")
        input("Waiting")
        # Generate random initial turbine positions in normalized form (x/D, y/D)
        x0 = [myrandom.uniform(0, xmax) for _ in range(N_T)]
        y0 = [myrandom.uniform(0, ymax) for _ in range(N_T)]

    if (len(a0)==0):
        print("No initial a provided to solver - setting each a0 = 0.3")
        a0 = 0.3

    if (len(yaw0)==0):
        print("No initial yaw provided to solver - setting each yaw midway between bounds.")
        yaw0 = (yawmin+yawmax)/2.0

    model.R0 = Param(initialize=0.5, domain=NonNegativeReals, units=u.m, doc="Turbine radius")     # TODO: this needs to be 0.5, but currently that causes ipopt failure
    model.D  = Param(initialize=2 * model.R0.value, units=u.m, doc="Rotor diameter (D)")
    model.A  = Param(initialize=np.pi * (0.5 * model.D.value)**2, units=u.m, doc="Rotor Area (A)")

    model.d0  = Param(initialize=0.0, units=u.m, doc="Near-wake distance")

    model.xmax = Param(initialize=xmax / model.D.value, domain=NonNegativeReals, doc="Dimensionless x-axis length")
    model.ymax = Param(initialize=ymax / model.D.value, domain=NonNegativeReals, doc="Dimensionless y-axis length")

    model.N_T = Param(initialize=N_T, domain=NonNegativeIntegers, doc="Number of turbines")
    model.I = Set(initialize=range(model.N_T.value), doc="Index set for turbines")

    model.N_W = Param(initialize=N_W, domain=NonNegativeIntegers, doc="Number of wind conditions")
    model.K = Set(initialize=range(model.N_W.value), doc="Index set for wind conditions")


    model.rho = Param(initialize=rho, domain=NonNegativeReals, units=u.kg/u.m**3, doc="Air density")
    model.d_min = Param(initialize=d_min, doc="Minimum spacing in x/D and y/D (3D)")
    model.xi = Param(initialize=xi, domain=NonNegativeReals, doc="Wake spreading constant")
    model.k = Param(initialize=k, doc="Logistic wake interaction decay constant")
    model.a = Param(model.I, initialize=a0, doc="Axial induction factor")
    model.U_wind = Param(initialize=U_wind, domain=NonNegativeReals, mutable=True, units=u.m/u.s, doc="Wind speed")

    # Wind direction in radians (meteorological convention)
    model.theta_wind = Param(model.K, initialize=theta_wind_list, mutable=True, doc="Wind direction (0°=west to east, 90°=south to north)")


    # JK new parameters
    model.ad = Param(initialize=0.0, doc="Wake deflection parameter ad")
    model.bd = Param(initialize=0.1, doc="Wake deflection parameter bd")
    #model.yaw_min = yawmin
    #model.yaw_max = yawmax
    #model.Ct = 4 * model.a * (1 - model.a)
    #model.Cp = 4 * model.a * (1 - model.a)**2



    # Define decision variables in normalized coordinates
    if fix_pos:
        model.x = Param(model.I, initialize=x0, doc="Normalized x-location (x/D)")
        model.y = Param(model.I, initialize=y0, doc="Normalized y-location (y/D)")
    else:
        model.x = Var(model.I, initialize=x0, bounds=(0, model.xmax.value), within=NonNegativeReals, doc="Normalized x-location (x/D)")
        model.y = Var(model.I, initialize=y0, bounds=(0, model.ymax.value), within=NonNegativeReals, doc="Normalized y-location (y/D)")

    if fix_ctrl:
        model.yaw = Param(model.I, model.K, initialize=yaw0, doc="Normalized y-location (y/D)")
    else:
        model.yaw = Var(model.I, model.K, initialize=yaw0, bounds=(yawmin, yawmax), within=NonNegativeReals, doc="Normalized y-location (y/D)")


    def Ct_rule(model, i):
        return 4 * model.a[i] * (1 - model.a[i])
    model.Ct = Expression(model.I, rule=Ct_rule)

    def Cp_rule(model, i):
        return 4 * model.a[i] * (1 - model.a[i])**2
    model.Cp = Expression(model.I, rule=Cp_rule)



    # Compute normalized distances
    def distance_rule(model, i, j):
        return (  (model.x[i] - model.x[j])**2 + (model.y[i] - model.y[j])**2  + 1e-6 )**0.5
    model.dist = Expression(model.I, model.I, rule=distance_rule)

    # Define cosine and sine components of theta_ij
    def cos_theta_rule(model, i, j):
        return (model.x[i] - model.x[j]) / (model.dist[i, j] + 1e-6)  # Avoid div-by-zero
    model.cos_theta = Expression(model.I, model.I, rule=cos_theta_rule)

    def sin_theta_rule(model, i, j):
        return (model.y[i] - model.y[j]) / (model.dist[i, j] + 1e-6)
    model.sin_theta = Expression(model.I, model.I, rule=sin_theta_rule)

    #JK trig identities:
    #https://duckduckgo.com/?t=h_&q=cos+cos+%2B+sin+sin&iax=images&ia=images&iai=http%3A%2F%2Faleph0.clarku.edu%2F~djoyce%2Ftrig%2Fsumformulas.jpg

    def dij_rule(model, i, j, k):
        return model.dist[i, j] * (model.cos_theta[i, j] * cos(model.theta_wind[k]) + model.sin_theta[i, j] * sin(model.theta_wind[k]))
    model.d = Expression(model.I, model.I, model.K, rule=dij_rule)

    def rij_rule(model, i, j, k):
        return model.dist[i, j] * (model.sin_theta[i, j] * cos(model.theta_wind[k]) - model.cos_theta[i, j] * sin(model.theta_wind[k]))
    model.r = Expression(model.I, model.I, model.K, rule=rij_rule)

    def soft_switch_rule(model, i, j, k):
        return 1 / (1 + exp(-model.k * model.d[i, j, k]))  # Already normalized by D
    model.soft_switch = Expression(model.I, model.I, model.K, rule=soft_switch_rule)


    def yaw_wind_rule(model, i, k):
        return model.theta_wind[k] - model.yaw[i,k] + np.radians(180)
    model.yaw_wind = Expression(model.I, model.K, rule=yaw_wind_rule)


    def sigma_y0_rule(model, i, k):
        return abs(   model.R0*cos(model.yaw_wind[i,k])/np.sqrt(2.0)   )      # TODO: fix the absolute values
    model.sigma_y0 = Expression(model.I, model.K, rule=sigma_y0_rule)

    def sigma_y_rule(model, i, j, k):
        return model.sigma_y0[i,k] + model.xi*( abs(model.d[i,j,k]) - model.d0 ) + 1e-6           # TODO: check the role of i, j here     #NOTE it actually looks like we should put sigma_y0[j] here. Compare with broadcasting, also compare our broadcasting with similar code from Natalie to see if the pytorch code convention is even correct.
    model.sigma_y = Expression(model.I, model.I, model.K, rule=sigma_y_rule)

    def phi_rule(model, i, k):
        return 0.3*model.yaw_wind[i,k] /  cos(model.yaw_wind[i,k]) * (1.0 - sqrt(1.0 - model.Ct[i]*cos(model.yaw_wind[i,k]) + 1e-6))    # TODO: get rid of torch fn
    model.phi = Expression(model.I, model.K, rule=phi_rule)


    def C0_rule(model, i):
        return 1.0 - sqrt(1.0 - model.Ct[i] + 1e-6)
    model.C0 = Expression(model.I, rule=C0_rule)



    def E0_rule(model, i):
        return model.C0[i]**2 - 3.0*e**(1/12)*model.C0[i] + 3*e**(1/3)    # change these to exp functions
    model.E0 = Expression(model.I, rule=E0_rule)


    def deltaf_factor1_rule(model, i, k):
        return sqrt( model.sigma_y0[i,k] / (model.xi*model.Ct[i]) + 1e-6 )
    model.deltaf_factor1 = Expression(model.I, model.K, rule=deltaf_factor1_rule)


    def factor2_lognum_rule(model, i, j, k):      # TODO: check the roles of i,j
        return ( 1.6+sqrt(model.Ct[i] + 1e-6) )*( 1.6*sqrt(model.sigma_y[i,j,k]/model.sigma_y0[i,k] + 1e-6 )-sqrt(model.Ct[i] + 1e-6) )   # TODO: needs to be checked for case when Cp not all constant - may not be correct after broadcasting
    model.factor2_lognum = Expression(model.I, model.I, model.K, rule=factor2_lognum_rule)

    def factor2_logdenom_rule(model, i, j, k):
        return (1.6-sqrt(model.Ct[i] + 1e-6))*(1.6*sqrt(model.sigma_y[i,j,k]/model.sigma_y0[i,k] + 1e-6 )+sqrt(model.Ct[i] + 1e-6))  + 1e-6     # TODO: check the roles of i and j, also above
    model.factor2_logdenom = Expression(model.I, model.I, model.K, rule=factor2_logdenom_rule)

    def deltaf_factor2_rule(model, i, j, k):
        return log(  model.factor2_lognum[i,j,k] / model.factor2_logdenom[i,j,k]  + 1e-6  )    # TODO: check the log, used to be torch.log
    model.deltaf_factor2 = Expression(model.I, model.I, model.K, rule=deltaf_factor2_rule)


    def deltaf_rule(model, i, j, k):
        return model.ad*model.D + model.bd*model.d[i,j,k] + tan(model.phi[i,k])*model.d0 + model.phi[i,k]/5.2 * model.E0[i] * model.deltaf_factor1[i,k]*model.deltaf_factor2[i,j,k]      # TODO must be rescaled by D. REMEMBER - d and d0 here should already be rescaled. It may come down to just scaling E0
    model.deltaf = Expression(model.I, model.I, model.K, rule=deltaf_rule)


    def exp_term_rule(model, i, j, k):
        return exp(  -((model.r[i,j,k] - model.deltaf[i,j,k])**2) / ((np.sqrt(2.0)*model.sigma_y[i,j,k])**2 + 1e-6 )    )
    model.exp_term = Expression(model.I, model.I, model.K, rule=exp_term_rule)


    def wake_def_rule(model, i, j, k):
        return (1.0 - sqrt(1.0 - model.sigma_y0[i,k]/model.sigma_y[i,j,k] * model.Ct[i] + 1e-6)) * model.exp_term[i,j,k]   # TODO: needs to be checked for case when Cp not all constant - may not be correct after broadcasting
    model.wake_def = Expression(model.I, model.I, model.K, rule=wake_def_rule)


    def wake_def_switch_rule(model, i, j, k):
        if i == j:
            return 0  # No self-wake
        return (model.soft_switch[i,j,k] * model.wake_def[i,j,k]**2.0)
    model.wake_def_switch = Expression(model.I, model.I, model.K, rule=wake_def_switch_rule)


    def wake_def_sum_rule(model, j, k):
        return sum(wake_def_switch_rule(model, i, j, k) for i in model.I if j != i)
    model.wake_def_sum = Expression(model.I, model.K, rule=wake_def_sum_rule)


    def V_eff_rule(model, i, k):
        return model.U_wind * (1 - sqrt( model.wake_def_sum[i,k] + 1e-6 ))
    model.V_eff = Expression(model.I, model.K, rule=V_eff_rule)


    def P_rule(model, i, k):
        return 0.5 * model.rho * model.A * model.Cp[i] * model.V_eff[i,k]**3
    model.P = Expression(model.I, model.K, rule=P_rule)


    def obj_scene_rule(model, k):
        return sum( model.P[i,k] for i in model.I )
    model.obj_scene = Expression(model.K, rule=obj_scene_rule)

    def obj_rule(model):
        return sum( model.obj_scene[k] for k in model.K )
    model.obj = Objective(rule=obj_rule, sense=maximize)


    return model




def analyze_wind_deficit(model, output_csv = "wake_matrix.csv", output_plot = "wind_deficit_heatmap.png"):

    ## Analyze wind speed deficit for a given turbine layout.
    ## - Fix turbine positions
    ## - Solve the model
    ## - Extract and visualize wind speed deficit
    ## - Save wake interaction matrix as CSV

    fig, ax = plt.subplots(figsize=(10, 10))

    # Extract wind speed deficits
    num_turbines = model.N_T.value
    wind_deficits = [value(model.V_eff[i]) for i in range(num_turbines)]
    # Heat map for wind speed deficits
    x_coords = [value(model.x[i]) for i in model.I]
    y_coords = [value(model.y[i]) for i in model.I]
    sct=ax.scatter(x_coords, y_coords, c=wind_deficits, cmap='coolwarm', s=200, edgecolors='k')
    fig.colorbar(sct, ax=ax, label="Effective wind speed (m/s)")
    ax.set_xlabel("X/D")
    ax.set_ylabel("Y/D")
    ax.set_title("Effective Wind Speed")

    wind_angle = model.theta_wind.value  # Wind direction in degrees (meteorological convention)
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
                sigma_matrix[i, j] = value(model.soft_switch[i,j]) # calculates wake interaction matrix
                dij_matrix[i, j] = value(model.d[i,j])
                effective_matrix[i, j] = value(model.V_eff[i])

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
    x_coords = [value(optimized_model.x[i]) for i in range(optimized_model.N_T.value)]
    y_coords = [value(optimized_model.y[i]) for i in range(optimized_model.N_T.value)]
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

def solve_model(model):
    solver = SolverFactory('ipopt')
    opts = {'halt_on_ampl_error': 'yes', "max_iter":15000}
    results = solver.solve(model,  tee=True, options=opts, symbolic_solver_labels=True)  #warmstart=True,
    #assert(results.solver.termination_condition == TerminationCondition.optimal or results.solver.termination_condition == TerminationCondition.feasible)
    return results
    # solver docs: https://pyomo.readthedocs.io/en/stable/howto/solver_recipes.html






if __name__ == "__main__":
    myrandom = random.Random(6548)

    R0 = 63.0
    D  = R0*2.0
    xmax = 1000.0/D    # All distance values are normalized outside the routine for building model
    ymax = 1000.0/D
    N_T = 40
    yawmin=np.radians(180.0-30.0)
    yawmax=np.radians(180.0+30.0)

    x0 = [myrandom.uniform(0, xmax) for _ in range(N_T)]
    y0 = [myrandom.uniform(0, ymax) for _ in range(N_T)]
    #yaw0 = [(yawmax+yawmin)/2.0 for _ in range(N_T)]

    n_wind = 36
    theta_wind_list = np.radians(  np.array(360.0*torch.rand(n_wind))  ).tolist()

    model = multi_joint_position_control_model(N_T = N_T, k=10.0, theta_wind_list=theta_wind_list, xmax=xmax, ymax=ymax, x0=x0, y0=y0,  fix_pos=False, fix_ctrl=False)

    obj0 = value(model.obj)

    start = time.time()
    results = solve_model(model)
    end   = time.time()

    obj_opt = value(model.obj)

    plot_layout(model, "layout.pdf", 0.0, xmax, 0.0, ymax)
