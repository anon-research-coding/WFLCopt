import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import plotting
#from WFmodels import WFmodel_first, WFmodel_torch, WFmodel_scaled
import pickle


from WFmodelsJoint import WFmodel_joint
#from WF_joint_models import WFmodel_joint
import argparse
import make_wind_rose
import math
import time
import parallelogram
import scipy

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":

    # Case parameters
    # N_T = 64 case1, 80 case2
    # d_min = 3.0 case1, 4.0 case2
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_T',         type=int,   default=80)
    parser.add_argument('--problemtype', type=str,   default="decomp")
    parser.add_argument('--method',      type=str,   default="SLSQP")
    parser.add_argument('--xi',          type=float, default=0.033)
    parser.add_argument('--k',           type=float, default=5.0)
    parser.add_argument('--n_wind',      type=int,   default=36)
    parser.add_argument('--yaw_center',  type=float, default=0.0)
    parser.add_argument('--yaw_spread',  type=float, default=30.0)
    parser.add_argument('--obj_scale',   type=float, default=1.0)
    parser.add_argument('--tol',         type=float, default=1e-7)
    parser.add_argument('--d_min',       type=float, default=4.0)
    parser.add_argument('--seed',        type=int,   default=7)
    parser.add_argument('--case',        type=int,   default=2)
    parser.add_argument('--index',       type=int,   default=9999)

    args = parser.parse_args()

    filename ="Parallelogram_WFopt_out"  # "Box_Scaled_WFopt_out"
    for (k,v) in vars(args).items():
        filename += "__" + k + "--" + str(v)

    figname = "./png/" + filename + ".png"
    pname = "./p/" + filename + ".p"


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize turbine positions
    N_T = args.N_T
    R0 = 63.0
    D  = 2*R0 #126.0

    yawmin = np.radians(args.yaw_center-args.yaw_spread)     # Chen 2022 "the yaw angles are allowed to change between 30 to 30 according to previous researches and realistic experiments"
    yawmax = np.radians(args.yaw_center+args.yaw_spread)
    a0 = 0.3*torch.ones(N_T)    # initial axial induction factors
    yaw0  = (yawmax+yawmin)/2*torch.ones(N_T)    # initial yaw positions

    k = args.k
    rho = 1.225
    n_wind = args.n_wind
    U_wind =  8.0
    d_min = args.d_min #3.0
    xi = args.xi

    """
    Generate Wind conditions and probabilities
    """
    bins, centers, probs, density = make_wind_rose.uniform_wind_rose(n_wind, 12345)
    theta_wind_list = np.radians( centers ).tolist()
    theta_wind_prob_list = probs.tolist()
    #theta_wind_list = np.radians(  np.array(360.0*torch.rand(n_wind))  ).tolist()     #np.radians(  60.0*torch.rand(n_wind)  )
    #theta_wind_prob_list = [ 1/len(theta_wind_list) for _ in range(len(theta_wind_list)) ]
    theta_wind_pairs_zip = zip(theta_wind_list, theta_wind_prob_list)
    theta_wind_pairs_list = [(x,y) for (x,y) in theta_wind_pairs_zip]



    """
    Generate feasible sets and random initial feasible layouts
    """

    # Case 1 is a square
    if args.case == 1:
        # Generate initial coordinates in a square
        min_feasible_width = math.isqrt(N_T)*d_min*D
        xmax = 1.3*min_feasible_width / D
        ymax = 1.3*min_feasible_width / D
        nx = math.isqrt(N_T) - 1
        ny = math.isqrt(N_T) - 1
        xpts = torch.arange(0,xmax+xmax/nx,xmax/nx)
        ypts = torch.arange(0,ymax+ymax/ny,ymax/ny)
        x0,y0 = torch.meshgrid(xpts,ypts)
        x0 = x0.flatten()
        y0 = y0.flatten()
        l0 = torch.cat((x0,y0))

        # Overwrites the above: generate nonuniform initial coordinates in the same square
        x0_list = []
        y0_list = []
        x0_list.append( xmax*torch.rand(1).item() )
        y0_list.append( ymax*torch.rand(1).item() )
        success = 1
        trial = 0
        while success < N_T:
            trial += 1
            x0_list.append( xmax*torch.rand(1).item() )
            y0_list.append( ymax*torch.rand(1).item() )


            X = torch.Tensor(x0_list).unsqueeze(0).repeat(len(x0_list),1).T
            Y = torch.Tensor(y0_list).unsqueeze(0).repeat(len(y0_list),1).T
            distsq = (X - X.T)**2 + (Y - Y.T)**2
            distsq.fill_diagonal_(9999.0)
            mindistsq = torch.min(distsq)
            if mindistsq > d_min**2:
                success +=1
            else:
                del x0_list[-1]
                del y0_list[-1]
        l0 = torch.cat((torch.Tensor(x0_list),torch.Tensor(y0_list)))

    # Case 2 is the Horns Rev parallelogram
    elif args.case == 2:
        _, vertices = parallelogram.get_parallelogram_vertices()
        vertices = 1.20*vertices

        A_par, b_par = parallelogram.get_constraint(vertices)

        A_par = np.concatenate(  (np.identity(N_T)*A_par[0,0], np.identity(N_T)*A_par[0,1]), axis=1  )

        A_par = scipy.sparse.csr_array(A_par)
        b_par = np.concatenate( (b_par[0]*np.ones(N_T), b_par[1]*np.ones(N_T)) )

        b_u = -b_par[1]
        b_l =  b_par[0]

        xmax = vertices[:,0].max()
        ymax = vertices[:,1].max()

        # Generate nonuniform initial coordinates in the parallelogram
        x0_list = []
        y0_list = []
        feas = False
        while feas==False:
            x0 = xmax*torch.rand(1).item()
            y0 = ymax*torch.rand(1).item()
            l0 = np.array([x0,y0])
            feas = parallelogram.check_feas(l0, vertices)

        x0_list.append( x0 )
        y0_list.append( y0 )
        success = 1
        trial = 0
        while success < N_T:
            trial += 1

            feas = False
            while feas==False:
                x0 = xmax*torch.rand(1).item()
                y0 = ymax*torch.rand(1).item()
                l0 = np.array([x0,y0])
                feas = parallelogram.check_feas(l0, vertices)

            x0_list.append( x0 )
            y0_list.append( y0 )

            X = torch.Tensor(x0_list).unsqueeze(0).repeat(len(x0_list),1).T
            Y = torch.Tensor(y0_list).unsqueeze(0).repeat(len(y0_list),1).T
            distsq = (X - X.T)**2 + (Y - Y.T)**2
            distsq.fill_diagonal_(9999.0)
            mindistsq = torch.min(distsq)
            if mindistsq > d_min**2:
                success +=1
                print("Success = {}".format(success))
            else:
                del x0_list[-1]
                del y0_list[-1]
        l0 = torch.cat((torch.Tensor(x0_list),torch.Tensor(y0_list)))

    else:
        print("Invalid case number")
        quit()



    # Reset the seed after data gen
    np.random.seed(7)
    torch.manual_seed(7)

    wf_model = WFmodel_joint(N_T, k, rho, d_min, xmax, ymax, yawmin, yawmax, xi, U_wind, args.obj_scale)
    if args.case==2: wf_model.A_pgram = A_par; wf_model.b_l = b_l;  wf_model.b_u = b_u


    #solver_name_splits = args.solver.split("-")
    problemtype = args.problemtype  # solver_name_splits[0]
    method      = args.method       # solver_name_splits[1]

    print("problemtype")
    print( problemtype )
    print("method")
    print( method )

    l = l0.double()
    a = a0.double()
    yaw = yaw0.double()
    l.requires_grad=True
    a.requires_grad=True
    yaw.requires_grad=True





    # A matrix where each row is a yaw vector for another wind scenario
    start = time.time()
    if problemtype=="master":
        obj_list, viol_list, running_time_list = wf_model.master_scipy_opt_alt(l, a, yaw, theta_wind_pairs_list, method=method, tol=args.tol)     #method="SLSQP")
    elif problemtype=="decomp":
        obj_list, viol_list, running_time_list = wf_model.danskins_scipy_opt(l, a, yaw, theta_wind_pairs_list, method=method, tol=args.tol)
    elif problemtype=="ADMM":
        obj_list, viol_list, running_time_list = wf_model.ADMM_scipy_opt(l, a, yaw, theta_wind_pairs_list, mu=1.0,  method=method, tol=args.tol)
    else:
        input("Error: Invalid problem type")
        quit()
    end = time.time()



    # Reverse the internal objective scaling applied in the wf_model class for the optimization
    obj_list = [y/args.obj_scale for y in obj_list]





    plt.plot(range(len(obj_list)), obj_list)
    plt.ylabel(r"$\mathbb{E}_{\theta}\sum P_{\theta}$")
    plt.xlabel(r"Iteration (Total Runtime = {})".format(end-start))
    plt.legend()
    #plt.show()
    plt.savefig(figname)


    outdict = {}
    outdict["obj_list"] = obj_list
    outdict["viol_list"] = viol_list
    outdict["running_time_list"] = running_time_list
    with open(pname, 'wb') as handle:
        pickle.dump(outdict, handle, protocol=pickle.HIGHEST_PROTOCOL)



    # NOTEs on updating to Horns rev
    # Make sure xmax, ymax are update in this file so they can have valid values passed to the WF class
    # Then the xmax, yxmax should be kept as bounds in scipy functions, and added if not yet in.




    #V_eff_stack = torch.stack(V_eff_list)
    #Vmin = V_eff_stack.min()
    #Vmax = V_eff_stack.max()
    #plotting.video_wind_deficit_with_control(x_list, y_list, yaw_list, V_eff_list, wf_model.U_wind, wf_model.theta_wind, 0, xmax, 0, ymax, Vmin,Vmax, wf_model.yaw_min, wf_model.yaw_max,  wf_model.d0, mp4_filename = "video")
