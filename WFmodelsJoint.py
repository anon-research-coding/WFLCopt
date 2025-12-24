import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import plotting
import pickle
from torch import vmap

import Nets
import math


from scipy.stats import weibull_min, multivariate_normal
import logging
logging.basicConfig(level=logging.INFO)

import scipy
import copy
import time

from utils import OptimizeResult

class WFmodel_joint:
    def __init__(self, N_T, k, rho, d_min, xmax, ymax, yawmin, yawmax, xi, U_wind, obj_scale=1.0):
        self.N_T = N_T
        self.I = range(self.N_T)

        # Physical parameters
        self.d0 = 0.0
        self.d_min = d_min
        self.xmax = xmax         # length of feasible region in x dimension
        self.ymax = ymax         # length of feasible region in y dimension
        self.xi = xi             # wake spreading constant
        self.k = k               # wake interaction logistic parameter by Natalie
        self.rho = rho
        self.R0 = 0.5 #R0        # Turbine radius
        self.D = 1.0             # Unit diameter assumed everywhere, assumed already factored into X_dim and Y_dim

        self.A = (np.pi * (0.5 * self.D)**2)   # Disk area, see e.g. Eq (3) of Park

        # TODO: NEED VALUES. These are made up. Recheck paper?
        self.ad = 0.0
        self.bd = 0.1

        self.yaw_min = yawmin      # Chen 2022 "the yaw angles are allowed to change between 30 to 30 according to previous researches and realistic experiments"
        self.yaw_max = yawmax

        self.a_min = 0.3
        self.a_max = 0.7

        self.A_pgram = None
        self.b_l = None
        self.b_u = None

        # wind parameters
        self.U_wind = U_wind

        self.apply_patch_abs = True
        self.penalty_weight = 1e2
        self.obj_scale = obj_scale
        self.relu = torch.nn.ReLU()

        self.yaw0_list_cache = []   # Global variable to set/access hotstarts from within scipy / solvers
        self.inner_step_count = 0
        self.gradl_exp_cache = 0.0

        self.obj_list_cache = []
        self.viol_list_cache = []
        self.inner_time_list_cache = []       # Time to solve inner problem by BFGS, usually to look at hot start timing
        self.running_time_list_cache = []     # Total solving time at the current iteration of the outer method

        self.n_iter_total_list = []
        self.n_iter_max_list = []
        self.n_iter_min_list = []

    def patch_abs(self, x):
        if self.apply_patch_abs:
            return torch.abs(x)
        else:
            return x

    def compute_power(self, l, a,yaw, theta_wind):

        N = len(l)//2
        x = l[:N]
        y = l[N:]

        e = np.exp(1)

        # Our primary 1D variables are made vertical
        a = a.unsqueeze(1)
        yaw = yaw.unsqueeze(1)

        # New variables based on Chen/Song 2022 admm
        # Our d,r is their x,y (notation analogous to Park SCP)
        X = x.unsqueeze(0).repeat(len(x),1).T
        Y = y.unsqueeze(0).repeat(len(x),1).T

        Ct = 4 * a * (1 - a)
        Cp = 4 * a * (1 - a)**2

        dist = torch.sqrt(  (X - X.T)**2 + (Y - Y.T)**2 + 1e-6 )
        cos_theta = (X - X.T) / (dist + 1e-6)
        sin_theta = (Y - Y.T) / (dist + 1e-6)


        d = dist * (cos_theta * np.cos(theta_wind) + sin_theta * np.sin(theta_wind))
        r = dist * (sin_theta * np.cos(theta_wind) - cos_theta * np.sin(theta_wind))

        soft_switch = 1 / (1 + torch.exp(-self.k * d))

        yaw_wind = theta_wind - yaw + np.radians(180)

        sigma_y0 = self.patch_abs(   self.R0*torch.cos(yaw_wind)/np.sqrt(2.0)   )

        sigma_y  = sigma_y0 + self.xi*( self.patch_abs(d) - self.d0 ) + 1e-6

        phi = 0.3*yaw_wind /  torch.cos(yaw_wind) * (1.0 - torch.sqrt(1.0 - Ct*torch.cos(yaw_wind) + 1e-6))
        C0 = 1.0 - torch.sqrt(1.0 - Ct + 1e-6)
        E0 = C0**2 - 3.0*e**(1/12)*C0 + 3*e**(1/3)
        deltaf_factor1 = torch.sqrt(sigma_y0 / (self.xi*Ct) + 1e-6)
        factor2_lognum   = (1.6+torch.sqrt(Ct + 1e-6))*(1.6*torch.sqrt(sigma_y/sigma_y0 + 1e-6 )-torch.sqrt(Ct + 1e-6))   # TODO: needs to be checked in case when Cp not all constant - may not be correct after broadcasting
        factor2_logdenom = (1.6-torch.sqrt(Ct + 1e-6))*(1.6*torch.sqrt(sigma_y/sigma_y0 + 1e-6 )+torch.sqrt(Ct + 1e-6))  + 1e-6
        deltaf_factor2 = torch.log(  factor2_lognum / factor2_logdenom  + 1e-6  )
        deltaf = self.ad*self.D + self.bd*d + torch.tan(phi)*self.d0 + phi/5.2 * E0 * deltaf_factor1*deltaf_factor2

        exp_term = torch.exp(  -((r - deltaf)**2) / ((np.sqrt(2.0)*sigma_y)**2 + 1e-6 )    )
        wake_def    = (1.0 - torch.sqrt(1.0 - sigma_y0/sigma_y * Ct + 1e-6)) * exp_term   # TODO: needs to be checked for case when Cp not all constant - may not be correct after broadcasting
        wake_def_switch = (soft_switch * wake_def**2.0)

        # Applying this trick to wake_def_switch.T sums over column instead of row
        wake_def_sum = (wake_def_switch.T).flatten()[1:].view(self.N_T-1, self.N_T+1)[:,:-1].reshape(self.N_T, self.N_T-1).sum(1).unsqueeze(1)

        V_eff = self.U_wind * (1 - torch.sqrt( wake_def_sum + 1e-6 ))   # This 1e-6 prevents nan gradients in testing.
        P = 0.5 * self.rho * self.A * Cp * V_eff**3
        obj = self.obj_scale*P.sum()
        return obj, P, V_eff, deltaf, sigma_y, d, r


    def interdist(self, l):

        x = l[:self.N_T]
        y = l[ self.N_T:2*self.N_T]
        X = x.unsqueeze(0).repeat(len(x),1).T
        Y = y.unsqueeze(0).repeat(len(x),1).T
        distsq = (X - X.T)**2 + (Y - Y.T)**2
        return torch.cat(tuple([distsq[i,i+1:] for i in range(len(distsq))]))


    def interdist_viol(self, l):

        interdistances = self.interdist(l)
        viols = self.relu(self.d_min**2 - interdistances)
        return viols.sum()



    def compute_grad_l(self,l,obj):
        grad_l = torch.autograd.grad(obj, l, retain_graph=True)[0]
        return grad_l

    def compute_step_l(self,l,grad_l, alpha=0.01):
        l = l + alpha*grad_l
        return l

    def compute_grad_step_l(self,l,obj, alpha=0.01):
        grad_l = torch.autograd.grad(obj, l, retain_graph=True)[0]
        l = l + alpha*grad_l
        return l

    def compute_grad_yaw(self, yaw, obj):
        grad_yaw = torch.autograd.grad(obj, yaw, retain_graph=True)[0]
        return grad_yaw

    def compute_grad_step_yaw(self, yaw, obj, alpha=0.01):
        grad_yaw = torch.autograd.grad(obj, yaw, retain_graph=True)[0]
        yaw = yaw + alpha*grad_yaw
        return yaw

    def compute_projection_l(self, l):
        N = len(l)//2
        x = l[:N]
        y = l[N:]
        x = torch.clamp(x, 0, self.xmax)
        y = torch.clamp(y, 0, self.ymax)
        l = torch.cat((x,y))
        return l

    def compute_projection_yaw(self, yaw):
        yaw = torch.clamp(yaw, self.yaw_min, self.yaw_max)
        return yaw

    # Returns a list of values for the independent variables, per optimization step
    def yaw_pgd(self, l, a, yaw, theta_wind,  nsteps=100, alpha=0.01, precompute = False):

        if precompute:

            l_d = l.detach()
            a_d = a.detach()
            yaw_d = yaw.detach()

            yaw, _ = self.scipy_yaw_opt(l_d, a_d, yaw_d, theta_wind)
            yaw.requires_grad=True

        obj_list = []
        l_list = []        # These will stay constant
        yaw_list = []      # This changes

        obj, P, V_eff, deltaf, sigma_y, d, r = self.compute_power(l, a,yaw, theta_wind)
        for i in range(nsteps):
            obj_list.append(obj.item())
            l_list.append( l )
            yaw_list.append( yaw )
            yaw = self.compute_grad_step_yaw(yaw, obj, alpha=alpha)
            yaw = self.compute_projection_yaw(yaw)
            obj, P, V_eff, deltaf, sigma_y, d, r = self.compute_power(l, a, yaw, theta_wind)
        return obj, obj_list, l_list, yaw_list


    def compute_expected_power(self, l, a, yaw, theta_wind_list):
        n_wind = len(theta_wind_list)
        obj_exp = 0.0
        for theta_wind, prob_wind in theta_wind_list:
            obj, _, _, _, _, _, _ = self.compute_power(l, a, yaw, theta_wind)
            obj_exp += obj*prob_wind
        return obj_exp

    # Compute the expected power given a matrix of different yaws per scenario
    def compute_multi_expected_power(self, l, a, multi_yaw_mat, theta_wind_list):
        n_wind = len(theta_wind_list)
        obj_exp = 0.0
        for i in range(n_wind):
            theta_wind, prob_wind = theta_wind_list[i]
            obj, _, _, _, _, _, _ = self.compute_power(l, a, multi_yaw_mat[i], theta_wind)
            obj_exp += obj*prob_wind

        self.obj_list_cache.append( obj_exp.item() )
        self.viol_list_cache.append( self.interdist_viol(l).item()  )
        self.running_time_list_cache.append(time.time() - self.start_time)
        return obj_exp

    def danskins_optimal_expected_power(self, l, a, yaw0_list, theta_wind_list, reverse_sign=False):
        n_wind = len( theta_wind_list )

        obj_exp = 0.0
        gradl_exp = 0.0
        yaw_opt_list = []
        for i in range(len(theta_wind_list)):
            theta_wind, prob_wind = theta_wind_list[i]
            yaw_opt, _ = self.scipy_yaw_opt(l.detach(), a.detach(), yaw0_list[i].detach(), theta_wind)
            obj_opt, _, _, _, _, _, _  = self.compute_power(l, a, yaw_opt, theta_wind)
            grad_l = self.compute_grad_l(l, obj_opt)

            obj_exp += obj_opt*prob_wind
            gradl_exp += grad_l*prob_wind

            yaw_opt_list.append(yaw_opt)

        self.obj_list_cache.append(obj_exp.item())
        self.viol_list_cache.append( self.interdist_viol(l).item()  )
        self.running_time_list_cache.append(time.time() - self.start_time)

        sgn = -1.0 if reverse_sign else 1.0
        offset = 10e5 if reverse_sign else 0.0

        return sgn*obj_exp, sgn*gradl_exp, yaw_opt_list



    # Adapted from danskins_optimal_expected_power
    def danskins_optimal_expected_power_scipy(self, l, a, yaw0_list, theta_wind_list, add_penalty=False, reverse_sign=False):
        n_wind = len( theta_wind_list )

        obj_exp = 0.0
        gradl_exp = 0.0
        yaw_opt_list = []
        n_iter_list = []

        start = time.time()
        for i in range(n_wind):
            theta_wind, prob_wind = theta_wind_list[i]
            yaw_opt, n_iter = self.scipy_yaw_opt(l, a, yaw0_list[i], theta_wind)
            l.requires_grad = True
            obj_opt, _, _, _, _, _, _  = self.compute_power(l, a, yaw_opt, theta_wind)
            grad_l = self.compute_grad_l(l, obj_opt)
            l.requires_grad = False

            obj_exp += obj_opt*prob_wind
            gradl_exp += grad_l*prob_wind

            yaw_opt_list.append(yaw_opt)
            n_iter_list.append(n_iter)
        end = time.time()

        l.requires_grad = True
        viol = self.interdist_viol(l)


        self.yaw0_list_cache = yaw_opt_list
        self.gradl_exp_cache = gradl_exp
        self.inner_step_count +=1
        self.obj_list_cache.append(obj_exp.item())
        self.viol_list_cache.append(viol.item())
        self.running_time_list_cache.append(time.time() - self.start_time)
        self.inner_time_list_cache.append(end-start)
        self.n_iter_total_list.append( sum(n_iter_list) )
        self.n_iter_max_list.append( max(n_iter_list) )
        self.n_iter_min_list.append( min(n_iter_list) )
        print("f = {} at step {} taking {}s".format(obj_exp, self.inner_step_count, end-start))

        sgn = -1.0 if reverse_sign else 1.0
        offset = 10e5 if reverse_sign else 0.0

        if add_penalty:
            penalty = self.penalty_weight*viol
            penalty_grad = torch.autograd.grad(penalty, l, retain_graph=True)[0]
            obj_exp -= penalty
            gradl_exp -= penalty_grad
        l.requires_grad = False


        return sgn*obj_exp, sgn*gradl_exp, yaw_opt_list       #obj_opt_list, yaw_opt_list, obj_gradx_list, obj_grady_list



    def danskins_decomp_pgd(self, l, a, yaw, theta_wind_list, nsteps_outer=100,  alpha_outer=0.00005):

        obj_list = []
        l_list = []
        yaw0_list = [yaw.clone() for _ in range(len(theta_wind_list))]

        obj_exp, gradl_exp, yaw_opt_list = self.danskins_optimal_expected_power(l, a, yaw0_list, theta_wind_list)
        for i in range(nsteps_outer):

            obj_list.append(obj_exp.item())
            l_list.append( l )

            l = self.compute_step_l(l, gradl_exp, alpha=alpha_outer)
            l = self.compute_projection_l( l )

            start = time.time()                                                                          # TODO: hot start
            obj_exp, gradl_exp, yaw_opt_list = self.danskins_optimal_expected_power(l, a, yaw_opt_list, theta_wind_list)
            end = time.time()


            grad = gradl_exp
            gradnorm = torch.norm(grad,p=2).item()

            print("Total objective = {}, time = {}, alpha = {}, ||grad||={}, step = {}".format(obj_exp, end-start, alpha_outer, gradnorm, alpha_outer*gradnorm))

        return obj_list, l_list



    def acc_danskins_decomp_pgd(self, l, a, yaw, theta_wind_list,
                                nboundupdate=100,
                                reltol=1e-4,
                                abstol=0.0,
                                maxiters=1e3,
                                algo='fast',
                                disp=False):

        yaw0_list = [yaw.clone() for _ in range(len(theta_wind_list))]

        p = self.compute_projection_l( l )
        # counting variable for number of iterations
        k = 0
        # lower bound for the cost function
        low = -np.inf

        # setup for accelerated algorithm
        if algo == 'fast':
            y = p
            ###f, grad = fun(p)
            f, grad, yaw_opt_list = self.danskins_optimal_expected_power(p, a, yaw0_list, theta_wind_list, reverse_sign=True)
            f = f.item()
            # starting guess for gradient scaling parameter 1 / | nabla f |
            ###s = 1.0 / np.linalg.norm(gradl_total)
            s = 1.0 / torch.norm(grad)
            # refine by backtracking search
            while True:
                y_new = self.compute_projection_l( y - s * grad )
                f_new, grad_new, yaw_opt_list = self.danskins_optimal_expected_power(y_new, a, yaw_opt_list, theta_wind_list, reverse_sign=True)
                f_new = f_new.item()
                if f_new < f + torch.dot(y_new - y, grad) + \
                        0.5 * torch.norm(y_new - y)**2 / s:
                    break
                s *= 0.8
            # reduce s by some factor as optimal s might become smaller during
            # the course of optimization
            s /= 3.0
            print("Chose initial stepsize {}".format(s))
        else:
            f, grad, yaw_opt_list = self.danskins_optimal_expected_power(p, a, yaw0_list, theta_wind_list, reverse_sign=True)
            f = f.item()

        while k < maxiters:

            print("Main iter {}".format(k))

            k += 1

            # update lower bound on cost function
            # initialize at beginning (k=1) and then every nboundupdateth iteration
            if (k % nboundupdate == 0) or (k == 1):
                if algo =='fast':
                    f, grad, yaw_opt_list = self.danskins_optimal_expected_power(p, a, yaw_opt_list, theta_wind_list, reverse_sign=True)
                    f = f.item()

                i = torch.argmin(grad)
                low = max((low, (f - torch.sum(p * grad) + grad[i]).item() ))

                gap = f - low

                if disp:
                    print('%g: f %e, gap %e, relgap %e' % (k, f, gap, gap/low if low > 0 else np.inf))

            if algo == 'fast':

                print("Calculating optimal expected power")
                start = time.time()
                f, grad, yaw_opt_list = self.danskins_optimal_expected_power(p, a, yaw_opt_list, theta_wind_list, reverse_sign=True)
                f = f.item()
                print("f = {}".format(-f))
                end = time.time()
                print("{}s".format(end-start))
                p, pold = self.compute_projection_l( y - s * grad ), p
                y = p + k/(k+3.0) * (p - pold)
            else:
                # see e.g section 4.2 in http://www.web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
                s = 1.0 / torch.norm(grad)
                z = self.compute_projection_l( p - s * grad )
                fnew, gradnew, yaw_opt_list = self.danskins_optimal_expected_power(z, a, yaw_opt_list, theta_wind_list, reverse_sign=True)
                fnew = fnew.item()

                while fnew > f + torch.dot(z - p, grad) + \
                        0.5 * torch.norm(z - p)**2 / s:

                    print("Line search iteration")

                    s *= 0.5
                    z = self.compute_projection_l( p - s * grad )
                    fnew, gradnew, yaw_opt_list = self.danskins_optimal_expected_power(z, a, yaw_opt_list, theta_wind_list, reverse_sign=True)

                p = z
                f, grad = fnew, gradnew
                print("f = {}".format(-f))
        else:
            print('warning: maxiters reached before convergence')
        if disp:
            print('cost %e, low %e, gap %e' % (f, low, gap))

        return OptimizeResult(x=p, fun=f, nit=k, success=True), self.obj_list_cache



    def danskins_decomp_BGFS(self, l, a, yaw, theta_wind_list, tol=1e-8):

        l = l.detach()  # Otherwise ->  RuntimeError: Can't call numpy() on Tensor that requires grad
        a = a.detach()

        yaw0_list = [yaw.clone() for _ in range(len(theta_wind_list))]
        self.yaw0_list_cache = yaw0_list

        f = lambda l: -self.danskins_optimal_expected_power_scipy(torch.DoubleTensor(l), a, self.yaw0_list_cache, theta_wind_list)[0].detach()
        g = lambda l: -self.gradl_exp_cache

        l_opt = scipy.optimize.minimize(f, l,
                                        jac = g,
                                        method='L-BFGS-B',
                                        bounds=[(0,self.xmax) for _ in range(len(yaw))]+[(0,self.ymax) for _ in range(len(yaw))],
                                        tol=tol,
                                        options={'disp': True}
                                        ).x

        return l_opt, self.obj_list_cache, self.inner_time_list_cache, self.n_iter_total_list, self.n_iter_max_list, self.n_iter_min_list

    def yaw_gradient(self, l, a,yaw, theta_wind):
        yaw.requires_grad = True
        l =  l.double()
        a =  a.double()
        yaw = yaw.double()

        obj, P, V_eff, deltaf, sigma_y, d, r = self.compute_power(l, a, yaw, theta_wind)
        yawgrad = self.compute_grad_yaw(yaw, obj)
        return yawgrad


    def master_var_gradient(self, z, a, theta_wind_list):
        N_T = self.N_T
        n_wind = len(theta_wind_list)

        a =  a.double()
        z = z.detach().double()
        z.requires_grad = True


        obj = self.compute_multi_expected_power(z[:2*N_T], a, z[2*N_T:].view(n_wind, N_T), theta_wind_list)
        grad = torch.autograd.grad(obj, z, retain_graph=True)[0]

        return grad

    # Objective function of the ADMM subproblem: P - eps
    # z = [x;y;yaw;eps]
    def admm_var_subproblem_obj(self, z, a, theta_wind_pair):
        N_T = self.N_T

        theta_wind, theta_prob = theta_wind_pair

        a =  a.double()
        z = z.detach().double()
        z.requires_grad = True

        eps = z[-1]
        obj = -theta_prob*self.compute_power(z[:2*N_T], a, z[2*N_T:3*N_T], theta_wind)[0] + eps
        grad = torch.autograd.grad(obj, z, retain_graph=True)[0]

        return obj.item(), grad

    # compute the objective value and cache the gradient as a class attribute
    # This is intended for use with scipy where objective values are tested first,
    #      so we can pass a function which retrieves the gradient rather than recomputing the objective
    def yaw_obj_gradient_cache(self, l, a,yaw, theta_wind):
        yaw.requires_grad = True
        l =  l.double()
        a =  a.double()
        yaw = yaw.double()

        obj, P, V_eff, deltaf, sigma_y, d, r = self.compute_power(l, a, yaw, theta_wind)
        yawgrad = self.compute_grad_yaw(yaw, obj)
        self.yawgrad_cache = yawgrad

        return obj.item()


    # Takes all Tensor inputs
    def scipy_yaw_opt(self, l, a, yaw0, theta_wind):

        """
        Strategy 1: compute raw objective for f calls and recompute it during gradient calls
                    Makes sense when runtime is dominated by line search
        """
        #f = lambda yaw: -self.compute_power(l, a, torch.DoubleTensor(yaw), theta_wind)[0]
        #g = lambda yaw: -self.yaw_gradient( l, a, torch.DoubleTensor(yaw), theta_wind)

        """
        Strategy 2: compute gradient at each f call and cache it for fast gradient calls
                    Makes sense when gradient steps are more prevalent than linesearch steps
        """
        #f = lambda yaw: -self.yaw_obj_gradient_cache(l, a, torch.Tensor(yaw).double(), theta_wind)
        f = lambda yaw: -self.yaw_obj_gradient_cache(l, a, torch.DoubleTensor(yaw), theta_wind)
        g = lambda yaw: -self.yawgrad_cache


        scipy_result = scipy.optimize.minimize(f, yaw0.detach().numpy(),
                                            jac = g,
                                            method='L-BFGS-B',
                                            bounds=[(self.yaw_min,self.yaw_max) for _ in range(len(yaw0))],
                                            #callback=callbackF,
                                            #tol=1e-5,
                                            #options={'ftol': 1e-6}
                                            )

        yaw_scipy = scipy_result.x
        n_iter    = scipy_result.nit

        return torch.DoubleTensor(yaw_scipy), n_iter



    def master_scipy_opt_alt(self, l, a, yaw, theta_wind_list, method="SLSQP", tol=1e-8):
        print("Inside master_scipy_opt_alt")

        # Reset globals
        # This will be filled by each call to the objective compute_multi_expected_power
        self.obj_list_cache = []
        self.viol_list_cache = []
        self.running_time_list_cache = []
        self.start_time = time.time()

        N_T = len(l)//2
        n_wind = len(theta_wind_list)

        # Initialize yaws to the locally optimal ones
        yaw_opt_list = []
        yaw0=yaw
        for i in range(n_wind):
            theta_wind, prob_wind = theta_wind_list[i]
            yaw_opt, n_iter = self.scipy_yaw_opt(l, a, yaw0, theta_wind)
            yaw_opt_list.append(yaw_opt)
        multi_yaw_mat = torch.stack(yaw_opt_list)                         # yaw.unsqueeze(0).repeat(n_wind,1)
        print("Precomputed yaws")


        # z = [l, multi_yaw_mat] will be the master variable
        z = torch.cat( (l,multi_yaw_mat.flatten()) )
        #l = z[:2*N_T]

        f = lambda z: -self.compute_multi_expected_power(torch.DoubleTensor(z[:2*N_T]), a, torch.DoubleTensor(z[2*N_T:]).view(n_wind, N_T), theta_wind_list).item()
        g = lambda z: -self.master_var_gradient(torch.DoubleTensor(z), a, theta_wind_list)

        def nc(z):
            l = z[:2*N_T]
            return self.interdist(l)

        def nc_jac(z):
            z = torch.DoubleTensor(z)
            jac = torch.autograd.functional.jacobian(nc, z)
            return jac.numpy()

        lb = (self.d_min**2)*np.ones( (N_T**2 - N_T)//2 )
        ub = np.inf*np.ones( (N_T**2 - N_T)//2 )
        interdist_constraint = scipy.optimize.NonlinearConstraint(lambda z: nc(torch.DoubleTensor(z)).numpy(), lb, ub, jac=nc_jac)

        bounds = [(0,self.xmax) for _ in range(N_T)] + [(0,self.ymax) for _ in range(N_T)] + [(self.yaw_min,self.yaw_max) for _ in range(N_T)]*n_wind

        if (self.A_pgram != None):
            A_pgram_exp = np.zeros((N_T, N_T*(n_wind+2)))
            A_pgram_exp[:,:2*N_T] = self.A_pgram.toarray()
            A_pgram_exp = scipy.sparse.csr_array(A_pgram_exp)

        pgram_constraints = [scipy.optimize.LinearConstraint(A_pgram_exp, lb=self.b_l, ub=self.b_u,)] if (self.A_pgram != None) else []

        constraints = [interdist_constraint] + pgram_constraints

        scipy_result = scipy.optimize.minimize(f, z.detach().numpy(),
                                            jac = g,
                                            method=method,
                                            bounds=bounds,
                                            constraints = constraints,
                                            #callback=callbackF,
                                            tol=tol,
                                            options={'maxiter': 1000}
                                            )
        return self.obj_list_cache, self.viol_list_cache, self.running_time_list_cache



    def danskins_scipy_opt(self, l, a, yaw, theta_wind_list, method="SLSQP", tol=1e-8):

        #reset globals
        self.obj_list_cache = []
        self.viol_list_cache = []
        self.running_time_list_cache = []
        self.start_time = time.time()

        N_T = len(l)//2

        l = l.detach()  # Otherwise ->  RuntimeError: Can't call numpy() on Tensor that requires grad
        a = a.detach()

        yaw0_list = [yaw.clone() for _ in range(len(theta_wind_list))]
        self.yaw0_list_cache = yaw0_list

        f = lambda l: -self.danskins_optimal_expected_power_scipy(torch.DoubleTensor(l), a, self.yaw0_list_cache, theta_wind_list)[0].detach()
        #g = lambda l: -self.danskins_optimal_expected_power_scipy(torch.Tensor(l).double(), a, self.yaw0_list_cache, theta_wind_list)[1]
        g = lambda l: -self.gradl_exp_cache

        def nc(l):
            return self.interdist(l)

        def nc_jac(l):
            l = torch.DoubleTensor(l)
            jac = torch.autograd.functional.jacobian(nc, l)
            return jac.numpy()

        lb = (self.d_min**2)*np.ones((N_T**2 - N_T)//2)
        ub =          np.inf*np.ones((N_T**2 - N_T)//2)
        interdist_constraint = scipy.optimize.NonlinearConstraint(lambda l: nc(torch.DoubleTensor(l)).numpy(), lb, ub, jac=nc_jac)

        bounds = [(0,self.xmax) for _ in range(N_T)] + [(0,self.ymax) for _ in range(N_T)]

        pgram_constraints = [scipy.optimize.LinearConstraint(self.A_pgram, lb=self.b_l, ub=self.b_u)] if (self.A_pgram != None) else []


        if method=="L-BFGS-B":
            f = lambda l: -self.danskins_optimal_expected_power_scipy(torch.DoubleTensor(l), a, self.yaw0_list_cache, theta_wind_list, add_penalty=True)[0].detach()
            #g = lambda l: -self.danskins_optimal_expected_power_scipy(torch.Tensor(l).double(), a, self.yaw0_list_cache, theta_wind_list)[1]
            g = lambda l: -self.gradl_exp_cache
            constraints = []
        else:
            f = lambda l: -self.danskins_optimal_expected_power_scipy(torch.DoubleTensor(l), a, self.yaw0_list_cache, theta_wind_list)[0].detach()
            #g = lambda l: -self.danskins_optimal_expected_power_scipy(torch.Tensor(l).double(), a, self.yaw0_list_cache, theta_wind_list)[1]
            g = lambda l: -self.gradl_exp_cache
            constraints = [interdist_constraint] + pgram_constraints


        scipy_result = scipy.optimize.minimize(f, l.detach().numpy(),
                                            jac = g,
                                            method=method,
                                            bounds=bounds,
                                            constraints = constraints,
                                            #callback=callbackF,
                                            tol=tol,
                                            options={'maxiter': 1000}
                                            )

        return self.obj_list_cache, self.viol_list_cache, self.running_time_list_cache



    # The (method) field in this one will specify the lower-level solver
    def ADMM_scipy_opt(self, l, a, yaw, theta_wind_list, mu=10.0, method="SLSQP", tol=1e-5, max_iter=1000):

        print("inside ADMM_scipy_opt")

        #reset globals
        self.obj_list_cache = []
        self.viol_list_cache = []
        self.running_time_list_cache = []
        self.start_time = time.time()

        N_T = len(l)//2
        n_wind = len(theta_wind_list)

        # Initialize l_k and yaw_k for each scenario
        l_list      = [l.clone()          for _ in range(n_wind)]
        yaw_list    = [yaw.clone()        for _ in range(n_wind)]
        lmda_l_list = [torch.zeros(2*N_T) for _ in range(n_wind)]  # Matches the dimension of l
        eps_list    = [torch.zeros(1)     for _ in range(n_wind)]

        iter = 0
        while iter < max_iter:

            print("iter {}".format(iter))

            # Update linking variables
            l_link = (1/(2*mu))*(torch.stack(l_list) - torch.stack(lmda_l_list)).mean(0).detach()

            # Solve the independent subproblems
            l_new_list = []
            yaw_new_list = []
            lmda_l_new_list = []
            eps_new_list = []
            for k in range(len(theta_wind_list)):
                print("k = {}".format(k))
                theta_wind_pair       = theta_wind_list[k]
                theta_wind, prob_wind = theta_wind_pair

                l   = l_list[k]
                yaw = yaw_list[k]
                eps = eps_list[k]
                lmda_l = lmda_l_list[k]


                # Solve the subproblem for this scenario
                z = self.ADMM_subproblem(l, a, yaw, eps, theta_wind_pair, l_link, lmda_l, mu, method="SLSQP", tol=1e-5)
                z = torch.DoubleTensor(z)
                l   = z[ :2*N_T ]
                yaw = z[  2*N_T:3*N_T ]
                eps = z[-1].unsqueeze(0)


                # Update lagrange multipliers
                lmda_l = lmda_l + 2*mu*(l_link - l)


                l_new_list.append(l)
                yaw_new_list.append(yaw)
                lmda_l_new_list.append(lmda_l)
                eps_new_list.append(eps)


            # Update the subproblem variables
            l_list = l_new_list
            yaw_list = yaw_new_list
            lmda_l_list = lmda_l_new_list
            eps_list = eps_new_list

            iter +=1



            l_resids =  torch.norm(   torch.stack(l_list) - l_link, dim=1  )
            l_resid_sum = l_resids.sum()
            print("l_resid_sum = {}".format(l_resid_sum))




        l = l.detach()  # Otherwise ->  RuntimeError: Can't call numpy() on Tensor that requires grad
        a = a.detach()

        yaw_list = [yaw.clone() for _ in range(len(theta_wind_list))]
        self.yaw0_list_cache = yaw0_list


        return self.obj_list_cache, self.viol_list_cache, self.running_time_list_cache


    # The (method) field in this one will specify the lower-level solver
    # Parameters:
    #     mu is the augmented lagrangian penalty
    #     lambda_l are the lagrange multipliers
    #     theta_wind is the wind angle for this scenario
    #     l_link are parameters in this subproblem
    # l, yaw are initial values for its optimization, used for warmstarting
    # master variable z = [x;y;yaw;eps]
    # TODO: make sure a is double
    def ADMM_subproblem(self, l, a, yaw, eps, theta_wind_pair, l_link, lmda_l, mu, method="SLSQP", tol=1e-5):

        #reset globals
        self.obj_list_cache = []
        self.viol_list_cache = []
        self.running_time_list_cache = []
        self.start_time = time.time()

        N_T = len(l)//2
        theta_wind, theta_prob = theta_wind_pair

        l = l.detach()  # Otherwise ->  RuntimeError: Can't call numpy() on Tensor that requires grad
        a = a.detach()

        def nc_interdist(z):
            l = z[:2*N_T]
            return self.interdist(l)
        def nc_jac_interdist(z):
            z = torch.DoubleTensor(z)
            jac = torch.autograd.functional.jacobian(nc_interdist, z)
            return jac.numpy()
        lb_interdist = (self.d_min**2)*np.ones((N_T**2 - N_T)//2)
        ub_interdist =          np.inf*np.ones((N_T**2 - N_T)//2)
        interdist_constraint = scipy.optimize.NonlinearConstraint(lambda z: nc_interdist(torch.DoubleTensor(z)).numpy(), lb_interdist, ub_interdist, jac=nc_jac_interdist)

        # This quantity should be bounded above by 0 (constraint is 0>=RHS)
        def nc_link(z):
            l = z[:2*N_T]
            eps = z[-1]
            RHS = (  (l_link-l)*lmda_l + mu*(l_link-l)**2  ).sum() - eps
            return RHS

        def nc_jac_link(z):
            z = torch.DoubleTensor(z)
            jac = torch.autograd.functional.jacobian(nc_link, z)
            return jac.numpy()

        lb_link = -np.inf*np.ones(1)
        ub_link =        np.zeros(1)
        link_constraint = scipy.optimize.NonlinearConstraint(lambda z: nc_link(torch.DoubleTensor(z)).numpy(), lb_link, ub_link, jac=nc_jac_link)

        f = lambda z: self.admm_var_subproblem_obj(torch.DoubleTensor(z), a, theta_wind_pair)
        #g = lambda l: -self.gradl_exp_cache  # TODO
        constraints = [interdist_constraint, link_constraint]
        bounds = [(0,self.xmax) for _ in range(N_T)] + [(0,self.ymax) for _ in range(N_T)] + [(self.yaw_min, self.yaw_max) for _ in range(N_T)] + [(-np.inf, np.inf)]

        z = torch.cat((l,yaw,eps))
        scipy_result = scipy.optimize.minimize(f, z.detach().numpy(),
                                            jac = True,
                                            method=method,
                                            bounds=bounds,
                                            constraints = constraints,
                                            #callback=callbackF,
                                            tol=tol,
                                            options={'maxiter': 1000}
                                            )

        return scipy_result.x



if __name__ == "__main__":

    np.random.seed(0)
    torch.manual_seed(0)

    # Initialize turbine positions
    N_T = 20
    R0 = 63.0
    D  = 2*R0 #126.0
    xmax = 1000.0 / D
    ymax = 1000.0 / D
    k = 50.0
    rho = 1.225

    # Make up a wind direction and magnitude
    U_wind =  8.0
    theta_wind = torch.Tensor( [np.radians(0.0)] ) #np.radians(270.0)

    d_min = 3.0
    myrandom = random.Random(6548)

    xrand = xmax*torch.rand(N_T,2)    # position x is a 2-vector
    x0 = xrand[:,0]       #[myrandom.uniform(0, xmax) for _ in range(N_T)]
    y0 = xrand[:,1]       #[myrandom.uniform(0, ymax) for _ in range(N_T)]
    a0 = 0.3*torch.ones(N_T)    # initial axial induction factors
    yaw0  = torch.zeros(N_T)    # initial yaw positions
    x0 = x0.double()
    y0 = y0.double()
    a0 = a0.double()
    yaw0 = yaw0.double()
    xi = 0.033



    """
    Compute power from WFmodelJoint on the random inputs
    """
    wf_model = WFmodelJoint(N_T, k, rho, R0, d_min, xmax, ymax, xi, U_wind)

    this_obj, this_P, this_V_eff, this_deltaf = vmap(wf_model.compute_obj)(x0.unsqueeze(0).repeat(3,1), y0.unsqueeze(0).repeat(3,1), a0.unsqueeze(0).repeat(3,1), yaw0.unsqueeze(0).repeat(3,1), theta_wind.unsqueeze(0).repeat(3,1))
    print("this_obj")
    print( this_obj )
    print("this_P")
    print( this_P )


    """
    Plot velocity deficits and yaw
    """
    V_eff0  = this_V_eff[0]
    deltaf0 = this_deltaf[0]
    Vmin = -5.0   # For calibrating plot colors only
    Vmax = 8.0   #
    print("V_eff0")
    print( V_eff0 )
    print("len(V_eff0)")
    print( len(V_eff0) )
    plotting.paint_wind_deficit_JK(x0, y0, V_eff0, wf_model.U_wind, theta_wind, 0, xmax, 0, ymax, Vmin,Vmax, output_plot = "plt/winddef.png")
