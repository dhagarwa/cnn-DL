#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:24:09 2017

@author: dhwanit
"""

#LEARNS NN FOR MAT Inversion 
############################################################################################
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as spio
import scipy
import sys
import time
###########################################################################################
epochs = 200
mat_size = 88
batch_size = mat_size +2000
reg_parameter = 0.0
backtracking_lr = 1
training_instances = 2000 #Not used
test_instances = 400
category = 12
r = 2  #rank update for random input matrix
r_net=4
filename= 'diagpluslr256.txt'

#sys.stdout = open(filename, "w+")
print('size:', mat_size, ' epochs:', epochs, ' r_net:', r_net, ' batch size:', batch_size, ' reg_parameter:',reg_parameter )
###########################################################################################
from functools import reduce
from torch.optim import Optimizer
from math import isinf

class LBFGSB(Optimizer):
    """Implements L-BFGS algorithm.

.. warning::
    This optimizer doesn't support per-parameter options and parameter
    groups (there can be only one).

.. warning::
    Right now all parameters have to be on a single device. This will be
    improved in the future.

.. note::
    This is a very memory intensive optimizer (it requires additional
    ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
    try reducing the history size, or use a different algorithm.

Arguments:
    lr (float): learning rate (default: 1)
    max_iter (int): maximal number of iterations per optimization step
        (default: 20)
    max_eval (int): maximal number of function evaluations per optimization
        step (default: max_iter * 1.25).
    tolerance_grad (float): termination tolerance on first order optimality
        (default: 1e-5).
    tolerance_change (float): termination tolerance on function value/parameter
        changes (default: 1e-9).
    line_search_fn (str): line search methods, currently available
        ['backtracking', 'goldstein', 'weak_wolfe']
    bounds (list of tuples of tensor): bounds[i][0], bounds[i][1] are elementwise
        lowerbound and upperbound of param[i], respectively
    history_size (int): update history size (default: 100).
"""

    def __init__(self, params, lr=1, max_iter=20, max_eval=None,
                 tolerance_grad=1e-5, tolerance_change=1e-16, history_size=100,
                 line_search_fn=None, bounds=None):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(lr=lr, max_iter=max_iter, max_eval=max_eval,
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                        history_size=history_size, line_search_fn=line_search_fn, bounds=bounds)
        super(LBFGSB, self).__init__(params, defaults)
    
        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")
    
        self._params = self.param_groups[0]['params']
        self._bounds = [(None, None)] * len(self._params) if bounds is None else bounds
        self._numel_cache = None
    
    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache
    
    def _gather_flat_grad(self):
        return torch.cat(
            tuple(param.grad.data.view(-1) for param in self._params), 0)
    
    def _add_grad(self, step_size, update):
        offset = 0
        #print('Learning rate', step_size)
        for p in self._params:
            numel = p.numel()
            p.data.add_(step_size, update[offset:offset + numel].resize_(p.size()))
            offset += numel
        assert offset == self._numel()
    
    def step(self, closure):
        """Performs a single optimization step.
    
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1
    
        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']
    
        state = self.state['global_state']
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)
    
        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = orig_loss.data[0]
        current_evals = 1
        state['func_evals'] += 1
    
        flat_grad = self._gather_flat_grad()
        abs_grad_sum = flat_grad.abs().sum()
    
        if abs_grad_sum <= tolerance_grad:
            return loss
    
        # variables cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')
    
        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state['n_iter'] += 1
    
            ############################################################
            # compute gradient descent direction
            ############################################################
            if state['n_iter'] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                H_diag = 1
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                if ys > 1e-10:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
    
                    # store new direction/step
                    old_dirs.append(s)
                    old_stps.append(y)
    
                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)
    
                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)
    
                if 'ro' not in state:
                    state['ro'] = [None] * history_size
                    state['al'] = [None] * history_size
                ro = state['ro']
                al = state['al']
    
                for i in range(num_old):
                    ro[i] = 1. / old_stps[i].dot(old_dirs[i])
    
                # iteration in L-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_dirs[i].dot(q) * ro[i]
                    q.add_(-al[i], old_stps[i])
    
                # multiply by initial Hessian
                # r/d is the final direction
                d = r = torch.mul(q, H_diag)
                for i in range(num_old):
                    be_i = old_stps[i].dot(r) * ro[i]
                    r.add_(al[i] - be_i, old_dirs[i])
    
            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone()
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss
    
            ############################################################
            # compute step length
            ############################################################
            # directional derivative
            gtd = flat_grad.dot(d)  # g * d
    
            # check that progress can be made along that direction
            if gtd > -tolerance_change:
                break
    
            # reset initial guess for step size
            if state['n_iter'] == 1:
                t = min(1., 1. / abs_grad_sum) * lr
            else:
                t = lr
    
            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                # raise RuntimeError("line search function is not supported yet")
                if line_search_fn == 'weak_wolfe':
                    t = self._weak_wolfe(closure, d)
                elif line_search_fn == 'goldstein':
                    t = self._goldstein(closure, d)
                elif line_search_fn == 'backtracking':
                    t = self._backtracking(closure, d)
                    #print('Learning rate', t)
                self._add_grad(t, d)
            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
            if n_iter != max_iter:
                # re-evaluate function only if not in last iteration
                # the reason we do this: in a stochastic setting,
                # no use to re-evaluate that function here
                loss = closure().data[0]
                flat_grad = self._gather_flat_grad()
                abs_grad_sum = flat_grad.abs().sum()
                ls_func_evals = 1
    
            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals
    
            ############################################################
            # check conditions
            ############################################################
            
            if n_iter == max_iter:
                print('Learning rate:', t)
                break
    
            if current_evals >= max_eval:
                print('Learning rate:', t)
                break
    
            if abs_grad_sum <= tolerance_grad:
                print('Learning rate:', t)
                break
    
            if d.mul(t).abs_().sum() <= tolerance_change:
                print('Learning rate:', t)
                break
    
            if abs(loss - prev_loss) < tolerance_change:
                print('Learning rate:', t)
                break
    
        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss
    
        return orig_loss
    
    def _copy_param(self):
        original_param_data_list = []
        for p in self._params:
            param_data = p.data.new(p.size())
            param_data.copy_(p.data)
            original_param_data_list.append(param_data)
        return original_param_data_list
    
    def _set_param(self, param_data_list):
        for i in range(len(param_data_list)):
            self._params[i].data.copy_(param_data_list[i])
    
    def _set_param_incremental(self, alpha, d):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.data.copy_(p.data + alpha*d[offset:offset + numel].resize_(p.size()))
            offset += numel
        assert offset == self._numel()
    
    def _directional_derivative(self, d):
        deriv = 0.0
        offset = 0
        for p in self._params:
            numel = p.numel()
            deriv += torch.sum(p.grad.data * d[offset:offset + numel].resize_(p.size()))
            offset += numel
        assert offset == self._numel()
        return deriv
    
    def _max_alpha(self, d):
        offset = 0
        max_alpha = float('inf')
        for p, bnd in zip(self._params, self._bounds):
            numel = p.numel()
            l_bnd, u_bnd = bnd
            p_grad = d[offset:offset + numel].resize_(p.size())
            if l_bnd is not None:
                from_l_bnd = ((l_bnd-p.data)/p_grad)[p_grad<0]
                min_l_bnd = torch.min(from_l_bnd) if from_l_bnd.numel() > 0 else max_alpha
            if u_bnd is not None:
                from_u_bnd = ((u_bnd-p.data)/p_grad)[p_grad>0]
                min_u_bnd = torch.min(from_u_bnd) if from_u_bnd.numel() > 0 else max_alpha
            max_alpha = min(max_alpha, min_l_bnd, min_u_bnd)
        return max_alpha
    
    
    def _backtracking(self, closure, d):
        # 0 < rho < 0.5 and 0 < w < 1
        rho = 1e-4
        w = 0.5
    
        original_param_data_list = self._copy_param()
        phi_0 = closure().data[0]
        phi_0_prime = self._directional_derivative(d)
        alpha_k = backtracking_lr
        while True:
            self._set_param_incremental(alpha_k, d)
            phi_k = closure().data[0]
            self._set_param(original_param_data_list)
            if phi_k <= phi_0 + rho * alpha_k * phi_0_prime:
                break
            else:
                alpha_k *= w
                
        #print('Learning rate: ', alpha_k)
        return alpha_k
    
    
    def _goldstein(self, closure, d):
        # 0 < rho < 0.5 and t > 1
        rho = 1e-4
        t = 2.0
    
        original_param_data_list = self._copy_param()
        phi_0 = closure().data[0]
        phi_0_prime = self._directional_derivative(d)
        a_k = 0.0
        b_k = self._max_alpha(d)
        alpha_k = min(1e4, (a_k + b_k) / 2.0)
        while True:
            self._set_param_incremental(alpha_k, d)
            phi_k = closure().data[0]
            self._set_param(original_param_data_list)
            if phi_k <= phi_0 + rho*alpha_k*phi_0_prime:
                if phi_k >= phi_0 + (1-rho)*alpha_k*phi_0_prime:
                    break
                else:
                    a_k = alpha_k
                    alpha_k = t*alpha_k if isinf(b_k) else (a_k + b_k) / 2.0
            else:
                b_k = alpha_k
                alpha_k = (a_k + b_k)/2.0
            if torch.sum(torch.abs(alpha_k * d)) < self.param_groups[0]['tolerance_grad']:
                break
            if abs(b_k-a_k) < 1e-6:
                break
        return alpha_k
    
    
    def _weak_wolfe(self, closure, d):
        # 0 < rho < 0.5 and rho < sigma < 1
        rho = 1e-4
        sigma = 0.9
    
        original_param_data_list = self._copy_param()
        phi_0 = closure().data[0]
        phi_0_prime = self._directional_derivative(d)
        a_k = 0.0
        b_k = self._max_alpha(d)
        alpha_k = min(1e4, (a_k + b_k) / 2.0)
        while True:
            self._set_param_incremental(alpha_k, d)
            phi_k = closure().data[0]
            phi_k_prime = self._directional_derivative(d)
            self._set_param(original_param_data_list)
            if phi_k <= phi_0 + rho*alpha_k*phi_0_prime:
                if phi_k_prime >= sigma*phi_0_prime:
                    break
                else:
                    alpha_hat = alpha_k + (alpha_k - a_k) * phi_k_prime / (phi_0_prime - phi_k_prime)
                    a_k = alpha_k
                    phi_0 = phi_k
                    phi_0_prime = phi_k_prime
                    alpha_k = alpha_hat
            else:
                alpha_hat = a_k + 0.5*(alpha_k-a_k)/(1+(phi_0-phi_k)/((alpha_k-a_k)*phi_0_prime))
                b_k = alpha_k
                alpha_k = alpha_hat
            if torch.sum(torch.abs(alpha_k * d)) < self.param_groups[0]['tolerance_grad']:
                break
            if abs(b_k-a_k) < 1e-6:
                break
        return alpha_k
###########################################################################################



###########################################################################################
class Net(nn.Module):

    def __init__(self, category, N, r):
        super(Net, self).__init__()
        self.ds = None
        if category==1:
            self.fc1 = nn.Linear(mat_size, mat_size, bias=False)
            #self.fc2 = nn.Linear(50, 2)
            #self.conv1 = nn.Conv1d(1, 1, 4096, stride=4096) #input channels, output channels, filter size, stride
            self.lr = 0.5
            
            # create error function
            self.error_function = torch.nn.MSELoss(size_average=False)
            # create optimiser, using simple stochastic gradient descent
            #self.optimiser = torch.optim.SGD(self.parameters(), self.lr, weight_decay=reg_parameter)
            self.optimiser = LBFGSB(self.parameters(), self.lr, line_search_fn='backtracking')
            self.category=1
            
        elif category==2:
            self.conv1 = nn.Conv1d(1, 16, 4, stride=1) #input 128x1, output 125x16#input channels, output channels, filter size, stride
            self.pool1 = nn.MaxPool1d(3, stride=2)#output 62x16
            self.conv2 = nn.Conv1d(16, 16, 4, stride=2) #output 30x16
            self.pool2 = nn.MaxPool1d(2, stride=2) #output 15x16
            self.conv3 = nn.Conv1d(16, 16, 8, stride=1) #output 8x16
            self.lr = 0.01
            self.category = 2
            self.error_function = torch.nn.MSELoss(size_average=False)
            self.optimiser = LBFGSB(self.parameters(),  line_search_fn='backtracking')
            #self.optimiser = torch.optim.SGD(self.parameters(), self.lr)
            
        elif category==3:
            self.conv1 = nn.Conv1d(1, 16, 4, stride=1) #input 128x1, output 125x16#input channels, output channels, filter size, stride
            self.pool1 = nn.MaxPool1d(3, stride=2)#output 62x16
            self.conv2 = nn.Conv1d(16, 16, 4, stride=2) #output 30x16
            self.conv3 = nn.Conv1d(16, 1, 20, stride=1)#output 11x1
            self.fc1 = nn.Linear(11, 11, bias=False) #output 11x1
            self.conv4 = nn.Conv1d(1, 16, 4, stride=1) #output 8x16
            self.pool2 = nn.MaxPool1d(2, stride=2) #output 15x16
            self.conv5 = nn.Conv1d(16, 16, 8, stride=1) #output 8x16
            self.lr = 0.01
            self.category = 3
            self.error_function = torch.nn.MSELoss(size_average=False)
            self.optimiser = LBFGSB(self.parameters(),  line_search_fn='backtracking')
            #self.optimiser = torch.optim.SGD(self.parameters(), self.lr)

        elif category==4:
            self.conv1 = nn.Conv1d(1, 128, 1, stride=1)
            self.conv2 = nn.Conv1d(128, 64, 1, stride=1)
            self.conv3 = nn.Conv1d(64, 32, 1, stride=1)
            self.conv4 = nn.Conv1d(32, 16, 1, stride=1)
            self.conv5 = nn.Conv1d(16, 8, 1, stride=1)
            self.conv6 = nn.Conv1d(8, 4, 1, stride=1)
            self.conv7 = nn.Conv1d(4, 2, 1, stride=1)
            self.conv8 = nn.Conv1d(2, 1, 1, stride=1)
            self.lr = 0.1
            self.category = 4
            self.error_function = torch.nn.MSELoss(size_average=False)
            self.optimiser = LBFGSB(self.parameters(),  line_search_fn='backtracking')
        
        elif category==5:
            self.conv1 = nn.Conv1d(1, 128, 1, stride=1, bias=False)
            self.conv2 = nn.Conv1d(128, 1, 1, stride=1, bias=False)
            self.lr = 0.1
            self.category = 5
            self.error_function = torch.nn.MSELoss(size_average=False)
            self.optimiser = LBFGSB(self.parameters(),  line_search_fn='backtracking')
            
        elif category==6:
            self.conv1 = nn.Conv1d(1, 4, 4, stride=1) # output 125x4
            #Do channel mixing, output-125x16
            self.pool1 = nn.MaxPool1d(3, stride=2)#output 62x16
            self.conv2 = nn.Conv1d(16, 16, 4, stride=2) #output 30x16
            self.conv3 = nn.Conv1d(16, 1, 20, stride=1)#output 11x1
            self.fc1 = nn.Linear(11, 11, bias=False) #output 11x1
            self.conv4 = nn.Conv1d(1, 16, 4, stride=1) #output 8x16
            self.pool2 = nn.MaxPool1d(2, stride=2) #output 15x16
            self.conv5 = nn.Conv1d(16, 16, 8, stride=1) #output 8x16
            self.conv6 = nn.Conv1d(16, 16, 8, stride=1) #output 8x16
            self.conv7 = nn.Conv1d(16, 16, 8, stride=1) #output 8x16
            self.lr = 0.1
            self.category = 6
            self.error_function = torch.nn.MSELoss(size_average=False)
            self.optimiser = LBFGSB(self.parameters(),  line_search_fn='backtracking')   
            
            
        elif category==7:
            self.r1_l1_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r1_l1_pool1 = nn.MaxPool1d(2, stride=2)
            self.r1_l2_fc1 = nn.Linear(128, 10, bias=False)
            self.r1_l2_fc2 = nn.Linear(10, 10, bias=False)
            self.r1_l2_fc3 = nn.Linear(10, 128, bias=False)
            self.r1_l3_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r2_l1_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r2_l1_pool1 = nn.MaxPool1d(2, stride=2)
            self.r2_l2_fc1 = nn.Linear(64, 10, bias=False)
            self.r2_l2_fc2 = nn.Linear(10, 10, bias=False)
            self.r2_l2_fc3 = nn.Linear(10, 64, bias=False)
            self.r2_l2_exp = nn.Conv1d(1, 2, 1, stride=1, bias=False)
            self.r2_l3_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r2_l3_exp = nn.Conv1d(1, 2, 1, stride=1, bias=False )
            self.r3_l1_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r3_l1_pool1 = nn.MaxPool1d(2, stride=2)
            self.r3_l1_exp = nn.Conv1d(1, 8, 1, stride=1, bias=False)
            self.r3_l2_fc1 = nn.Linear(32, 10, bias=False)
            self.r3_l2_fc2 = nn.Linear(10, 10, bias=False)
            self.r3_l2_fc3 = nn.Linear(10, 32, bias=False)
            self.r3_l2_exp = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r3_l3_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r3_l3_exp = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.lr = 0.1
            self.category = 7
            self.error_function = torch.nn.MSELoss(size_average=False)
            self.optimiser = LBFGSB(self.parameters(),  line_search_fn='backtracking')   
            
        #has diagonal scaling     
        elif category==8:
            
            self.ds=True #diagonal scaling true or false
            self.r1_l1_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r1_l1_pool1 = nn.MaxPool1d(2, stride=2)
            self.r1_l2_fc1 = nn.Linear(128, 10, bias=False)
            self.r1_l2_fc2 = nn.Linear(10, 10, bias=False)
            self.r1_l2_fc3 = nn.Linear(10, 128, bias=False)
            self.r1_l3_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r1_l3_ds = nn.Parameter(torch.randn(128), requires_grad=True)
            self.r2_l1_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r2_l1_pool1 = nn.MaxPool1d(2, stride=2)
            self.r2_l2_fc1 = nn.Linear(64, 10, bias=False)
            self.r2_l2_fc2 = nn.Linear(10, 10, bias=False)
            self.r2_l2_fc3 = nn.Linear(10, 64, bias=False)
            self.r2_l2_exp = nn.Conv1d(1, 2, 1, stride=1, bias=False)
            self.r2_l3_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r2_l3_exp = nn.Conv1d(1, 2, 1, stride=1, bias=False )
            self.r3_l1_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r3_l1_pool1 = nn.MaxPool1d(2, stride=2)
            self.r3_l1_exp = nn.Conv1d(1, 8, 1, stride=1, bias=False)
            self.r3_l2_fc1 = nn.Linear(32, 10, bias=False)
            self.r3_l2_fc2 = nn.Linear(10, 10, bias=False)
            self.r3_l2_fc3 = nn.Linear(10, 32, bias=False)
            self.r3_l2_exp = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r3_l3_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r3_l3_exp = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.lr = 0.1
            self.category = 8
            self.error_function = torch.nn.MSELoss(size_average=False)
            self.optimiser = LBFGSB(self.parameters(),  line_search_fn='backtracking')   
            
        elif category==9: #basically 7  with only r1_l2 and r1_l3 used
            self.r1_l1_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r1_l1_pool1 = nn.MaxPool1d(2, stride=2)
            self.r1_l2_fc1 = nn.Linear(128, 10, bias=False)
            self.r1_l2_fc2 = nn.Linear(10, 10, bias=False)
            self.r1_l2_fc3 = nn.Linear(10, 128, bias=False)
            self.r1_l3_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r2_l1_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r2_l1_pool1 = nn.MaxPool1d(2, stride=2)
            self.r2_l2_fc1 = nn.Linear(64, 10, bias=False)
            self.r2_l2_fc2 = nn.Linear(10, 10, bias=False)
            self.r2_l2_fc3 = nn.Linear(10, 64, bias=False)
            self.r2_l2_exp = nn.Conv1d(1, 2, 1, stride=1, bias=False)
            self.r2_l3_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r2_l3_exp = nn.Conv1d(1, 2, 1, stride=1, bias=False )
            self.r3_l1_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r3_l1_pool1 = nn.MaxPool1d(2, stride=2)
            self.r3_l1_exp = nn.Conv1d(1, 8, 1, stride=1, bias=False)
            self.r3_l2_fc1 = nn.Linear(32, 10, bias=False)
            self.r3_l2_fc2 = nn.Linear(10, 10, bias=False)
            self.r3_l2_fc3 = nn.Linear(10, 32, bias=False)
            self.r3_l2_exp = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r3_l3_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r3_l3_exp = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.lr = 0.1
            self.category = 9
            self.error_function = torch.nn.MSELoss(size_average=False)
            self.optimiser = LBFGSB(self.parameters(),  line_search_fn='backtracking')              

        elif category==10: #generalised version of network 8, i.e with diagonal scaling 
            self.ds=True #diagonal scaling true or false
            self.r1_l1_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r1_l1_pool1 = nn.MaxPool1d(2, stride=2)
            self.r1_l2_fc1 = nn.Linear(N, r, bias=False)
            self.r1_l2_fc2 = nn.Linear(r, r, bias=False)
            self.r1_l2_fc3 = nn.Linear(r, N, bias=False)
            self.r1_l3_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r1_l3_ds = nn.Parameter(torch.randn(N), requires_grad=True)
            self.r2_l1_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r2_l1_pool1 = nn.MaxPool1d(2, stride=2)
            self.r2_l2_fc1 = nn.Linear(N//2, r, bias=False)
            self.r2_l2_fc2 = nn.Linear(r, r, bias=False)
            self.r2_l2_fc3 = nn.Linear(r, N//2, bias=False)
            self.r2_l2_exp = nn.Conv1d(1, 2, 1, stride=1, bias=False)
            self.r2_l3_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r2_l3_exp = nn.Conv1d(1, 2, 1, stride=1, bias=False )
            self.r3_l1_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r3_l1_pool1 = nn.MaxPool1d(2, stride=2)
            self.r3_l1_exp = nn.Conv1d(1, 8, 1, stride=1, bias=False)
            self.r3_l2_fc1 = nn.Linear(N//4, r, bias=False)
            self.r3_l2_fc2 = nn.Linear(r, r, bias=False)
            self.r3_l2_fc3 = nn.Linear(r, N//4, bias=False)
            self.r3_l2_exp = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r3_l3_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r3_l3_exp = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.lr = 0.1
            self.category = 10
            self.N = N
            self.r = r
            self.error_function = torch.nn.MSELoss(size_average=False)
            self.optimiser = LBFGSB(self.parameters(),  line_search_fn='backtracking')              
            

        elif category==11: #generalised version of network 8, i.e with diagonal scaling 
            self.ds=True #diagonal scaling true or false
            self.r1_l1_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r1_l1_pool1 = nn.MaxPool1d(2, stride=2)
            self.r1_l2_fc1 = nn.Linear(N, r, bias=False)
            self.r1_l2_fc2 = nn.Linear(r, r, bias=False)
            self.r1_l2_fc3 = nn.Linear(r, N, bias=False)
            self.r1_l3_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r1_l3_ds = nn.Parameter(torch.randn(N), requires_grad=True)
            self.r2_l1_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r2_l1_pool1 = nn.MaxPool1d(2, stride=2)
            self.r2_l2_fc1 = nn.Linear(N//2, r, bias=False)
            self.r2_l2_fc2 = nn.Linear(r, r, bias=False)
            self.r2_l2_fc3 = nn.Linear(r, N//2, bias=False)
            self.r2_l2_exp = nn.Conv1d(1, 2, 1, stride=1, bias=False)
            self.r2_l3_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r2_l3_exp = nn.Conv1d(1, 2, 1, stride=1, bias=False )
            self.r3_l1_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r3_l1_pool1 = nn.MaxPool1d(2, stride=2)
            self.r3_l1_exp = nn.Conv1d(1, 8, 1, stride=1, bias=False)
            self.r3_l2_fc1 = nn.Linear(N//4, r, bias=False)
            self.r3_l2_fc2 = nn.Linear(r, r, bias=False)
            self.r3_l2_fc3 = nn.Linear(r, N//4, bias=False)
            self.r3_l2_exp = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r3_l3_conv1 = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r3_l3_exp = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.lr = 0.1
            self.category = 11
            self.N = N
            self.r = r
            self.error_function = torch.nn.MSELoss(size_average=False)
            self.optimiser = LBFGSB(self.parameters(),  line_search_fn='backtracking')   


        elif category==12: #generalised version of network 8, i.e with block diagonal scaling 
            self.ds=True #block diagonal scaling true or false
            self.r1_l1_conv1 = nn.Conv1d(1, 4, 5, stride=1, padding=2, bias=False)
            self.r1_l1_pool1 = nn.MaxPool1d(2, stride=2)
            self.r1_l2_fc1 = nn.Linear(N, r, bias=False)
            self.r1_l2_fc2 = nn.Linear(r, r, bias=False)
            self.r1_l2_fc3 = nn.Linear(r, N, bias=False)
            self.r1_l3_conv1 = nn.Conv1d(1, 4, 5, stride=1, padding=2, bias=False)
            self.r1_l3_bds = nn.Parameter(torch.randn(r, r), requires_grad=True)
            self.r2_l1_conv1 = nn.Conv1d(1, 4, 5, stride=1, padding=2, bias=False)
            self.r2_l1_pool1 = nn.MaxPool1d(2, stride=2)
            self.r2_l2_fc1 = nn.Linear(N//2, r, bias=False)
            self.r2_l2_fc2 = nn.Linear(r, r, bias=False)
            self.r2_l2_fc3 = nn.Linear(r, N//2, bias=False)
            self.r2_l2_exp = nn.Conv1d(1, 2, 1, stride=1, bias=False)
            self.r2_l3_conv1 = nn.Conv1d(1, 4, 5, stride=1, padding=2, bias=False)
            self.r2_l3_exp = nn.Conv1d(1, 2, 1, stride=1, bias=False )
            self.r3_l1_conv1 = nn.Conv1d(1, 4, 5, stride=1, padding=2, bias=False)
            self.r3_l1_pool1 = nn.MaxPool1d(2, stride=2)
            self.r3_l1_exp = nn.Conv1d(1, 8, 1, stride=1, bias=False)
            self.r3_l2_fc1 = nn.Linear(N//4, r, bias=False)
            self.r3_l2_fc2 = nn.Linear(r, r, bias=False)
            self.r3_l2_fc3 = nn.Linear(r, N//4, bias=False)
            self.r3_l2_exp = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.r3_l3_conv1 = nn.Conv1d(1, 4, 5, stride=1, padding=2, bias=False)
            self.r3_l3_exp = nn.Conv1d(1, 4, 1, stride=1, bias=False)
            self.lr = 0.1
            self.category = 12
            self.N = N
            self.r = r
            self.error_function = torch.nn.MSELoss(size_average=False)
            self.optimiser = LBFGSB(self.parameters(),  line_search_fn='backtracking')  
   
    def full_channel_mix(self, input, channels, length):
        for i in range(channels):
            for j in range(channels):
                if i==0 and j==0:
                    output = input[:, i, :].contiguous().view(-1, 1, length ) + input[:,j, : ].contiguous().view(-1, 1, length)
            
                else:
                    output = torch.cat((output, input[:, i, :].contiguous().view(-1, 1, length ) + input[:,j, : ].contiguous().view(-1, 1, length)), 1)
                
                
        return output
            
  

    def add_channels(self, input, channels, length):
        for i in range(channels):
            if i==0 :
                output = input[:, i, :].contiguous().view(-1, 1, length ) 
            
            else:
                output = output + input[:,i, : ].contiguous().view(-1, 1, length)
                
                
        return output
                  
        
    def forward(self, x):
        if self.category==1:
            #x = F.sigmoid(self.fc1(x))
            x = self.fc1(x)
            #x = self.conv1(x)
            out = x
            
        elif self.category==2:
            x = x.contiguous().view(-1, 1, 128)
            x = self.conv1(x)
            x = F.relu(self.pool1(x))
            x = self.conv2(x)
            x = F.relu(self.pool2(x))
            x = self.conv3(x)
            x = x.view(-1, self.num_flat_features(x))
            out = x
            
        elif self.category==3:
            x = x.contiguous().view(-1, 1, 128)
            out1 = self.conv1(x)
            out2 = F.relu(self.pool1(out1))
            out3 = F.relu(self.conv2(out2))
            out4 = F.relu(self.conv3(out3))
            out5 = out4.contiguous().view(-1, 11)
            out6 = self.fc1(out5)
            out6 = out6.contiguous().view(-1, 1, 11)
            out7 = self.conv4(out6)
            out8 = self.pool2(out3)
            out9 = F.relu(self.conv5(out8))
            out = out7.contiguous().view(-1, 128) + out9.view(-1, 128)
            #out = out7.contiguous().view(-1, 128)
            
            
        elif self.category==4:
            x = x.contiguous().view(-1, 1, 128)
            out1 = self.conv1(x)
            out2 = self.conv2(out1)
            out3 = self.conv3(out2)
            out4 = self.conv4(out3)
            out5 = self.conv5(out4)
            out6 = self.conv6(out5)
            out7 = self.conv7(out6)
            out8 = self.conv8(out7)
            out=out8.view(-1, 128)
            
        elif self.category==5:
            x = x.contiguous().view(-1, 1, 128)
            out1 = self.conv1(x)
            out2 = self.conv2(out1)
            out = out2.view(-1, 128)
            
        elif self.category==6:
            x = x.contiguous().view(-1, 1, 128)
            out1 = self.conv1(x) #output 125x4
            out2 = self.full_channel_mix(out1, 4, 125) #output 125x16
            out3 = F.relu(self.pool1(out2)) #output 62x16
            out4 = self.conv2(out3) # output 30x16
            out5 = self.conv3(out4) #output 11x1
            out6 = self.fc1(out5) #output 11x1
            out7 = self.conv4(out6) #output 8x16
            out8 = F.relu(self.pool2(out4)) #output 15x16
            out9 = self.conv5(out8) #output 8x16
            out10 = self.conv6(out8) #output 8x16
            out11 = self.conv7(out8) #output 8x16
            out = out9.contiguous().view(-1, 128) + out7.view(-1, 128) + out10.contiguous().view(-1, 128)  +out11.contiguous().view(-1, 128) 
            
        
        elif self.category==7:
             x = x.contiguous().view(-1, 1, 128)
             r1_l1_out1 = self.r1_l1_conv1(x)
             r1_l1_out2 = self.r1_l1_pool1(r1_l1_out1)
             r1_l1_out = self.add_channels(r1_l1_out2, 4, 64)
             r1_l2_out1 = self.r1_l2_fc1(x.contiguous().view(-1, 128))
             r1_l2_out2 = self.r1_l2_fc2(r1_l2_out1)
             r1_l2_out = self.r1_l2_fc3(r1_l2_out2)
             r1_l3_out1 = self.r1_l3_conv1(x)
             r1_l3_out = self.add_channels(r1_l3_out1, 4, 128)
             r1_l3_out = r1_l3_out.contiguous().view(-1, 128)
             r2_l1_out1 = self.r2_l1_conv1(r1_l1_out)
             r2_l1_out2 = self.r2_l1_pool1(r2_l1_out1)
             r2_l1_out = self.add_channels(r2_l1_out2, 4, 32)
             r2_l2_out1 = self.r2_l2_fc1(r1_l1_out.contiguous().view(-1, 64))
             r2_l2_out2 = self.r2_l2_fc2(r2_l2_out1)
             r2_l2_out3 = self.r2_l2_fc3(r2_l2_out2)
             r2_l2_out4 = self.r2_l2_exp(r2_l2_out3.contiguous().view(-1, 1, 64))
             r2_l2_out = r2_l2_out4.contiguous().view(-1, 128)
             r2_l3_out1 = self.r2_l3_conv1(r1_l1_out)
             r2_l3_out2 = self.add_channels(r2_l3_out1, 4, 64)
             r2_l3_out3 = self.r2_l3_exp(r2_l3_out2)
             r2_l3_out = r2_l3_out3.contiguous().view(-1, 128)
             r3_l1_out1 = self.r3_l1_conv1(r2_l1_out)
             r3_l1_out2 = self.r3_l1_pool1(r3_l1_out1)
             r3_l1_out3 = self.add_channels(r3_l1_out2, 4, 16)
             r3_l1_out4 = self.r3_l1_exp(r3_l1_out3)
             r3_l1_out = r3_l1_out4.contiguous().view(-1, 128)
             r3_l2_out1 = self.r3_l2_fc1(r2_l1_out.contiguous().view(-1,32))
             r3_l2_out2 = self.r3_l2_fc2(r3_l2_out1)
             r3_l2_out3 = self.r3_l2_fc3(r3_l2_out2)
             r3_l2_out4 = self.r3_l2_exp(r3_l2_out3.contiguous().view(-1, 1, 32))
             r3_l2_out = r3_l2_out4.contiguous().view(-1, 128)
             r3_l3_out1 = self.r3_l3_conv1(r2_l1_out)
             r3_l3_out2 = self.add_channels(r3_l3_out1, 4, 32)
             r3_l3_out3 = self.r3_l3_exp(r3_l3_out2)
             r3_l3_out = r3_l3_out3.contiguous().view(-1, 128)
             out = r1_l2_out + r1_l3_out + r2_l2_out + r2_l3_out + r3_l1_out + r3_l2_out + r3_l3_out
             
        
        elif self.category==8:
             x = x.contiguous().view(-1, 1, 128)
             r1_l1_out1 = self.r1_l1_conv1(x)
             r1_l1_out2 = self.r1_l1_pool1(r1_l1_out1)
             r1_l1_out = self.add_channels(r1_l1_out2, 4, 64)
             r1_l2_out1 = self.r1_l2_fc1(x.contiguous().view(-1, 128))
             r1_l2_out2 = self.r1_l2_fc2(r1_l2_out1)
             r1_l2_out = self.r1_l2_fc3(r1_l2_out2)
             r1_l3_out1 = self.r1_l3_conv1(x)
             r1_l3_out = self.add_channels(r1_l3_out1, 4, 128) 
             r1_l3_out = r1_l3_out.contiguous().view(-1, 128) * self.r1_l3_ds
             r2_l1_out1 = self.r2_l1_conv1(r1_l1_out)
             r2_l1_out2 = self.r2_l1_pool1(r2_l1_out1)
             r2_l1_out = self.add_channels(r2_l1_out2, 4, 32)
             r2_l2_out1 = self.r2_l2_fc1(r1_l1_out.contiguous().view(-1, 64))
             r2_l2_out2 = self.r2_l2_fc2(r2_l2_out1)
             r2_l2_out3 = self.r2_l2_fc3(r2_l2_out2)
             r2_l2_out4 = self.r2_l2_exp(r2_l2_out3.contiguous().view(-1, 1, 64))
             r2_l2_out = r2_l2_out4.contiguous().view(-1, 128)
             r2_l3_out1 = self.r2_l3_conv1(r1_l1_out)
             r2_l3_out2 = self.add_channels(r2_l3_out1, 4, 64)
             r2_l3_out3 = self.r2_l3_exp(r2_l3_out2)
             r2_l3_out = r2_l3_out3.contiguous().view(-1, 128)
             r3_l1_out1 = self.r3_l1_conv1(r2_l1_out)
             r3_l1_out2 = self.r3_l1_pool1(r3_l1_out1)
             r3_l1_out3 = self.add_channels(r3_l1_out2, 4, 16)
             r3_l1_out4 = self.r3_l1_exp(r3_l1_out3)
             r3_l1_out = r3_l1_out4.contiguous().view(-1, 128)
             r3_l2_out1 = self.r3_l2_fc1(r2_l1_out.contiguous().view(-1,32))
             r3_l2_out2 = self.r3_l2_fc2(r3_l2_out1)
             r3_l2_out3 = self.r3_l2_fc3(r3_l2_out2)
             r3_l2_out4 = self.r3_l2_exp(r3_l2_out3.contiguous().view(-1, 1, 32))
             r3_l2_out = r3_l2_out4.contiguous().view(-1, 128)
             r3_l3_out1 = self.r3_l3_conv1(r2_l1_out)
             r3_l3_out2 = self.add_channels(r3_l3_out1, 4, 32)
             r3_l3_out3 = self.r3_l3_exp(r3_l3_out2)
             r3_l3_out = r3_l3_out3.contiguous().view(-1, 128)
             out = r1_l2_out + r1_l3_out + r2_l2_out + r2_l3_out + r3_l1_out + r3_l2_out + r3_l3_out
             
        elif self.category==9: #same as 7 but only r1_l2 and r1_l3
             x = x.contiguous().view(-1, 1, 128)

             r1_l2_out1 = self.r1_l2_fc1(x.contiguous().view(-1, 128))
             r1_l2_out2 = self.r1_l2_fc2(r1_l2_out1)
             r1_l2_out = self.r1_l2_fc3(r1_l2_out2)
             r1_l3_out1 = self.r1_l3_conv1(x)
             r1_l3_out = self.add_channels(r1_l3_out1, 4, 128)
             r1_l3_out = r1_l3_out.contiguous().view(-1, 128)

             out = r1_l2_out + r1_l3_out 



        elif self.category==10:
             x = x.contiguous().view(-1, 1, self.N)
             r1_l1_out1 = self.r1_l1_conv1(x)
             r1_l1_out2 = self.r1_l1_pool1(r1_l1_out1)
             r1_l1_out = self.add_channels(r1_l1_out2, 4, self.N//2)
             r1_l2_out1 = self.r1_l2_fc1(x.contiguous().view(-1, self.N))
             r1_l2_out2 = self.r1_l2_fc2(r1_l2_out1)
             r1_l2_out = self.r1_l2_fc3(r1_l2_out2)
             r1_l3_out1 = self.r1_l3_conv1(x)
             r1_l3_out = self.add_channels(r1_l3_out1, 4, self.N) 
             r1_l3_out = r1_l3_out.contiguous().view(-1, self.N) * self.r1_l3_ds
             r2_l1_out1 = self.r2_l1_conv1(r1_l1_out)
             r2_l1_out2 = self.r2_l1_pool1(r2_l1_out1)
             r2_l1_out = self.add_channels(r2_l1_out2, 4, self.N//4)
             r2_l2_out1 = self.r2_l2_fc1(r1_l1_out.contiguous().view(-1, self.N//2))
             r2_l2_out2 = self.r2_l2_fc2(r2_l2_out1)
             r2_l2_out3 = self.r2_l2_fc3(r2_l2_out2)
             r2_l2_out4 = self.r2_l2_exp(r2_l2_out3.contiguous().view(-1, 1, self.N//2))
             r2_l2_out = r2_l2_out4.contiguous().view(-1, self.N)
             r2_l3_out1 = self.r2_l3_conv1(r1_l1_out)
             r2_l3_out2 = self.add_channels(r2_l3_out1, 4, self.N//2)
             r2_l3_out3 = self.r2_l3_exp(r2_l3_out2)
             r2_l3_out = r2_l3_out3.contiguous().view(-1, self.N)
             r3_l1_out1 = self.r3_l1_conv1(r2_l1_out)
             r3_l1_out2 = self.r3_l1_pool1(r3_l1_out1)
             r3_l1_out3 = self.add_channels(r3_l1_out2, 4, self.N//8)
             r3_l1_out4 = self.r3_l1_exp(r3_l1_out3)
             r3_l1_out = r3_l1_out4.contiguous().view(-1, self.N)
             r3_l2_out1 = self.r3_l2_fc1(r2_l1_out.contiguous().view(-1,self.N//4))
             r3_l2_out2 = self.r3_l2_fc2(r3_l2_out1)
             r3_l2_out3 = self.r3_l2_fc3(r3_l2_out2)
             r3_l2_out4 = self.r3_l2_exp(r3_l2_out3.contiguous().view(-1, 1, self.N//4))
             r3_l2_out = r3_l2_out4.contiguous().view(-1, self.N)
             r3_l3_out1 = self.r3_l3_conv1(r2_l1_out)
             r3_l3_out2 = self.add_channels(r3_l3_out1, 4, self.N//4)
             r3_l3_out3 = self.r3_l3_exp(r3_l3_out2)
             r3_l3_out = r3_l3_out3.contiguous().view(-1, self.N)
             out = r1_l2_out + r1_l3_out + r2_l2_out + r2_l3_out + r3_l1_out + r3_l2_out + r3_l3_out
             

        elif self.category==11:
             x = x.contiguous().view(-1, 1, self.N)
             r1_l2_out1 = self.r1_l2_fc1(x.contiguous().view(-1, self.N))
             r1_l2_out2 = self.r1_l2_fc2(r1_l2_out1)
             r1_l2_out = self.r1_l2_fc3(r1_l2_out2)
             r1_l3_out1 = self.r1_l3_conv1(x)
             r1_l3_out = self.add_channels(r1_l3_out1, 4, self.N) 
             r1_l3_out = r1_l3_out.contiguous().view(-1, self.N) * self.r1_l3_ds

             out = r1_l2_out + r1_l3_out 


        elif self.category==12:
             x = x.contiguous().view(-1, 1, self.N)
             r1_l1_out1 = self.r1_l1_conv1(x)
             r1_l1_out2 = self.r1_l1_pool1(r1_l1_out1)
             r1_l1_out = self.add_channels(r1_l1_out2, 4, self.N//2)
             r1_l2_out1 = self.r1_l2_fc1(x.contiguous().view(-1, self.N))
             r1_l2_out2 = self.r1_l2_fc2(r1_l2_out1)
             r1_l2_out = self.r1_l2_fc3(r1_l2_out2)
             r1_l3_out1 = self.r1_l3_conv1(x)
             r1_l3_out2 = r1_l3_out1.permute(0, 2, 1)
             r1_l3_out3 = torch.mm(r1_l3_out2.contiguous().view(-1, 4), self.r1_l3_bds)
             r1_l3_out4 = r1_l3_out3.contiguous().view(-1, self.N, 4)
             r1_l3_out4 = r1_l3_out4.permute(0, 2, 1)
             r1_l3_out = self.add_channels(r1_l3_out4, 4, self.N) 
             r1_l3_out = r1_l3_out.contiguous().view(-1, self.N)
             r2_l1_out1 = self.r2_l1_conv1(r1_l1_out)
             r2_l1_out2 = self.r2_l1_pool1(r2_l1_out1)
             r2_l1_out = self.add_channels(r2_l1_out2, 4, self.N//4)
             r2_l2_out1 = self.r2_l2_fc1(r1_l1_out.contiguous().view(-1, self.N//2))
             r2_l2_out2 = self.r2_l2_fc2(r2_l2_out1)
             r2_l2_out3 = self.r2_l2_fc3(r2_l2_out2)
             r2_l2_out4 = self.r2_l2_exp(r2_l2_out3.contiguous().view(-1, 1, self.N//2))
             r2_l2_out = r2_l2_out4.contiguous().view(-1, self.N)
             r2_l3_out1 = self.r2_l3_conv1(r1_l1_out)
             r2_l3_out2 = self.add_channels(r2_l3_out1, 4, self.N//2)
             r2_l3_out3 = self.r2_l3_exp(r2_l3_out2)
             r2_l3_out = r2_l3_out3.contiguous().view(-1, self.N)
             r3_l1_out1 = self.r3_l1_conv1(r2_l1_out)
             r3_l1_out2 = self.r3_l1_pool1(r3_l1_out1)
             r3_l1_out3 = self.add_channels(r3_l1_out2, 4, self.N//8)
             r3_l1_out4 = self.r3_l1_exp(r3_l1_out3)
             r3_l1_out = r3_l1_out4.contiguous().view(-1, self.N)
             r3_l2_out1 = self.r3_l2_fc1(r2_l1_out.contiguous().view(-1,self.N//4))
             r3_l2_out2 = self.r3_l2_fc2(r3_l2_out1)
             r3_l2_out3 = self.r3_l2_fc3(r3_l2_out2)
             r3_l2_out4 = self.r3_l2_exp(r3_l2_out3.contiguous().view(-1, 1, self.N//4))
             r3_l2_out = r3_l2_out4.contiguous().view(-1, self.N)
             r3_l3_out1 = self.r3_l3_conv1(r2_l1_out)
             r3_l3_out2 = self.add_channels(r3_l3_out1, 4, self.N//4)
             r3_l3_out3 = self.r3_l3_exp(r3_l3_out2)
             r3_l3_out = r3_l3_out3.contiguous().view(-1, self.N)
             out = r1_l2_out + r1_l3_out + r2_l2_out + r2_l3_out + r3_l1_out + r3_l2_out + r3_l3_out
             


            
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
	
    def adjust_lr(self, epoch):
        lr = self.lr * (0.1 ** (epoch // 50))
        for param_group in self.optimiser.param_groups:
            param_group['lr'] = lr


    def train(self, inputs_list, targets_list):
        # calculate the output of the network
        output = self.forward(inputs_list)
        #print(output)
        # create a Variable out of the target vector, doesn't need gradients calculated
        # also shift to GPU, remove .cuda. if not desired
        target_variable = Variable(targets_list.data, requires_grad=False)
        
        # calculate error
        loss = self.error_function(output, target_variable)
        #if flag ==1:
        #    print(loss.data[0])

        # zero gradients, perform a backward pass, and update the weights.
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        return (loss.data[0])

    def l2_penalty(self):
        l2_reg = None
        for W in self.parameters():
            if l2_reg is None:
                l2_reg = W.norm(2)
            else:
                l2_reg = l2_reg + W.norm(2)
 
        return l2_reg           

    def train_bfgs(self, inputs_list, targets_list):
        # calculate the output of the network
        def closure():
            self.optimiser.zero_grad()
            output = self.forward(inputs_list)
            #print(output)
            # create a Variable out of the target vector, doesn't need gradients calculated
            # also shift to GPU, remove .cuda. if not desired
            target_variable = Variable(targets_list.data, requires_grad=False)
            
            # calculate error
            loss_without_reg = self.error_function(output, target_variable)
            l2_reg = self.l2_penalty()
            loss = loss_without_reg + reg_parameter*l2_reg
            #if flag ==1:
            #    print(loss.data[0])
    
            # zero gradients, perform a backward pass, and update the weights.
            #print('Loss',loss.data[0])
            loss.backward()
            return loss
        

        self.optimiser.zero_grad()
        self.optimiser.step(closure)
        self.optimiser.zero_grad()
        output = self.forward(inputs_list)
        target_variable = Variable(targets_list.data, requires_grad=False)
        loss_val = self.error_function(output, target_variable) + reg_parameter*self.l2_penalty()
        loss_val.backward()
        return (loss_val.data[0])

        
  ###########################################################################################      

########## Initialising network###################### ############ #############################      
net = Net(category, mat_size, r_net)
net.double()
print(net)

#params = list(net.parameters())
#print('Number of parameters: ')
#print(len(params))
#print(params[0].size())
#print(params[1].size())

#####################################################################################################
u1 = np.random.rand(mat_size, r)
v1 = np.random.rand(r, mat_size)


inputs1 = np.random.rand(training_instances, mat_size)
inputs2 = np.eye(mat_size, dtype=float)
inputs = np.concatenate((inputs2, inputs1), axis=0)
#A = np.matrix([[1., 2., 1.], [1., 0., 1.], [4., 2., 4.]])
#filename = 'laplaceMatrices.mat'
#filename = 'lapDLPN1024.mat'
#filename = 'lapDLPN1024.mat'
#mat = spio.loadmat(filename)       #Change file name appropriately
#A = mat['lapDL']                # input_data is numpy matrix LaplaceDL
#A = mat['DLP']
#print('rank:', np.linalg.matrix_rank(A))
#A = np.eye(mat_size) + A
A = np.diag(np.random.rand(mat_size)) + u1@v1 
#A = np.diag(np.ones(128)) + u1@v1 
print(A)
#A = np.random.rand(mat_size, mat_size)
Ainv = np.linalg.inv(A)
print('Condition number:' , np.linalg.cond(A))
targets = np.transpose(Ainv@np.transpose(inputs))
inputs_tensor = Variable(torch.from_numpy(inputs), requires_grad=False)
inputs_tensor = inputs_tensor.type(torch.DoubleTensor)
#print('Inputs tensor: ', inputs_tensor)
T = torch.from_numpy(A)
T = T.type(torch.DoubleTensor)
#print('True Matrix: ', T)
targets_tensor = Variable(torch.from_numpy(targets))
targets_tensor = targets_tensor.type(torch.DoubleTensor)
#print('Targets tensor: ', targets_tensor)

#epochs is the number of times the training data set is used for training
mi, ni = inputs_tensor.size()
mt, nt = targets_tensor.size()
ini_gradnorm=0.0
ini_loss = 0.0
weightnorm = 0.0

#---------------------Test instances initialisation--------------

target_output = Variable(torch.randn(test_instances,ni)).double()
#print('target output', target_output)
test_input = Variable(T.mm(target_output.data.permute(1, 0)).permute(1, 0))
#print('test input:',test_input)
#----------------------------------------------------------------
start = time.time()
for e in range(epochs):
    
    #net.adjust_lr(e)
    for i in range(mi//batch_size):
        inputs = inputs_tensor[i*batch_size:(i+1)*batch_size, :].view(batch_size, ni)
        targets = targets_tensor[i*batch_size:(i+1)*batch_size, :].contiguous().view(batch_size, nt)
        #loss = net.train(inputs, targets)
        #print('Loss:', loss)
        if e==0:
            output = net.forward(inputs)
            target_variable = Variable(targets.data, requires_grad=False)
            ini_loss = net.error_function(output, target_variable) + reg_parameter*net.l2_penalty()
            ini_loss.backward()
            ini_gradnorm = 0.0
            for w in net.parameters():
                ini_gradnorm += torch.norm(w.grad, 2)
         
            print('Initial Loss:', ini_loss.data)
            print('Initial grad norm:', ini_gradnorm)
            
        loss = net.train_bfgs(inputs, targets)
        if e%1==0:
            print('--------------EPOCH ', e, '------------')
            print('Absolute Loss:', loss)
            print('Relative Loss:', loss/ini_loss)
            
            gradnorm = 0.0
            weightnorm = 0.0
            for w in net.parameters():
                gradnorm += torch.norm(w.grad, 2)
                weightnorm += torch.norm(w.data, 2)
                #print('weights:', w.data)
                    
                    
                #print('Grad:', norm(w.grad))
                
            print('Relative Grad Norm:', gradnorm/ini_gradnorm)    
            if e+1==epochs:
                print('Weights norm', weightnorm)
                
                
    test_output = net(test_input)
    #print('test output', test_output)
    error = torch.max(torch.div((target_output.data - test_output.data).norm(2, 1),  target_output.data.norm(2, 1)))
    print('Relative Test error:', error)
    print('Time elapsed:', time.time() - start)
#print('Relative error:', error) 
#
#print('------------Printing TEST CASE ERROR:----------')        
#target_output = Variable(torch.randn(test_instances,ni)).double()
##print('target output', target_output)
#test_input = Variable(T.mm(target_output.data.permute(1, 0)).permute(1, 0))
##print('test input:',test_input)
#test_output = net(test_input)
##print('test output', test_output)
#error = torch.max(torch.div((target_output.data - test_output.data).norm(2, 1),  target_output.data.norm(2, 1)))
#print('Relative error:', error) 

#sys.stdout.flush()