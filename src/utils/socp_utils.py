#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 10:45:24 2025

: Project contributors"""

import time
import numpy as np
import torch

def generate_random_socp(dim,nSOC,n_As,nIneq,nEq,seed=1):
    '''
    Generate random general socp:
        min_x      q.T * x
        s.t.       ||A_i * x + b_i || <= c_i.T * x + d_i
                   Fx = g

    Parameters
    ----------
    dim : Dim of variable
    nSOC : #Cone constraints
    n_As :  List, n_As[i]: # rows of A_i
    nEq : #Equality constraints
    seed : Random seed. The default is 1.

    Returns
    -------
    For all the parameters, requires_grad = True by default
    
    q : Torch tensor with size (dim).
    A : List, A[i]: Torch tensor with size (n_As[i],dim).
    b : List, b[i]: Torch tensor with size (n_As[i]).
    c : List, c[i]: Torch tensor with size (dim).
    d : List, d[i]: Torch tensor with size ().
    F : Torch tensor with size (nEq,dim).
    g : Torch tensor with size (nEq).

    '''
    
    torch.manual_seed(seed)
 
    
    q = torch.randn(dim,dtype=torch.float64)
    q.requires_grad_(True)
    A = []
    b = []
    c = []
    d = []
    x0 = torch.randn(dim,dtype=torch.float64)
    for i in range(nSOC):
        A_i = torch.randn(n_As[i], dim, dtype=torch.float64, requires_grad=True)
        
        b_i = torch.randn(n_As[i], dtype=torch.float64, requires_grad=True)

        # A_i = torch.eye(n_As[i], dtype=torch.float64)
        # b_i = torch.zeros(n_As[i], dtype=torch.float64)
        
        A_i.requires_grad_(True)
        b_i.requires_grad_(True)
        A.append(A_i)
        b.append(b_i)
        
        # c_i = torch.zeros(dim, dtype=torch.float64)
        c_i = torch.randn(dim, dtype=torch.float64, requires_grad=True)
        c_i.requires_grad_(True)
        
        c.append(c_i)
        
        d_i = torch.norm(A[i] @ x0 + b[i], 2) - c[i] @ x0
        d_i = d_i.detach().clone()
        d_i.requires_grad_(True)
        
        # d_i = 4 * abs(torch.randn(1, dtype=torch.float64, requires_grad=True))
        
        # d_i.retain_grad()
        d.append(d_i)
    if nIneq>0:
        G = torch.randn(nIneq, dim, dtype=torch.float64)
        h = G @ x0 + torch.ones(nIneq)
        G.requires_grad_(True)
        h.requires_grad_(True)
    else:
        G,h = None, None
        
    if nEq>0:
        F = torch.randn(nEq, dim, dtype=torch.float64)
        g = F @ x0
    
        F.requires_grad_(True)
        g.requires_grad_(True)
    else:
        F,g = None,None
    
    
    

    return q,A,b,c,d,F,g,G,h
    

def to_scs(A,b,c,d,F,g):
    '''
    TODO: detailed
    Transform the general socp into scs format
    '''
    nSOC = len(A)
    G = []
    h = []
    for i in range(nSOC):
        G_i = torch.vstack((-A[i],-c[i]))
        h_i = torch.hstack((b[i],d[i]))
        G.append(G_i)
        h.append(h_i)
    G = torch.cat(G, dim=0)
    h = torch.cat(h, dim=0)
    if F is not None:
        G = torch.vstack( ( torch.vstack((G,F)), torch.zeros(G.shape[1]) ))
        h = torch.hstack( ( torch.hstack((h,g)), torch.zeros(1)))
    G.requires_grad_(True)
    h.requires_grad_(True)
    return G,h
