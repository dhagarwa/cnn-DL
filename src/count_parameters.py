#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:17:46 2018

@author: dhwanit
"""

N = 1024
r = 4
m = 4
par1 = 2*N*r + r*r + 2*(N//2)*r + r*r + 2*(N//4)*r + r*r + N + 32 
par2 = 2*N*r + r*r + 2*(N//2)*r + r*r + 2*(N//4)*r + r*r + 6*5*4 + 20 + 16
print('Number of parameters:', par1)
print('number of parameters:', par2)