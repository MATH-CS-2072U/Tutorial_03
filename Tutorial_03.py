import numpy as np
from Gauss import *
from LU import *

A = np.array([[8, 2, 7],[4,-1,4],[-4,-0.5,1]])
r = np.array([[2],[-3],[1]])
Aa = np.concatenate((A,r),1)

Ua = Gauss(Aa)
x = backsub(Ua)
print('Solution found by Gauss elimination:')
print(Ua)
print(x)

L, U = LU(A)
print('Factors L and U:')
print(L)
print(U)

# Step 1: forward substitution:
y = LUforwardsub(L,r)
print(y)

# Step 2: backward substitution:
x = LUbacksub(U,y)
print(x)
