## Stat 280 Problem Set 4
## Ningyin(Ariel) Xu
## Problem 4
import numpy as np
import matplotlib.pyplot as plt
import os

'''
Problem 4d:
'''

def initialization_nm(A):
	upperbd = 2/(np.linalg.norm(A, ord = 2)**2)
	lowerbd = 0
	alpha = (upperbd - lowerbd) * np.random.random_sample() + lowerbd
	X0 = alpha*A.T
	return X0

# def F_X(A, X):
# 	Xinv = np.linalg.inv(X)
# 	FX = Xinv - A
# 	return FX

def stop_condition(X, X_last):
	X_diff = X - X_last
	stopcond = np.linalg.norm(X_diff, ord = "fro")
	return stopcond

def newton_inverse(A, epislon, X0):
	m,n = X0.shape
	I = np.identity(m)
	X = X0@(2*I - A@X0)
	X_last = X0
	stopcond = stop_condition(X, X_last)
	k = 1 #since we already run one iteration
	while stopcond > epislon:
		X_last = X
		X = X@(2*I - A@X)
		stopcond = stop_condition(X, X_last)
		k += 1

	return (X, k)

def test_accur(X_ast, actualInv):
	return np.linalg.norm(X_ast - actualInv, ord="fro")


if __name__ == "__main__":
	epi = 10**-8

	A = np.random.randint(-1000,1000,size = (2,2))
	actual_Ainv = np.linalg.inv(A)

	A_d = np.diag(np.diag(np.random.randn(10,10)))
	actual_Adinv = np.linalg.inv(A_d)

	X0 = initialization_nm(A)
	Xast_A,k1 = newton_inverse(A, epi, X0)
	accur_A = test_accur(Xast_A, actual_Ainv)
	print(A, Xast_A, accur_A, k1)

	X0d = initialization_nm(A_d)
	Xast_Ad,k2 = newton_inverse(A_d, epi, X0d)
	accur_Ad = test_accur(Xast_Ad, actual_Adinv)
	print(A_d, Xast_Ad, accur_Ad, k2)

	Als = [np.random.randn(10,10), np.random.randn(100,100), 
		   np.random.randn(1000,1000), np.random.randn(10000,10000)]
	for A in Als:
		actual_inv = np.linalg.inv(A)
		X0 = initialization_nm(A)
		Xast,k = newton_inverse(A, epi, X0)
		accur = test_accur(Xast, actual_inv)
		print(accur, k1)




