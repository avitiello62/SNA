import numpy as np
import time
import networkx as nx



def my_hits(G, epsilon=1e-8, normalized=True):
	L = nx.adjacency_matrix(G)
	L_T = L.transpose()	
	n = L.shape[0]
	epsilon_matrix = epsilon * np.ones(n)
	h = np.ones(n, dtype=np.float64)
	a = np.ones(n, dtype=np.float64)

	while True:
		a_old = a
		h_old = h

		a = L_T*h_old
		a_max = a.max(axis=0)
		a = a/a_max if a_max>0 else a
		h = L*a
		h_max = h.max(axis=0)
		h = h/h_max if h_max>0 else h

		#print("h: ", h)
		#print("a: ", a)
		#input("Continue?")

		if (((abs(h - h_old)) < epsilon_matrix).all()) and (((abs(a - a_old)) < epsilon_matrix).all()):
				break
	return h,a
	




