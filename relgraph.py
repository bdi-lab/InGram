from scipy.sparse import csr_matrix
from tqdm import tqdm
from utils import *
import numpy as np
import math
import igraph
def create_relation_graph(triplet, num_ent, num_rel):
	ind_h = triplet[:,:2]
	ind_t = triplet[:,1:]
	

	E_h = csr_matrix((np.ones(len(ind_h)), (ind_h[:, 0], ind_h[:, 1])), shape=(num_ent, 2 * num_rel))
	E_t = csr_matrix((np.ones(len(ind_t)), (ind_t[:, 1], ind_t[:, 0])), shape=(num_ent, 2 * num_rel))

	diag_vals_h = E_h.sum(axis=1).A1
	diag_vals_h[diag_vals_h!=0] = 1/(diag_vals_h[diag_vals_h!=0]**2)

	diag_vals_t = E_t.sum(axis=1).A1
	diag_vals_t[diag_vals_t!=0] = 1/(diag_vals_t[diag_vals_t!=0]**2)


	D_h_inv = csr_matrix((diag_vals_h, (np.arange(num_ent), np.arange(num_ent))), shape=(num_ent, num_ent))
	D_t_inv = csr_matrix((diag_vals_t, (np.arange(num_ent), np.arange(num_ent))), shape=(num_ent, num_ent))


	A_h = E_h.transpose() @ D_h_inv @ E_h
	A_t = E_t.transpose() @ D_t_inv @ E_t

	return A_h + A_t

def get_relation_triplets(G_rel, B):
	rel_triplets = []
	for tup in G_rel.get_edgelist():
		h,t = tup
		tupid = G_rel.get_eid(h,t)
		w = G_rel.es[tupid]["weight"]
		rel_triplets.append((int(h), int(t), float(w)))
	rel_triplets = np.array(rel_triplets)

	nnz = len(rel_triplets)

	temp = (-rel_triplets[:,2]).argsort()
	weight_ranks = np.empty_like(temp)
	weight_ranks[temp] = np.arange(nnz) + 1

	relation_triplets = []
	for idx,triplet in enumerate(rel_triplets):
		h,t,w = triplet
		rk = int(math.ceil(weight_ranks[idx]/nnz*B))-1
		relation_triplets.append([int(h), int(t), rk])
		assert rk >= 0
		assert rk < B
	
	return np.array(relation_triplets)

def generate_relation_triplets(triplet, num_ent, num_rel, B):
	A = create_relation_graph(triplet, num_ent, num_rel)
	G_rel = igraph.Graph.Weighted_Adjacency(A)
	relation_triplets = get_relation_triplets(G_rel, B)
	return relation_triplets