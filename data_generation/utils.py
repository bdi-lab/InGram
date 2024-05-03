import networkx as nx
import random

def remove_duplicate(x):
	return list(dict.fromkeys(x))

def read_KG(path):
	entity = []
	relation = []
	triplet = []
	with open(path, 'r') as f:
		for line in f.readlines():
			h, r, t = line.strip().split('\t')
			entity.append(h)
			entity.append(t)
			relation.append(r)
			triplet.append((h, r, t))
	return remove_duplicate(entity), remove_duplicate(relation), remove_duplicate(triplet)

def gather(x):
	ent = []
	rel = []
	for h, r, t in x:
		ent.append(h)
		ent.append(t)
		rel.append(r)
	return remove_duplicate(ent), remove_duplicate(rel)

def check_no_overlap(x, y):
	assert len(set(x).intersection(set(y))) == 0
	print("Done: Check no overlap")

def write(path, x):
	with open(path, 'w') as f:
		for h, r, t in x:
			f.write(f"{h}\t{r}\t{t}\n")

def gather_neighbor(triplet, x, thr):
	res = []
	for h, r, t in triplet:
		if h == x:
			res.append(t)
		elif t == x:
			res.append(h)
	res = remove_duplicate(res)
	if len(res) > thr:
		res = random.sample(res, thr)
	return res

def sample_2hop(triplet, x, thr):
	sample = set()
	for e in x:
		neighbor = set([e])
		neighbor_1hop = gather_neighbor(triplet, e, thr)
		neighbor = neighbor.union(set(neighbor_1hop))

		for e1 in neighbor_1hop:
			neighbor_2hop = gather_neighbor(triplet, e1, thr)
			neighbor = neighbor.union(set(neighbor_2hop))

		sample = sample.union(neighbor)
	return sample

def merge(x, y, p):
	if p >= 1:
		return y
	elif p <= 0:
		return x
	else:
		num_tot = min(len(x) / (1 - p), len(y) / p)
		random.shuffle(x)
		random.shuffle(y)
		return x[:int(num_tot * (1 - p))] + y[:int(num_tot * p)]

def gcc(triplet):
	edge = []
	for h, r, t in triplet:
		edge.append((h, t))
	G = nx.Graph()
	G.add_edges_from(edge)
	largest_cc = max(nx.connected_components(G), key=len)
	return largest_cc
