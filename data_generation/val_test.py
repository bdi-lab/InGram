import igraph
import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data')
parser.add_argument('--seed', default = 5, type = int)
args = parser.parse_args()
random.seed(args.seed)

full_data = args.data
if full_data not in os.listdir():
    raise ValueError
print(f"PROCESSING {full_data}")
test = []
test_graph = []
test_rel = set()
test_r2ht = {}
test_q = {}
test_hcon = {}
test_tcon = {}
with open(f"./{full_data}/kg_inference.txt") as f:
    for line in f.readlines():
        h,r,t = line.strip().split()
        test.append((h,r,t))
        if r in test_r2ht:
            test_r2ht[r].append((h,t))
        else:
            test_r2ht[r] = [(h,t)]
        if (h,'_',t) in test_q:
            test_q[(h,'_',t)].append(r)
        else:
            test_q[(h,'_',t)] = [r]
        test_rel.add(r)
        test_graph.append((h,t))
G_test = igraph.Graph.TupleList(test_graph, directed = True)
spanning_test = G_test.spanning_tree()

num_test = len(test)
test_msg = set()
test = set(test)

for e in spanning_test.es:
    h,t = e.tuple
    h = spanning_test.vs[h]["name"]
    t = spanning_test.vs[t]["name"]
    r = random.choice(test_q[(h,'_',t)])
    test_msg.add((h, r, t))
    test_rel.discard(r)
    test.discard((h,r,t))
for r in test_rel:
    h,t = random.choice(test_r2ht[r])
    test_msg.add((h,r,t))
    test.discard((h,r,t))
left_test = sorted(list(test))
test_msg = sorted(list(test_msg))
random.shuffle(left_test)
remainings = int(num_test * 0.6) - len(test_msg)
test_msg += left_test[:remainings]
left_test = left_test[remainings:]

final_valid = left_test[:len(left_test)//2]
final_test = left_test[len(left_test)//2:]

with open(f"{full_data}/msg.txt", "w") as f:
    for h,r,t in test_msg:
        f.write(f"{h}\t{r}\t{t}\n")
with open(f"{full_data}/valid.txt", "w") as f:
    for h,r,t in final_valid:
        f.write(f"{h}\t{r}\t{t}\n")
with open(f"{full_data}/test.txt", "w") as f:
    for h,r,t in final_test:
        f.write(f"{h}\t{r}\t{t}\n")
