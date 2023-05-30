import torch
from relgraph import generate_relation_triplets

def initialize(target, msg, d_e, d_r, B):

    init_emb_ent = torch.zeros((target.num_ent, d_e)).cuda()
    init_emb_rel = torch.zeros((2*target.num_rel, d_r)).cuda()
    gain = torch.nn.init.calculate_gain('relu')
    torch.nn.init.xavier_normal_(init_emb_ent, gain = gain)
    torch.nn.init.xavier_normal_(init_emb_rel, gain = gain)
    relation_triplets = generate_relation_triplets(msg, target.num_ent, target.num_rel, B)

    relation_triplets = torch.tensor(relation_triplets).cuda()

    return init_emb_ent, init_emb_rel, relation_triplets