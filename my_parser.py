import argparse
import json

def parse(test=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default = "./data/", type = str)
    parser.add_argument('--data_name', default = 'NL-100', type = str)
    parser.add_argument('--exp', default = 'exp', type = str)
    parser.add_argument('-m', '--margin', default = 2, type = float)
    parser.add_argument('-lr', '--learning_rate', default=5e-4, type = float)
    parser.add_argument('-nle', '--num_layer_ent', default = 2, type = int)
    parser.add_argument('-nlr', '--num_layer_rel', default = 2, type = int)
    parser.add_argument('-d_e', '--dimension_entity', default = 32, type = int)
    parser.add_argument('-d_r', '--dimension_relation', default = 32, type = int)
    parser.add_argument('-hdr_e', '--hidden_dimension_ratio_entity', default = 8, type = int)
    parser.add_argument('-hdr_r', '--hidden_dimension_ratio_relation', default = 4, type = int)
    parser.add_argument('-b', '--num_bin', default = 10, type = int)
    parser.add_argument('-e', '--num_epoch', default = 10000, type = int)
    if test:
        parser.add_argument('--target_epoch', default = 6600, type = int)
    parser.add_argument('-v', '--validation_epoch', default = 200, type = int)
    parser.add_argument('--num_head', default = 8, type = int)
    parser.add_argument('--num_neg', default = 10, type = int)
    parser.add_argument('--best', action = 'store_true')
    if not test:
        parser.add_argument('--no_write', action = 'store_true')

    args = parser.parse_args()

    if test and args.best:
        remaining_args = []
        with open(f"./ckpt/best/{args.data_name}/config.json") as f:
            configs = json.load(f)
        for key in vars(args).keys():
            if key in configs:
                vars(args)[key] = configs[key]
            else:
                remaining_args.append(key)

    return args