import os

dataset_names = ['NL', 'WK', 'FB']
dataset_types = ['25', '50', '75', '100']

for dataset_name in dataset_names:
    for dataset_type in dataset_types:
        dataset = dataset_name + "-" + dataset_type
        if os.path.isdir(f"./{dataset}/"):
            continue
        os.makedirs(f"./{dataset}/")
        os.system(f'cp ../../data_230118/{dataset}/train.txt {dataset}/train.txt')
        os.system(f'cp ../../data_230118/{dataset}/valid.txt {dataset}/valid.txt')
        os.system(f'cp ../../data_230118/{dataset}/msg.txt {dataset}/msg.txt')
        os.system(f'cp ../../data_230118/{dataset}/test.txt {dataset}/test.txt')

datasets = ['NL-0']

for dataset in datasets:
    if os.path.isdir(f"./{dataset}/"):
        continue
    os.makedirs(f"./{dataset}/")
    os.system(f'cp ../../data_230118/{dataset}/train.txt {dataset}/train.txt')
    os.system(f'cp ../../data_230118/{dataset}/valid.txt {dataset}/valid.txt')
    os.system(f'cp ../../data_230118/{dataset}/msg.txt {dataset}/msg.txt')
    os.system(f'cp ../../data_230118/{dataset}/test.txt {dataset}/test.txt')
