# InGram: Inductive Knowledge Graph Embedding via Relation Graphs
This code is the official implementation of the following [paper](https://proceedings.mlr.press/v202/lee23c.html):

> Jaejun Lee, Chanyoung Chung, and Joyce Jiyoung Whang, InGram: Inductive Knowledge Graph Embedding via Relation Graphs, The 40th International Conference on Machine Learning (ICML), 2023.

All codes are written by Jaejun Lee (jjlee98@kaist.ac.kr). When you use this code or data, please cite our paper.

```bibtex
@inproceedings{ingram,
	author={Jaejun Lee and Chanyoung Chung and Joyce Jiyoung Whang},
	title={{I}n{G}ram: Inductive Knowledge Graph Embedding via Relation Graphs},
	booktitle={Proceedings of the 40th International Conference on Machine Learning},
	year={2023},
	pages={18796--18809}
}
```

## Requirements

We used Python 3.8 and PyTorch 1.12.1 with cudatoolkit 11.3.

You can install all requirements with:

```shell
pip install -r requirements.txt
```

## Reproducing the Reported Results

We used NVIDIA RTX A6000, NVIDIA GeForce RTX 2080 Ti, and NVIDIA GeForce RTX 3090 for all our experiments. We provide the checkpoints we used to produce the inductive link prediction results on 14 datasets. If you want to use the checkpoints, place the unzipped ckpt folder in the same directory with the codes.

You can download the checkpoints from https://drive.google.com/file/d/1aZrx2dYNPT7j4TGVBOGqHMdHRwFUBqx5/view?usp=sharing.

The command to reproduce the results in our paper:

```python
python3 test.py --best --data_name [dataset_name]
```

## Training from Scratch

To train InGram from scratch, run `train.py` with arguments. Please refer to `my_parser.py` for the examples of the arguments. Please tune the hyperparameters of our model using the range provided in Appendix C of the paper because the best hyperparameters may be different due to randomness.

The list of arguments of `train.py`:
- `--data_name`: name of the dataset
- `--exp`: experiment name
- `-m, --margin`: $\gamma$
- `-lr, --learning_rate`: learning rate
- `-nle, --num_layer_ent`: $\widehat{L}$
- `-nlr, --num_layer_rel`: $L$
- `-d_e, --dimension_entity`: $\widehat{d}$
- `-d_r, --dimension_relation`: $d$
- `-hdr_e, --hidden_dimension_ratio_entity`: $\widehat{d'}/\widehat{d}$
- `-hdr_r, --hidden_dimension_ratio_relation`: $d'/d$
- `-b, --num_bin`: $B$
- `-e, --num_epoch`: number of epochs to run
- `--target_epoch`: the epoch to run test (only used for test.py)
- `-v, --validation_epoch`: duration for the validation
- `--num_head`: $\widehat{K}=K$
- `--num_neg`: number of negative triplets per triplet
- `--best`: use the provided checkpoints (only used for test.py)
- `--no_write`: don't save the checkpoints (only used for train.py)
