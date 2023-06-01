# InGram: Inductive Knowledge Graph Embedding via Relation Graphs
This code is the official implementation of the following [paper](https://arxiv.org/abs/2305.19987):

> Jaejun Lee, Chanyoung Chung, and Joyce Jiyoung Whang, InGram: Inductive Knowledge Graph Embedding via Relation Graphs, To appear in the 40th International Conference on Machine Learning (ICML), 2023.

All codes are written by Jaejun Lee (jjlee98@kaist.ac.kr). When you use this code or data, please cite our paper.

```bibtex
@article{ingram,
  author={Jaejun Lee and Chanyoung Chung and Joyce Jiyoung Whang},
  title={{I}n{G}ram: Inductive Knowledge Graph Embedding via Relation Graphs},
  year={2023},
  journal={arXiv preprint arXiv:2305.19987},
  doi = {10.48550/arXiv.2305.19987}
}
```

## Requirements

We used Python 3.8 and PyTorch 1.12.1 with cudatoolkit 11.3.

You can install all requirements with:

```shell
pip install -r requirements.txt
```

## Reproducing Results

We used NVIDIA RTX A6000, NVIDIA GeForce RTX 2080 Ti, and NVIDIA GeForce RTX 3090 for all our experiments.

We provide the checkpoints we used to produce the inductive link prediction results on 14 datasets.

You can download the checkpoints in https://drive.google.com/file/d/1aZrx2dYNPT7j4TGVBOGqHMdHRwFUBqx5/view?usp=sharing.

For usage, place the unzipped ckpt folder in the same directory with the codes.

The commands to reproduce the results in our paper:

```python
python3 test.py --best --data_name [dataset_name]
```

## Training from Scratch

To train InGram from scratch, run `train.py` with arguments.

Please tune our model in your machine in the range provided in Appendix C, because the best hyperparameters may be different due to randomness.

The list of arguments of `train.py` and their brief descriptions:
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

Please refer to `my_parser.py` for the examples of the arguments.
