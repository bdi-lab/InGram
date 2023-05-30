# InGram: Inductive Knowledge Graph Embedding via Relation Graphs

This code is the official implementation of the paper, "InGram: Inductive Knowledge Graph Embedding via Relation Graphs (ICML 2023)".

Codes written by Jaejun Lee (jjlee98@kaist.ac.kr).

If you use this code or data, please cite our paper.

The bibtex is provided at the end of this file.

> Jaejun Lee, Chanyoung Chung and Joyce Jiyoung Whang, InGram: Inductive Knowledge Graph Embedding via Relation Graphs, To appear in 40th International Conference on Machine Learning (ICML 2023), 2023.

## Requirements

We used Python 3.8 and PyTorch 1.12.1 with cudatoolkit 11.3.

You can install all requirements with:

```shell
pip install -r requirements.txt
```

## Reproducing Results

We used NVIDIA RTX A6000, NVIDIA GeForce RTX 2080 Ti and NVIDIA GeForce RTX 3090 for all our experiments.

We provide checkpoints for the inductive link prediction results in Table 1 and 3.

You can download the checkpoints in https://drive.google.com/file/d/1aZrx2dYNPT7j4TGVBOGqHMdHRwFUBqx5/view?usp=sharing

For usage, place the unzipped checkpoint folder in the same directory with the codes.

The commands we used to get the results in our paper using the provided checkpoints:

```python
python3 test.py --best --data_name [dataset_name]
```

We provide the checkpoints for all datasets in our paper.


## Training new datasets

To train InGram on new datasets, run `train.py` with arguments.

We suggest you to use the provided checkpoints, or to tune our model from scratch in your machine in the range provided in our paper's Appendix for further usage, because the best hyperparameters may be different due to randomness.

The list of arguments and their brief descriptions:\
--data_name: name of the dataset. Ex. NL-100, WK-75\
--exp: experiment name\
-m, --margin: $\gamma$\
-lr, --learning_rate: learning rate. Ex. 1e-3\
-nle, --num_layer_ent: $\widehat{L}$\
-nlr, --num_layer_rel: $L$\
-d_e, --dimension_entity: $\widehat{d}$ \
-d_r, --dimension_relation: $d$\
-hdr_e, --hidden_dimension_ratio_entity: $\widehat{d'}/\widehat{d}$\
-hdr_r, --hidden_dimension_ratio_relation: $d'/d$\
-b, --num_bin: $B$\
-e, --num_epoch: number of epochs to run\
--target_epoch: the epoch to run test on (only used for test.py)\
-v, --validation_epoch: duration for the validation\
--num_head: $\widehat{K}=K$\
--num_neg: number of negative triplets per triplet\
--best: use the provided checkpoints (only used for test.py)\
--no_write: don't save the checkpoints (only used for train.py)

Refer to our paper for notations.

## Citation

If you use this code or data, please cite our paper.

```bibtex
@inproceedings{ingram,
  author={Jaejun Lee and Chanyoung Chung and Joyce Jiyoung Whang},
  title={InGram: Inductive Knowledge Graph Embedding via Relation Graphs},
  year={2023},
  booktitle={arXiv preprint},
  doi = {},
  pages={}
}
```
