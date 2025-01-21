# GUIM

### Introduction

This code is the implementation of  GUIM

### Code Strcuture

```
.
├── GUIM
│   ├── config.py
│   ├── data
│   ├── dataset
│   ├── log
│   ├── main.py
│   ├── model
│   ├── parameter_parser.py
│   ├── result
│   ├── task
│   ├── unlearning
│   └── utils
└──  README.md
```

### Environment prepare

The code runs well under python 3.8.10. The required packages are as follows:

```
lightgbm==4.4.0
matplotlib==3.7.5
networkx==3.1
numba==0.58.1
numpy==1.24.4
ogb==1.3.6
PyYAML==6.0.1
PyYAML==6.0.1
scikit_learn==1.3.2
scipy==1.14.0
torch==2.2.1
torch_geometric==2.5.3
torch_scatter==2.1.2+pt22cu118
torch_sparse==0.6.18+pt22cu118
tqdm==4.66.4
```



### Quick Start

For regular unlearning task(e.g., Node Unlearning with ratio 0.1 on ogbn-arxiv using SGC with Budget ratio 0.05)

```
python main.py --dataset_name "ogbn-arxiv" --unlearning_methods "GUIM" --base_model "SGC" --unlearning_epochs 50 --Budget 0.05
```

