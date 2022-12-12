# RPNalgorithm
## Introduction
This repository contains implementations of the RPN and SapN algorithms based on the transformer libraries of pytorch and HuggingFace[HuggingFace's transformers](https://github.com/huggingface/transformers).It increases the generalization performance of the model by producing a limited perturbation of the word vector.RPN improves the performance of RoBERTa and TextCNN on various Natural Language Understanding tasks.

## Instructions for use
### Installation

```
pip install -r requirements.txt
```

### use

```
usage: python -m torch.distributed.launch --nproc_per_node [N] main.py [-h] [-tp TRAIN_DATA_PATH] [-vp VAL_DATA_PATH] [-tep TEST_DATA_PATH] [-g GPUS] [-e N] [-lr Ne-N] [-eps Ne-N] [-b N] [-m MODEL] [-md MODE] [-ns NOISE]
               [-sc SCALE] [-pr PROB] [-as N] [-tk TASKS_KINDS] [-nl NUM_LABELS] [--local_rank LOCAL_RANK]


optional arguments:
  -h, --help            show this help message and exit
  -tp TRAIN_DATA_PATH, --train_data_path TRAIN_DATA_PATH
                        which data path
  -vp VAL_DATA_PATH, --val_data_path VAL_DATA_PATH
                        which data path
  -tep TEST_DATA_PATH, --test_data_path TEST_DATA_PATH
                        which data path
  -g GPUS, --gpus GPUS  which gpu to have
  -e N, --epochs N      number of total epochs to run
  -lr Ne-N, --lr Ne-N   model value of learning rate
  -eps Ne-N, --eps Ne-N
                        model value of precision
  -b N, --batch_size N  number of batchsize
  -m MODEL, --model MODEL
                        which model use XLNet or Roberta
  -md MODE, --mode MODE
                        which mode use test or val
  -ns NOISE, --noise NOISE
                        which noise use else,None(FreeLB),SAP ,RPN or whole
  -sc SCALE, --scale SCALE
                        Variance of Gaussian distribution
  -pr PROB, --prob PROB
                        SAP and RPN selection probability
  -as N, --adv_step N   number of adv step
  -tk TASKS_KINDS, --tasks_kinds TASKS_KINDS
                        which tasks_kinds use Classification or MultipleChoice
  -nl NUM_LABELS, --num_labels NUM_LABELS
                        If you use Classification.Please input num_labels
```

For example, in the EP task using RoBERTa+RPN.

```
python -m torch.distributed.launch --nproc_per_node 2 main.py --train_data_path data/dataset_emoji_train_data.csv --val_data_path data/dataset_emoji_val_data.csv --test_data_path data/dataset_emoji_test_data.csv --model Roberta --gpus 2 --lr 3e-5 --epochs 3 --batch_size 512 --noise RPN --prob 0.3 --adv_step 3
```

