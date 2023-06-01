
<!-- # ![](http://latex.codecogs.com/svg.latex?EG^3P): Explanation Graph Generation via Generative Pre-training over Synthetic Graphs -->
# $EG^3P$: Explanation Graph Generation via Generative Pre-training over Synthetic Graphs
The repository which contains the code and pre-trained models for our paper: Explanation Graph Generation via Generative Pre-training over Synthetic Graphs.(ACL2023-Findings)

## Overview



## Quickstart

### Prepara the environment

```
conda creative -n EG3P  python=3.8 
pip install -r requirement.txt
```




### Corpus Construction





### Training
* Process the data

    * DATASET_PATH: The path of the dataset path(.src/.tgt)
    * MODEL_DIR_PATH: The path of the model dir
```
bash process_dataset.sh DATASET_PATH MODEL_DIR_PATH
```

* Pretraining

    * DATASET_PATH: The path of the dataset path(processed)
    * MODEL_PATH : The path of the model file(xx.pt)


```
bash pretrain.sh DATASET_PATH MODEL_PATH
```



* Fine-tuning
    * TASK_NAME: The name of downstream task: ExplaGraphs / CSQA / OBQA
    * DATASET_PATH: The path of the dataset path(processed)
    * MODEL_NAME: The name of the pre-trained model. (only used for naming the checkpoint_dir)
    * MODEL_PATH: The path of the model file(xx.pt)
    

```
bash finetune.sh TASK_NAME DATASET_PATH MODEL_NAME MODEL_PATH 
```

## Evaluate

* 
