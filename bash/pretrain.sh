
DATE=`date '+%m-%d'`

## The location of the checkpoint dir
CHECKPOINT_DIR_NAME="pretrain_${DATE}"

## The Path of the processed dataset
DATA_DIR=$1

## The Path of the model  (The file of model.pt)
MODEL_PATH=$2




time=$(date "+%Y%m%d-%H%M%S")

echo "############################################"
echo "Time: ${time}"
echo "Mission: pretrain"
echo "Train Checkpoint dir: $CHECKPOINT_DIR_NAME"
echo "--------------------------------------------"
echo "Tips: $TIPS"
echo "############################################"

source ~/.bashrc

conda activate fs

nvidia-smi

python ../src/run_model.py 
 --dataset-dir "${DATA_DIR}"\
 --exp-dir ../checkpoints/"${CHECKPOINT_DIR_NAME}"\
 --model-arch bart_large\
 --model-path "${MODEL_PATH}"\
 --max-tokens 1536\
 --valid-interval 1000\
 --gradient-accumulation 4\
 --save-interval 10000\
 --warmup-update 7500\
 --total-num-update 50000




