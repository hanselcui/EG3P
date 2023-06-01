
DATE=`date '+%m-%d'`
TASK_NAME=$1       # 命名checkpoint_dir/subTask
DATA_DIR=$2
MODEL_NAME=$3     # 命名checkpoint_dir/subTask
MODEL_PATH=$4



## --------------------- 以下不需要修改 ------------------------- ##

source ~/.bashrc

conda activate EG3P

CHECKPOINT_DIR="${TASK_NAME}_${MODEL_NAME}_${DATE}"

mkdir "../checkpoints/$CHECKPOINT_DIR"

time=$(date "+%Y%m%d-%H%M%S")

echo " "
echo "####################################################################################################################################"
echo "--------------------------------------------"
echo "Time: ${time}"
echo "Mission: $TASK_NAME"
echo "Train Checkpoint dir: $CHECKPOINT_DIR"
echo "Model path: $MODEL_PATH"
echo "--------------------------------------------"
echo "Tips: $TIPS"
echo "####################################################################################################################################"
echo " "

nvidia-smi



python ../src/run_model.py \
 --dataset-dir "$DATA_DIR"\
 --exp-dir ../checkpoints/"$CHECKPOINT_DIR"\
 --model-arch bart_large\
 --model-path "$MODEL_PATH"\
 --batch-size 16\
 --valid-interval 20\
 --save-interval 200\
 --warmup-update 1500\
 --total-num-update 10000

