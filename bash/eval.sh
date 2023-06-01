
DATE=`date '+%m-%d'`

## The location of the checkpoint dir
CHECKPOINT_DIR_NAME="pretrain_${DATE}"

## The Path of the processed dataset
DATA_DIR=$1

## The Path of the model  (The file of model.pt)
MODEL_PATH=$2

## The Path of the generated file
PREDICT_DIR=$3




time=$(date "+%Y%m%d-%H%M%S")

echo "############################################"
echo "Time: ${time}"
echo "Mission: evaluation"
echo "Train Checkpoint dir: $CHECKPOINT_DIR_NAME"
echo "--------------------------------------------"
echo "Tips: $TIPS"
echo "############################################"

source ~/.bashrc

conda activate fs

nvidia-smi

python ../src/evaluate.py eval \
 --dataset-dir "${DATA_DIR}"\
 --model-path "${MODEL_PATH}"\
 --predict-dir "${PREDICT_DIR}"\
 --sub-dir valid \




