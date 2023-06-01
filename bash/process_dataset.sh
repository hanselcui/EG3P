
source ~/.bashrc
conda activate EG3P



## Tthe Path of the dataset (xx.src, xx.tgt)
DATASET_PATH=$1
## The Path of the model
MODEL_PATH=$2



echo "dealing ${DATASET_PATH}"
echo "${DATASET_PATH}" 
echo "${MODEL_PATH}"

python ../src/bpe_encoder.py \
            --encoder-json $MODEL_PATH/encoder.json \
            --vocab-bpe $MODEL_PATH/vocab.bpe \
            --inputs $DATASET_PATH/train.src \
            --outputs $DATASET_PATH/train.bpe.src \
            --workers 48 \
            --keep-empty

python ../src/bpe_encoder.py \
            --encoder-json $MODEL_PATH/encoder.json \
            --vocab-bpe $MODEL_PATH/vocab.bpe \
            --inputs $DATASET_PATH/train.tgt \
            --outputs $DATASET_PATH/train.bpe.tgt \
            --workers 48 \
            --keep-empty
echo "train set deal"

python ../src/bpe_encoder.py \
            --encoder-json $MODEL_PATH/encoder.json \
            --vocab-bpe $MODEL_PATH/vocab.bpe \
            --inputs $DATASET_PATH/valid.src \
            --outputs $DATASET_PATH/valid.bpe.src \
            --workers 48 \
            --keep-empty

python ../src/bpe_encoder.py \
            --encoder-json $MODEL_PATH/encoder.json \
            --vocab-bpe $MODEL_PATH/vocab.bpe \
            --inputs $DATASET_PATH/valid.tgt \
            --outputs $DATASET_PATH/valid.bpe.tgt \
            --workers 48 \
            --keep-empty
echo "valid set deal"

python ../src/bpe_encoder.py \
            --encoder-json $MODEL_PATH/encoder.json \
            --vocab-bpe $MODEL_PATH/vocab.bpe \
            --inputs $DATASET_PATH/test.src \
            --outputs $DATASET_PATH/test.bpe.src \
            --workers 48 \
            --keep-empty

python ../src/bpe_encoder.py \
            --encoder-json $MODEL_PATH/encoder.json \
            --vocab-bpe $MODEL_PATH/vocab.bpe \
            --inputs $DATASET_PATH/test.tgt \
            --outputs $DATASET_PATH/test.bpe.tgt \
            --workers 48 \
            --keep-empty
echo "test set deal"


fairseq-preprocess --source-lang "src" --target-lang "tgt" \
    --trainpref $DATASET_PATH/train.bpe \
    --validpref $DATASET_PATH/valid.bpe \
    --testpref $DATASET_PATH/test.bpe \
    --destdir $DATASET_PATH/bin \
    --workers 48 \
    --srcdict $MODEL_PATH/dict.txt \
    --tgtdict $MODEL_PATH/dict.txt


echo "process finished"
