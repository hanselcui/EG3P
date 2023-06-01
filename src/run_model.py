import sys
import os
from argparse import ArgumentParser
from fairseq_cli.train import cli_main as fairseq_train
from fairseq_cli.generate import cli_main as fairseq_generate
import logging
import shlex
import re
from model_eval import evaluate_generate_file


logger = logging.getLogger(__name__)


def set_train_parser(parser_group):
    train_parser = parser_group.add_parser("train")
    train_parser.add_argument("--dataset-dir", type=str, required=True, default="",
                              help="dataset directory where train.src is located in")
    train_parser.add_argument("--exp-dir", type=str, default="checkpoints",
                              help="experiment directory which stores the checkpoint weights")
    train_parser.add_argument("--model-path", type=str, default="bart.base/model.pt",
                              help="the directory of pre-trained model path")
    train_parser.add_argument("--model-arch", type=str, default="bart_large", choices=["bart_large", "bart_base"])
    train_parser.add_argument("--max-tokens", type=int, default=1536,
                              help="if you train a large model on 16GB memory, max-tokens should be empirically "
                                   "set as 1536, and can be near-linearly increased according to your GPU memory.")
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--gradient-accumulation", type=int, default=8,
                              help="the accumulation steps to arrive a equal batch size on one card, and"
                                   " you can also reduce it to a proper value for you.")
    train_parser.add_argument("--total-num-update", type=int, default=50000,
                              help="the total optimization training steps")
    train_parser.add_argument("--learning-rate", type=float, default=3e-5,
                              help="the peak learning rate for model training")
    # train_parser.add_argument("--save-step", type=int, default=500)
    train_parser.add_argument("--save-interval", type=int, default=500)
    train_parser.add_argument("--valid-interval", type=int, default=500)
    train_parser.add_argument("--warmup-update", type=int, default=15000)
    train_parser.add_argument("--wandb", type=int, default=0)




def train(args):

    '''
        If the batch_size is set, ignore the setting of max_tokens.
    '''
    if args.batch_size is not None:
        args.max_tokens = None
    
    print(args)

    cmd = f"""
        fairseq-train {args.dataset_dir}/bin \
        --save-dir {args.exp_dir} \
        --restore-file {args.model_path} \
        --arch {args.model_arch}  \
        --memory-efficient-fp16	\
        --task translation  \
        --criterion label_smoothed_cross_entropy  \
        --source-lang src  \
        --target-lang tgt  \
        --truncate-source  \
        --label-smoothing 0.1  \
        --max-source-positions 1024 \
        --max-tokens {args.max_tokens}  \
        --update-freq {args.gradient_accumulation} \
        --max-update {args.total_num_update}  \
        --required-batch-size-multiple 1  \
        --dropout 0.1  \
        --attention-dropout 0.1  \
        --relu-dropout 0.0  \
        --weight-decay 0.01  \
        --optimizer adam  \
        --adam-eps 1e-08  \
        --clip-norm 0.1  \
        --lr-scheduler polynomial_decay  \
        --lr {args.learning_rate}  \
        --total-num-update {args.total_num_update}  \
        --warmup-updates {args.warmup_update}  \
        --ddp-backend no_c10d  \
        --num-workers 20  \
        --reset-meters  \
        --reset-optimizer \
        --reset-dataloader \
        --share-all-embeddings \
        --layernorm-embedding \
        --share-decoder-input-output-embed  \
        --skip-invalid-size-inputs-valid-test  \
        --log-format json  \
        --log-interval 10  \
        --save-interval-updates	{args.save_interval} \
        --validate-interval	{args.valid_interval} \
        --save-interval	{args.save_interval} \
        --patience 200 \
        --report-accuracy \

    """

    sys.argv = shlex.split(cmd)

    logger.info("Begin to train model for dataset {}".format(args.dataset_dir))
    logger.info("Running command {}".format(re.sub("\s+", " ", cmd.replace("\n", " "))))
    
    fairseq_train()


if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    set_train_parser(subparsers)

    args = parser.parse_args()

    train(args)