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




def set_eval_parser(parser_group):
    eval_parser = parser_group.add_parser("eval")
    eval_parser.add_argument("--dataset-dir", type=str, required=True, default="",
                             help="dataset directory where train.src is located in")
    eval_parser.add_argument("--model-path", type=str, default="tapex.base/model.pt",
                             help="the directory of fine-tuned model path such as tapex.base.wikisql/model.pt")
    eval_parser.add_argument("--sub-dir", type=str, default="valid", choices=["train", "valid"],
                             help="the directory of pre-trained model path, and the default should be in"
                                  "{bart.base, bart.large, tapex.base, tapex.large}.")
    eval_parser.add_argument("--max-tokens", type=int, default=1536 * 4,
                             help="the max tokens can be larger than training when in inference.")
    eval_parser.add_argument("--predict-dir", type=str, default="predict",
                             help="the predict folder of generated result.")
    eval_parser.add_argument("--checkpoint-dir", type=str, default="predict",
                             help="the predict folder of all_result.txt.")
    


def evaluate(args):

    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)

    cmd = f"""
        fairseq-generate 
        --path {args.model_path} \
        {args.dataset_dir}/bin \
        --truncate-source \
        --gen-subset {args.sub_dir} \
        --max-tokens {args.max_tokens} \
        --nbest 1 \
        --source-lang src \
        --target-lang tgt \
        --results-path {args.predict_dir} \
        --beam 5 \
        --bpe gpt2 \
        --remove-bpe \
        --num-workers 20 \
        --skip-invalid-size-inputs-valid-test
    """
    sys.argv = shlex.split(cmd)
    logger.info("Begin to evaluate model on the {} subset of dataset {}".format(args.sub_dir, args.dataset_dir))
    logger.info("Running command {}".format(re.sub("\s+", " ", cmd.replace("\n", " "))))

    fairseq_generate()
    generate_file = os.path.join(args.predict_dir, "generate-{}.txt".format(args.sub_dir))
    evaluate_generate_file(generate_file, target_delimiter="|", name = args.model_path, checkpoint_dir=args.checkpoint_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    set_eval_parser(subparsers)

    args = parser.parse_args()
    evaluate(args)