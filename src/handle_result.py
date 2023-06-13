import pandas as pd
import os
from argparse import ArgumentParser


def set_path_parser(parser_group):
    path_parser = parser_group.add_parser('handle_result')
    path_parser.add_argument("--predict-path", type=str, required=True, default='',
                            help="predict folder where the result of output is in")
    path_parser.add_argument("--output-path", type=str, required=True, default='',
                            help="output folder after handling the predict")

def handle_result(args):
    predict_dir = args.predict_path + "/generate-valid.txt_model.pt.csv"
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    df = pd.read_csv(predict_dir, sep='\t', usecols=[1])
    output_dir = args.output_path + '/predict.txt_model.pt.csv'
    df.to_csv(output_dir, sep='\t', index=False)
    


if __name__ == "__main__":
    parser = ArgumentParser()
    subparser = parser.add_subparsers(dest="subcommand")
    set_path_parser(subparser)

    args = parser.parse_args()
    handle_result(args)

