# generate_yaml.py
# Author: Julie Kallini

# For importing utils
import sys
sys.path.append("..")

from jinja2 import Template
from utils import PERTURBATIONS, CHECKPOINT_WRITE_PATH, \
    PAREN_MODELS, PAREN_MODEL_PATH
import argparse
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Generate yaml for training',
        description='Generate train and dataset yaml configs for mistral training')
    parser.add_argument('perturbation_type',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=PERTURBATIONS.keys(),
                        help='Perturbation function used to transform BabyLM dataset')
    parser.add_argument('train_set',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=["100M", "10M"],
                        help='BabyLM train set')
    parser.add_argument('random_seed', type=int, help="Random seed")
    parser.add_argument('paren_model',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=list(PAREN_MODELS.keys()) + ["randinit"],
                        help='Parenthesis model')
    parser.add_argument('-np', '--no_pos_encodings', action='store_true',
                        help="Train GPT-2 with no positional encodings")

    # Get args
    args = parser.parse_args()
    if args.paren_model != "randinit":
        paren_model_path = PAREN_MODEL_PATH + PAREN_MODELS[args.paren_model] + "/checkpoint-5000"
    else:
        paren_model_path = "null"
    paren_model_name = args.paren_model
    no_pos_encodings_str = "-no-positional-encodings" if args.no_pos_encodings else ""
    no_pos_encodings_underscore = "_no_positional_encodings" if args.no_pos_encodings else ""

    # Create directory for yaml
    yaml_directory = f"conf/babylm_{args.perturbation_type}_{args.train_set}_{paren_model_name}{no_pos_encodings_underscore}/seed{args.random_seed}"
    if not os.path.exists(yaml_directory):
        os.makedirs(yaml_directory)

    print("Generating GPT-2 model yaml file...")

    # Get model template, which varies due to changes in vocab size
    model_temp_file = open("conf/template/gpt2-small-template.yaml")
    lines = model_temp_file.readlines()
    model_temp_file.close()

    # Fill model template
    tokenizer = PERTURBATIONS[args.perturbation_type]["gpt2_tokenizer"]
    vocab_size = len(tokenizer)
    model_template = Template("".join(lines))
    model_conf = model_template.render(
        perturbation=args.perturbation_type,
        vocab_size=vocab_size,
        paren_model=paren_model_name,
        paren_model_path=paren_model_path,
        no_pos_encodings=no_pos_encodings_str,
    )

    # Write model yaml to file
    model_file = open(
        f"conf/babylm_{args.perturbation_type}_{args.train_set}_{paren_model_name}{no_pos_encodings_underscore}/gpt2{no_pos_encodings_str}-small-{args.perturbation_type}-{paren_model_name}.yaml", "w")
    model_file.write(model_conf)
    model_file.close()

    print("Generating train yaml file...")

    # Get train template file
    train_temp_file = open("conf/template/babylm_train_template.yaml")
    lines = train_temp_file.readlines()
    train_temp_file.close()

    # Fill train template file
    train_template = Template("".join(lines))
    train_conf = train_template.render(
        perturbation=args.perturbation_type,
        seed=args.random_seed,
        ckpt_path=CHECKPOINT_WRITE_PATH,
        train_set=args.train_set,
        paren_model=paren_model_name,
        no_pos_encodings=no_pos_encodings_str,
        no_pos_encodings_underscore=no_pos_encodings_underscore,
    )

    # Write train yaml to file
    train_file = open(yaml_directory + \
        f"/train_{args.perturbation_type}_{args.train_set}_{paren_model_name}{no_pos_encodings_underscore}_seed{args.random_seed}.yaml", "w")
    train_file.write(train_conf)
    train_file.close()

    print("Generating dataset yaml file...")

    # Get dataset temp file
    dataset_temp_file = open("conf/template/babylm_dataset_template.yaml")
    lines = dataset_temp_file.readlines()
    dataset_temp_file.close()

    # Fill dataset template file
    dataset_template = Template("".join(lines))
    dataset_conf = dataset_template.render(
        perturbation=args.perturbation_type,
        train_set=args.train_set,
        seed=args.random_seed,
    )

    # Write dataset yaml to file
    dataset_file = open(yaml_directory + \
        f"/dataset_{args.perturbation_type}_{args.train_set}_seed{args.random_seed}.yaml", "w")
    dataset_file.write(dataset_conf)
    dataset_file.close()

    # Create directory for model checkpoints
    ckpt_directory = CHECKPOINT_WRITE_PATH + f"/babylm_{args.perturbation_type}_{args.train_set}_{paren_model_name}{no_pos_encodings_underscore}"
    if not os.path.exists(ckpt_directory):
        os.makedirs(ckpt_directory)