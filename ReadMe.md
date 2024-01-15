# 💥 Mission: Impossible Language Models 💥

This is the code repository for the paper "[Mission: Impossible Language Models](https://arxiv.org/abs/2401.06416)".

If you use our code, please cite our paper:

```
@misc{kallini2024mission,
      title={Mission: Impossible Language Models}, 
      author={Julie Kallini and Isabel Papadimitriou and Richard Futrell and Kyle Mahowald and Christopher Potts},
      year={2024},
      eprint={2401.06416},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

This repository contains the code necessary to fully replicate our paper, including the scripts to create impossible language datasets, train GPT-2 models, and run all experiments. We also include the notebooks to generate the result graphs in our paper.

Let's get started!

## Setup

First, clone the repo and install dependencies:

```
git clone https://github.com/jkallini/mission-impossible-language-models.git
cd mission-impossible-language-models
pip install -r requirements.txt
```

## Impossible Language Dataset Creation

The scripts for creating impossible language datasets are located in the `data/` directory.
First, you must download a copy of the [BabyLM dataset](https://babylm.github.io/), which we use for our experiments. 
Then, make sure to set `BABYLM_DATA_PATH` in the `utils.py` file to the path on your system where your BabyLM dataset is located.

After downloading the BabyLM dataset, you will need to tag it with morphological features and part-of-speech tags. You can use our `tag.py` script.

With the tagged data, you can easily recreate one of the impossible language datasets described in our paper. These are predefined and listed in the `PERTURBATIONS` section at the end of `utils.py`. Here is an
example for the PartialReverse language from the paper:

```
python3 perturb.py reverse_partial 100M
```

This will create a perturbed version of the 100M BabyLM train set. You may use `perturb.py` or `perturb.sh` to perturb multiple splits at the same time.

### Defining New Impossible Languages

You can also define your own impossible languages! They are described by four attributes:

1. `perturbation_function`: function mapping tagged sentences to sequences of GPT-2 tokens.
2. `affect_function`: function that determines whether an input sentences is "affected" or altered by the perturbation.
3. `filter_function`: function that determines whether an input sentence should be included in the final dataset.
4. `gpt2_tokenizer`: tokenizer used to perturb this dataset.

You can add these definitions to `utils.py`, where the existing perturbations are located.


## Model Training

To train GPT-2 models, we use [`mistral`](https://github.com/stanford-crfm/mistral). If you would like to train GPT-2s with `mistral` as well, please follow their steps for installation. You may download their repo anywhere on your system.

Next, make sure to change the following constants in `utils.py`:
- `CHECKPOINT_WRITE_PATH`: the path where your training checkpoints will be written.
- `CHECKPOINT_READ_PATH`: the path where you will read training checkpoints when running experiments.

Our training scripts are in the `training/` directory.
Once you have `mistral` installed, set `MISTRAL_PATH` to the path of your library in `prepare_training.sh`. Then, you can use this script to generate the config files that you will use to launch `mistral` training runs.

Our scripts will create the config files and move them to the location of your `mistral` directory—you will only need to launch the training run. Here's an example command to launch training for the PartialReverse language using the 100M training set with the random seed set to 41:

```
CUDA_VISIBLE_DEVICES=0 python3 train.py --config conf/train_reverse_partial_100M_randinit_seed41.yaml --nnodes 1 --nproc_per_node 1 --training_arguments.fp16 true --training_arguments.warmup_steps 300 --training_arguments.max_steps 3000
```

## Experiments

The main paper includes three experiments: perplexities, surprisals, and causal interventions. The appendix also includes a constituency probing experiment.

The scripts to run each of these experiments is separated into their own subdirectories:

1. `perplexities/`: code to run perplexity experiments. You may use `perplexities.py` or `perplexities.sh` to run experiments for multiple languages at the same time.
2. `hop_surprisal/`: code to run surprisal experiments for the *Hop languages, in `hop_surprisal.py`.
3. `hop_interventions/`: code to run interchange intervention experiments for the *Hop languages. First generate the agreement data using `create_agreement_data.py`, then run the intervention experiments using `hop_interventions.py`.
You will need to separately clone and install [`align-transformers`](https://github.com/frankaging/align-transformers) (recently renamed to `pyvene`) and set `PATH_TO_ALIGN_TRANSFORMERS` to the path where the library is located on your system.
4. `edge_probing/`: code to run constituency probing experiments. Use `get_constituency_parses.py` and `load_phrase_data.py` to prepare the test data, and use `edge_probing.py` to run the experiments.

Each directory contains python notebooks to generate the result graphs shown in the paper.