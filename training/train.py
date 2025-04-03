# train.py
# Author: Julie Kallini

import sys
sys.path.append('..')

import os
import argparse
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from models.modeling_gpt2 import GPT2LMHeadModel
from models.configuration_gpt2 import GPT2Config
from babylm_dataset import BabyLMCorpus
from utils import PERTURBATIONS, CHECKPOINT_PATH
from transformers.trainer_utils import get_last_checkpoint

os.environ["WANDB_PROJECT"] = "mission-impossible"

def main(args):

    # Load the custom BabyLMCorpus dataset
    builder = BabyLMCorpus(config_name=f"babylm_{args.perturbation_type}_{args.train_set}")
    builder.download_and_prepare()
    dataset = builder.as_dataset()

    # Initialize the tokenizer
    tokenizer = PERTURBATIONS[args.perturbation_type]["gpt2_tokenizer"]
    tokenizer.pad_token = tokenizer.eos_token

    # Configure the GPT-2 model
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.max_length,   
        n_ctx=args.max_length,
        n_embd=768,
        n_layer=12,
        n_head=12,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        has_positional_encodings=not args.no_pos_encodings,
        geometric_attention=args.geometric_attention,
        alibi=args.alibi,
        rope=args.rope,
        _attn_implementation_autoset=False,
        _attn_implementation="eager",
    )

    print("Perturbation type:", args.perturbation_type)
    print("Training set:", args.train_set)
    print("Run name:", args.run_name)
    print("Seed:", args.random_seed)
    print(config)

    # Initialize the GPT-2 model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel(config).to(device)

    # Specify the run name and checkpoint + logging directory
    run_name = args.run_name if args.run_name is not None else f"test"
    result_dir = f"{CHECKPOINT_PATH}/{args.perturbation_type}_{args.train_set}/{run_name}_seed{args.random_seed}"
    checkpoint_dir = f"{result_dir}/checkpoints"
    logging_dir = f"{result_dir}/logs"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    # Calculate appropriate gradient accumulation steps for effective batch size
    gradient_accumulation_steps = args.effective_batch_size // args.per_device_train_batch_size
    print(f"Effective batch size: {args.effective_batch_size}")
    print(f"Per-device batch size: {args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        overwrite_output_dir=True,
        do_train=True,
        eval_strategy="steps",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=False,
        prediction_loss_only=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        lr_scheduler_type="linear",
        warmup_steps=args.warmup_steps,
        run_name=run_name,
        logging_dir=logging_dir,
        logging_first_step=True,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        ignore_data_skip=False,
        seed=args.random_seed,
        fp16=True,
        report_to="wandb" if not args.disable_wandb else [],
    )

    # Data collator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    train_dataset = dataset["train"].shuffle(seed=args.random_seed)
    eval_dataset = dataset["validation"].shuffle(seed=args.random_seed)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    resume_from_checkpoint = False
    if args.resume_from_checkpoint:
        last_checkpoint = get_last_checkpoint(checkpoint_dir)
        if last_checkpoint is not None:
            print(f"Resuming from checkpoint: {last_checkpoint}")
            resume_from_checkpoint = last_checkpoint
        else:
            print("No checkpoint found. Training from scratch.")

    # Start training
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT-2 model from scratch on a custom dataset")

    # BabyLM dataset arguments
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
    
    # GPT-2 architecture arguments
    parser.add_argument('--no_pos_encodings', action='store_true', help="No positional encodings")
    parser.add_argument('--geometric_attention', action='store_true', help="Use geometric attention")
    parser.add_argument('--alibi', action='store_true', help="Use alibi position biases")
    parser.add_argument('--rope', action='store_true', help="Use RoPE (Rotary Positional Embeddings)")

    # Training arguments
    parser.add_argument('--run_name', type=str, default="run_name", help="Run name")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per GPU/TPU core for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size per GPU/TPU core for evaluation")
    parser.add_argument("--effective_batch_size", type=int, default=512, help="Effective batch size for training")
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta 1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta 2")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--max_steps", type=int, default=3000, help="Maximum number of training steps")
    parser.add_argument("--warmup_steps", type=int, default=300, help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Steps interval for logging")
    parser.add_argument("--eval_steps", type=int, default=500, help="Steps interval for evaluation")
    parser.add_argument("--save_steps", type=int, default=100, help="Steps interval for saving checkpoints")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument('--random_seed', type=int, help="Random seed", default=42)
    parser.add_argument('--resume_from_checkpoint', action='store_true', help="Resume training from checkpoint")

    args = parser.parse_args()
    main(args)