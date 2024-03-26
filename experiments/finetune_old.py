from runlora.modeling import RunLoRAModel
from runlora import RunLoRACollection
from modeling_llama import LlamaForSequenceClassification

from transformers import (
    AutoConfig, 
    AutoModelForCausalLM,
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    EvalPrediction,
    PretrainedConfig,
    default_data_collator,
    Trainer,
    TrainingArguments,
    set_seed
)
from datasets import load_dataset
import evaluate
import torch
import torch.utils.benchmark as benchmark
import torch.nn as nn
from peft import LoraConfig, get_peft_model
import numpy as np

from argparse import ArgumentParser
import gc
import pandas as pd
import os
import logging
logging.basicConfig(level=logging.INFO, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args(args):
    parser = ArgumentParser()

    parser.add_argument('-m', '--model_name_or_config',
                        type=str,
                        required=True,
                        help='path to config file or name'
                        'of model available in transformers hub')
    parser.add_argument('--batch-size', type=int, default=10, help='batch size')
    parser.add_argument('-r', '--lora-r', type=int, default=8,
                        help='rank of LoRA adapter')
    parser.add_argument('-a', '--lora-alpha', type=int, default=8,
                        help='LoRA scaling factor')
    parser.add_argument('-d', '--lora-dropout', type=float, default=0.,
                        help='dropout applied to LoRA adapter input')
    parser.add_argument('--dtype', type=str, default='fp32',
                        help='dtype of parameters and activations')
    parser.add_argument("--target-modules",
                        action="extend",
                        nargs="+", type=str,
                        help='list of modules eligible for LoRA adapters')
    parser.add_argument('--runlora-fwd', default='forward2', help='forward operator for runlora operation'
                        '(best choice for your model can be determined with model_exp.py script)')
    parser.add_argument('--runlora-bwd', default='backward5', help='backward operator for runlora operation'
                        '(best choice for your model can be determined with model_exp.py script)')
    parser.add_argument("--learning-rate",
                        type=float,
                        default=2e-5)
    parser.add_argument("--epochs", 
                        type=int, 
                        default=5, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--task_name",
                        type=str,
                        default=None,
                        help="The name of the glue task to train on.",
                        choices=list(task_to_keys.keys()))
    parser.add_argument('-v', '--verbose', action='store_true') 

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--lora', action='store_true') 
    group.add_argument('--runlora', action='store_true') 
    
    parser.add_argument('-o', "--out", type=str, required=False,
                        help='prefix of output dir for trained models')
    parser.add_argument("--with-wandb",
                        action="store_true",
                        help="enable experiment logging to wandb")

    args = parser.parse_args(args)

    if args.verbose:
        logging.info(args)

    if args.dtype in ['fp32', 'float32']:
        args.dtype = torch.float
    elif args.dtype in ['fp16', 'float16']:
        args.dtype = torch.half
    elif args.dtype in ['bf16', 'bfloat16']:
        if not torch.cuda.is_bf16_supported():
            raise ValueError('BFloat16 is not supported on your machine.')
        else:
            args.dtype = torch.bfloat16
    else:
        raise ValueError(f'{args.dtype} is not a supported dtype.')
    assert args.lora_r > 0, "LoRA rank must be positive"
    assert len(args.target_modules) > 0, 'target_modules is empty'
    if not args.out:
        args.out = \
            args.model_name_or_config.split('/')[-1].split('.')[0] + \
            f'b{args.batch_size}r{args.lora_r}'

    return args


def reset_memory(reset_stats=True):
    gc.collect()
    torch.cuda.empty_cache()
    if reset_stats:
        torch.cuda.reset_peak_memory_stats()


def get_model_and_tokenizer(args, num_labels):
    if os.path.exists(args.model_name_or_config):
        config = AutoConfig.from_pretrained(args.model_name_or_config, finetuning_task=args.task_name, num_labels=num_labels)
        if 'llama' in args.model_name_or_config:
            # Llama with FlashAttention
            model = LlamaForSequenceClassification(config)
        elif 'opt' in args.model_name_or_config:
            model = AutoModelForSequenceClassification(config=config,
                                                       attn_implementation="flash_attention_2")
        else:
            # some models do not support flash attention
            model = AutoModelForSequenceClassification(config=config)
        # TODO: add llama tokenizer like in relora
        tokenizer = None
    else:
        config = AutoConfig.from_pretrained(args.model_name_or_config, finetuning_task=args.task_name, num_labels=num_labels)
        if 'opt' in args.model_name_or_config:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_config,
                config=config,
                attn_implementation="flash_attention_2"
                # TODO: enable/disable cache?
                # takes more time but requires less memory
                # use_cache=False,
            )
        else:
            # some models don't support flash attention
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_config,
                config=config,
                # TODO: enable/disable cache?
                # takes more time but requires less memory
                # use_cache=False,
            )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_config)

    return model, tokenizer

def main(args):
    if args.runlora:
        lora_impl = 'RunLoRA'
    elif args.lora:
        lora_impl = 'LoRA'
    else:
        lora_impl = None
        logging.info('No low-rank adapter implementation is specified. Training of all parameters is performed.')
    
    device = torch.device("cuda")
    # torch.set_default_device(device)

    dataset = load_dataset("glue", args.task_name)    
    is_regression = args.task_name == "stsb"
    if not is_regression:
        label_list = dataset["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    model, tokenizer = get_model_and_tokenizer(args, num_labels=num_labels)
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if lora_impl:

        if args.verbose:
            logging.info(f'Total params before {lora_impl} transform: {params}, '
                         f'Trainable params before {lora_impl} transform: {trainable_params}')

        if args.runlora:
    
            run_lora_mapping = {}
            run_lora_collection = RunLoRACollection()
            for module_name, module in model.named_modules():
                if isinstance(module, nn.Linear) and any(trgt in module_name for trgt in args.target_modules):
                    run_lora_mapping[module_name] =run_lora_collection[(args.runlora_fwd, args.runlora_bwd)]

            # this is a wrapper
            # model is changed, so we put model as argument to Trainer
            runlora_model = RunLoRAModel(model,
                             run_lora_mapping,
                             lora_r=args.lora_r,
                             lora_alpha=args.lora_alpha,
                             target_modules=args.target_modules)

        elif args.lora:

            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                # the eval metrics are not estimated without this parameter lol
                task_type="SEQ_CLS",
            )
            model = get_peft_model(model, lora_config)


    model = model.to(device)
    model = model.to(args.dtype)
    reset_memory()
    if args.verbose:
        logging.info(f'Allocated GPU Memory for Model: {torch.cuda.memory_allocated() / 2**20} MB')

    if lora_impl:
        
        if args.runlora: 
            runlora_model.prepare_for_finetuning()

        # peft.lora and runlora disables gradients for all parameters except for adapters
        # enable training of last layer for downstream tasks like glue
        for name, param in model.named_parameters():
            if 'classifier' in name or 'score' in name:
                param.requires_grad = True

        for name, param in model.named_parameters():
            print(name, param.requires_grad, param.dtype)
        
        params = sum(p.numel() for p in model.parameters())
        trainable_params_runlora = \
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        if args.verbose:
            logging.info(f'Total params after {lora_impl} transform: {params}, '
                         f'Trainable params after {lora_impl} transform: {trainable_params_runlora}')
            logging.info(model)
    
        assert trainable_params_runlora < trainable_params, \
                f"Number of trainable params after {lora_impl} transform increased!"

    # Training
    model.train()
    training_args = {
        'num_train_epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'per_device_train_batch_size': args.batch_size,
        'per_device_eval_batch_size': args.batch_size,
        # 'bf16': True if args.dtype == torch.bfloat16 else False,
        'bf16': True,
        'logging_first_step': True,
        'logging_steps': 500,
        'evaluation_strategy': 'steps',
        'eval_steps': 500,
        'save_strategy': 'no',
        'seed': 42
    }
    training_args = TrainingArguments(f"./tmp/{args.task_name}", **training_args)
    if args.with_wandb:
        model_name = args.model_name_or_config.split('/')[-1].rstrip('.json')
        training_args.run_name=f"{model_name}.{args.task_name}.{lora_impl}"
        training_args.report_to = ['wandb']
    else:
        training_args.report_to = []
    set_seed(training_args.seed)

    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    label_to_id = None
    if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id and not is_regression:
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            print(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    max_seq_length = 128
    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding="max_length", max_length=max_seq_length, truncation=True)

        return result

    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on dataset",
    )
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
    test_dataset = tokenized_datasets["test_matched" if args.task_name == "mnli" else "test"]

    metric = evaluate.load("glue", args.task_name)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator
    )

    train_result = trainer.train(resume_from_checkpoint=None)

    for name, param in model.named_parameters():
        print(name, param.requires_grad, param.dtype)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    logging.info(f"Train metrics: {metrics}")

    tasks = [args.task_name]
    eval_datasets = [eval_dataset]
    if args.task_name == "mnli":
        tasks.append("mnli-mm")
        valid_mm_dataset = tokenized_datasets["validation_mismatched"]
        eval_datasets.append(valid_mm_dataset)
        combined = {}

    for eval_dataset, task in zip(eval_datasets, tasks):
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)

        if task == "mnli-mm":
            metrics = {k + "_mm": v for k, v in metrics.items()}
        if task is not None and "mnli" in task:
            combined.update(metrics)

        logging.info(f"Eval Metrics: {metrics}")
        if task == 'mnli': 
            logging.info(f"Eval Metrics: {combined}")

    # if args.out:
    #     os.makedirs(os.path.join(args.out, args.task_name), exist_ok=True)
    #     # trainer.save_model(os.path.join(args.out, args.task_name))  # Saves the tokenizer too for easy upload
    #     torch.save(model.state_dict((os.path.join(args.out, args.task_name)+f'.e{args.epochs}')))


if __name__ == "__main__":
    args = parse_args(None)
    main(args)
