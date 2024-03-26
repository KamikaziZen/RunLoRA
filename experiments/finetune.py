from runlora.modeling import RunLoRAModel
from runlora import RunLoRACollection

from modeling_llama import LlamaForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, 
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    OPTForCausalLM,
    set_seed
)
from datasets import load_dataset
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
import math

from argparse import ArgumentParser
import gc
import os
import wandb
import logging
logging.basicConfig(level=logging.INFO, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args(args):
    parser = ArgumentParser()

    parser.add_argument('-m', '--model_name_or_config',
                        type=str,
                        required=True,
                        help='path to config file or name'
                        'of model available in transformers hub')
    parser.add_argument(
        "--dataset-name",
        help="The name of the dataset to use (via the datasets library).",
        required=True, 
        type=str
    )
    parser.add_argument(
        '--dataset-config-name',
        default=None, 
        help="The configuration name of the dataset to use (via the datasets library).",
        type=str
    )
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
    parser.add_argument("--group-texts", action='store_true',
                        help='concaternate texts up to maximum length')
    parser.add_argument('--block-size',
                        type=int,
                        help="input sequence length after tokenization."
                             "the training dataset will be truncated in block of this size for training.")
    parser.add_argument('-v', '--verbose', action='store_true') 

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--lora', action='store_true') 
    group.add_argument('--runlora', action='store_true') 
    
    parser.add_argument('-o', "--out", type=str, required=False,
                        help='prefix of output dir for trained models')
    parser.add_argument("--with-wandb",
                        action="store_true",
                        help="enable experiment logging to wandb")
    parser.add_argument('--random-seed', type=int, default=42)

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


def get_model_and_tokenizer(args):
    if os.path.exists(args.model_name_or_config):
        
        config = AutoConfig.from_pretrained(args.model_name_or_config)
        if 'llama' in args.model_name_or_config:
            # Llama with FlashAttention
            model = LlamaForCausalLM(config)
        elif 'opt' in args.model_name_or_config:
            # OPT with FalshAttention
            model = OPTForCausalLM(config, 
                                   torch_dtype=args.dtype,
                                   attn_implementation="flash_attention_2")
        else:
            model = AutoModelForCausalLM.from_config(config, torch_dtype=args.dtype)

        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        model.resize_token_embeddings(len(tokenizer))
        
    else:
        
        if 'opt' in args.model_name_or_config:
            # OPT with FlashAttention
            model = OPTForCausalLM.from_pretrained(
                args.model_name_or_config, 
                torch_dtype=args.dtype,
                attn_implementation="flash_attention_2",
                device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_config,
                torch_dtype=args.dtype,
                device_map='auto'
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

    # Model and adapters
    model, tokenizer = get_model_and_tokenizer(args)
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
                    # modify this part if optimal forward and backward functions are not the same for all layers
                    run_lora_mapping[module_name] = run_lora_collection[(args.runlora_fwd, args.runlora_bwd)]

            torch.manual_seed(args.random_seed)

            # this is a wrapper
            # model is changed, so we put model as argument to Trainer
            runlora_model = RunLoRAModel(model,
                             run_lora_mapping,
                             lora_r=args.lora_r,
                             lora_alpha=args.lora_alpha,
                             lora_dropout=args.lora_dropout,
                             target_modules=args.target_modules)

        elif args.lora:

            torch.manual_seed(args.random_seed)
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                # the eval metrics are not estimated without this parameter lol
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

    model = model.to(args.dtype)
    model = model.to(device)
    reset_memory()

    if args.verbose:
        logging.info(f'Allocated GPU Memory for Model: {torch.cuda.memory_allocated() / 2**20} MB')

    if lora_impl:
        
        if args.runlora: 
            runlora_model.prepare_for_finetuning()
        
        params = sum(p.numel() for p in model.parameters())
        trainable_params_adapters = \
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        if args.verbose:
            logging.info(f'Total params after {lora_impl} transform: {params}, '
                         f'Trainable params after {lora_impl} transform: {trainable_params_adapters}')
            logging.info(model)
    
        assert trainable_params_adapters < trainable_params, \
                f"Number of trainable params after {lora_impl} transform increased!"


    # # Do we need it? 
    # model.resize_token_embeddings(len(tokenizer))

    # Data
    datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    if "validation" not in datasets.keys():
        datasets["validation"] = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[:10%]",
        )
        datasets["train"] = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[10%:]",
        )

    if 'quotes' in args.dataset_name:
        text_column_name = 'quote'
    elif "text" in datasets['train'].column_names:
        text_column_name = 'text'
    else:
        text_column_name = datasets['train'].column_names[0]
    tokenized_datasets = datasets.map(
        lambda samples: tokenizer(samples[text_column_name]),
        remove_columns=datasets['train'].column_names,
        batched=True,
    )

    if not args.block_size:
        block_size = model.config.max_sequence_length \
            if hasattr(model.config, 'max_sequence_length') \
            else model.config.max_position_embeddings
    else:
        block_size = args.block_size
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    if args.group_texts:
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
        )

    # batch = tokenizer("Two things are infinite: ", return_tensors='pt').to(device)

    # with torch.no_grad():
    #     output_tokens = model.generate(**batch, max_new_tokens=50)

    # print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))

    for name, param in model.named_parameters():
        print(name, param.requires_grad, param.dtype)

    # Training
    training_args = {
        'num_train_epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'per_device_train_batch_size': args.batch_size,
        'per_device_eval_batch_size': args.batch_size,
        'bf16': True if args.dtype == torch.bfloat16 else False,
        'warmup_steps': 100,
        'logging_first_step': True,
        'logging_steps': 100,
        'evaluation_strategy': 'steps',
        'eval_steps': .3,
        'save_strategy': 'no',
        'seed': args.random_seed,
        'output_dir': 'outputs',
        # 'max_steps': 1 # for debugging
    }
    training_args = TrainingArguments(**training_args)
    if args.with_wandb:
        model_name = args.model_name_or_config.split('/')[-1].rstrip('.json')
        training_args.run_name=f"{model_name}.r{args.lora_r}.a{args.lora_alpha}.{lora_impl}.{args.dataset_name.split('/')[-1]}"
        training_args.report_to = ['wandb']
    else:
        training_args.report_to = []
    set_seed(training_args.seed)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Eval Before training ***")

        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(datasets["validation"])
        perplexity = math.exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        wandb.log({'eval/perplexity': metrics["perplexity"],
                   'eval/samples': metrics["eval_samples"]})

    # # Debugging
    # from runlora.modeling import RunLoRALinear
    # from functools import partial
    # def fhook(idx, module, input, output):
    #     print('forward through module:', module)
    #     for inp in input:
    #         if inp is not None:
    #             if isinstance(inp, int):
    #                 print('input: int')
    #             else:
    #                 print('input:', inp.shape, inp.dtype)
    #         else:
    #             print('input: None')
    #     if type(output) is tuple:
    #         for otp in output:
    #             if otp is not None:
    #                 if type(otp) is tuple:
    #                     print('output: tuple')
    #                 else:
    #                     print('output:', otp.shape, otp.dtype)
    #             else:
    #                 print('output: None')
    #     else:
    #         if 'shape' in dir(output):
    #             print('output:', output.shape, output.dtype)
    #         else:
    #             print('output:', type(output))
    #     print()
    # def bhook(idx, module, input, output):
    #     print('backward through module:', module)
    #     for inp in input:
    #         if inp is not None:
    #             print('input:', inp.shape, inp.dtype)
    #         else:
    #             print('None')
    #     if type(output) is tuple:
    #         for otp in output:
    #             if otp is not None:
    #                 print('output:', otp.shape, otp.dtype)
    #             else:
    #                 print('output: None')
    #     else:
    #         print('output:', output.shape, output.dtype)
    #     print()
    # i, j = 0 , 0
    # f_handles = []
    # b_handles = []
    # for name, m in model.named_modules():
    #     # if isinstance(m, (nn.Linear, RunLoRALinear)):
    #     handle = m.register_forward_hook(partial(fhook, i))
    #     i += 1
    #     f_handles.append(handle)
    #     # print(list(p[0] for p in m.named_parameters()))
    #     handle = m.register_backward_hook(partial(bhook, j))
    #     b_handles.append(handle)
    #     j += 1

    model.train()
    train_result = trainer.train(resume_from_checkpoint=None)

    wandb.config.update(args)
    wandb.config.update({'trainable_parameters': trainable_params_adapters})

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Eval After Training ***")

        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(datasets["validation"])
        perplexity = math.exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        wandb.log({'eval/perplexity': metrics["perplexity"],
                   'eval/samples': metrics["eval_samples"]})

    for name, param in model.named_parameters():
        print(name, param.requires_grad, param.dtype)

    # with torch.no_grad():
    #     output_tokens = model.generate(**batch, max_new_tokens=50)

    # print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
    
    # if args.out:
    #     os.makedirs(os.path.join(args.out, args.task_name), exist_ok=True)
    #     # trainer.save_model(os.path.join(args.out, args.task_name))  # Saves the tokenizer too for easy upload
    #     torch.save(model.state_dict((os.path.join(args.out, args.task_name)+f'.e{args.epochs}')))


if __name__ == "__main__":
    args = parse_args(None)
    main(args)
