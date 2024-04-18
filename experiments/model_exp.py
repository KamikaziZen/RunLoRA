from runlora.modeling import RunLoRAModel
from runlora import RunLoRACollection
from modeling_llama import LlamaForCausalLM

from transformers import (
    AutoConfig, 
    AutoModelForCausalLM,
    OPTForCausalLM,
    BitsAndBytesConfig
)
import torch
import torch.utils.benchmark as benchmark
from peft import LoraConfig, get_peft_model

from argparse import ArgumentParser
import gc
import pandas as pd
import os
import warnings
import logging
logging.basicConfig(level=logging.INFO, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')


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
    parser.add_argument('--sequence-length',
                        type=int,
                        required=False,
                        help="sequence length of a batch, "
                             "defaults to model.config.max_sequence_length or model.config.max_position_embeddings")
    parser.add_argument("--criterions",
                        action="extend",
                        nargs="+", type=str,
                        help='criterions for best forward-backward'
                        'pair estimation')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--q4', action='store_true') 
    group.add_argument('--q8', action='store_true') 
    parser.add_argument('--min-run-time', type=float, default=10.,
                        help='min time in seconds for running consecutive'
                        'experiments in mean runtime estimation')
    parser.add_argument('--log-model-scheme', action='store_true',
                        help='log resulting model scheme with optimized LoraOperations')
    parser.add_argument('-v', '--verbose', action='store_true') 
    parser.add_argument('-o', "--out", type=str, required=False,
                        help='prefix of output file with test results')

    args = parser.parse_args(args)

    if args.verbose:
        logging.info(args)

    type_string = args.dtype
    if type_string in ['fp32', 'float32']:
        args.dtype = torch.float
    elif type_string in ['fp16', 'float16']:
        args.dtype = torch.half
    elif type_string in ['bf16', 'bfloat16']:
        if not torch.cuda.is_bf16_supported():
            raise ValueError('BFloat16 is not supported on your machine.')
        else:
            args.dtype = torch.bfloat16
    else:
        raise ValueError(f'{type_string} is not a supported dtype.')
        
    assert args.lora_r > 0, "LoRA rank must be positive"
    assert len(args.target_modules) > 0, 'target_modules is empty'
    if args.q4:
        bits = '.q4'
    elif args.q8:
        bits = '.q8'
    else:
        bits = ''
    if not args.out:
        args.out = \
            args.model_name_or_config.split('/')[-1].rstrip('.json') + \
            f"b{args.batch_size}s{args.sequence_length or 'max'}" \
            f"r{args.lora_r}.{type_string}{bits}"

    return args


def reset_memory(reset_stats=True):
    gc.collect()
    torch.cuda.empty_cache()
    if reset_stats:
        torch.cuda.reset_peak_memory_stats()


def get_model(args):
    if args.q4 or args.q8:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.q4,
            load_in_8bit=args.q8,
           # bnb_4bit_quant_type="nf4",
           # bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=args.dtype
        )
    else:
        quantization_config=None
    
    if os.path.exists(args.model_name_or_config):
        config = AutoConfig.from_pretrained(args.model_name_or_config)
        if 'llama' in args.model_name_or_config:
            warnings.warn('Llama from config does not support quantization. Use model class from huggingface hub.')
            # Llama with FlashAttention
            model = LlamaForCausalLM(config)
            model = model.to(args.dtype)
        elif 'opt' in args.model_name_or_config:
            # OPT with FalshAttention
            model = OPTForCausalLM(config, 
                                   torch_dtype=args.dtype,
                                   attn_implementation="flash_attention_2")
        else:
            model = AutoModelForCausalLM.from_config(config, torch_dtype=args.dtype)
    else:
        if 'opt' in args.model_name_or_config:
            # OPT with FlashAttention
            model = OPTForCausalLM.from_pretrained(args.model_name_or_config, 
                                                   torch_dtype=args.dtype,
                                                   attn_implementation="flash_attention_2",
                                                   quantization_config=quantization_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_config,
                torch_dtype=args.dtype,
                quantization_config=quantization_config
            )

    return model


def bench_model(model, args):
    # generating random tokens as input batch
    # TODO: put batch generation in setup?
    config = model.config
    max_sequence_length = config.max_sequence_length if hasattr(config, 'max_sequence_length') else config.max_position_embeddings
    if 'roberta' in config._name_or_path:
        # I have no idea, but it doesn't work otherwise
        max_sequence_length -= 2
    if args.sequence_length:
        if args.sequence_length > max_sequence_length:
            raise ValueError('Sequence length can not be larger than maximum sequence length of the model')
        else:
            max_sequence_length = args.sequence_length
    
    input_ids = torch.randint(low=0, high=config.vocab_size,
                              size=(args.batch_size, max_sequence_length))
    labels = input_ids.clone()

    reset_memory()

    bench = benchmark.Timer(
        stmt='model(input_ids, labels=labels).loss.backward()',
        # stmt='with torch.autocast(device_type="cuda"): model(input_ids, labels=labels).loss.backward()',
        # setup='reset_memory(reset_stats=False)',
        globals={'input_ids': input_ids, 'labels': labels, 'model': model})

    # warmup
    warmup_measure = bench.blocked_autorange(min_run_time=args.min_run_time)
    assert len(warmup_measure.times) >= 10, \
        'Number of measurements is less than 10, increase min_run_time!'
    
    reset_memory()
    max_mem_prev = torch.cuda.max_memory_allocated()
    max_res_prev = torch.cuda.max_memory_reserved()

    # benchmarking
    measure = bench.blocked_autorange(min_run_time=args.min_run_time)
    print("Computing mean with {} measurments, {} runs per measurment".format(
        len(measure.times), measure.number_per_run))

    max_mem = torch.cuda.max_memory_allocated()
    max_res = torch.cuda.max_memory_reserved()

    del input_ids, labels, bench
    reset_memory()

    print("Mean time: {} us".format(measure.mean * 1000000))
    print("Max Allocated Overhead: {} MB".format((max_mem - max_mem_prev) / 2**20))
    print("Max Reserved Overhead:{} MB".format((max_res - max_res_prev) / 2**20))
    print()

    return {'mean_time_us': measure.mean * 1000000,
            'max_mem_overhead_MB': (max_mem - max_mem_prev) / 2**20,
            'max_mem_res_overhead_MB': (max_res - max_res_prev) / 2**20,
            'msrs/runs': f'{len(measure.times)}/{measure.number_per_run}',
            'max_sequence_length': max_sequence_length,}


def main(args):
    rows = []

    device = torch.device("cuda")
    torch.set_default_device(device)

    model = get_model(args)

    # looking for the best lora operator for given shapes
    run_lora_collection = RunLoRACollection(min_run_time=args.min_run_time/2)
    run_lora_mapping = \
        run_lora_collection.optimize_for_model(
            model,
            n_batch=args.batch_size,
            lora_r=args.lora_r,
            target_modules=args.target_modules,
            criterions=args.criterions,
            quant=hasattr(model.config, 'quantization_config'))

    del model, run_lora_collection
    reset_memory()
    if args.verbose:
        logging.info(f'Allocated GPU Memory: {torch.cuda.memory_allocated() / 2**20} MB')

    # RunLoRA
    for criterion in args.criterions:

        # manage putting model to cuda only after replacing modules?
        model = get_model(args)
        params = sum(p.numel() for p in model.parameters())
        trainable_params = \
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        if args.verbose:
            logging.info(f'Total params before RunLoRA transform: {params}, '
                         f'Trainable params before RunLoRA transform: {trainable_params}')

        model = RunLoRAModel(model,
                             run_lora_mapping[criterion],
                             lora_r=args.lora_r,
                             lora_alpha=args.lora_alpha,
                             lora_dtype=args.dtype,
                             target_modules=args.target_modules)
        # this is already done by passing lora_dtype
        # model = model.to(args.dtype)
        # memory is not immediately cleaned after runlora transform
        reset_memory()
        # print(model)
        if args.verbose:
            logging.info(f'Allocated GPU Memory after loading the model: {torch.cuda.memory_allocated() / 2**20} MB')

        # Every parameter except for lora adapters is set to requires_grad=False
        model.prepare_for_finetuning()
        params = sum(p.numel() for p in model.parameters())
        trainable_params_runlora = \
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        if args.verbose:
            logging.info(f'Total params after RunLoRA transform: {params}, '
                         f'Trainable params after RunLoRA transform: {trainable_params_runlora}')

        if not(args.q4 or args.q8):
            assert trainable_params_runlora < trainable_params, \
                f"Number of trainable params after RunLoRA transform increased from {trainable_params} to {trainable_params_runlora}!"

        # for name, param in model.named_parameters():
        #     print(name, param.dtype, param.requires_grad)
        print(model)

        stats = bench_model(model, args)
        rows.append({'criterion': criterion,
                     **vars(args), **stats})

        # Logging model structure 
        if args.log_model_scheme:
            with open(f'{args.out}_{criterion}.scheme', 'w') as f:
                f.write(model.__repr__())

        del model
        reset_memory()

    del run_lora_mapping
    
    reset_memory()
    if args.verbose:
        logging.info(f'Allocated GPU Memory: {torch.cuda.memory_allocated() / 2**20}MB')

    # Vanilla LoRA
    model = get_model(args)
    params = sum(p.numel() for p in model.parameters())
    trainable_params = \
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.verbose:
        logging.info(f'Total params before LoRA transform: {params}, '
                     f'Trainable params before LoRA transform: {trainable_params}')

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model = model.to(args.dtype)
    
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad, param.dtype)
    # print(model)
        
    # memory is not immediately cleaned after peft transform
    reset_memory()
    if args.verbose:
        logging.info(f'Allocated GPU Memory after loading the model: {torch.cuda.max_memory_allocated() / 2**20} MB')

    params = sum(p.numel() for p in model.parameters())
    trainable_params_lora = \
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.verbose:
        logging.info(f'Total params after LoRA transform: {params}, '
                     f'Trainable params after LoRA transform: {trainable_params_lora}')

    if not (args.q4 or args.q8):
        assert trainable_params_lora < trainable_params, \
            f"Number of trainable params after LoRA transform increased from {trainable_params} to {trainable_params_lora}!"

    assert trainable_params_lora == trainable_params_runlora, \
        "Number of trainable params after LoRA and RunLoRA transforms do not match!"

    stats = bench_model(model, args)
    rows.append({**vars(args), **stats})

    del model
    if args.verbose:
        logging.info(f'Max GPU Memory Reserved: {torch.cuda.max_memory_reserved() / 2**20} MB')
    reset_memory()

    # Results
    df = pd.DataFrame.from_records(rows).drop(columns=['out', 'criterions'])
    df.sort_values(['mean_time_us', 'max_mem_overhead_MB'],
                   ascending=[True, True], inplace=True)
    df.to_csv(args.out+'.csv')
    print(df[['model_name_or_config',
              'criterion', 'mean_time_us',
              'max_mem_overhead_MB', 'msrs/runs']])


if __name__ == "__main__":
    args = parse_args(None)
    main(args)
