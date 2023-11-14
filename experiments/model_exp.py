from lightlora.modeling import LightLoRAModel
from transformers import AutoConfig, AutoModelForCausalLM
from lightlora import LightLoRACollection
import torch
import torch.utils.benchmark as benchmark
from peft import LoraConfig, get_peft_model
from argparse import ArgumentParser
import gc
import pandas as pd


def parse_args(args):
    parser = ArgumentParser()

    parser.add_argument('-m', '--model_name_or_config',
                        type=str,
                        required=True)
    parser.add_argument('--n_batch', type=int, default=10)
    parser.add_argument('-r', '--lora_r', type=int, default=8)
    parser.add_argument('-a', '--lora_alpha', type=int, default=8)
    parser.add_argument('-d', '--lora_dropout', type=float, default=0.)
    parser.add_argument("--target_modules",
                        action="extend",
                        nargs="+", type=str)
    parser.add_argument("--criterions",
                        action="extend",
                        nargs="+", type=str)
    parser.add_argument('--min_run_time', type=float, default=10.)
    parser.add_argument('-o', "--out", type=str, required=False)

    args = parser.parse_args(args)

    print(args)
    assert args.lora_r > 0, "LoRA rank must be positive"
    assert len(args.target_modules) > 0, 'target_modules is empty'
    if not args.out:
        args.out = \
            args.model_name_or_config.split('/')[-1].split('.')[0] + \
            f'b{args.n_batch}_r{args.lora_r}'

    return args


def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def bench_model(model, config, args):
    # generating random tokens as input batch
    # TODO: put batch generation in setup?
    input_ids = torch.randint(low=0, high=config.vocab_size, 
                              size=(args.n_batch, config.max_sequence_length))
    labels = input_ids.clone()

    reset_memory()
    max_mem_prev = torch.cuda.max_memory_allocated()

    bench = benchmark.Timer(
        stmt='model(input_ids, labels=labels).loss.backward()',
        globals={'input_ids': input_ids, 'labels': labels, 'model': model})

    # warmup
    warmup_mesure = bench.blocked_autorange(min_run_time=10.0)
    assert len(warmup_mesure.times) >= 1, \
        'Number of measurements for warmup is less than 1, increase min_run_time!'

    # benchmarking
    measure = bench.blocked_autorange(min_run_time=args.min_run_time)
    print("Computing mean with {} measurments, {} runs per measurment".format(
        len(measure.times), measure.number_per_run))
    assert len(measure.times) >= 10, \
        'Number of measurements is less than 10, increase min_run_time!'

    max_mem = torch.cuda.max_memory_allocated()

    del input_ids, labels, bench
    reset_memory()

    print("Mean time: {} us".format(measure.mean * 1000000))
    print("Max mem overhead: {} MB".format((max_mem - max_mem_prev) / 2**20))
    print()

    return {'mean_time_us': measure.mean * 1000000,
            'max_mem_overhead_MB': (max_mem - max_mem_prev) / 2**20,
            'msrs/runs': f'{len(measure.times)}/{measure.number_per_run}'}


def main(args):
    rows = []

    device = torch.device("cuda")
    torch.set_default_device(device)

    dtype = torch.float
    torch.set_default_dtype(dtype)

    config = AutoConfig.from_pretrained(args.model_name_or_config)
    model = AutoModelForCausalLM.from_config(config)

    # looking for the best lora operator for given shapes
    light_lora_collection = LightLoRACollection()
    light_lora_mapping = \
        light_lora_collection.optimize_for_model(
            model,
            n_batch=args.n_batch,
            lora_r=args.lora_r,
            target_modules=args.target_modules,
            criterions=args.criterions)

    del model, light_lora_collection
    reset_memory()

    # LightLoRA
    for criterion in args.criterions:

        # manage putting model to cuda only after replacing modules?
        model = AutoModelForCausalLM.from_config(config)
        params = sum(p.numel() for p in model.parameters())
        trainable_params = \
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total params before LightLoRA transform: {params}, '
              f'Trainable params before LightLoRA transform: {trainable_params}')

        model = LightLoRAModel(model,
                               light_lora_mapping[criterion],
                               lora_r=args.lora_r,
                               lora_alpha=args.lora_alpha,
                               target_modules=args.target_modules)
        # Every parameter except for lora adapters is set to requires_grad=False
        model.prepare_for_finetuning()
        params = sum(p.numel() for p in model.parameters())
        trainable_params_lightlora = \
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total params after LightLoRA transform: {params}, '
              f'Trainable params after LightLoRA transform: {trainable_params_lightlora}')
        print(model)

        assert trainable_params_lightlora < trainable_params, \
            "Number of trainable params after LightLoRA transform increased!"

        stats = bench_model(model, config, args)
        rows.append({'criterion': criterion,
                     **vars(args), **stats})

        del model
        reset_memory()

    del light_lora_mapping
    reset_memory()

    # Vanilla LoRA
    model = AutoModelForCausalLM.from_config(config)
    params = sum(p.numel() for p in model.parameters())
    trainable_params = \
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params before LoRA transform: {params}, '
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
    print(model)
    params = sum(p.numel() for p in model.parameters())
    trainable_params_lora = \
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params after LoRA transform: {params}, '
          f'Trainable params after LoRA transform: {trainable_params_lora}')

    assert trainable_params_lora < trainable_params, \
        "Number of trainable params after LoRA transform increased!"

    assert trainable_params_lora == trainable_params_lightlora, \
        "Number of trainable params after LoRA and LightLoRA transforms do not match!"

    stats = bench_model(model, config, args)
    rows.append({**vars(args), **stats})

    del model
    reset_memory()

    # Results
    df = pd.DataFrame.from_records(rows).drop(columns=['out', 'criterions'])
    df.sort_values(['mean_time_us', 'max_mem_overhead_MB'],
                   ascending=[True, True], inplace=True)
    df.to_csv(args.out+'.csv')
    print(args)
    print(df[['model_name_or_config',
              'criterion', 'mean_time_us',
              'max_mem_overhead_MB', 'msrs/runs']])


if __name__ == "__main__":
    args = parse_args(None)
    main(args)
