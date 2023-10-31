import torch
from lightlora import light_lora_collection
import torch.utils.benchmark as benchmark
from argparse import ArgumentParser
import pandas as pd


# TODO: default config class which reads from config file
def parse_args(args):
    parser = ArgumentParser(prog="Parameters for LightLora")
    parser.add_argument("--n_batch", type=int, default=1)
    parser.add_argument("--n_seq", type=int, default=4096)
    parser.add_argument("--n_in", type=int, default=1024)
    parser.add_argument("--n_out", type=int, default=1024)
    parser.add_argument("--n_rank", type=int, default=32)
    parser.add_argument("--b_is_None", choices=["True", "False"],
                        default="True")
    parser.add_argument("--x_req_grad", choices=["True", "False"],
                        default="True")
    parser.add_argument("--u_req_grad", choices=["True", "False"],
                        default="True")
    parser.add_argument("--v_req_grad", choices=["True", "False"],
                        default="True")
    parser.add_argument("--b_req_grad", choices=["True", "False"],
                        default="True")
    parser.add_argument("--dtype", default="float")
    parser.add_argument("--short", choices=["True", "False"], default="True")
    parser.add_argument('-o', "--out", type=str, default='out')

    args = parser.parse_args(args)
    return args


def mytimeit(statement, glbls):
    glbls['w'].grad = None
    glbls['x'].grad = None
    glbls['u'].grad = None
    glbls['v'].grad = None
    if 'b' in glbls and glbls['b']:
        glbls['b'].grad = None
    bench = benchmark.Timer(stmt=statement, globals=glbls)
    _ = bench.blocked_autorange(min_run_time=1.0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    measure = bench.blocked_autorange(min_run_time=3.0)
    print("Evaluating \"{}\"".format(statement))
    print("Computing mean with {} measurments, {} runs per measurment".format(
        len(measure.times), measure.number_per_run))
    print("Mean time: {} us".format(measure.mean * 1000000))
    print("Max mem: {} MB".format(torch.cuda.max_memory_allocated()/2**20))
    print()
    return {'mean_time': measure.mean * 1000000,
            'max_mem_MB': torch.cuda.max_memory_allocated() / 2**20,
            'msrs/runs': f'{len(measure.times)}/{measure.number_per_run}'}


def mytimeit_lightlora(*vars, path_f, path_b):
    x, w, u, v, b = vars
    print("path_f={} path_b={}".format(path_f, path_b))
    # global light_lora
    light_lora = light_lora_collection[path_f, path_b]
    glbls = {'w': w, 'x': x, 'u': u, 'v': v, 'b': b, 'light_lora': light_lora}
    return mytimeit("light_lora.apply(x, w, u, v, b).sum().backward()", glbls)


def mytimeit_lightlora_fwd(*vars, path_f, path_b):
    x, w, u, v, b = vars
    print("path_f={} path_b={}".format(path_f, path_b))
    global light_lora
    light_lora = light_lora_collection[path_f, path_b]
    glbls = {'w': w, 'x': x, 'u': u, 'v': v, 'b': b, 'light_lora': light_lora}
    return mytimeit("light_lora.apply(x, w, u, v, b)", glbls)


def mytimeit_lightlora_bwd(*vars, path_f, path_b):
    x, w, u, v, b = vars
    print("path_f={} path_b={}".format(path_f, path_b))
    global light_lora
    y = light_lora_collection[path_f, path_b].apply(x, w, u, v, b)
    glbls = {'y': y}
    return mytimeit("y.sum().backward(retain_graph=True)", glbls)


def main(args):
    rows = []

    if args.dtype == "float" or args.dtype == "fp32":
        dtype = torch.float
    elif args.dtype == "half" or args.dtype == "fp16":
        dtype = torch.half
    elif args.dtype == "bfloat" or args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError("Incorrect value of dtype")

    device = torch.device("cuda")
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    w = torch.nn.Parameter(torch.randn(args.n_in, args.n_out),
                           requires_grad=True)
    x = torch.randn(args.n_batch, args.n_seq, args.n_in,
                    requires_grad=args.x_req_grad == "True")
    u = torch.randn(args.n_in, args.n_rank,
                    requires_grad=args.u_req_grad == "True")
    v = torch.randn(args.n_rank, args.n_out,
                    requires_grad=args.v_req_grad == "True")
    if args.b_is_None == "True":
        b = None
    else:
        b = torch.randn(args.n_out, requires_grad=args.b_req_grad == "True")
    variables = [x, w, u, v, b]
    print("x.shape={} w.shape={} u.shape={} v.shape={} b.shape={}".format(
        x.shape, w.shape, u.shape, v.shape,
        b.shape if b is not None else None))
    print()

    # not used!
    # if x.requires_grad:
    #     baseline_nflops = 6 * prod(x.shape) * w.shape[1]
    # else:
    #     baseline_nflops = 4 * prod(x.shape) * w.shape[1]

    # Now W shall not accumulate gradient any more
    w.requires_grad = False

    if b is not None:
        glbls = {'x': x, 'w': w, 'u': u, 'v': v, 'b': b}
        stmt_all = [
            "(x@u)",
            "(w+u@v)",
            "w.addmm(u, v)",
            "(x@w+(x@u)@v+b)",
            "xx=x.reshape(-1,x.shape[-1]); (b.addmm("
            "xx,w).addmm_(xx.mm(u), v).reshape(*x.shape[:-1],w.shape[-1]))",
            "(x@(w+u@v)+b)",
            "(x@(w.addmm(u,v))+b)",
            "xx=x.reshape(-1, x.shape[-1]); (b.addmm(xx,w.addmm(u,v)))",
        ]
    else:
        glbls = {'x': x, 'w': w, 'u': u, 'v': v}
        stmt_all = [
            "(x@u)",
            "(w+u@v)",
            "w.addmm(u, v)",
            "(x@w+(x@u)@v)",
            "xx=x.reshape(-1,x.shape[-1]); (xx.mm(w)"
            ".addmm_(xx.mm(u),v).reshape(*x.shape[:-1],w.shape[-1]))",
            "(x@(w+u@v))",
            "(x@(w.addmm(u,v)))",
            "xx=x.reshape(-1,x.shape[-1]); (xx.mm(w.addmm(u,v)))",
        ]

    for stmt in stmt_all:
        timestats = mytimeit(stmt, glbls)
        rows.append({'statement': stmt,
                     'path_f': 'statement',
                     **vars(args), **timestats})
        stmt2 = stmt + ".sum().backward()"
        timestats = mytimeit(stmt2, glbls)
        rows.append({'statement': stmt2,
                     'path_f': 'statement',
                     'path_b': 'autograd',
                     **vars(args), **timestats})

    # Find the fastest forward+backward
    if args.short == "True":
        fwd_keys = light_lora_collection.forward_keys_short
        bwd_keys = light_lora_collection.backward_keys_short
    else:
        fwd_keys = light_lora_collection.forward_keys
        bwd_keys = light_lora_collection.backward_keys

    # forward1 and backward1 as a baseline
    fast_f = fwd_keys[0]
    fast_b = bwd_keys[0]
    timestats = mytimeit_lightlora(*variables,
                                   path_f=fast_f, path_b=fast_b)
    rows.append({'path_f': fast_f, 'path_b': fast_b,
                 'statement': "light_lora.apply(x, w, u, v, b)"
                 ".sum().backward()",
                 **vars(args), **timestats})

    fast_mean = torch.inf
    for path_f in fwd_keys:
        timestats = mytimeit_lightlora_fwd(*variables,
                                           path_f=path_f, path_b=fast_b)
        rows.append({'path_f': path_f,  # no backward path
                     'statement': "light_lora.apply(x, w, u, v, b)",
                     **vars(args), **timestats})
        if fast_mean > timestats['mean_time']:
            fast_f = path_f
            fast_mean = timestats['mean_time']
    fast_mean = torch.inf
    for path_b in bwd_keys:
        timestats = mytimeit_lightlora(*variables,
                                       path_f=fast_f, path_b=path_b)
        rows.append({'path_f': fast_f, 'path_b': path_b,
                     'statement': "light_lora.apply(x, w, u, v, b)"
                     ".sum().backward()",
                     **vars(args), **timestats})
        if fast_mean > timestats['mean_time']:
            fast_b = path_b
            fast_mean = timestats['mean_time']

    df = pd.DataFrame.from_records(rows).drop(columns="out")
    df.sort_values(['mean_time', 'max_mem_MB'],
                   ascending=[True, True], inplace=True)
    # removing duplicates with a forward1-backward1 baseline if any
    df.drop_duplicates(subset=['statement', 'path_f', 'path_b'],
                       keep='last', inplace=True)
    df.to_csv(args.out)
    print(args)
    print(df[['statement', 'mean_time', 'max_mem_MB',
              'path_f', 'path_b', 'msrs/runs']])


if __name__ == "__main__":
    args = parse_args(None)
    main(args)
