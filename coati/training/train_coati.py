import os
import copy, argparse, json, pickle
import sys
import torch
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DistributedDataParallel
from time import time as get_time
from torch import autocast
from coati.common import *
from coati.data.dataset import COATI_dataset

from coati.models.autograd_funs.autograd_funs import all_gather
from coati.models.encoding.tokenizers.trie_tokenizer import TrieTokenizer
from coati.models.encoding.tokenizers import get_vocab
from coati.models.encoding.clip_e2e import clip_ar_xform, e3gnn_smiles_clip_e2e
from coati.models.encoding.clip_e2e import clip_loss as clip_loss_module
from coati.training.logger import COATILogger
from coati.common.util import makedir, utc_epoch_now


def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def serialize_model(
    train_args,
    dataset_summary,
    model_state_dict,
    model_kwargs,
    optimizer_state_dict=None,
    **kwargs,
):
    d = pickle.dumps(
        {
            "train_args": train_args,
            "dataset_summary": dataset_summary,
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "model_kwargs": model_kwargs,
            **kwargs,
        },
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    print("Model Document size (MB): ", sys.getsizeof(d) / (1024 * 1024))
    return d


def train_autoencoder(gpu, args):
    """
    gpu: an integer indexing the GPU which will be used in this process.

    args.nr: index of node
    args.gpus: gpus per node
    args.world_size: # nodes * # gpus
    """
    rank = args.nr * args.gpus + gpu
    print(f"train autoencoder rank {rank} reporting in.")

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,  # Computed by the mp.spawn caller.
        rank=rank,
    )

    if rank == 0:
        output_path = os.path.join(args.output_dir, args.exp_name, args.run_name)
        makedir(output_path)
        with open(os.path.join(output_path, "params.json"), "w") as f:
            json.dump(vars(args), f)
    req_fields = ["smiles", "atoms", "coords"]
    # try to recall dataset, train and test.
    dataset = COATI_dataset(cache_dir=args.data_dir, fields=req_fields)
    tokenizer = TrieTokenizer(n_seq=args.n_seq, **get_vocab(args.tokenizer_vocab))
    token_entropy_unit = np.log(float(len(tokenizer.keys))) / np.log(2.0)

    if rank == 0:
        logger = COATILogger(
            model_name="e3gnn_smiles_clip_e2e",
            run_time=args.run_name,
            output_path=args.output_dir,
            model_path=args.model_dir,
            args=vars(args),
            dataset="",
        )
        logger.start()

    device = torch.device("cuda:" + str(gpu))
    torch.cuda.device(gpu)
    dtype = eval("torch." + args.dtype)
    torch.set_default_dtype(dtype)
    print("Using device:", device)

    xform_routine = lambda X: clip_ar_xform(
        X,
        tokenizer=tokenizer,
        device=device,
        p_dataset=args.p_dataset,
        p_formula=args.p_formula,
        p_fim=args.p_fim,
        p_graph=args.p_graph,
        p_clip=args.p_clip,
        p_clip_cut=args.p_clip_cut,
        p_randsmiles=args.p_randsmiles,
    )

    kwargs = {
        "n_layer_xformer": args.n_layer_xformer,
        "n_layer_e3gnn": args.n_layer_e3gnn,
        "n_hidden_e3nn": args.n_hidden_e3nn,
        "n_hidden_xformer": args.n_hidden_xformer,
        "n_embd_common": args.n_embd_common,
        "biases": args.biases,
        "n_head": args.n_head,
        "n_seq": args.max_n_seq,
        "n_tok": tokenizer.n_token,  # base
        "torch_emb": args.torch_emb,
        "norm_clips": args.norm_clips,
        "norm_embed": args.norm_embed,
        "token_mlp": args.token_mlp,
    }

    if not args.do_clip:
        kwargs["use_point_encoder"] = False

    model_kwargs = kwargs.copy()  # avoid saving the device.
    kwargs["device"] = device

    model = e3gnn_smiles_clip_e2e(**kwargs)
    if rank == 0:
        print("end-to-end clip autogregressive model: ", model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
        eps=1e-8,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
    ngrad_updates = 0
    offline_losses = {"batch_losses": [], "ar_losses": [], "clip_losses": []}
    n_toks = 0  # Number of tokens done(period)
    if rank == 0:
        makedir(args.model_dir)

    if not (args.resume_document is None):
        # Only rank0 has the logger.
        with open(args.resume_document, "rb") as f_in:
            model_doc = pickle.load(f_in)

        if "n_toks_processed" in model_doc:
            n_toks = model_doc["n_toks_processed"]
        if "n_grads_processed" in model_doc:
            ngrad_updates = model_doc["n_grads_processed"]
        model_dict_ = model_doc["model"]
        new_names = [
            k.replace("module.", "") if k.startswith("module.") else k
            for k in model_dict_.keys()
        ]
        model_dict = {
            new_name: t for new_name, t in zip(new_names, model_dict_.values())
        }
        if args.load_transformer_only:
            print("loading ONLY TRANSFORMER from checkpoint")
            xformer_dict = {
                new_name: t
                for new_name, t in zip(new_names, model_dict.values())
                if new_name.split(".")[0] == "xformer"
            }
            smiles_to_clip_dict = {
                new_name: t
                for new_name, t in zip(new_names, model_dict.values())
                if new_name.split(".")[0] == "smiles_to_clip"
            }
            model.xformer.load_state_dict(xformer_dict, strict=False)
            model.smiles_to_clip.load_state_dict(smiles_to_clip_dict, strict=False)
        else:
            model.load_state_dict(model_dict, strict=False)

        if args.resume_optimizer:
            try:
                optimizer.load_state_dict(model_doc["optimizer"])
                optimizer_to(optimizer, device)
            except Exception as Ex:
                print("failed to resume optimizer", Ex)
                pass
        else:
            pass
        print("Loaded from checkpoint. ")

    model = DistributedDataParallel(
        model, device_ids=[gpu], find_unused_parameters=True
    )
    clip_computer = clip_loss_module()

    def do_epoch(epoch, dataset, partition="train"):
        nonlocal ngrad_updates, n_toks, offline_losses
        res = {"loss": 0, "counter": 0, "loss_arr": []}

        t0 = get_time()
        ng = 0

        def do_minibatch(i, batch_data):
            nonlocal ngrad_updates, ng, t0, res, optimizer, model, epoch, n_toks, offline_losses

            if partition == "train":
                model.train()
                optimizer.zero_grad()
            else:
                model.eval()

            TRAIN_FIELDS = ["smiles", "atoms", "coords"]
            if not all([X in batch_data.keys() for X in TRAIN_FIELDS]):
                print("Bad MiniBatch...")
                return
            if not batch_data["tokens"].shape[0] == batch_data["atoms"].shape[0]:
                print("a row was lost, skipping batch")
                return
            if not batch_data["y_next"].shape[0] == batch_data["atoms"].shape[0]:
                print("a row was lost, skipping batch")
                return

            if partition == "train":
                h_e3gnn, h_xformer, logits, bad_rows = model.module.forward_dist(
                    batch_data["raw_tokens"],
                    batch_data["tokens"],
                    batch_data["atoms"],
                    batch_data["coords"],
                    tokenizer,
                    p_clip_emb_smi=args.p_clip_emb_smi,
                )
            if partition == "test":
                with torch.no_grad():
                    h_e3gnn, h_xformer, logits, bad_rows = model.module.forward_dist(
                        batch_data["raw_tokens"],
                        batch_data["tokens"],
                        batch_data["atoms"],
                        batch_data["coords"],
                        tokenizer,
                        p_clip_emb_smi=args.p_clip_emb_smi,
                    )

            bad_rows = all_gather(bad_rows)
            all_h_xformer = all_gather(h_xformer)
            all_h_e3gnn = all_gather(h_e3gnn)

            ar_loss_ = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch_data["y_next"].view(-1),
                ignore_index=-1,
            )
            ar_loss = ar_loss_.mean()

            if args.do_clip:
                clip_loss_ = clip_computer(all_h_xformer, all_h_e3gnn, bad_rows)
                clip_loss = clip_loss_.mean()
                loss = ar_loss + clip_loss * token_entropy_unit
            else:
                loss = ar_loss

            if partition == "train":
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

            if rank == 0:
                ngrad_updates += batch_data["atoms"].shape[0]
                ng += batch_data["atoms"].shape[0]
                n_toks += (batch_data["tokens"] > 0).sum().item()

            if ngrad_updates * args.world_size > args.ngrad_to_save and rank == 0:
                ngrad_updates = 0
                if args.data_parallel:
                    msd = model.module.state_dict()
                else:
                    msd = model.state_dict()
                model_doc = serialize_model(
                    train_args=vars(args),
                    dataset_summary=dataset.summary,
                    model_state_dict=copy.deepcopy(msd),
                    model_kwargs=model_kwargs,
                    optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),
                    n_toks_processed=n_toks,
                    n_grads_processed=ngrad_updates,
                    offline_loss=offline_losses,
                )
                logger.log_pytorch(
                    model_doc,
                    tags={"train_epoch": str(epoch), "dataset_epoch": str(epoch)},
                )
                del model_doc, msd

            if (i % args.log_batch_loss) == 0 and rank == 0:
                step_batch_loss = logger.log_metric(
                    partition + "_batch_loss",
                    loss.item(),
                    dataset_epoch=epoch,
                    step=i,
                    tags={"n_toks": n_toks},
                )
                step_ar_loss = logger.log_metric(
                    partition + "_ar_loss",
                    ar_loss.item(),
                    dataset_epoch=epoch,
                    step=i,
                    tags={"n_toks": n_toks},
                )
                if args.do_clip:
                    step_clip_loss = logger.log_metric(
                        partition + "_clip_loss",
                        clip_loss.item(),
                        dataset_epoch=epoch,
                        step=i,
                        tags={"n_toks": n_toks},
                    )

                offline_losses["batch_losses"].append(step_batch_loss)
                offline_losses["ar_losses"].append(step_ar_loss)
                if args.do_clip:
                    offline_losses["clip_losses"].append(step_clip_loss)

            res["loss"] += loss.item() * args.batch_size
            res["counter"] += args.batch_size
            res["loss_arr"].append(loss.item())

            prefix = ""
            if partition != "train":
                prefix = ">> %s \t" % partition
            if i % args.log_interval == 0 and rank == 0:
                print(
                    prefix
                    + "run_time %s Epoch %d \t it %d \t toks %im <tok/batch> %.0f \t ar_l: %.2f, clip_l %.6f, loss %.4f \t grads_ps %.4f"
                    % (
                        logger.run_time,
                        epoch,
                        i,
                        int(n_toks / 1e6),
                        n_toks
                        / ((1 + i) * args.batch_size * args.world_size * (1 + epoch)),
                        ar_loss,
                        clip_loss if args.do_clip else -1,
                        sum(res["loss_arr"][-10:]) / len(res["loss_arr"][-10:]),
                        ng / (get_time() - t0),
                    )
                )

            del batch_data
            return

        epoch_iter = enumerate(
            iter(
                dataset.get_data_pipe(
                    batch_size=args.batch_size,
                    partition=partition,
                    distributed_rankmod_total=args.world_size,
                    distributed_rankmod_rank=rank,
                    required_fields=["smiles"],
                    xform_routine=xform_routine,
                )
            )
        )
        for i, batch_data in epoch_iter:
            do_minibatch(i, batch_data)

        # -------------------
        # end minibatch loop
        if partition == "train":
            lr_scheduler.step()
        if res["counter"] == 0:
            return

        if rank == 0:
            print(f"epoch completed in {ng} grads and {get_time()-t0} seconds")
            logger.log_metric(
                partition + " epoch mean loss",
                res["loss"] / res["counter"],
                dataset_epoch=epoch,
            )
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        return res["loss"] / res["counter"]

    # Loop over epochs.
    res = {
        "epochs": [],
        "losses": [],
        "best_test": 1e10,
        "best_epoch": 0,
        "best_model": None,
    }
    for epoch in range(0, args.n_epochs):
        do_epoch(epoch, dataset, partition="train")
        if epoch % args.test_interval == 0 and epoch > 0 and rank == 0:
            test_loss = do_epoch(epoch, dataset, partition="test")
            if test_loss is None:
                continue
            # Add plots of the foldover vs. actual.
            res["epochs"].append(epoch)
            res["losses"].append(test_loss)
            if test_loss < res["best_test"]:
                res["best_test"] = test_loss
                res["best_epoch"] = epoch
                if args.data_parallel:
                    msd = model.module.state_dict()
                else:
                    msd = model.state_dict()
                res["best_model"] = copy.deepcopy(msd)
                del msd
            print("test loss: %.4f \t epoch %d" % (test_loss, epoch))
            print(
                "Best: test loss: %.4f \t epoch %d"
                % (res["best_test"], res["best_epoch"])
            )

    if rank == 0:
        print("SAVING MODEL TO ", output_path + args.model_filename + ".pt")
        model_doc = serialize_model(
            train_args=vars(args),
            dataset_summary=dataset.summary,
            model_state_dict=res["best_model"],
            model_kwargs=model_kwargs,
            optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),
            n_toks_processed=n_toks,
            n_grads_processed=ngrad_updates,
        )
        logger.log_pytorch(model_doc, tags={"best": "best"})


def do_args():
    parser = argparse.ArgumentParser(description="token_transformer")
    parser.add_argument("--exp_name", type=str, default="token_transformer")
    parser.add_argument("--run_name", type=str, default=str(int(utc_epoch_now())))
    parser.add_argument("--output_dir", type=str, default="COATI_outputs")
    parser.add_argument("--model_dir", type=str, default="COATI_models")
    parser.add_argument("--data_dir", type=str, default="COATI_data")

    # ddp options.
    parser.add_argument(
        "-ws", "--world_size", default=1, type=int, help="total number of processes"
    )
    parser.add_argument(
        "-nr", "--nr", default=0, type=int, help="ranking within the nodes"
    )
    parser.add_argument(
        "-n", "--nodes", default=1, type=int, metavar="N", help="number of nodes"
    )
    parser.add_argument(
        "-g",
        "--gpus",
        default=torch.cuda.device_count(),
        type=int,
        help="number of gpus per node",
    )

    parser.add_argument(
        "--device", type=str, default="cuda", help="pytorch backend device."
    )
    parser.add_argument("--dtype", type=str, default="float", help="default data type")
    parser.add_argument("--log_batch_loss", default=25, help="steps per tnet log")
    parser.add_argument(
        "--code_features",
        default=["protein", "secondary", "library"],
        help="one hot encoded additional dimensions.",
    )
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument(
        "--recipe",
        type=list,
        default=[
            {"collection": "geom_drugs", "n_samples": 6_000_000, "filter": {}},
        ],
    )

    parser.add_argument("--n_layer_e3gnn", type=int, default=4)
    parser.add_argument("--n_hidden_e3nn", type=int, default=128)
    parser.add_argument("--msg_cutoff_e3nn", type=float, default=10.0)
    parser.add_argument("--n_hidden_xformer", type=int, default=128)
    parser.add_argument("--n_embd_common", type=int, default=128)
    parser.add_argument("--n_layer_xformer", type=int, default=16)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument(
        "--biases", type=bool, default=True, help="Use biases in the xformer."
    )
    parser.add_argument("--n_seq", type=int, default=200)
    parser.add_argument("--tokenizer_vocab", type=str, default="Jan8")
    parser.add_argument("--torch_emb", type=bool, default=False)
    parser.add_argument(
        "--load_transformer_only",
        type=bool,
        default=False,
        help="load trained transformer but use fresh point encoder",
    )

    parser.add_argument("--p_dataset", type=float, default=0.3)
    parser.add_argument("--p_formula", type=float, default=0.3)
    parser.add_argument("--p_fim", type=float, default=0.5)
    parser.add_argument("--p_graph", type=float, default=0.3)
    parser.add_argument("--p_clip", type=float, default=0.3)
    parser.add_argument("--p_clip_cut", type=float, default=0.3)

    parser.add_argument("--p_clip_emb_smi", type=float, default=0.4)
    parser.add_argument("--p_randsmiles", type=float, default=0.5)

    parser.add_argument(
        "--norm_clips", type=bool, default=False, help="normalize the clip vectors"
    )
    parser.add_argument(
        "--token_mlp",
        type=bool,
        default=False,
        help="Do we use an MLP or just hclip as a token.",
    )
    parser.add_argument(
        "--norm_embed", type=bool, default=False, help="Layernorm after embedding"
    )
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--clip_grad", type=float, default=10.0)

    parser.add_argument(
        "--do_clip",
        type=bool,
        default=True,
        help="If false, do not use clip loss during training.",
    )

    parser.add_argument(
        "--test_frac", type=float, default=0.02, help="test data fraction"
    )
    parser.add_argument(
        "--valid_frac", type=float, default=0.02, help="test data fraction"
    )
    parser.add_argument(
        "--test_interval",
        type=int,
        default=1,
        metavar="N",
        help="how many epochs to wait before logging test",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--ngrad_to_save", default=2e6, help="ngrad updates between model saves."
    )

    parser.add_argument(
        "--resume_document", default=None, help="Restore from an S3 document"
    )
    parser.add_argument(
        "--resume_optimizer",
        type=bool,
        default=False,
        help="Restore opt. from an S3 document",
    )

    args, unparsed_args = parser.parse_known_args()
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    if len(unparsed_args):
        print("Warning... unparsed: ", unparsed_args)
    return args


if __name__ == "__main__":
    args = do_args()
    train_autoencoder(args)
