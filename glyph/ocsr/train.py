"""OCSR trainer for the SMILES-only baseline.

Single-GPU and torchrun-DDP launches are both supported. The DDP path is
activated when launched via ``torchrun`` (``RANK``/``WORLD_SIZE`` set in the
environment); otherwise it falls back to single-process training.

Training data can be PubChem rendered dynamically with Indigo, optionally
mixed with the USPTO 680k auxiliary dataset, across all visible GPUs. The
target format is SMILES-only ``chartok``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from glyph.paths import checkpoints_dir, data_path

from .baseline_config import BaselineRecipe, SwinLargeRecipe
from .eval import evaluate_pairs, load_uspto_benchmark  # noqa: F401 (kept for callers)
from .model import OCSRModel, OCSRModelConfig
from .smiles_tokenizer import CharSmilesTokenizer
from .train_data import PubChemIndigoDataset, UsptoMolDataset, collate_batch


def _wandb_login() -> None:
    """Fail fast when online training is started without W&B credentials.

    A network-free run (``WANDB_MODE=offline`` or ``disabled``) needs no key, so
    the requirement is only enforced for online logging.
    """

    os.environ.setdefault("WANDB_DISABLE_GIT", "true")
    if os.environ.get("WANDB_MODE", "online") in {"offline", "disabled"}:
        return
    key = os.environ.get("WANDB_API_KEY")
    if not key or key.startswith("this-will-be-replaced"):
        raise RuntimeError(
            "WANDB_API_KEY must be set for online OCSR training; "
            "set WANDB_MODE=offline (or disabled) for a network-free run"
        )


def _data_path_arg(value: str) -> str:
    """Resolve a training-data path argument.

    A ``hub:<key>`` value auto-downloads a frozen enrichment CSV from the
    ``EdisonScientific/glyph-datasets`` dataset repo (e.g. ``hub:adjacent_ring``),
    so no manual ``huggingface-cli`` step is needed. Every other value is treated
    as a filesystem path and resolved below ``GLYPH_DATA_ROOT`` when relative.
    """

    if value.startswith("hub:"):
        from .train_data import resolve_ocsr_train_csv

        return str(resolve_ocsr_train_csv(value[len("hub:") :]))
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else data_path(path))


def _ddp_env() -> tuple[bool, int, int, int]:
    """Return (is_ddp, rank, world_size, local_rank)."""

    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        return (
            True,
            int(os.environ.get("RANK", "0")),
            int(os.environ["WORLD_SIZE"]),
            int(os.environ.get("LOCAL_RANK", "0")),
        )
    return False, 0, 1, 0


def _set_seed(seed: int, rank: int = 0) -> None:
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _build_model(recipe: BaselineRecipe) -> OCSRModel:
    cfg = OCSRModelConfig(
        encoder_name=recipe.encoder_name,
        input_size=recipe.input_size,
        embed_dim=recipe.embed_dim,
        dec_num_layers=recipe.dec_num_layers,
        dec_attn_heads=recipe.dec_attn_heads,
        dec_ff_dim=recipe.dec_ff_dim,
        max_len=min(recipe.max_target_len, 256),
        vocab_size=recipe.vocab_size,
        batch_size=recipe.batch_size,
        seed=recipe.seed,
        pad_id=recipe.pad_id,
        sos_id=recipe.sos_id,
        eos_id=recipe.eos_id,
        backbone_pretrained=recipe.backbone_pretrained,
    )
    return OCSRModel(cfg)


def _greedy_decode(
    model: nn.Module,
    image_batch: torch.Tensor,
    tokenizer: CharSmilesTokenizer,
    max_len: int,
) -> list[str]:
    inner = cast(
        "OCSRModel",
        model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model,
    )
    inner.eval()
    device = image_batch.device
    with torch.no_grad():
        memory = inner.encoder(image_batch)
        ids = torch.full(
            (image_batch.shape[0], 1), tokenizer.SOS_ID, dtype=torch.long, device=device
        )
        for _ in range(max_len - 1):
            logits = inner.decoder(ids, memory)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            if bool((nxt.squeeze(-1) == tokenizer.EOS_ID).all().item()):
                break
    return [tokenizer.decode(r) for r in ids.cpu().tolist()]


def _eval_subset(
    model: nn.Module,
    tokenizer: CharSmilesTokenizer,
    recipe: BaselineRecipe,
    device: torch.device,
    n_eval: int,
) -> dict[str, Any]:
    from .bench_image import load_benchmark_samples, stack_batch

    samples = load_benchmark_samples(n=n_eval, image_size=recipe.input_size)
    batch = stack_batch(samples)
    images = batch["images"].to(device)
    preds = _greedy_decode(model, images, tokenizer, max_len=min(recipe.max_target_len, 256))
    result = evaluate_pairs(preds, batch["smiles"])
    mode_collapse_ratio = (
        max(preds.count(prediction) / len(preds) for prediction in set(preds)) if preds else 0.0
    )
    return {
        "uspto/canon_exact_match": result.canon_exact_match,
        "uspto/valid_smiles_rate": result.valid_smiles_rate,
        "uspto/n_invalid_gt": result.n_invalid_gt,
        "eval/n_examples": result.n,
        "eval/mode_collapse_ratio": mode_collapse_ratio,
        "eval/sample_preds": list(zip(batch["image_ids"], batch["smiles"], preds, strict=False))[
            :8
        ],
    }


def _build_dataset(
    args: argparse.Namespace,
    tokenizer: CharSmilesTokenizer,
    max_target_len: int,
    recipe: BaselineRecipe,
):
    from .train_data import IndigoRenderOptions

    parts = []
    if args.pubchem_csv:
        parts.append(
            PubChemIndigoDataset(
                csv_path=args.pubchem_csv,
                tokenizer=tokenizer,
                max_target_len=max_target_len,
                max_rows=args.max_pubchem_rows,
                render_opts=IndigoRenderOptions(
                    image_size=recipe.input_size,
                    randomize=True,
                    augment=args.augment,
                    p_clean=args.aug_p_clean,
                    p_each=args.aug_p_each,
                ),
            )
        )
    # Extra PubChem CSV (e.g., adjacent-ring mined subset). Repeats so its
    # effective sampling weight is meaningful when ConcatDataset draws uniformly.
    extra_pc = getattr(args, "extra_pubchem_csv", None)
    extra_pc_repeat = getattr(args, "extra_pubchem_repeat", 1)
    if extra_pc:
        from torch.utils.data import ConcatDataset as _ConcatDataset

        ds_extra = PubChemIndigoDataset(
            csv_path=extra_pc,
            tokenizer=tokenizer,
            max_target_len=max_target_len,
            max_rows=getattr(args, "max_extra_pubchem_rows", None),
            render_opts=IndigoRenderOptions(
                image_size=recipe.input_size,
                randomize=True,
                augment=args.augment,
                p_clean=args.aug_p_clean,
                p_each=args.aug_p_each,
            ),
        )
        if extra_pc_repeat > 1:
            ds_extra = _ConcatDataset([ds_extra] * int(extra_pc_repeat))
        parts.append(ds_extra)
    if args.uspto_csv:
        parts.append(
            UsptoMolDataset(
                csv_path=args.uspto_csv,
                image_root=args.uspto_image_root,
                tokenizer=tokenizer,
                max_target_len=max_target_len,
                image_size=recipe.input_size,
                max_rows=args.max_uspto_rows,
                path_strip_prefix=args.uspto_path_strip_prefix or None,
                augment=args.augment,
                aug_p_clean=args.aug_p_clean,
                aug_p_each=args.aug_p_each,
            )
        )
    # Stereo-200K (white-bg cropped). Same image-loader as USPTO, paths relative to image_root.
    stereo_csv = getattr(args, "stereo_csv", None)
    stereo_image_root: Any = getattr(args, "stereo_image_root", None)
    stereo_repeat = getattr(args, "stereo_repeat", 1)
    if stereo_csv:
        from torch.utils.data import ConcatDataset as _ConcatDataset

        ds_stereo = UsptoMolDataset(
            csv_path=stereo_csv,
            image_root=stereo_image_root,
            tokenizer=tokenizer,
            max_target_len=max_target_len,
            image_size=recipe.input_size,
            max_rows=getattr(args, "max_stereo_rows", None),
            path_strip_prefix=None,
            augment=args.augment,
            aug_p_clean=args.aug_p_clean,
            aug_p_each=args.aug_p_each,
        )
        if stereo_repeat > 1:
            ds_stereo = _ConcatDataset([ds_stereo] * int(stereo_repeat))
        parts.append(ds_stereo)
    # Real-image dataset (CLEF, JPO, etc). Also USPTO-format (SMILES,file_path).
    realimg_csv = getattr(args, "realimg_csv", None)
    realimg_image_root: Any = getattr(args, "realimg_image_root", None)
    realimg_repeat = getattr(args, "realimg_repeat", 1)
    if realimg_csv:
        from torch.utils.data import ConcatDataset as _ConcatDataset

        ds_real = UsptoMolDataset(
            csv_path=realimg_csv,
            image_root=realimg_image_root,
            tokenizer=tokenizer,
            max_target_len=max_target_len,
            image_size=recipe.input_size,
            max_rows=getattr(args, "max_realimg_rows", None),
            path_strip_prefix=None,
            augment=args.augment,
            aug_p_clean=args.aug_p_clean,
            aug_p_each=args.aug_p_each,
        )
        if realimg_repeat > 1:
            ds_real = _ConcatDataset([ds_real] * int(realimg_repeat))
        parts.append(ds_real)
    rdkit_csv = getattr(args, "rdkit_pubchem_csv", None)
    if rdkit_csv:
        from .rdkit_renderer import RDKitRenderDataset

        ds_rdkit = RDKitRenderDataset(
            csv_path=rdkit_csv,
            tokenizer=tokenizer,
            max_target_len=max_target_len,
            image_size=recipe.input_size,
            max_rows=getattr(args, "max_rdkit_pubchem_rows", None),
            augment=args.augment,
            aug_p_clean=args.aug_p_clean,
            aug_p_each=args.aug_p_each,
            randomize=True,
        )
        parts.append(ds_rdkit)
    if not parts:
        raise ValueError("Must provide at least --pubchem_csv or --uspto_csv")
    if len(parts) == 1:
        return parts[0]
    return ConcatDataset(parts)


def train(args: argparse.Namespace) -> dict[str, Any]:
    is_ddp, rank, world_size, local_rank = _ddp_env()
    if is_ddp:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if getattr(args, "encoder_name", None):
        # Create a dynamic recipe with overridden encoder_name
        recipe_cls = type("DynamicRecipe", (BaselineRecipe,), {"encoder_name": args.encoder_name})
    else:
        recipe_cls = SwinLargeRecipe if getattr(args, "swin_large", False) else BaselineRecipe
    recipe = recipe_cls(backbone_pretrained=args.backbone_pretrained)
    if getattr(args, "input_size", None):
        recipe.input_size = args.input_size
    _set_seed(recipe.seed, rank=rank)
    tokenizer = CharSmilesTokenizer()
    target_format = "chartok_smiles_only"
    max_target_len = min(recipe.max_target_len, 256)

    model = _build_model(recipe).to(device)
    if is_ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )
    enc_lr = args.encoder_lr if args.encoder_lr is not None else recipe.encoder_lr
    dec_lr = args.decoder_lr if args.decoder_lr is not None else recipe.decoder_lr
    inner_for_params = (
        model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    )
    enc_params = list(inner_for_params.encoder.parameters())
    dec_params = list(inner_for_params.decoder.parameters())
    optim = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": enc_lr},
            {"params": dec_params, "lr": dec_lr},
        ],
        lr=dec_lr,
        weight_decay=recipe.weight_decay,
    )
    # Cosine LR schedule with linear warmup. Defaults match BaselineRecipe.
    total_steps = max(args.max_steps, 1)
    warmup_steps = max(int(recipe.warmup_ratio * total_steps), 1)

    def _lr_lambda(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return step_idx / warmup_steps
        progress = (step_idx - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, _lr_lambda) if args.cosine_lr else None
    ls = args.label_smoothing if args.label_smoothing is not None else recipe.label_smoothing
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_ID, label_smoothing=ls)

    if args.resume_from:
        if rank == 0:
            print(f"resuming from {args.resume_from}", flush=True)
        state = torch.load(args.resume_from, map_location=device, weights_only=False)
        inner = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        inner.load_state_dict(state["model"], strict=True)
        # We deliberately do NOT restore optimizer/scheduler state. The cosine
        # schedule is a fresh schedule against the new ``--max_steps``, and the
        # learning-rate search starts over.

    if args.uspto_weighted_sampler:
        if not args.uspto_csv:
            raise ValueError("--uspto_weighted_sampler requires --uspto_csv")
        from torch.utils.data import WeightedRandomSampler

        from .train_data import UsptoMolDataset

        ds = UsptoMolDataset(
            csv_path=args.uspto_csv,
            image_root=args.uspto_image_root,
            tokenizer=tokenizer,
            max_target_len=max_target_len,
            image_size=recipe.input_size,
            max_rows=args.max_uspto_rows,
            path_strip_prefix=args.uspto_path_strip_prefix or None,
            augment=args.augment,
            aug_p_clean=args.aug_p_clean,
            aug_p_each=args.aug_p_each,
        )
        weights = ds.compute_sample_weights(
            stereo_weight=args.stereo_weight,
            long_threshold=args.long_threshold,
            long_weight=args.long_weight,
        )
        if rank == 0:
            n_long = sum(1 for w in weights if w >= args.long_weight)
            n_stereo = sum(1 for w in weights if w >= args.stereo_weight)
            print(
                f"USPTO weighted sampler: n={len(ds)} long_weight={n_long} stereo_weight>=stereo_weight_count={n_stereo}",
                flush=True,
            )
        # Each rank gets its own sampler over the same dataset, with disjoint
        # random draws via per-rank generator seeding.
        gen = torch.Generator()
        gen.manual_seed(recipe.seed + rank)
        per_rank_samples = max(args.max_steps * args.batch_size, len(ds))
        sampler = WeightedRandomSampler(
            weights,
            num_samples=per_rank_samples,
            replacement=True,
            generator=gen,
        )
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
            drop_last=True,
            persistent_workers=(args.num_workers > 0),
        )
    else:
        ds = _build_dataset(args, tokenizer, max_target_len, recipe)
        sampler = (
            DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
            if is_ddp
            else None
        )
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
            drop_last=True,
            persistent_workers=(args.num_workers > 0),
        )

    run = None
    if rank == 0:
        _wandb_login()
        run = wandb.init(
            project=recipe.wandb_project,
            entity=recipe.wandb_entity,
            name=args.run_name,
            config={
                **asdict(recipe),
                "git_commit": args.git_commit,
                "target_format": target_format,
                "deviation_from_public_recipe": "SMILES-only chartok baseline; chartok_coords,edges not yet wired",
                "pubchem_csv": str(args.pubchem_csv) if args.pubchem_csv else None,
                "uspto_csv": str(args.uspto_csv) if args.uspto_csv else None,
                "uspto_image_root": str(args.uspto_image_root) if args.uspto_image_root else None,
                "max_pubchem_rows": args.max_pubchem_rows,
                "max_uspto_rows": args.max_uspto_rows,
                "world_size": world_size,
                "is_ddp": is_ddp,
                "n_eval_examples": args.eval_n,
                "max_steps": args.max_steps,
                "batch_size_per_gpu": args.batch_size,
                "effective_batch_size": args.batch_size * world_size,
            },
            mode=os.environ.get("WANDB_MODE") or "online",
        )

    if rank == 0:
        print(
            f"[rank {rank}/{world_size}] device={device} dataset_size={len(ds)} "
            f"steps={args.max_steps} batch={args.batch_size} "
            f"backbone_pretrained={recipe.backbone_pretrained} cosine_lr={bool(args.cosine_lr)} "
            f"resume_from={args.resume_from} enc_lr={enc_lr} dec_lr={dec_lr} ls={ls} "
            f"weighted_sampler={bool(args.uspto_weighted_sampler)}",
            flush=True,
        )

    step = 0
    t0 = time.time()
    last_eval: dict[str, Any] | None = None
    epoch = 0
    done = False
    while not done:
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        for batch in dl:
            step += 1
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target_ids"].to(device, non_blocking=True)
            inputs = targets[:, :-1]
            labels = targets[:, 1:]

            if args.bf16_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(images, inputs)
                    loss = loss_fn(logits.reshape(-1, recipe.vocab_size), labels.reshape(-1))
            else:
                logits = model(images, inputs)
                loss = loss_fn(logits.reshape(-1, recipe.vocab_size), labels.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError(f"NaN/Inf loss at step {step}: {loss.item()}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=recipe.grad_clip_max_norm
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"NaN/Inf grad_norm at step {step}")
            optim.step()
            if scheduler is not None:
                scheduler.step()
            optim.zero_grad(set_to_none=True)

            if rank == 0:
                elapsed = time.time() - t0
                cur_lr = optim.param_groups[0]["lr"]
                wandb.log(
                    {
                        "train/loss": float(loss.detach().cpu()),
                        "train/grad_norm": float(grad_norm.detach().cpu()),
                        "train/step": step,
                        "train/epoch": epoch,
                        "train/lr": cur_lr,
                        "train/examples_per_sec": (step * args.batch_size * world_size)
                        / max(elapsed, 1e-6),
                    },
                    step=step,
                )
                if (
                    args.checkpoint_every_steps
                    and step % args.checkpoint_every_steps == 0
                    and args.write_checkpoint
                ):
                    outdir = Path(args.output_dir)
                    outdir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = outdir / f"step{step}.pth"
                    inner = (
                        model.module
                        if isinstance(model, nn.parallel.DistributedDataParallel)
                        else model
                    )
                    torch.save(
                        {"model": inner.state_dict(), "step": step, "recipe": asdict(recipe)},
                        ckpt_path,
                    )
                if args.eval_n > 0 and (step == 1 or step % args.eval_every_steps == 0):
                    last_eval = _eval_subset(model, tokenizer, recipe, device, args.eval_n)
                    flat = {k: v for k, v in last_eval.items() if not isinstance(v, list)}
                    wandb.log(flat, step=step)
                    if step >= args.min_step_for_kill:
                        if last_eval["uspto/valid_smiles_rate"] < 0.05:
                            raise RuntimeError(
                                f"Kill criterion hit at step {step}: valid_smiles_rate="
                                f"{last_eval['uspto/valid_smiles_rate']:.4f} < 0.05"
                            )
                        if last_eval["eval/mode_collapse_ratio"] > 0.5:
                            raise RuntimeError(
                                f"Kill criterion hit at step {step}: mode_collapse_ratio="
                                f"{last_eval['eval/mode_collapse_ratio']:.4f} > 0.5"
                            )

            if step >= args.max_steps:
                done = True
                break
        epoch += 1

    ckpt_path = None
    if rank == 0 and args.write_checkpoint:
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        ckpt_path = outdir / f"step{step}.pth"
        inner = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        torch.save({"model": inner.state_dict(), "step": step, "recipe": asdict(recipe)}, ckpt_path)
        wandb.log({"checkpoint_path": str(ckpt_path)}, step=step)

    summary: dict[str, Any] = {
        "rank": rank,
        "world_size": world_size,
        "is_ddp": is_ddp,
        "run_name": args.run_name,
        "wandb_url": run.url if run else None,
        "steps_completed": step,
        "elapsed_s": time.time() - t0,
        "checkpoint_path": str(ckpt_path) if ckpt_path else None,
        "last_eval": last_eval,
        "target_format": target_format,
        "git_commit": args.git_commit,
    }
    if rank == 0 and run is not None:
        wandb.finish()
    if is_ddp:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return summary


def _default_run_name() -> str:
    return time.strftime("mn-baseline-swinb384-%Y%m%d-%H%M%S")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--pubchem_csv", type=_data_path_arg, default=None)
    p.add_argument("--uspto_csv", type=_data_path_arg, default=None)
    p.add_argument("--uspto_image_root", type=_data_path_arg, default=None)
    p.add_argument("--uspto_path_strip_prefix", type=str, default="uspto_mol/")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_pubchem_rows", type=int, default=None)
    p.add_argument("--max_uspto_rows", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=60)
    p.add_argument(
        "--eval_n",
        type=int,
        default=64,
        help=(
            "In-training eval probe size (USPTO benchmark subset). 0 disables the "
            "probe entirely, so a launch check runs with no eval-benchmark download."
        ),
    )
    p.add_argument("--eval_every_steps", type=int, default=20)
    p.add_argument("--backbone_pretrained", action="store_true")
    p.add_argument("--run_name", type=str, default=_default_run_name())
    p.add_argument(
        "--output_dir",
        type=_data_path_arg,
        default=str(checkpoints_dir("ocsr", "preflight")),
    )
    p.add_argument("--write_checkpoint", action="store_true")
    p.add_argument("--git_commit", type=str, default="unknown")
    p.add_argument(
        "--min_step_for_kill",
        type=int,
        default=2000,
        help=(
            "Number of training steps before the eval-based kill criteria are "
            "enforced. Defaults to 2000 (rough proxy for 'epoch 2'). Preflight "
            "runs set this very high so they never trigger the kill."
        ),
    )
    p.add_argument("--cosine_lr", action="store_true", help="Use cosine LR with 2 percent warmup.")
    p.add_argument(
        "--resume_from",
        type=_data_path_arg,
        default=None,
        help="Optional path to a self-trained checkpoint .pth to warm-start from.",
    )
    p.add_argument(
        "--checkpoint_every_steps",
        type=int,
        default=0,
        help="If >0 and --write_checkpoint is set, save a checkpoint every N steps.",
    )
    p.add_argument(
        "--encoder_lr",
        type=float,
        default=None,
        help="Override encoder LR. Defaults to recipe.encoder_lr.",
    )
    p.add_argument(
        "--decoder_lr",
        type=float,
        default=None,
        help="Override decoder LR. Defaults to recipe.decoder_lr.",
    )
    p.add_argument(
        "--label_smoothing",
        type=float,
        default=None,
        help="Override label smoothing. Defaults to recipe.label_smoothing.",
    )
    p.add_argument(
        "--uspto_weighted_sampler",
        action="store_true",
        help="Use USPTO-only weighted sampler that oversamples stereo + long rows.",
    )
    p.add_argument("--stereo_weight", type=float, default=4.0)
    p.add_argument("--long_threshold", type=int, default=80)
    p.add_argument("--long_weight", type=float, default=4.0)
    p.add_argument(
        "--augment",
        action="store_true",
        help="Enable chemistry-safe image augmentation on both PubChem and USPTO.",
    )
    p.add_argument("--aug_p_clean", type=float, default=0.3)
    p.add_argument("--aug_p_each", type=float, default=0.4)
    p.add_argument("--bf16_amp", action="store_true", help="Train with bfloat16 autocast.")
    p.add_argument(
        "--swin_large", action="store_true", help="Use Swin-Large 384 instead of Swin-Base."
    )
    p.add_argument(
        "--encoder_name", type=str, default=None, help="Override vision encoder model name."
    )
    p.add_argument("--input_size", type=int, default=None, help="Override input image size.")
    p.add_argument(
        "--extra_pubchem_csv",
        type=_data_path_arg,
        default=None,
        help="Optional second PubChem-format CSV (e.g., adjacent-ring mined). Concatenated to main PubChem.",
    )
    p.add_argument(
        "--extra_pubchem_repeat",
        type=int,
        default=1,
        help="Repeat the extra PubChem dataset N times to oversample.",
    )
    p.add_argument("--max_extra_pubchem_rows", type=int, default=None)
    p.add_argument(
        "--stereo_csv",
        type=_data_path_arg,
        default=None,
        help="Optional Stereo-200K-format CSV (SMILES,file_path). Treated as USPTO-format dataset.",
    )
    p.add_argument("--stereo_image_root", type=_data_path_arg, default=None)
    p.add_argument("--stereo_repeat", type=int, default=1)
    p.add_argument("--max_stereo_rows", type=int, default=None)
    p.add_argument(
        "--realimg_csv",
        type=_data_path_arg,
        default=None,
        help="Optional real-image USPTO-format CSV (CLEF/JPO etc).",
    )
    p.add_argument("--realimg_image_root", type=_data_path_arg, default=None)
    p.add_argument("--realimg_repeat", type=int, default=1)
    p.add_argument("--max_realimg_rows", type=int, default=None)
    p.add_argument(
        "--rdkit_pubchem_csv",
        type=_data_path_arg,
        default=None,
        help="PubChem-format CSV rendered with RDKit (instead of/alongside Indigo).",
    )
    p.add_argument("--max_rdkit_pubchem_rows", type=int, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    summary = train(args)
    if summary["rank"] == 0:
        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
