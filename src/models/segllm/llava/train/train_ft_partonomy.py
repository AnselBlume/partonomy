#!/usr/bin/env python
"""
Fine-tune SegLLM on the ExplanatorySegDataset.
Author: you
"""

import os, sys, time, json, torch, wandb, shutil, types
from functools import partial
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
import transformers, deepspeed
from llava.train.llava_trainer import LLaVATrainer
from llava.model import LlavaLlamaForCausalLM, LlavaConfig
from llava.train.seg_register.register_dataset import Register as COCORegister
from llava import conversation as conversation_lib

from utils.explanatory_seg_dataset import ExplanatorySegDataset
from utils.explanatory_seg_datasets_adapter import ExplanatorySegDatasetsAdapter
from utils.question_type import QuestionType


# ---------- Helpers -----------------------------------------------------------

def parse_args():
    import argparse
    p = argparse.ArgumentParser("SegLLM ExplanatorySeg finetuning")
    p.add_argument("--output_dir", type=str, default="./runs/segllm_explanatory")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--precision", type=str, default="bf16", choices=["fp32","fp16","bf16"])
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="segllm-explanatory")
    p.add_argument("--vision_tower", default="openai/clip-vit-large-patch14")
    p.add_argument("--image_size", type=int, default=1024)
    p.add_argument("--local_rank", type=int, default=0)
    p.add_argument("--dataset_json", required=True,
                   help="Path to ExplanatorySeg dataset JSON (train split)")
    p.add_argument("--val_dataset_json", default=None)
    p.add_argument("--partonomy_dataset_split", default="partimagenet|pascal_part|paco_lvis|partonomy")
    p.add_argument("--sample_one_question_per_image", action="store_true")
    p.add_argument("--random_seed", type=int, default=42)
    return p.parse_args()


# ---------- Dataset adapters --------------------------------------------------

def make_explanatory_datasets(args, tokenizer):
    splits = args.partonomy_dataset_split.split("|")
    train_qtypes = [
        QuestionType.IDENTIFICATION_WITH_LABEL,
        QuestionType.POSITIVE_WITH_LABEL,
        QuestionType.NEGATIVE_WITH_LABEL,
    ]
    train_sets = []
    for qt in train_qtypes:
        for split in splits:
            dataset_path = args.dataset_json.format(dataset_split=split)
            print(f">> Loading {dataset_path}")
            train_sets.append(
                ExplanatorySegDataset(
                    dataset_path,
                    tokenizer,
                    args.vision_tower,
                    qt,
                    sample_one_question_per_image=args.sample_one_question_per_image,
                    random_seed=args.random_seed
                )
            )
    train_adapter = ExplanatorySegDatasetsAdapter(train_sets, indexing_strategy="concatenate")

    val_adapter = None
    if args.val_dataset_json:
        val_sets = []
        for split in splits:
            dataset_path = args.val_dataset_json.format(dataset_split=split)
            val_sets.append(
                ExplanatorySegDataset(
                    dataset_path,
                    tokenizer,
                    args.vision_tower,
                    QuestionType.IDENTIFICATION_WITH_LABEL,
                    model_str="segllm"
                )
            )
        val_adapter = ExplanatorySegDatasetsAdapter(val_sets, indexing_strategy="concatenate", inference=True)
    return train_adapter, val_adapter


# ---------- Training loop -----------------------------------------------------

def main():
    args = parse_args()
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda", args.local_rank)
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(args.output_dir) if args.local_rank == 0 else None

    if args.wandb and args.local_rank == 0:
        wandb.init(project=args.wandb_project, name=os.path.basename(args.output_dir), config=vars(args))

    # ----- Load model & tokenizer -----
    model_name = "Marlo-Z/SegLLM/all_data_checkpoint"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, padding_side="right", use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token
    model = LlavaLlamaForCausalLM.from_pretrained(model_name)
    model.config.use_cache = False

    # enable bf16/fp16
    if args.precision == "bf16":
        model.to(torch.bfloat16)
    elif args.precision == "fp16":
        model.to(torch.float16)
    model.to(device)

    # attach processors
    vision_tower = model.get_vision_tower()
    data_args = types.SimpleNamespace()
    data_args.image_processor = vision_tower.image_processor
    data_args.mask_processor = model.get_segmentator().process_images
    data_args.register = COCORegister(data_args)
    data_args.is_multimodal = True
    data_args.image_aspect_ratio = "square"
    data_args.image_grid_pinpoints = None
    model.config.image_aspect_ratio = "square"

    # ----- Load dataset -----
    train_adapter, val_adapter = make_explanatory_datasets(args, tokenizer)
    train_loader = torch.utils.data.DataLoader(
        train_adapter, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_adapter, batch_size=1, shuffle=False, num_workers=2
    ) if val_adapter else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.precision in ["fp16","bf16"])

    # ----- Training -----
    step = 0
    model.train()
    for epoch in range(args.epochs):
        for batch in train_loader:
            step += 1
            batch = {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if args.precision=="bf16" else torch.float16, enabled=args.precision!="fp32"):
                out = model(**batch)
                loss = out["loss"] if isinstance(out, dict) else out
            scaler.scale(loss / args.grad_accum_steps).backward()
            if step % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            if step % 10 == 0 and args.local_rank == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                if args.wandb:
                    wandb.log({"train/loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=step)
            if step % 100 == 0 and args.local_rank == 0:
                print(f"[{epoch}|{step}] loss={loss.item():.4f}")

        # simple eval
        if val_loader and args.local_rank == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for vbatch in val_loader:
                    vbatch = {k:v.to(device) if torch.is_tensor(v) else v for k,v in vbatch.items()}
                    out = model(**vbatch)
                    vloss = out["loss"] if isinstance(out, dict) else out
                    val_loss += vloss.item()
                val_loss /= len(val_loader)
            writer.add_scalar("val/loss", val_loss, epoch)
            if args.wandb:
                wandb.log({"val/loss": val_loss}, step=step)
            model.train()

        # save checkpoint
        if args.local_rank == 0:
            ckpt_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, os.path.join(ckpt_dir, "trainer_state.pt"))
            print(f"Saved checkpoint to {ckpt_dir}")

    if args.local_rank == 0:
        writer.close()
        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
