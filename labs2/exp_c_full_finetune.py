# Solution C — Full fine-tuning @384
#
# **Author: Vitaliy**
#
# We take a clean copy of the baseline model, scale the position embeddings to $24 \times 24$ ($384$ px),
# configure differential learning rates, and train for 11 epochs (budget limit of 99,275 training images total).
# If a saved checkpoint exists, we load it immediately to avoid unnecessary retraining.
# `Deit_starter.py` calls `run_c`; running this file directly does the same.
#
# Original run (Colab T4), fixed `evaluate()` @384: **97.58%** (baseline @224: 97.20%).

import copy

import torch.optim as optim

from common import MAX_EPOCHS, evaluate, resize_pos_embed_, save_result
from training import imagenet_ce_loss, load_checkpoint, make_loader, save_checkpoint, train

# Checkpoint file naming constants
MODEL_CHECKPOINT_FILE = "model_384_full.pth"
LOG_CHECKPOINT_FILE = "history_full.json"


def run_c(base_model, resolution=384):
    """Execute Solution C: Full end-to-end fine-tuning at target resolution (default: 384 px)."""
    # 1. Clone baseline model, resize pos_embed to target resolution grid, and unfreeze all weights
    cloned_base_model = copy.deepcopy(base_model)
    full_finetuned_model = resize_pos_embed_(cloned_base_model, resolution)
    full_finetuned_model.requires_grad_(True)

    # 2. Checkpoint detection: load pre-existing weights and training log if available
    training_history = load_checkpoint(full_finetuned_model, MODEL_CHECKPOINT_FILE, LOG_CHECKPOINT_FILE)

    # 3. If no saved checkpoint exists, execute end-to-end training
    if training_history is None:
        all_named_parameters = dict(full_finetuned_model.named_parameters())

        # Group 1: Pre-trained Transformer backbone (blocks 0–11)
        # Gentle learning rate (1e-5) preserves ImageNet-1k representations while adjusting attention projections
        backbone_parameters = [
            param
            for name, param in all_named_parameters.items()
            if 'head' not in name and 'pos_embed' not in name
        ]

        # Group 2: Resized positional embedding
        # 10x higher learning rate (1e-4) allows rapid coordinate recalibration to the 24x24 grid
        pos_embed_parameters = [
            param
            for name, param in all_named_parameters.items()
            if 'pos_embed' in name
        ]

        # Group 3: Classification heads (head and head_dist)
        # 10x higher learning rate (1e-4) enables rapid convergence on the 10 Imagewoof dog classes
        head_parameters = [
            param
            for name, param in all_named_parameters.items()
            if 'head' in name
        ]

        # Configure AdamW optimizer with differential learning rates across groups
        optimizer = optim.AdamW([
            {'params': backbone_parameters, 'lr': 1e-5},
            {'params': pos_embed_parameters, 'lr': 1e-4},
            {'params': head_parameters, 'lr': 1e-4},
        ], weight_decay=0.05)

        # Cosine Annealing learning rate schedule across the 11-epoch budget
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

        # 4. Train across all 11 epochs with intermediate validation evaluations
        train_loader = make_loader(resolution)
        eval_callback = lambda model: evaluate(model, resolution)
        training_history = train(
            full_finetuned_model,
            loader=train_loader,
            loss_fn=imagenet_ce_loss,
            optimizer=optimizer,
            epochs=MAX_EPOCHS,
            scheduler=scheduler,
            eval_fn=eval_callback,
            label="C",
        )

        # Save trained weights and complete history log
        save_checkpoint(full_finetuned_model, training_history, MODEL_CHECKPOINT_FILE, LOG_CHECKPOINT_FILE)

    # 5. Evaluate final top-1 accuracy on the validation split
    final_accuracy = evaluate(full_finetuned_model, resolution)
    print(f"Solution C (Full Fine-Tuning @{resolution}): {final_accuracy:.2%}")
    save_result(f"c_full_finetune_{resolution}", final_accuracy)

    return full_finetuned_model, final_accuracy


if __name__ == "__main__":
    from common import load_pretrained
    run_c(load_pretrained())
