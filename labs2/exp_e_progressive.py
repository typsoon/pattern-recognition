# # Solution E — Progressive resolution curriculum
#
# **Author: Vitaliy**
#
# Instead of jumping straight to 384, we transition through `224 -> 256 -> 320 -> 384`.
# To avoid positional embedding degradation (blur) from sequential interpolation, we always keep
# a pristine 224 model copy and interpolate directly from it to the targeted grid sizes.
# `Deit_starter.py` calls `run_e`; running this file directly does the same.
#
# Original run (Colab T4), fixed `evaluate()` @384: **97.38%** (baseline @224: 97.20%).

import copy

import torch.optim as optim

from common import MAX_EPOCHS, evaluate, resize_pos_embed_, save_result
from training import imagenet_ce_loss, load_checkpoint, make_loader, save_checkpoint, train

# Checkpoint file naming constants
MODEL_CHECKPOINT_FILE = "model_prog.pth"
LOG_CHECKPOINT_FILE = "history_prog.json"

# Curriculum schedule designed to strictly respect the 100,000 image compute budget:
# (stage_resolution, stage_epochs, stage_learning_rate)
SCHEDULE = [
    (224, 4, 5e-5),  # Stage 1: 4 epochs @ 224x224 (Grid: 14x14)
    (256, 4, 2e-5),  # Stage 2: 4 epochs @ 256x256 (Grid: 16x16)
    (320, 1, 1e-5),  # Stage 3: 1 epoch  @ 320x320 (Grid: 20x20)
    (384, 2, 5e-6),  # Stage 4: 2 epochs @ 384x384 (Grid: 24x24)
]  # Total = 11 epochs * 9,025 images = 99,275 images (< 100,000 cap)


def run_e(base_model, curriculum_schedule=SCHEDULE):
    """Train a copy of `base_model` progressively through `curriculum_schedule` stages."""
    # 1. Enforce compute budget constraint
    total_curriculum_epochs = sum(epochs for _, epochs, _ in curriculum_schedule)
    assert total_curriculum_epochs <= MAX_EPOCHS, (
        f"Budget exceeded: {total_curriculum_epochs} epochs > {MAX_EPOCHS} allowed"
    )

    # Extract target evaluation resolution from the final curriculum stage (384 px)
    final_target_resolution, _, _ = curriculum_schedule[-1]
    patch_size = base_model.patch_embed.patch_size[0]

    # 2. Checkpoint detection: create a container model at final resolution to probe for saved weights
    checkpoint_probe_model = copy.deepcopy(base_model)
    progressive_model = resize_pos_embed_(checkpoint_probe_model, final_target_resolution)
    curriculum_history = load_checkpoint(progressive_model, MODEL_CHECKPOINT_FILE, LOG_CHECKPOINT_FILE)

    # 3. If no pre-existing checkpoint is found, execute the multi-stage curriculum
    if curriculum_history is None:
        progressive_model = copy.deepcopy(base_model)
        curriculum_history = []

        for stage_resolution, stage_epochs, stage_learning_rate in curriculum_schedule:
            grid_dimension = stage_resolution // patch_size
            print(
                f"\nTransitioning to Resolution: {stage_resolution} "
                f"(Grid: {grid_dimension}x{grid_dimension}) with LR: {stage_learning_rate}"
            )

            # 3.1. Avoid compounding interpolation blur by cloning the pristine 14x14 base model
            # and interpolating positional embeddings directly to the new stage resolution.
            pristine_base_model = copy.deepcopy(base_model)
            stage_model = resize_pos_embed_(pristine_base_model, stage_resolution)

            # 3.2. Extract trained backbone and classification head weights from the previous stage,
            # deliberately omitting pos_embed so the freshly interpolated tensor is preserved.
            previous_stage_weights = progressive_model.state_dict()
            transferred_weights = {
                param_name: param_tensor
                for param_name, param_tensor in previous_stage_weights.items()
                if param_name != 'pos_embed'
            }

            # 3.3. Load matching weights into stage model (strict=False ignores missing pos_embed)
            stage_model.load_state_dict(transferred_weights, strict=False)

            # 3.4. Unfreeze all model parameters for training during this stage
            stage_model.requires_grad_(True)
            progressive_model = stage_model

            # 3.5. Configure stage optimizer with decaying learning rate
            stage_optimizer = optim.AdamW(
                progressive_model.parameters(), 
                lr=stage_learning_rate, 
                weight_decay=0.05
            )

            # 3.6. Train on the current stage's resolution and data loader
            stage_loader = make_loader(stage_resolution)
            stage_eval_callback = lambda model: evaluate(model, stage_resolution)
            stage_history = train(
                progressive_model,
                loader=stage_loader,
                loss_fn=imagenet_ce_loss,
                optimizer=stage_optimizer,
                epochs=stage_epochs,
                eval_fn=stage_eval_callback,
                label=f"E@{stage_resolution}",
            )

            # 3.7. Annotate epoch metric entries with the active stage resolution
            for epoch_metric in stage_history:
                history_entry = dict(epoch_metric, resolution=stage_resolution)
                curriculum_history.append(history_entry)

        # 4. Serialize final weights and complete curriculum history log
        save_checkpoint(progressive_model, curriculum_history, MODEL_CHECKPOINT_FILE, LOG_CHECKPOINT_FILE)

    # 5. Evaluate final validation accuracy at target resolution (384x384)
    final_accuracy = evaluate(progressive_model, final_target_resolution)
    print(f"Solution E (Progressive Curriculum @{final_target_resolution}): {final_accuracy:.2%}")
    save_result(f"e_progressive_{final_target_resolution}", final_accuracy)
    
    return progressive_model, final_accuracy


if __name__ == "__main__":
    from common import load_pretrained
    run_e(load_pretrained())
