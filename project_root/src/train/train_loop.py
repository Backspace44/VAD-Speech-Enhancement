"""Epoch-level training loop helpers."""

from __future__ import annotations

import torch
from torch.amp import autocast
from tqdm import tqdm

from src import config
from src.train.checkpointing import save_best_model_checkpoint, save_checkpoint
from src.train.train_setup import clip_gradients, step_scheduler_batch


def run_training(
    *,
    model,
    optimizer,
    scheduler,
    scheduler_type: str,
    criterion,
    train_loader,
    val_loader,
    start_epoch: int,
    logger,
    writer,
    scaler,
    args,
    exp_dir,
):
    """Run the train/validation loop and return summary state."""
    best_val_loss = float("inf")
    best_epoch = None
    early_stop_counter = 0
    early_stop_patience = config.TRAINING_CONFIG.get("early_stopping_patience", 10)
    batch_save_interval = 50
    history: list[dict] = []

    for epoch in range(start_epoch, config.TRAINING_CONFIG["num_epochs"]):
        logger.info(f"\n{'='*70}")
        logger.info(f"EPOCH {epoch + 1}/{config.TRAINING_CONFIG['num_epochs']}")
        logger.info(f"{'='*70}")

        avg_train_loss, num_batches_processed = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_type=scheduler_type,
            criterion=criterion,
            train_loader=train_loader,
            epoch=epoch,
            logger=logger,
            writer=writer,
            scaler=scaler,
            args=args,
            exp_dir=exp_dir,
            batch_save_interval=batch_save_interval,
        )
        logger.info(f"Train Loss: {avg_train_loss:.6f} (processed {num_batches_processed} batches)")

        avg_val_loss = validate_one_epoch(
            model=model,
            criterion=criterion,
            val_loader=val_loader,
            scaler=scaler,
        )
        logger.info(f"Val Loss: {avg_val_loss:.6f}")

        improved = avg_val_loss < best_val_loss
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": float(avg_train_loss),
            "val_loss": float(avg_val_loss),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "batches_processed": int(num_batches_processed),
            "best_val_loss_so_far": float(min(best_val_loss, avg_val_loss)),
            "improved": bool(improved),
        }
        history.append(epoch_record)

        if writer:
            writer.add_scalar("train_loss", avg_train_loss, epoch)
            writer.add_scalar("val_loss", avg_val_loss, epoch)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            elif scheduler_type != "OneCycleLR":
                scheduler.step()

        if improved:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            early_stop_counter = 0
            checkpoint_path = save_best_model_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                avg_train_loss=avg_train_loss,
                avg_val_loss=avg_val_loss,
                exp_dir=exp_dir,
                cli_args=vars(args),
            )
            logger.info(f" Best model saved: {checkpoint_path}")
        else:
            early_stop_counter += 1
            logger.info(f"No improvement ({early_stop_counter}/{early_stop_patience})")
            if early_stop_counter >= early_stop_patience:
                logger.info("Early stopping triggered!")
                break

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


def train_one_epoch(
    *,
    model,
    optimizer,
    scheduler,
    scheduler_type: str,
    criterion,
    train_loader,
    epoch: int,
    logger,
    writer,
    scaler,
    args,
    exp_dir,
    batch_save_interval: int,
):
    """Train one epoch and handle periodic checkpointing."""
    model.train()
    train_loss = 0.0
    num_batches_processed = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch_idx, batch in enumerate(pbar):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            logger.info(f"\nReached max_batches limit ({args.max_batches}). Saving checkpoint...")
            avg_loss_so_far = train_loss / batch_idx if batch_idx > 0 else train_loss
            checkpoint_path = save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                batch_idx,
                avg_loss_so_far,
                exp_dir,
                name=f"checkpoint_epoch{epoch+1}_batch{batch_idx}_final",
                extra_state={"config": vars(args)},
            )
            logger.info(f" Final checkpoint saved: {checkpoint_path.name}")
            break

        noisy_mag = batch["noisy_mag"].to(config.DEVICE)
        ideal_mask = batch["ideal_mask"].to(config.DEVICE)
        optimizer.zero_grad()

        if scaler:
            with autocast("cuda"):
                predicted_mask = model(noisy_mag)
                loss = criterion(predicted_mask, ideal_mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = clip_gradients(model)
            scaler.step(optimizer)
            scaler.update()
            step_scheduler_batch(scheduler, scheduler_type)
        else:
            predicted_mask = model(noisy_mag)
            loss = criterion(predicted_mask, ideal_mask)
            loss.backward()
            grad_norm = clip_gradients(model)
            optimizer.step()
            step_scheduler_batch(scheduler, scheduler_type)

        train_loss += loss.item()
        num_batches_processed = batch_idx + 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if writer:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("batch_loss", loss.item(), global_step)
            if grad_norm is not None:
                writer.add_scalar("grad_norm", float(grad_norm), global_step)
            writer.add_scalar("batch_learning_rate", optimizer.param_groups[0]["lr"], global_step)

        if (batch_idx + 1) % batch_save_interval == 0:
            avg_loss_so_far = train_loss / (batch_idx + 1)
            checkpoint_path = save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                batch_idx,
                avg_loss_so_far,
                exp_dir,
                name=f"checkpoint_epoch{epoch+1}_batch{batch_idx+1}",
                extra_state={"config": vars(args)},
            )
            logger.info(f"\n Checkpoint saved at batch {batch_idx+1}: {checkpoint_path.name}")

    avg_train_loss = train_loss / num_batches_processed if num_batches_processed > 0 else 0.0
    return avg_train_loss, num_batches_processed


def validate_one_epoch(*, model, criterion, val_loader, scaler):
    """Run one validation pass and return average loss."""
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            noisy_mag = batch["noisy_mag"].to(config.DEVICE)
            ideal_mask = batch["ideal_mask"].to(config.DEVICE)

            if scaler:
                with autocast("cuda"):
                    predicted_mask = model(noisy_mag)
                    loss = criterion(predicted_mask, ideal_mask)
            else:
                predicted_mask = model(noisy_mag)
                loss = criterion(predicted_mask, ideal_mask)
            val_loss += loss.item()

    return val_loss / len(val_loader)
