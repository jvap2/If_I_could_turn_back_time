"""
training_loop_v2.py
====================
Modified train_val_loop_HPO with ema_start_epoch parameter.

Key change vs training_loop.py:
  - EMA updates only fire when epoch >= ema_start_epoch (default = prune_epoch).
    Prior to that epoch, no EMA tracking occurs — the EMA model is initialised
    but never updated, so its parameters stay at the pre-training init weights
    until the gate opens.  After ema_start_epoch the loop updates every epoch
    and saves the best EMA checkpoint separately from the best raw checkpoint.

Upload this file alongside custom_optimizer.py / custom_schedulers.py /
cuda_helpers.py to your Drive folder so the Colab notebook can import it.
"""

from custom_optimizer import (
    Prune_Score_v3,
    Prune_Score_Reset,
    train_one_step_prune_HPO,
    train_one_step_prune,
)
from time import time
from cuda_helpers import get_memory_free_MiB
import torch
import torch.nn as nn


def train_val_loop_HPO(
    model, train_dataloader, val_dataloader,
    optimizer, loss_fn, scheduler,
    accuracy, top5accuracy, writer, device,
    experiment_name, model_name, timestamp,
    train_filename, val_filename, log_filename,
    sparsity_filename, prune_filename, debug_filename, jenks_filename,
    prune_count=0, one_update=False, EPOCHS=100, sparsity=0.0,
    prune_epoch_list=None, prune_epoch=0, prune_between=1, prune_ratio=0.5,
    one_shot=False, mask=True, mag_prune=False, bias_prune=False,
    kill_velocity=False, l2=0.0, lambda_=0.0, warmup_epochs=0, min_epochs=1,
    elem_bias=False, accum_steps=1, weight_reset=False,
    ema_model=None, ema_start_epoch=0,
):
    """
    ema_start_epoch : int
        Epoch number at which EMA updates begin.  Set equal to prune_epoch so
        that EMA only tracks the pruned model.  Before this epoch the ema_model
        parameter is ignored entirely (no updates, no validation inference).
    """
    no_jenks = False
    l2 = True
    mag_prune = True
    epoch = 0

    names = [
        name for name, layer in model.named_modules()
        if isinstance(layer, (nn.Conv2d, nn.Linear))
    ]
    imp_names = [names[0], names[-1]]

    print(f'Prune epoch list: {prune_epoch_list}')
    print(f'Prune epoch: {prune_epoch} | EMA starts: epoch {ema_start_epoch}')

    max_val_acc     = 0.0
    max_ema_val_acc = 0.0

    while (sparsity < prune_ratio and epoch < EPOCHS) or epoch <= min_epochs:
        epoch += 1
        print(f'Epoch: {epoch}')
        model.train()

        with open(train_filename, 'a') as f:
            print(f'Epoch: {epoch}| LR: {scheduler.get_last_lr()}', file=f)

        count = 0
        train_loss, train_acc, train_top5acc = 0.0, 0.0, 0.0
        start = time()
        print(f'Memory free: {get_memory_free_MiB(0)} MiB')

        if sparsity >= prune_ratio:
            no_jenks = True

        # ── Pruning ───────────────────────────────────────────────────────────
        at_prune = (epoch == prune_epoch or
                    (epoch > prune_epoch and (epoch - prune_epoch) % prune_between == 0))
        if at_prune:
            if not weight_reset:
                if one_shot and epoch == prune_epoch:
                    print('Pruning weights (one-shot Jenks)')
                    Prune_Score_v3(
                        model, optimizer, epoch, imp_names, prune_epoch_list,
                        mask=True, mag_prune=mag_prune, filter_based=False,
                        bias_prune=bias_prune, prune_file=prune_filename,
                    )
                    prune_count += 1
                elif (not one_shot and epoch >= prune_epoch
                      and sparsity < prune_ratio and epoch % 5 == 0):
                    print('Pruning weights (iterative)')
                    Prune_Score_v3(
                        model, optimizer, epoch, imp_names, prune_epoch_list,
                        mask=True, mag_prune=mag_prune, filter_based=False,
                        bias_prune=bias_prune, prune_file=prune_filename,
                    )
                    prune_count += 1
            else:
                if one_shot and epoch == prune_epoch:
                    print('Pruning weights with weight reset')
                    Prune_Score_Reset(
                        model, optimizer, epoch, imp_names, prune_epoch_list,
                        mask=True, mag_prune=mag_prune, filter_based=False,
                        bias_prune=bias_prune, prune_file=prune_filename,
                    )
                    prune_count += 1

            non_zero = sum(torch.count_nonzero(p)
                           for p in model.parameters() if p.dim() in [2, 4])
            total_p  = sum(p.numel()
                           for p in model.parameters() if p.dim() in [2, 4])
            sparsity = 1 - non_zero / total_p
            with open(sparsity_filename, 'a') as f:
                print(f'Epoch: {epoch}| Sparsity: {sparsity:.5f}', file=f)

        # ── Training step ─────────────────────────────────────────────────────
        ema_active = (ema_model is not None and epoch >= ema_start_epoch)

        if one_update:
            count += 1
            torch.cuda.empty_cache()
            acc, acc5, loss = train_one_step_prune_HPO(
                model, train_dataloader, optimizer, loss_fn, epoch, warmup_epochs,
                prune_epochs=prune_epoch, no_jenks=no_jenks, bias_prune=bias_prune,
                filter_based=False, mask=mask, L2=l2, lambda_=lambda_,
                debug=True, debugfile=debug_filename, jenksfile=jenks_filename,
                mag=False, elem_bias=elem_bias, accumulation_steps=accum_steps,
            )
            if mask and epoch > prune_epoch:
                for _, param in model.named_parameters():
                    param.data = param.data * optimizer.state[param]['mask']
            if ema_active:
                ema_model.update_parameters(model)
            l2_reg   = sum(torch.norm(p) ** 2 for p in model.parameters())
            lr_prune = sum(torch.norm(p) ** 2 for p in model.parameters()
                           if p.dim() in [2, 4])
            with open(train_filename, 'a') as f:
                print(f'Iter: {count}| Loss: {loss:.5f}| Acc: {acc.item():.5f}'
                      f'| Top5: {acc5.item():.5f}| L2: {l2_reg:.5f}'
                      f'| Lp: {lr_prune:.5f}', file=f)
        else:
            for X, y in train_dataloader:
                torch.cuda.empty_cache()
                count += 1
                X, y = X.to(device), y.to(device)
                acc, acc5, loss = train_one_step_prune(
                    model, X, y, optimizer, loss_fn, epoch, warmup_epochs,
                    prune_epochs=prune_epoch, no_jenks=no_jenks,
                    filter_based=False, mask=mask, L2=l2, lambda_=lambda_,
                    debug=True, debugfile=debug_filename, jenksfile=jenks_filename,
                )
                if mask and epoch > prune_epoch:
                    for _, param in model.named_parameters():
                        param.data = param.data * optimizer.state[param]['mask']
                if ema_active:
                    ema_model.update_parameters(model)
                train_loss     += loss.item()
                train_top5acc  += acc5.item()
                train_acc      += acc.item()
            l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
            with open(train_filename, 'a') as f:
                print(f'Iter: {count}| Loss: {train_loss/count:.5f}'
                      f'| Acc: {train_acc/count:.5f}'
                      f'| Top5: {train_top5acc/count:.5f}'
                      f'| L2: {l2_reg:.5f}', file=f)

        print(f'Epoch {epoch} time: {time()-start:.1f}s')
        scheduler.step()
        with open(log_filename, 'a') as f:
            print(f'Epoch: {epoch}| LR: {scheduler.get_last_lr()}', file=f)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        if ema_active:
            ema_model.eval()

        with torch.inference_mode():
            with open(val_filename, 'a') as f:
                print(f'Epoch: {epoch}', file=f)
            val_loss = val_acc = val_top5acc = 0.0
            ema_val_acc = ema_val_top5acc = 0.0
            count_val = 0

            for X, y in val_dataloader:
                count_val += 1
                X, y = X.to(device), y.to(device)
                y_pred   = model(X)
                val_loss += loss_fn(y_pred, y).item()
                val_acc  += accuracy(y_pred, y)
                val_top5acc += top5accuracy(y_pred, y)
                if ema_active:
                    ema_pred = ema_model(X)
                    ema_val_acc     += accuracy(ema_pred, y)
                    ema_val_top5acc += top5accuracy(ema_pred, y)
                with open(val_filename, 'a') as f:
                    print(f'Iter: {count_val}| Loss: {val_loss/count_val:.5f}'
                          f'| Acc: {val_acc/count_val:.5f}'
                          f'| Top5: {val_top5acc/count_val:.5f}', file=f)

            if val_acc / count_val > max_val_acc and epoch > prune_epoch:
                max_val_acc = val_acc / count_val
                torch.save(
                    model.state_dict(),
                    f'models/best_{timestamp}_{experiment_name}_{model_name}.pth',
                )

            if ema_active:
                ema_avg = ema_val_acc / count_val
                with open(val_filename, 'a') as f:
                    print(f'EMA Acc: {ema_avg:.5f}'
                          f'| EMA Top5: {ema_val_top5acc/count_val:.5f}', file=f)
                if ema_avg > max_ema_val_acc:
                    max_ema_val_acc = ema_avg
                    torch.save(
                        ema_model.module.state_dict(),
                        f'models/best_ema_{timestamp}_{experiment_name}_{model_name}.pth',
                    )

        writer.add_scalars('Loss',     {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc,  'val': val_acc},  epoch)

    # ── Final sparsity ────────────────────────────────────────────────────────
    non_zero = sum(torch.count_nonzero(p)
                   for p in model.parameters() if p.dim() in [2, 4])
    total_p  = sum(p.numel()
                   for p in model.parameters() if p.dim() in [2, 4])
    sparsity = 1 - non_zero / total_p
    with open(sparsity_filename, 'a') as f:
        print(f'Epoch: {epoch}| Final Sparsity: {sparsity:.5f}', file=f)
    with open(val_filename, 'a') as f:
        print(f'Best val acc: {max_val_acc:.5f} | Best EMA acc: {max_ema_val_acc:.5f}',
              file=f)
