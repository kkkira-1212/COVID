import torch
import torch.nn.functional as F


def to_device(data_dict, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data_dict.items()}


def todevice(data_dict, device):
    return to_device(data_dict, device)


def getregressiontarget(data):
    if data.get("X_next", None) is not None:
        return data["X_next"]
    for key in ("NewDeaths_ret_next", "return_next", "target_return", "y_reg"):
        if key in data:
            return data[key]
    raise KeyError("No regression target found (expected X_next or return-style target).")


def _count_coarse_groups(fine_batches, mapping, build_batch_mapping):
    total_loss_count = 0
    for batch_idx_fine in fine_batches:
        _, valid_mask, _, _, fine_to_coarse_map = build_batch_mapping(
            batch_idx_fine, mapping
        )
        if valid_mask.sum() > 0:
            total_loss_count += len(fine_to_coarse_map)
    return total_loss_count


def _update_multiscale_stats(stats, mse_fine, mse_coarse, lu_loss):
    stats["mse_fine_sum"] += mse_fine.item()
    stats["mse_coarse_sum"] += mse_coarse.item()
    if lu_loss is not None:
        stats["lu_sum"] += lu_loss.item()
        stats["lu_count"] += 1


def _log_multiscale_epoch(stats, n_batches, loss_scalar, use_lu, lambda_u, ep, epochs):
    if (ep + 1) % 10 != 0 and ep != 0 and ep != epochs - 1:
        return
    if n_batches > 0:
        mse_fine_avg = stats["mse_fine_sum"] / n_batches
        mse_coarse_avg = stats["mse_coarse_sum"] / n_batches
    else:
        mse_fine_avg = 0.0
        mse_coarse_avg = 0.0
    if use_lu and stats["lu_count"] > 0:
        lu_avg = stats["lu_sum"] / stats["lu_count"]
        print(
            f"  Loss: {loss_scalar:.6f}, Batches: {n_batches} "
            f"(mse_fine={mse_fine_avg:.6f}, mse_coarse={mse_coarse_avg:.6f}, "
            f"lu={lu_avg:.6f}, lambda_u={lambda_u})",
            flush=True
        )
    else:
        print(
            f"  Loss: {loss_scalar:.6f}, Batches: {n_batches} "
            f"(mse_fine={mse_fine_avg:.6f}, mse_coarse={mse_coarse_avg:.6f})",
            flush=True
        )


def _train_coarse_group(
    fine_indices_list,
    batch_idx_fine,
    unique_coarse_indices,
    coarse_idx,
    z_fine_batch,
    X_next_fine_batch,
    z_coarse_batch,
    X_next_coarse_batch,
    head,
    compute_lu_loss,
    lu_predictor,
    lu_detach_coarse,
    lambda_u,
    total_loss_count,
    idx_in_group,
    group_count,
    stats,
):
    coarse_pos = (unique_coarse_indices == coarse_idx).nonzero(as_tuple=True)[0]
    if len(coarse_pos) == 0:
        return 0.0, 0
    coarse_pos = coarse_pos[0]

    fine_positions_in_batch = []
    for fi in fine_indices_list:
        pos = torch.where(batch_idx_fine == fi)[0]
        if len(pos) > 0:
            fine_positions_in_batch.append(pos[0].item())

    if len(fine_positions_in_batch) == 0:
        return 0.0, 0

    fine_positions_tensor = torch.tensor(fine_positions_in_batch, device=z_fine_batch.device)
    z_fine_subset = z_fine_batch[fine_positions_tensor]
    X_next_fine_subset = X_next_fine_batch[fine_positions_tensor]
    z_coarse_subset = z_coarse_batch[coarse_pos:coarse_pos+1].expand(len(fine_positions_in_batch), -1)
    X_next_coarse_subset = X_next_coarse_batch[coarse_pos:coarse_pos+1]

    pred_fine_batch, pred_coarse_batch = head(z_fine_subset, z_coarse_subset)

    mse_fine = F.mse_loss(pred_fine_batch, X_next_fine_subset)
    mse_coarse = F.mse_loss(pred_coarse_batch.mean(dim=0, keepdim=True), X_next_coarse_subset)
    batch_loss = mse_fine + mse_coarse

    Lu = compute_lu_loss(
        z_fine_subset,
        z_coarse_subset,
        lu_predictor,
        lu_detach_coarse,
    )
    _update_multiscale_stats(stats, mse_fine, mse_coarse, Lu)
    if Lu is not None:
        batch_loss = batch_loss + lambda_u * Lu

    total_loss_scalar = batch_loss.item()
    if total_loss_count > 0:
        scaled_loss = batch_loss / total_loss_count
        is_last_in_batch = (idx_in_group == group_count - 1)
        scaled_loss.backward(retain_graph=not is_last_in_batch)
        del scaled_loss

    del pred_fine_batch, pred_coarse_batch, mse_fine, mse_coarse, batch_loss
    if Lu is not None:
        del Lu
    del z_fine_subset, X_next_fine_subset, z_coarse_subset, X_next_coarse_subset
    return total_loss_scalar, 1


def _validate_multiscale(enc_coarse, head, data, batch_size, device, roc_auc_score, np):
    with torch.no_grad():
        idx_coarse_val = data['idx_coarse_val']

        pred_coarse_val_list = []
        for i in range(0, len(idx_coarse_val), batch_size):
            end_idx = min(i + batch_size, len(idx_coarse_val))
            batch_idx = idx_coarse_val[i:end_idx]
            X_coarse_batch = data['X_coarse'][batch_idx].to(device)
            z_coarse_batch = enc_coarse(X_coarse_batch)
            _, pred_coarse_batch = head(z_coarse_batch, z_coarse_batch)
            pred_coarse_val_list.append(pred_coarse_batch.cpu())

        if len(pred_coarse_val_list) > 0:
            pred_coarse_val = torch.cat(pred_coarse_val_list, dim=0)
            pred_coarse_val_mean = pred_coarse_val.mean(dim=1)
            X_next_coarse_val_mean = data['X_next_coarse'][idx_coarse_val].mean(dim=1)
            residual_coarse = (X_next_coarse_val_mean - pred_coarse_val_mean).abs()
            res_val = residual_coarse.cpu().numpy()
            if data['y_coarse'] is not None:
                y_val = data['y_coarse'][idx_coarse_val].cpu().numpy()
                unique_classes = np.unique(y_val)
                if len(unique_classes) < 2:
                    return None
                try:
                    return roc_auc_score(y_val, res_val)
                except (ValueError, Exception):
                    return None
            return None
        return 0.0
