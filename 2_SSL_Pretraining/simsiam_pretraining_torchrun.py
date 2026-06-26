import os
import json
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from increased_Att_PreactRes_UNet_pxshuffle import Att_PreactRes_UNet_Encoder
from multiobject_ssl_dataset import (
    MultiObjectSSS_Dataset,
    ImageNetTransforms_weakstrong,
    PSPTransforms_weakstrong,
)


def setup() -> None:
    """Initialize the distributed environment from torchrun-provided env vars.

    torchrun sets MASTER_ADDR/MASTER_PORT, RANK, WORLD_SIZE and LOCAL_RANK.
    """
    if not dist.is_available():
        raise RuntimeError('torch.distributed is not available')

    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')

    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


class SimSiamEncoderWrapper(Att_PreactRes_UNet_Encoder):
    """Wrap the encoder with global average pooling.

    Converts the feature map output [B, 1024, 6, 6] -> [B, 1024].
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = super().forward(x)      # [B, 1024, 6, 6]
        x = self.avgpool(x)         # [B, 1024, 1, 1]
        x = torch.flatten(x, 1)     # [B, 1024]
        return x


class ProjectionMLP(nn.Module):
    """3-layer projector with BN (fc -> bn -> relu, x2, then fc -> bn).

    The final layer has no ReLU and uses a non-affine BN, as recommended by the
    SimSiam paper.
    """
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class PredictionMLP(nn.Module):
    """2-layer bottleneck predictor (fc -> bn -> relu -> fc).

    The output layer has no BN or ReLU.
    """
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimSiam(nn.Module):
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature/projector dimension (default: 2048).
        pred_dim: hidden dimension of the predictor (default: 512).
        """
        super().__init__()
        self.encoder = base_encoder

        # Encoder output dimension.
        prev_dim = 1024

        self.projector = ProjectionMLP(in_dim=prev_dim, hidden_dim=dim, out_dim=dim)
        self.predictor = PredictionMLP(in_dim=dim, hidden_dim=pred_dim, out_dim=dim)

    def forward(self, x1, x2):
        """
        Inputs:
            x1, x2: the two augmented views of a batch.
        Returns:
            p1, p2, z1, z2: predictor outputs and projector targets.
        """
        f1 = self.encoder(x1)   # [B, 1024]
        f2 = self.encoder(x2)   # [B, 1024]

        z1 = self.projector(f1)  # [B, 2048]
        z2 = self.projector(f2)  # [B, 2048]

        p1 = self.predictor(z1)  # [B, 2048]
        p2 = self.predictor(z2)  # [B, 2048]

        return p1, p2, z1, z2


def negative_cosine_similarity(p, z):
    """SimSiam loss: D(p, z) = -(p * z).sum(dim=1).mean() on L2-normalized vectors.

    The target z is detached (stop-gradient).
    """
    z = z.detach()

    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)

    return -(p * z).sum(dim=1).mean()


def adjust_learning_rate(optimizer, init_lr, epoch, num_epochs):
    """Decay the learning rate with a cosine schedule."""
    cur_lr = init_lr * 0.5 * (1. + np.cos(np.pi * epoch / num_epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def adjust_learning_rate_warmup(optimizer, init_lr, epoch, num_epochs, warmup_epochs=10):
    """Linear warmup followed by a cosine decay schedule."""
    if epoch < warmup_epochs:
        cur_lr = init_lr * epoch / warmup_epochs
    else:
        cur_lr = init_lr * 0.5 * (
            1. + np.cos(np.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs))
        )

    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def train_model(args):
    """Main training routine for multi-node multi-GPU SimSiam pretraining."""
    global_rank = args.rank
    local_rank = args.local_rank

    print(f"\nNode {args.node_rank}, Local GPU {local_rank}, Global Rank {global_rank}\n")

    device = local_rank

    # Per-object data paths are read from a JSON config file: a list of
    # {"memmap_path": ..., "mask_path": ...} entries (see object_configs_example.json).
    with open(args.config, 'r') as f:
        configs = json.load(f)

    path_sav_outputs = args.output_dir
    os.makedirs(path_sav_outputs, exist_ok=True)
    height_bbox = width_bbox = args.bbox_size
    batch_size = args.batch_size

    # Select the augmentation pipeline:
    #   'imagenet' -> standard ImageNet-style SSL augmentations
    #   'psp'      -> our novel phase-shift-profilometry augmentations
    if args.aug == 'psp':
        transform = PSPTransforms_weakstrong(
            num_phase_shifts=args.num_phase_shifts,
            base_crop_size=args.base_crop_size,
            final_size=args.bbox_size,
        )
    else:
        transform = ImageNetTransforms_weakstrong(final_size=args.bbox_size)

    dataset = MultiObjectSSS_Dataset(
        object_configs=configs,
        width_bbox=width_bbox,
        height_bbox=height_bbox,
        data_augmentations=transform,
    )

    # Use the global rank for the DistributedSampler so each GPU sees a unique shard.
    train_sampler = DistributedSampler(
        dataset,
        num_replicas=args.world_size,
        rank=global_rank,
        shuffle=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # must be False when using a DistributedSampler
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    if global_rank == 0:
        print(f'\nTotal dataset size: {len(dataset)}')
        print(f'Batches per GPU: {len(train_loader)}')
        print(f'Total batches across all GPUs: {len(train_loader) * args.world_size}\n')

    # Encoder hyper-parameters.
    num_phase_shifts = args.num_phase_shifts
    in_channels = num_phase_shifts * 2 * 3
    features = [128, 256, 384, 512]

    torch.cuda.set_device(local_rank)

    base_encoder = SimSiamEncoderWrapper(in_channels=in_channels, features=features)
    model = SimSiam(base_encoder)

    if global_rank == 0 and args.init_weights is not None:
        # Optionally load a saved initialization so all ranks start identically.
        model.load_state_dict(torch.load(args.init_weights, weights_only=True))
    # No barrier needed: DDP.__init__() broadcasts rank-0 params to all ranks.

    # Apply SyncBN (must be done before .to(device)).
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    tot_params = sum([p.data.nelement() for p in model.parameters()])
    print(f'\nSimSiam arch has {tot_params} parameters\n')

    # Fix the learning rate for the predictor (per the SimSiam recipe).
    fix_pred_lr = True
    if fix_pred_lr:
        optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False},
                        {'params': model.projector.parameters(), 'fix_lr': False},
                        {'params': model.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    weight_decay = 1e-4
    momentum = 0.9

    total_batch_size = batch_size * args.world_size

    # Linear LR scaling is intentionally not used for this large batch size.
    init_lr = args.init_lr

    if global_rank == 0:
        print(f'\nTotal batch size across all GPUs: {total_batch_size}')
        print(f'Initial lr: {init_lr}\n')

    optimizer = torch.optim.SGD(optim_params, init_lr, momentum=momentum, weight_decay=weight_decay)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False
    )

    num_epochs = args.epochs
    warmup_epochs = args.warmup_epochs

    best_model_path = os.path.join(path_sav_outputs, f'simsiam_encoder_{args.aug}.pth')
    trainhist_path = os.path.join(path_sav_outputs, f'simsiam_encoder_{args.aug}_trainhist.pkl')
    resume_ckpt_path = os.path.join(path_sav_outputs, f'simsiam_resume_ckpt_{args.aug}.pth')
    start_epoch = 1
    train_history = dict()
    best_train_loss = 1.0

    # Resume from checkpoint if requested (all ranks load from the shared filesystem).
    if args.resume and os.path.isfile(resume_ckpt_path):
        ckpt = torch.load(resume_ckpt_path, map_location=f'cuda:{local_rank}', weights_only=False)
        model.module.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_train_loss = ckpt['best_train_loss']
        train_history = ckpt['train_history']
        if global_rank == 0:
            print(f'\nResuming from epoch {start_epoch} (best loss so far: {best_train_loss:.5f})\n')
        dist.barrier(device_ids=[local_rank])  # ensure all ranks finished loading
    elif global_rank == 0:
        print('\nNo checkpoint found — starting fresh training\n')

    model.train()

    for epoch in range(start_epoch, num_epochs + 1):
        train_sampler.set_epoch(epoch)

        train_negcos_loss = 0.0

        # Standard deviation of the output z vectors (used to detect collapse).
        z1_std = 0.0
        z2_std = 0.0

        adjust_learning_rate_warmup(optimizer=optimizer, init_lr=init_lr, epoch=epoch, num_epochs=num_epochs, warmup_epochs=warmup_epochs)
        curr_lr = optimizer.param_groups[0]['lr']

        for x_weak, x_strong in train_loader:

            bs = x_weak.shape[0]
            x_weak = x_weak.reshape(bs, -1, height_bbox, width_bbox).to(device)
            x_strong = x_strong.reshape(bs, -1, height_bbox, width_bbox).to(device)

            p1, p2, z1, z2 = model(x1=x_weak, x2=x_strong)

            loss = negative_cosine_similarity(p1, z2) / 2 + negative_cosine_similarity(p2, z1) / 2

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_negcos_loss += loss.item()

            # Per-feature std along the batch dimension; a collapse drives this to 0.
            z1_detached = F.normalize(z1.detach(), dim=1)
            z2_detached = F.normalize(z2.detach(), dim=1)
            z1_std += z1_detached.std(dim=0).mean().item()
            z2_std += z2_detached.std(dim=0).mean().item()

            # Free the graph tensors immediately to avoid a transient memory spike.
            del x_weak, x_strong, p1, p2, z1, z2, loss, z1_detached, z2_detached

        # Reduce the local metrics across ranks as tensors.
        train_loss = torch.tensor([train_negcos_loss / len(train_loader)], device=local_rank)
        z1_std_val = torch.tensor([z1_std / len(train_loader)], device=local_rank)
        z2_std_val = torch.tensor([z2_std / len(train_loader)], device=local_rank)

        if dist.is_initialized():
            dist.reduce(tensor=train_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
            dist.reduce(tensor=z1_std_val, dst=0, op=torch.distributed.ReduceOp.SUM)
            dist.reduce(tensor=z2_std_val, dst=0, op=torch.distributed.ReduceOp.SUM)

        # Log aggregated metrics and checkpoint only on rank 0.
        if global_rank == 0:
            train_loss = train_loss / args.world_size
            z1_std_val = z1_std_val / args.world_size
            z2_std_val = z2_std_val / args.world_size

            print(
                f"{'-' * 130}\n[Node {args.node_rank}, Global Rank {global_rank}] Epoch {epoch:2d} | "
                f"Batch size per GPU: {batch_size} | "
                f"LR: {curr_lr:.6f} | Loss: {train_loss.item():.5f} | "
                f"std(z1) = {z1_std_val.item():.6f}, std(z2) = {z2_std_val.item():.6f}",
                flush=True,
            )

            train_history[epoch] = {
                'loss': train_loss.item(),
                'lr': curr_lr,
                'z1_std_val': z1_std_val.item(),
                'z2_std_val': z2_std_val.item()
            }

            # Overwrite the resume checkpoint every epoch so training can be
            # restarted from the latest completed epoch after any crash.
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_train_loss': best_train_loss,
                'train_history': train_history,
            }, resume_ckpt_path)

            if train_loss.item() < best_train_loss:
                best_train_loss = train_loss.item()
                print(f"Saving model with lowest train loss = {train_loss.item():.4f}\n")
                torch.save(model.module.state_dict(), best_model_path)

        dist.barrier(device_ids=[local_rank])  # ensure rank 0 finishes saving

    # Save the training metrics for later analysis.
    if global_rank == 0:
        with open(trainhist_path, "wb") as file:
            pickle.dump(train_history, file)

    print(f"\nFinished on Node {args.node_rank}, Local GPU {local_rank}, Global Rank {global_rank}.\n")


def main():
    parser = argparse.ArgumentParser(description='SimSiam DDP Training (torchrun)')

    # Only script-level flags here; torchrun supplies the distributed config via env vars.
    parser.add_argument('--config', required=True,
                        help='Path to a JSON file listing per-object {memmap_path, mask_path} entries')
    parser.add_argument('--output-dir', required=True,
                        help='Directory where checkpoints and the training history are written')
    parser.add_argument('--aug', choices=['imagenet', 'psp'], default='imagenet',
                        help="Augmentation pipeline: 'imagenet' (standard SSL augmentations) "
                             "or 'psp' (our novel phase-shift-profilometry augmentations)")
    parser.add_argument('--init-weights', default=None,
                        help='Optional path to a saved initial state_dict (loaded on rank 0; '
                             'DDP then broadcasts it to all ranks)')
    parser.add_argument('--batch-size', type=int, default=200,
                        help='Per-GPU batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Total number of training epochs')
    parser.add_argument('--warmup-epochs', type=int, default=20,
                        help='Number of linear LR warmup epochs')
    parser.add_argument('--init-lr', type=float, default=0.05,
                        help='Initial (peak) learning rate; no linear batch-size scaling is applied')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='DataLoader worker processes per GPU')
    parser.add_argument('--bbox-size', type=int, default=90,
                        help='Square crop size (H = W) fed to the encoder')
    parser.add_argument('--base-crop-size', type=int, default=80,
                        help="PSP augmentations only: initial random crop size before resizing to --bbox-size")
    parser.add_argument('--num-phase-shifts', type=int, default=4,
                        help='Number of phase shifts per direction (input channels = num_phase_shifts * 2 * 3)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the latest resume checkpoint')

    args = parser.parse_args()

    # torchrun env (fallbacks allow single-process debugging).
    args.rank = int(os.environ.get('RANK', '0'))
    args.world_size = int(os.environ.get('WORLD_SIZE', '1'))
    args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    args.local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', str(torch.cuda.device_count() or 1)))

    # Derive node_rank for logging.
    args.node_rank = args.rank // max(1, args.local_world_size)

    print(f"{'=' * 80}")
    print('DDP Training Configuration (torchrun):')
    print(f'  Global Rank: {args.rank}')
    print(f'  World Size (Total Processes): {args.world_size}')
    print(f'  Node Rank: {args.node_rank}')
    print(f'  Local Rank (GPU on node): {args.local_rank}')
    print(f'  GPUs per Node (LOCAL_WORLD_SIZE): {args.local_world_size}')
    print(f"{'=' * 80}", flush=True)

    setup()
    try:
        train_model(args)
    finally:
        cleanup()


if __name__ == '__main__':
    main()
