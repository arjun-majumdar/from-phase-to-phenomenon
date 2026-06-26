import os
import json
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import ssim, ms_ssim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from psp_multiobject_PSF_postSSL_dataset import MultiObject_PSFSSS_Dataset_LazyHDF5, MultiObject_PSFSSS_Dataset_Preloaded
from Att_PreAct_pxshuffle_UNet_Decoder_Only import Att_PreActRes_UNet_Pxshuffle_DecoderOnly, initialize_weights, icnr_init
from cost_functions import criterion_sharpened


def criterion(pred, target, w_l1=0.2, w_mse=0.2, w_wmse=0.3, w_ssim=0.1, w_log=0.2):
    '''
    Final Hybrid Loss for PSP-to-PSF Reconstruction
    w_wmse: Heavily penalizes peak intensity errors
    w_log:  Recovers the faint subsurface scattering tails
    '''
    # 1. Standard Pixel-wise Losses
    l1_loss = F.l1_loss(pred, target)
    mse_loss = F.mse_loss(pred, target)
    
    # 2. Weighted MSE (The Peak-Fixer)
    # We add a small constant to the weight to ensure background pixels 
    # still contribute, but peak pixels contribute ~10x more.
    # Weight is higher where target intensity is higher.
    weight = target + 0.1 
    wmse_loss = torch.mean(weight * (pred - target)**2)
    
    # 3. Log-L1 Loss (The Tail-Recoverer)
    eps = 1e-7
    log_l1 = F.l1_loss(torch.log(pred + eps), torch.log(target + eps))
    
    # 4. SSIM Loss (Structure)
    ssim_val = ssim(pred, target, data_range=1.0, size_average=True)
    ssim_loss = 1 - ssim_val
    
    # Total combined loss
    return (w_l1 * l1_loss) + \
           (w_mse * mse_loss) + \
           (w_wmse * wmse_loss) + \
           (w_ssim * ssim_loss) + \
           (w_log * log_l1)

def criterion_aggressive_sharpen(pred, target, w_l1=1.0, w_wmse=10.0, w_grad=50.0, w_ssim=0.5):
    '''
    Aggressively Scaled Loss for Tiny PSF Values
    '''
    
    # 1. Base L1 Loss (Keep this to stabilize the background)
    l1_loss = F.l1_loss(pred, target)
    
    # 2. Aggressive Weighted MSE 
    # Since target max is ~0.04, we normalize it so the peak is exactly 1.0.
    # Then we add 0.01 for the background. 
    # Now the peak is penalized 100x more than the background!
    max_val = torch.max(target) + 1e-8
    normalized_target = target / max_val
    weight = normalized_target + 0.01 
    
    wmse_loss = torch.mean(weight * (pred - target)**2)
    
    # 3. Scaled Gradient Loss
    # We multiply the inputs by 100 before taking the gradient to ensure
    # the resulting loss magnitude is large enough for the optimizer to care.
    def scaled_gradient_loss(p, t):
        p_scaled = p * 100.0
        t_scaled = t * 100.0
        
        dy_p = torch.abs(p_scaled[:, :, 1:, :] - p_scaled[:, :, :-1, :])
        dy_t = torch.abs(t_scaled[:, :, 1:, :] - t_scaled[:, :, :-1, :])
        
        dx_p = torch.abs(p_scaled[:, :, :, 1:] - p_scaled[:, :, :, :-1])
        dx_t = torch.abs(t_scaled[:, :, :, 1:] - t_scaled[:, :, :, :-1])
        
        return torch.mean(torch.abs(dy_p - dy_t)) + torch.mean(torch.abs(dx_p - dx_t))

    grad_loss = scaled_gradient_loss(pred, target)
    
    # 4. SSIM (Structure)
    ssim_val = ssim(pred, target, data_range=1.0, size_average=True)
    ssim_loss = 1 - ssim_val
    
    # Notice the massive multipliers. We are forcing the optimizer to look at the gradients.
    return (w_l1 * l1_loss) + \
           (w_wmse * wmse_loss) + \
           (w_grad * grad_loss) + \
           (w_ssim * ssim_loss)


def compute_uncertainty_weights(target_patches_dataset:torch.tensor) -> torch.tensor:
    """
    Compute weights based on pixel-wise variance across dataset.
    High variance regions get higher weight (harder to predict).

    Conceptual Detail: Batch Variance vs. Dataset Variance-
    Assuming target_patches_dataset -> target tensor passed into the loss function,
    its shape is (Batch, Channels, Height, Width). By operating on dim=0, we compute
    the variance across the current batch, not the entire dataset.
    Is this bad? Actually, no! Batch-adaptive weighting acts as a dynamic attention
    mechanism. Just be aware that if batch size is very small (e.g., 8 or 16), this
    variance map will be noisy. With a batch size of 256, it should be highly stable
    and work beautifully (in theory!).
    """
    # Compute variance across all samples for each pixel location
    pixel_variance = torch.var(target_patches_dataset, dim=0)  # (C, H, W)
    
    # Convert to weights: higher variance = higher weight
    weights = 1.0 + pixel_variance / (torch.mean(pixel_variance) + 1e-8)
    
    return weights

def weighted_pixel_loss(pred_patch, target_patch, weights, loss_type='mse'):
    '''
    Compute per-pixel loss with given weights.

    Args:
        pred_patch: (B, C, H, W) - Network prediction
        target_patch: (B, C, H, W) - Ground truth target
        weights: (B, C, H, W) - Weight for each pixel
        loss_type: 'mse', 'l1', or 'huber'

    Returns:
        weighted_loss: Scalar loss value
    '''
    # Compute per-pixel loss (element-wise)
    if loss_type == 'mse':
        pixel_loss = (pred_patch - target_patch) ** 2
    elif loss_type == 'l1':
        pixel_loss = torch.abs(pred_patch - target_patch)
    elif loss_type == 'huber':
        pixel_loss = F.huber_loss(pred_patch, target_patch, reduction='none', delta=1.0)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    # Apply weights to each pixel's loss
    weighted_pixel_loss = (weights*pixel_loss)

    # Return mean of all weighted pixel losses
    return torch.mean(weighted_pixel_loss)

def normalized_gradient_loss(p, t, max_vals):
    # Divide by max_vals per channel to normalize gradients relative to each channel's peak.
    # This prevents bright channels from exploding and dark channels from dying.
    p_norm = p / max_vals
    t_norm = t / max_vals

    dy_p = torch.abs(p_norm[:, :, 1:, :] - p_norm[:, :, :-1, :])
    dy_t = torch.abs(t_norm[:, :, 1:, :] - t_norm[:, :, :-1, :])

    dx_p = torch.abs(p_norm[:, :, :, 1:] - p_norm[:, :, :, :-1])
    dx_t = torch.abs(t_norm[:, :, :, 1:] - t_norm[:, :, :, :-1])

    return torch.mean(torch.abs(dy_p - dy_t)) + torch.mean(torch.abs(dx_p - dx_t))

def criterion_self_balanced_hybrid(
    pred:torch.tensor, target:torch.tensor,
    w_l1:float=1.0, w_wmse:float=5.0,
    w_grad:float=10.0, w_ssim:float=0.5,
    w_log:float=1.0, use_var:bool=True) -> torch.tensor:
    '''
    Self-Balancing Hybrid Loss: 
    Forces sharp peaks (Grad), preserves SSS tails (Log), AND prevents color collapse
    (max_c).

    PSF SSS loss function is now a highly sophisticated, physics-aware, self-balancing
    engine. It addresses the smoothing bias of neural networks, preserves the physical
    SSS glow, and adapts dynamically to the dataset.
    '''
    
    # Always compute max_c for gradient normalization to prevent color collapse
    max_c = torch.amax(target, dim=(0, 2, 3), keepdim=True) + 1e-8

    # 2. Weighted MSE (The Peak Fixer) 
    if use_var:
        # Data-Driven Weighting: Uses variance across the current batch
        # weights_var shape: (C, H, W)
        weights_var = compute_uncertainty_weights(target_patches_dataset=target)
        wmse_loss = weighted_pixel_loss(pred_patch=pred, target_patch=target, weights=weights_var, loss_type='mse')

    else:
        # Intensity-based weighting
        normalized_target = target / max_c
        weight = normalized_target + 0.05  
        wmse_loss = torch.mean(weight * (pred - target)**2)
        
    # 1. Base L1 Loss (Stabilizes the background)
    l1_loss = F.l1_loss(pred, target)

    # 3. Normalized Gradient Loss (The Texture/Sharpness Fixer)
    grad_loss = normalized_gradient_loss(p=pred, t=target, max_vals=max_c)
        
    # 4. Log-L1 Loss (The Tail Protector)
    eps = 1e-7
    log_l1 = F.l1_loss(torch.log(pred + eps), torch.log(target + eps))

    # 5. SSIM (Structure)
    ssim_val = ssim(pred, target, data_range=1.0, size_average=True)
    ssim_loss = 1 - ssim_val

    # Total combined loss
    return (w_l1 * l1_loss) + \
           (w_wmse * wmse_loss) + \
           (w_grad * grad_loss) + \
           (w_ssim * ssim_loss) + \
           (w_log * log_l1)


def setup(rank: int, world_size: int, master_addr: str, master_port: str) -> None:
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ.setdefault('NCCL_SOCKET_IFNAME', '^lo,docker,virbr')

    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    
    # Set device for this process
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    
    return None

def cleanup() -> None:
    dist.destroy_process_group()
    return None


def train_model(local_rank, args):
    """
    Main training function for multi-node multi-GPU training.
    
    Args:
        local_rank: Local GPU rank on this node (0 to gpus_per_node-1)
        args: Arguments containing node_rank, world_size, etc.
    """
    # Calculate global rank
    global_rank = args.node_rank * args.gpus_per_node + local_rank
    
    print(f"\nNode {args.node_rank}, Local GPU {local_rank}, Global Rank {global_rank}\n")
    
    # Setup distributed training
    setup(global_rank, args.world_size, args.master_addr, args.master_port)

    try:
        
        device = local_rank  # Local device on this node

        # Encoder hyper-params-
        num_phase_shifts = args.num_phase_shifts
        in_channels = num_phase_shifts * 2 * 3
        out_channels = 3
        features = [128, 256, 384, 512]

        # Make sure model is on correct device before DDP wrapper
        torch.cuda.set_device(local_rank)

        model_decoder_only = Att_PreActRes_UNet_Pxshuffle_DecoderOnly(
            in_channels=in_channels, out_channels=out_channels,
            features=features
        )

        # Apply weight initialization only to the decoder (not the encoder)-
        initialize_weights(model_decoder_only.decoder_stages)
        initialize_weights(model_decoder_only.out_conv)

        # Directory where checkpoints and the training history are written.
        path_sav_outputs = args.output_dir
        os.makedirs(path_sav_outputs, exist_ok=True)

        # Load pre-trained SimSiam encoder parameters from the given checkpoint.
        # Extract only Att_PreactRes_UNet_Encoder weights from the full SimSiam checkpoint-
        simsiam_state = torch.load(args.encoder_ckpt, weights_only=True)
        encoder_state = {}
        for k, v in simsiam_state.items():
            # Keep only encoder keys, excluding avgpool (added by SimSiamEncoderWrapper)
            if k.startswith('encoder.') and not k.startswith('encoder.avgpool'):
                # Strip the 'encoder.' prefix so keys match Att_PreactRes_UNet_Encoder directly
                encoder_state[k[len('encoder.'):]] = v

        model_decoder_only.trained_encoder.load_state_dict(encoder_state)
        print('\nSuccessfully loaded ImageNet data aug SimSiam pre-trained encoder parameters\n')

        # Apply SyncBN (must be done before .to(device))-
        model_decoder_only = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_decoder_only)
        model_decoder_only = model_decoder_only.to(device)

        # Wrap model with DDP for gradient synchronization across GPUs-
        model_decoder_only = DDP(model_decoder_only, device_ids=[local_rank], find_unused_parameters=True)

        tot_params = sum(p.numel() for p in model_decoder_only.parameters())
        print(f'\nModel has {tot_params} parameters\n')


        frozen_params = 0
        trainable_params = 0

        for layer_name, param in model_decoder_only.named_parameters():
            if 'trained_encoder' in layer_name:
                param.requires_grad = False
                frozen_params += param.nelement()
                # print(f'frozen: {layer_name} has {param.size()} params')
            else:
                trainable_params += param.nelement()
                # print(f'trainable: {layer_name} has {param.size()} params')

        print(f'\nfrozen parameters = {frozen_params}, trainable parameters = {trainable_params}\n')


        # Only train decoder parameters + output conv layeer!
        params_to_train = list(model_decoder_only.module.decoder_stages.parameters()) + list(model_decoder_only.module.out_conv.parameters())
        optimizer = torch.optim.Adam(params=params_to_train, lr=args.lr)

        # Resume from checkpoint if requested
        start_epoch = 1
        best_train_loss = 10

        # Output checkpoint path (also used for resuming).
        ckpt_path = os.path.join(path_sav_outputs, args.ckpt_name)

        if args.resume:
            checkpoint_path = ckpt_path

            if os.path.exists(checkpoint_path):
                map_location = {'cuda:0': f'cuda:{local_rank}'}
                saved_state = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
                model_decoder_only.module.load_state_dict(saved_state)
                
                start_epoch = args.resume_epoch + 1
                
                if global_rank == 0:
                    print(f'\nResumed from checkpoint at epoch {args.resume_epoch}')
                    print(f'Training will continue from epoch {start_epoch}\n')
            else:
                if global_rank == 0:
                    print(f'\nWARNING: No checkpoint found at {checkpoint_path}, training from scratch\n')


        # Per-object data paths are read from a JSON config file: a list of
        # {"memmap_path": ..., "psf_path": ...} entries (see object_configs_example.json).
        with open(args.config, 'r') as f:
            object_configs = json.load(f)

        width_bbox = height_bbox = args.bbox_size


        ###############################################################################
        # Dataset strategy:
        #   'lazy'      -> low RAM, reads PSF patches from HDF5 on-the-fly
        #   'preloaded' -> higher RAM, fastest __getitem__ (PSF held in memory)
        ###############################################################################

        if args.dataset_strategy == 'preloaded':
            MultiObject_PSFSSS_Dataset = MultiObject_PSFSSS_Dataset_Preloaded
        else:
            MultiObject_PSFSSS_Dataset = MultiObject_PSFSSS_Dataset_LazyHDF5

        dataset = MultiObject_PSFSSS_Dataset(
            object_configs=object_configs,
            width_bbox=width_bbox,
            height_bbox=height_bbox,
            data_augmentations=None
        )

        # IMPORTANT: Use global_rank for DistributedSampler, not local_rank
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, 
            num_replicas=args.world_size,  # Total number of GPUs across all nodes
            rank=global_rank,  # Global rank across all nodes
            shuffle=True    # shuffling is important for training, but must be done by DistributedSampler to ensure proper shuffling across epochs
        )

        # print(f'\nlength of dataset = {len(dataset)}\n')

        batch_size = args.batch_size

        # Standard DataLoader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Must be False when using DistributedSampler
            sampler=train_sampler,
            num_workers=args.num_workers,
            # Always True for GPU training.
            # Enables fast DMA transfer to VRAM.
            pin_memory=True,
            # Optional: If your epochs are short, this keeps workers alive
            # avoiding re-initialization overhead.
            # persistent_workers=True
        )

        if global_rank == 0:
            print(f'\nTotal dataset size: {len(dataset)}')
            print(f'Batches per GPU: {len(dataloader)}')
            print(f'Total batches across all GPUs: {len(dataloader) * args.world_size}\n')



        # Python dict to contain training metrics-
        train_history = dict()

        # Initialize parameter for saving 'best' parameters-
        # best_train_loss = 10

        # Training epochs-
        num_epochs = args.epochs

        model_decoder_only.train()

        for epoch in range(start_epoch, num_epochs + 1):
            # Set epoch for DistributedSampler to shuffle data differently each epoch
            train_sampler.set_epoch(epoch)
            
            total_loss_train = 0.0
            
            # curr_lr = optimizer.param_groups[0]['lr']

            for x, y in dataloader:
                
                # Get current batch-size-
                # bs = x.size(0)
                
                x = x.to(device).float()
                y = y.to(device).float()

                optimizer.zero_grad()
                out = model_decoder_only(x)
                y = y.permute(0,3,1,2)

                # loss = criterion(pred=out, target=y, w_l1=0.2, w_mse=0.2, w_wmse=0.3, w_ssim=0.1, w_log=0.2)
                # loss = criterion_sharpened(pred=out, target=y, w_l1=0.2, w_wmse=0.4, w_grad=0.3, w_ssim=0.2, w_log=0.1)
                # loss = criterion_aggressive_sharpen(pred=out, target=y, w_l1=1.0, w_wmse=10.0, w_grad=50.0, w_ssim=0.5)
                loss = criterion_self_balanced_hybrid(
                    pred=out, target=y,
                    w_l1=1.0, w_wmse=5.0,
                    w_grad=10.0, w_ssim=0.5,
                    w_log=1.0, use_var=True
                )

                loss.backward()
                optimizer.step()

                # Aggregate batch-level losses-
                total_loss_train += loss.item()

            # Convert to tensors for distributed reduction. To globally reduce local metrics across ranks, they should be Tensors-
            train_loss = torch.tensor([total_loss_train/len(dataloader)], device=local_rank)
            
            if torch.cuda.is_available():
                dist.reduce(tensor = train_loss, dst = 0, op = torch.distributed.ReduceOp.SUM)

            # will log the aggregated metrics only on the 0th GPU. Make sure "train_dataset" is of type
            # Dataset and not DataLoader to get the size of the full dataset and not of the local shard
            if global_rank == 0:
                train_loss_val = (train_loss / args.world_size).item()

                if epoch%2==0:
                    print(f'\nGT: min = {y.min().item():.4f}, max = {y.max().item():.4f}; Output: min = {out.min().item():.4f} & max = {out.max().item():.4f}\n')


                # Store model performance metrics in Python3 dict-
                train_history[epoch] = {
                    'train_sharpened_loss': train_loss_val,
                    # 'test_L1L2SSIM_loss': test_loss,
                    # 'lr': curr_lr
                }

                print(
                    f"Epoch = {epoch}; train sharp loss = {train_loss_val:.7f} ",
                    # f"test L1-L2-SSIM = {test_loss:.7f}"
                    # f"& LR = {curr_lr:.5f}"
                )

                # Save 'best' parameters so far-
                if train_loss_val < best_train_loss:
                    # update 'best_test_loss' variable to lowest loss encountered so far-
                    best_train_loss = train_loss_val

                    print(f'\nSaving model with lowest train loss = {best_train_loss:.9f}\n')

                    # Only rank 0 saves; write to a temp file then atomically rename.
                    tmp_path = ckpt_path + ".tmp"

                    try:
                        torch.save(model_decoder_only.module.state_dict(), tmp_path)
                        os.replace(tmp_path, ckpt_path)  # atomic rename
                    except Exception as e:
                        print(f"[rank0] ERROR: failed to save checkpoint to {ckpt_path}: {e}")

                        dist.barrier()  # Ensure rank 0 finishes saving

        # Save training metrics-
        if global_rank == 0:
            # Save training metrics as Python3 history for later analysis-
            trainhist_path = os.path.splitext(ckpt_path)[0] + "_trainhist.pkl"
            with open(trainhist_path, "wb") as file:
                pickle.dump(train_history, file)
        
        cleanup()
        print(f"\nFinished on Node {args.node_rank}, Local GPU {local_rank}, Global Rank {global_rank}.\n")

    # 2. Always destroy process group even on exception
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Decoder-only SSS-PSF training on a frozen SimSiam encoder (multi-node / multi-GPU)')

    # Data / I/O arguments
    parser.add_argument('--config', required=True,
                        help='Path to a JSON file listing per-object {memmap_path, psf_path} entries')
    parser.add_argument('--encoder-ckpt', required=True,
                        help='Path to the pre-trained SimSiam checkpoint to load the frozen encoder from')
    parser.add_argument('--output-dir', required=True,
                        help='Directory where the decoder checkpoint and training history are written')
    parser.add_argument('--ckpt-name', default='decoder_only.pth',
                        help='Filename for the saved decoder checkpoint (within --output-dir)')
    parser.add_argument('--dataset-strategy', choices=['lazy', 'preloaded'], default='lazy',
                        help="'lazy' reads PSF patches from HDF5 on-the-fly (low RAM); "
                             "'preloaded' holds all PSF data in RAM (fastest __getitem__)")

    # Training hyper-parameters
    parser.add_argument('--batch-size', type=int, default=128, help='Per-GPU batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Total training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Adam learning rate')
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader workers per GPU')
    parser.add_argument('--bbox-size', type=int, default=90, help='Square patch size (H = W)')
    parser.add_argument('--num-phase-shifts', type=int, default=4,
                        help='Phase shifts per direction; encoder input channels = num_phase_shifts * 2 * 3')

    # Multi-node arguments
    parser.add_argument('--node_rank', type=int, default=0,
                       help='Rank of the current node (0 for master, 1+ for workers)')
    parser.add_argument('--num_nodes', type=int, default=1,
                       help='Total number of nodes')
    parser.add_argument('--gpus_per_node', type=int, default=None,
                       help='Number of GPUs per node (default: all available)')
    parser.add_argument('--master_addr', type=str, default='localhost',
                       help='Address of master node')
    parser.add_argument('--master_port', type=str, default='12355',
                       help='Port on master node')
    parser.add_argument('--resume', action='store_true',
                   help='Resume training from checkpoint')
    parser.add_argument('--resume_epoch', type=int, default=65,
                   help='Epoch to resume from')
    
    args = parser.parse_args()
    
    # Determine GPUs per node
    if args.gpus_per_node is None:
        args.gpus_per_node = torch.cuda.device_count()
    
    # Calculate world size
    args.world_size = args.num_nodes * args.gpus_per_node
    
    print(f"\n{'=' * 80}")
    print(f"Multi-Node Multi-GPU Training Configuration:")
    print(f"  Node Rank: {args.node_rank}")
    print(f"  Number of Nodes: {args.num_nodes}")
    print(f"  GPUs per Node: {args.gpus_per_node}")
    print(f"  World Size (Total GPUs): {args.world_size}")
    print(f"  Master Address: {args.master_addr}")
    print(f"  Master Port: {args.master_port}")
    print(f"{'=' * 80}\n")
    
    # Spawn processes for each GPU on this node
    mp.spawn(
        fn=train_model,
        args=(args,),
        nprocs=args.gpus_per_node,
        join=True
    )


if __name__ == '__main__':
    main()
