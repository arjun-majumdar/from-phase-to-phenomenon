import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim


def criterion_sharpened(pred, target, w_l1=0.2, w_wmse=0.4, w_grad=0.3, w_ssim=0.2, w_log=0.1):
    '''
    Sharpened Loss for Texture Preservation
    
    Changes:
    - Removed w_mse (It is redundant with WMSE and causes blur)
    - Added w_grad (Forces the network to match the steepness/slope of the PSF)
    - Increased w_ssim (To prioritize structure over raw pixel values)
    '''
    
    # 1. Standard L1 (Robust regression)
    l1_loss = F.l1_loss(pred, target)
    
    # 2. Weighted MSE (Peak Intensity Fixer)
    # Penalizes errors in the bright center 10x more than the background
    weight = target + 0.1 
    wmse_loss = torch.mean(weight * (pred - target)**2)
    
    # 3. Log-L1 (Tail Recoverer)
    # Reduced weight slightly as tails are less critical for texture
    eps = 1e-7
    log_l1 = F.l1_loss(torch.log(pred + eps), torch.log(target + eps))
    
    # 4. SSIM (Structure) - Increased weight
    ssim_val = ssim(pred, target, data_range=1.0, size_average=True)
    ssim_loss = 1 - ssim_val
    
    # 5. NEW: Gradient Difference Loss (The Texture/Sharpness Fixer)
    # This calculates the derivative in X and Y directions.
    # If the network predicts a "smooth blob" but the target is a "sharp peak",
    # the gradients will look totally different, creating a huge loss.
    def gradient_loss(p, t):
        # Calculate gradients in X (horizontal) and Y (vertical)
        # We use simple finite differences
        dy_p = torch.abs(p[:, :, 1:, :] - p[:, :, :-1, :])
        dy_t = torch.abs(t[:, :, 1:, :] - t[:, :, :-1, :])
        
        dx_p = torch.abs(p[:, :, :, 1:] - p[:, :, :, :-1])
        dx_t = torch.abs(t[:, :, :, 1:] - t[:, :, :, :-1])
        
        # We penalize the L1 difference between the gradients
        grad_loss = torch.mean(torch.abs(dy_p - dy_t)) + torch.mean(torch.abs(dx_p - dx_t))
        return grad_loss

    grad_loss = gradient_loss(pred, target)

    # Total combined loss
    return (w_l1 * l1_loss) + \
           (w_wmse * wmse_loss) + \
           (w_grad * grad_loss) + \
           (w_ssim * ssim_loss) + \
           (w_log * log_l1)


def spatial_gradient_loss(pred, target):
    """
    Computes the L1 distance between the spatial gradients (edges) 
    of the prediction and the target. This forces the network to 
    learn sharp transitions and steep peaks.
    """
    # Define Sobel/Finite Difference kernels
    kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=pred.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=pred.device).view(1, 1, 3, 3)
    
    # Calculate gradients (assuming pred/target are Bx1xHxW or Bx3xHxW. 
    # If 3 channels, you can process them independently or convert to grayscale first).
    # Expanding kernels to match input channels if necessary:
    channels = pred.shape[1]
    kernel_x = kernel_x.repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.repeat(channels, 1, 1, 1)

    pred_grad_x = F.conv2d(pred, kernel_x, padding=1, groups=channels)
    pred_grad_y = F.conv2d(pred, kernel_y, padding=1, groups=channels)
    
    target_grad_x = F.conv2d(target, kernel_x, padding=1, groups=channels)
    target_grad_y = F.conv2d(target, kernel_y, padding=1, groups=channels)
    
    # Penalize the difference in edges
    grad_loss = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
    return grad_loss


def criterion(pred, target, w_l1=0.2, w_mse=0.1, w_wmse=0.3, w_ssim=0.25, w_log=0.05, w_grad=0.1):
    '''
    Improved Hybrid Loss for PSP-to-PSF Reconstruction
    '''
    # 1. Standard Pixel-wise
    l1_loss = F.l1_loss(pred, target)
    mse_loss = F.mse_loss(pred, target)
    
    # 2. FIXED Weighted MSE (The Peak-Fixer)
    # Target max is ~0.04. Multiplying by 250 maps the peak to ~10.0.
    # Weight at peak = 11.0. Weight at background = 1.0. 
    # Now the peak is TRULY penalized 11x more than the background.
    weight = (target * 250.0) + 1.0 
    wmse_loss = torch.mean(weight * (pred - target)**2)
    
    # 3. Log-L1 Loss (Tamed)
    # Reduced weight so it doesn't cannibalize the peak.
    eps = 1e-6
    log_l1 = F.l1_loss(torch.log(pred + eps), torch.log(target + eps))
    
    # 4. SSIM Loss (Structure - weight increased)
    # SSIM is excellent for preserving the asymmetric shape of the PSF
    ssim_val = ssim(pred, target, data_range=target.max().item(), size_average=True)
    ssim_loss = 1 - ssim_val
    
    # 5. NEW: Spatial Gradient Loss (The Sharpness-Enforcer)
    grad_loss = spatial_gradient_loss(pred, target)
    
    # Total combined loss
    return (w_l1 * l1_loss) + \
           (w_mse * mse_loss) + \
           (w_wmse * wmse_loss) + \
           (w_ssim * ssim_loss) + \
           (w_log * log_l1) + \
           (w_grad * grad_loss)
