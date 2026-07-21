import torch
import torch.nn as nn
import torch.nn.functional as F

class MappingLoss(nn.Module):
    def __init__(self, lambda_reg=1.0, lambda_cos=0.5, lambda_tv=0.1, eps=1e-6):
        """
        Combined loss function for vector field regression with integrated
        attention weighting inside the bounding box mask.
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_cos = lambda_cos
        self.lambda_tv = lambda_tv
        self.eps = eps
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred_offsets, gt_offsets, bbox_mask):
        # Calculate the number of valid pixels inside the mask
        N = torch.clamp(bbox_mask.sum(), min=1.0)

        # ==================================================
        # 🌟 Core Enhancement: Intra-Mask Attention Weighting (No Grad)
        # ==================================================
        with torch.no_grad():
            # A. Boundary-Aware Weighting
            # The L2 norm of gt_offsets represents the true distance from the pixel to the target edge.
            distance_to_edge = torch.norm(gt_offsets, dim=1, keepdim=True)
            sigma = 20.0  # Controls decay rate: closer pixels get higher weights (approaching 3.0 max)
            boundary_weight = 1.0 + 2.0 * torch.exp(-distance_to_edge / sigma)

            # B. Hard-Pixel Mining
            # Larger L2 error between predicted and ground-truth vectors incurs a higher penalty (up to +1.0)
            error_magnitude = torch.norm(pred_offsets - gt_offsets, dim=1, keepdim=True)
            hard_weight = 1.0 + (error_magnitude / (error_magnitude.max() + 1e-6))

            # Comprehensive dynamic weight (Shape: [B, 1, H, W])
            dynamic_weight = boundary_weight * hard_weight

        # ==================================================
        # 1. Smooth L1 Loss (Distance & Coordinate Regression)
        # ==================================================
        base_loss_reg = self.smooth_l1(pred_offsets, gt_offsets)
        # Apply dynamic weighting to the regression loss
        loss_reg = (base_loss_reg * dynamic_weight * bbox_mask).sum() / N

        # ==================================================
        # 2. Cosine Similarity Loss (Directional Consistency Constraint)
        # ==================================================
        cos_sim = F.cosine_similarity(pred_offsets, gt_offsets, dim=1, eps=1e-6)
        cos_sim = cos_sim.unsqueeze(1)  # [B, 1, H, W]
        # Apply dynamic weighting to the directional loss
        loss_cos = ((1.0 - cos_sim) * dynamic_weight * bbox_mask).sum() / N

        # ==================================================
        # 3. Charbonnier TV Loss (Highly Elastic Spatial Smoothing)
        # ==================================================
        diff_x = pred_offsets[:, :, :, 1:] - pred_offsets[:, :, :, :-1]
        diff_y = pred_offsets[:, :, 1:, :] - pred_offsets[:, :, :-1, :]

        tv_x = torch.sqrt(diff_x ** 2 + self.eps)
        tv_y = torch.sqrt(diff_y ** 2 + self.eps)

        mask_x = bbox_mask[:, :, :, 1:] * bbox_mask[:, :, :, :-1]
        mask_y = bbox_mask[:, :, 1:, :] * bbox_mask[:, :, :-1, :]

        loss_tv_x = (tv_x * mask_x).sum() / torch.clamp(mask_x.sum(), min=1.0)
        loss_tv_y = (tv_y * mask_y).sum() / torch.clamp(mask_y.sum(), min=1.0)
        loss_tv = loss_tv_x + loss_tv_y

        # ==================================================
        # Total Loss Fusion
        # ==================================================
        total_loss = (self.lambda_reg * loss_reg) + \
                     (self.lambda_cos * loss_cos) + \
                     (self.lambda_tv * loss_tv)

        return total_loss










