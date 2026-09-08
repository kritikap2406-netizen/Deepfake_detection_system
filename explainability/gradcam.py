"""
TrueSight — Grad-CAM / Grad-CAM++.

Implements both methods directly via forward/backward hooks rather than
depending on a third-party library's API. Our SpatialClassifier outputs
a single scalar logit (P(fake), not a per-class score vector), which
doesn't cleanly fit most Grad-CAM library APIs built around multi-class
`ClassifierOutputTarget(category_index)` selection — implementing this
ourselves keeps the math fully inspectable and avoids depending on
external API assumptions that don't match our model's output shape.

Hook target: backbone.bn2 — the batch-norm+activation layer immediately
after the final 1x1 conv (conv_head) and immediately before global
average pooling. This was chosen by directly inspecting the EfficientNet
architecture: it's the last spatial feature map (shape (B, 1280, 7, 7)
for 224x224 input, verified empirically) before the network collapses
to the pooled feature vector our classifier head operates on — the
standard, correct target for CNN Grad-CAM.

Grad-CAM (Selvaraju et al.): channel weights = global-average-pooled
gradient of the target score w.r.t. each channel of the feature map.
Grad-CAM++ (Chattopadhay et al.): a refinement using higher-order
gradient terms, generally producing tighter, more localized heatmaps
for objects with multiple instances/regions of interest.

Both are well-established published formulas — implemented here from
their standard equations, not from any particular codebase.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from spatial.model import SpatialClassifier


class GradCAMExtractor:
    def __init__(self, model: SpatialClassifier, target_layer: torch.nn.Module | None = None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or model.backbone.bn2

        self._activation: torch.Tensor | None = None
        self._gradient: torch.Tensor | None = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activation = output

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradient = grad_output[0]

    def _forward_backward(self, input_tensor: torch.Tensor) -> None:
        """input_tensor: (1, 3, 224, 224). Runs a forward + backward pass
        on the single fake-logit output, populating self._activation and
        self._gradient via the registered hooks."""
        if input_tensor.shape[0] != 1:
            raise ValueError("GradCAMExtractor processes one face crop at a time (batch size 1).")

        self.model.zero_grad()
        logit = self.model(input_tensor)  # shape (1,) — this IS the P(fake) logit, no class selection needed
        logit.backward()

    def _finalize_cam(self, cam: torch.Tensor, output_size: int = 224) -> torch.Tensor:
        """ReLU (per Grad-CAM, only positive influence matters), resize
        to the original input resolution, min-max normalize to [0,1]."""
        cam = F.relu(cam)
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W) for interpolate
        cam = F.interpolate(cam, size=(output_size, output_size), mode="bilinear", align_corners=False)
        cam = cam.squeeze()

        min_val, max_val = cam.min(), cam.max()
        if (max_val - min_val) < 1e-8:
            return torch.zeros_like(cam)
        return (cam - min_val) / (max_val - min_val)

    def compute_gradcam(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Standard Grad-CAM. Returns a (224, 224) heatmap in [0, 1]."""
        self._forward_backward(input_tensor)

        activation = self._activation[0]  # (C, H, W)
        gradient = self._gradient[0]      # (C, H, W)

        weights = gradient.mean(dim=(1, 2))  # (C,) — global average pool of gradient per channel
        cam = (weights[:, None, None] * activation).sum(dim=0)  # (H, W)

        return self._finalize_cam(cam)

    def compute_gradcam_plusplus(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Grad-CAM++. Returns a (224, 224) heatmap in [0, 1].

        Uses second- and third-order gradient terms to weight each
        spatial location's contribution to the channel weight, rather
        than a flat average — this generally produces sharper, more
        precisely localized activation than plain Grad-CAM.
        """
        self._forward_backward(input_tensor)

        activation = self._activation[0]  # (C, H, W)
        gradient = self._gradient[0]      # (C, H, W)

        grad_2 = gradient ** 2
        grad_3 = gradient ** 3

        # Standard Grad-CAM++ alpha coefficient (Chattopadhay et al.):
        # alpha_ij^k = grad2_ij^k / (2*grad2_ij^k + sum_ab(activation_ab^k * grad3_ab^k))
        sum_activation = activation.sum(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        denominator = 2 * grad_2 + sum_activation * grad_3
        denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
        alpha = grad_2 / denominator

        # Weight each channel by the alpha-weighted, ReLU'd gradient, summed spatially.
        weights = (alpha * F.relu(gradient)).sum(dim=(1, 2))  # (C,)
        cam = (weights[:, None, None] * activation).sum(dim=0)  # (H, W)

        return self._finalize_cam(cam)
