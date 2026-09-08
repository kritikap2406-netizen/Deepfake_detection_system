"""
TrueSight — adversarial robustness evaluation (FGSM / PGD).

Standard white-box adversarial attacks, implemented against our own
trained spatial classifier to measure how easily its predictions flip
under small, bounded pixel perturbations. This is a defensive robustness
measurement (how fragile is OUR detector), not an offensive tool aimed
at any external system — exactly the "Adversarial FGSM/PGD where
feasible" item the development plan calls for under Phase 18.

FGSM (Goodfellow et al.): a single step in the direction that increases
loss the fastest, scaled by epsilon.
    x_adv = x + epsilon * sign(grad_x loss(model(x), y))

PGD (Madry et al.): FGSM applied iteratively with a small step size,
re-projected back into the epsilon-ball around the original input after
each step — a stronger attack than single-step FGSM, since it can escape
local irregularities a single gradient step would miss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from spatial.model import SpatialClassifier


def fgsm_attack(model: SpatialClassifier, image: torch.Tensor, label: torch.Tensor, epsilon: float) -> torch.Tensor:
    """image: (1, 3, 224, 224) normalized tensor. label: (1,) in {0.,1.}.
    Returns a perturbed tensor of the same shape, NOT clamped to a valid
    pixel range (the input is already ImageNet-normalized, not raw
    [0,255], so "valid range" isn't a fixed interval here — clamping
    happens by the caller if working in pixel space instead).
    """
    image = image.clone().detach().requires_grad_(True)
    model.zero_grad()

    logit = model(image)
    loss = F.binary_cross_entropy_with_logits(logit, label)
    loss.backward()

    perturbation = epsilon * image.grad.sign()
    return (image + perturbation).detach()


def pgd_attack(
    model: SpatialClassifier,
    image: torch.Tensor,
    label: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    """Iterative FGSM with projection back into the epsilon-ball around
    the original input after each step."""
    original = image.clone().detach()
    perturbed = image.clone().detach()

    for _ in range(steps):
        perturbed.requires_grad_(True)
        model.zero_grad()

        logit = model(perturbed)
        loss = F.binary_cross_entropy_with_logits(logit, label)
        loss.backward()

        with torch.no_grad():
            perturbed = perturbed + alpha * perturbed.grad.sign()
            # Project back into the epsilon-ball around the original input.
            perturbation = torch.clamp(perturbed - original, min=-epsilon, max=epsilon)
            perturbed = (original + perturbation).detach()

    return perturbed


def attack_success_rate(
    model: SpatialClassifier,
    images: list[torch.Tensor],
    labels: list[torch.Tensor],
    attack_fn,
    **attack_kwargs,
) -> dict:
    """Runs attack_fn against every (image, label) pair and reports how
    often the model's prediction flips as a result. Only counts samples
    the model classified CORRECTLY before the attack — flipping an
    already-wrong prediction isn't a meaningful attack success.
    """
    model.eval()

    n_correct_clean = 0
    n_flipped = 0

    for image, label in zip(images, labels):
        clean_logit = model(image)
        clean_pred = (torch.sigmoid(clean_logit) >= 0.5).float()

        if clean_pred.item() != label.item():
            continue  # skip samples the model already gets wrong
        n_correct_clean += 1

        adv_image = attack_fn(model, image, label, **attack_kwargs)
        with torch.no_grad():
            adv_logit = model(adv_image)
        adv_pred = (torch.sigmoid(adv_logit) >= 0.5).float()

        if adv_pred.item() != clean_pred.item():
            n_flipped += 1

    success_rate = n_flipped / n_correct_clean if n_correct_clean > 0 else None

    return {
        "n_samples": len(images),
        "n_correct_clean": n_correct_clean,
        "n_flipped": n_flipped,
        "attack_success_rate": success_rate,
    }
