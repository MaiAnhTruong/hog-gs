#
# ECU: Evidence-Conditioned Uncertainty for soft structural dropout in sparse-view 3DGS.
#
# Upgrades the WGSD head: replaces the FIXED mapping rate = base*exp(beta*(front-witness)) (which we
# measured to be beta-sensitive - the same "hand-crafted coefficient" limitation D2GS admits) with a
# LEARNED mapping (UGOD-style differentiable soft dropout), but conditioned on CROSS-VIEW WITNESS
# EVIDENCE instead of UGOD's shape-only inputs (their stated limitation: "low dimensionality of the
# input fundamentally limits capacity" - position/rot/scale say nothing about multi-view support).
#
#   u_i = MLP(features_i) in (0,1)            features: evidence (witness/front/back/conf/rel) and/or
#                                             shape (PE(pos), scale, rot, opacity)  [ugod_style]
#   multiplier_i = (1-u_i) * omega_i          omega ~ concrete distribution (temperature tau),
#                                             clamped [0.2,0.8] for gradient flow (UGOD eq.)
#   opacity_train = opacity * multiplier      TRAIN-TIME ONLY (eval uses the plain model, so all
#                                             dropout variants are compared with identical inference).
#   loss += ecu_reg * (mean(u) - u_target)^2  rate anchor: prevents the degenerate u->0 collapse
#                                             (render loss alone prefers no suppression); keeps the
#                                             AVERAGE suppression at the DropGaussian-equivalent level
#                                             so only the ALLOCATION is learned (mirrors WGSD mean-1).
#
import math
import torch
from torch import nn


def _positional_encoding(x, n_freq=4):
    out = [x]
    for k in range(n_freq):
        out.append(torch.sin((2.0 ** k) * math.pi * x))
        out.append(torch.cos((2.0 ** k) * math.pi * x))
    return torch.cat(out, dim=1)


class EvidenceUncertaintyNet(nn.Module):
    """Small MLP -> per-Gaussian uncertainty u in (0,1). inputs: 'evidence' | 'shape' | 'both'."""

    EVIDENCE_DIM = 6   # witness, front_ratio, back_ratio, conf_seen_norm, rel_mean, opacity
    SHAPE_DIM = 3 * (1 + 2 * 4) + 3 + 4 + 1   # PE(pos,4) + scale(3) + rot(4) + opacity(1) = 35

    def __init__(self, inputs="evidence", hidden=64):
        super().__init__()
        self.inputs = str(inputs)
        in_dim = {"evidence": self.EVIDENCE_DIM, "shape": self.SHAPE_DIM,
                  "both": self.EVIDENCE_DIM + self.SHAPE_DIM}[self.inputs]
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def build_features(self, gaussians, swd_scores, scene_center, scene_scale):
        """All inputs detached (the net learns the mapping, not the model state)."""
        feats = []
        if self.inputs in ("evidence", "both"):
            if swd_scores is None:
                raise RuntimeError("[ECU] inputs include 'evidence' but SWD scores are missing (enable --swd_enable)")
            conf_norm = (swd_scores["conf_seen"] / swd_scores["conf_seen"].max().clamp(min=1.0)).clamp(0, 1)
            feats.append(torch.stack([
                swd_scores["witness"].clamp(0, 1),
                swd_scores["front_ratio"].clamp(0, 1),
                swd_scores["back_ratio"].clamp(0, 1),
                conf_norm,
                swd_scores["rel_mean"].clamp(0, 2) * 0.5,
                gaussians.get_opacity.detach().squeeze(1).clamp(0, 1),
            ], dim=1))
        if self.inputs in ("shape", "both"):
            pos = (gaussians.get_xyz.detach() - scene_center) / max(scene_scale, 1e-6)
            feats.append(torch.cat([
                _positional_encoding(pos.clamp(-2, 2), 4),
                gaussians.get_scaling.detach().clamp(max=10.0),
                gaussians.get_rotation.detach(),
                gaussians.get_opacity.detach().clamp(0, 1),
            ], dim=1))
        return torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]

    def forward(self, feats):
        return torch.sigmoid(self.mlp(feats)).squeeze(1)   # [N] u in (0,1)


def ecu_train_multiplier(u, tau=0.5, clamp_lo=0.2, clamp_hi=0.8):
    """UGOD-style concrete soft dropout, MEAN-1 RENORMALIZED. Returns [N,1] opacity multiplier.

    The raw (1-u)*omega has E[mult]~0.7 != 1 -> a systematic train/eval opacity mismatch that we
    measured to be catastrophic (all-mode collapse on the first Lightning round). Renormalizing the
    batch to mean 1 (same cure as WGSD) keeps E[opacity] unbiased so only the RELATIVE allocation
    of suppression across Gaussians is learned."""
    u_safe = u.clamp(1e-4, 1.0 - 1e-4)
    q = torch.rand_like(u).clamp(1e-4, 1.0 - 1e-4)
    omega = 1.0 - torch.sigmoid((torch.log(u_safe / (1 - u_safe)) + torch.log(q / (1 - q))) / tau)
    omega = omega.clamp(clamp_lo, clamp_hi)
    mult = (1.0 - u_safe) * omega
    mult = mult / mult.mean().clamp(min=1e-6)          # mean-1: unbiased E[opacity], allocation-only
    return mult.clamp(max=2.0).unsqueeze(1)
