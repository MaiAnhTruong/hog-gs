"""HOG v4 unit tests: depth-signal harm SIGN end-to-end, hybrid exponent, double-renorm mean-rate,
holdout-window fold exposure. No rasterizer."""
import os, sys, math
import torch
from torch import nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.hog import HOGConfig, HOGState

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

class Opt: pass
opt = Opt()
for k, v in dict(hog_enable=True, hog_mode="grad", hog_signal="depth", hog_fold_mode="blocked",
                 hog_gamma=1.0, hog_wgsd_beta=0.0, hog_holdout_window=30, hog_start=0,
                 hog_meta_interval=10, hog_rho=0.1, hog_max_rate=0.6, hog_folds=4,
                 hog_worst_lambda=0.5, hog_conf_threshold=0.5, hog_tau=0.05, hog_inner_steps=2,
                 hog_sigma=0.3, hog_lr=0.1, hog_momentum=0.9, hog_neff_h=0.1, iterations=10000,
                 hog_refresh=100).items():
    setattr(opt, k, v)

class FakeCam:
    def __init__(self, name, ang):
        self.image_name = name
        self.camera_center = torch.tensor([math.cos(ang), math.sin(ang), 0.0], device=dev)
        self.original_image = torch.full((3, 8, 8), 0.5, device=dev)
        self.depth_reliable = True
        self.invdepthmap = torch.full((8, 8), 0.5, device=dev)     # prior invdepth
        self.depth_mask = torch.ones(1, 8, 8, device=dev)
        self.depth_confidence = torch.ones(8, 8, device=dev)

cams = [FakeCam(f"v{i:02d}", 2 * math.pi * i / 12) for i in range(12)]

# (1) END-TO-END SIGN with DEPTH signal: Gaussian group A corrupts rendered depth away from prior,
#     group B pulls it toward the prior. rate(A) > rate(B); mean pinned to rho EXACTLY (double renorm).
class FakeG:
    def __init__(self):
        N = 600
        self._opacity = nn.Parameter(torch.zeros(N, 1, device=dev))
        self._xyz = torch.randn(N, 3, device=dev)
        self.optimizer = torch.optim.Adam([{"params": [self._opacity], "name": "opacity", "lr": 0.01}])
        self.bad = torch.zeros(N, device=dev); self.bad[:200] = 1.0
        self.good = torch.zeros(N, device=dev); self.good[200:400] = 1.0
    @property
    def get_xyz(self): return self._xyz

g = FakeG()
def fake_render(cam, rates):
    a = torch.sigmoid(g._opacity).squeeze(1)
    push = (a * g.bad).sum() * 0.001       # bad: pushes depth AWAY from prior (0.5 + push)
    pull = (a * g.good).sum() * 0.001      # good: pulls depth back toward prior
    depth = torch.full((1, 8, 8), 0.5, device=dev) + push - pull + 0.2   # offset: good reduces |err|
    rgb = torch.full((3, 8, 8), 0.5, device=dev)
    return {"render": rgb, "depth": depth}

st = HOGState(HOGConfig(opt), cams, model_path=os.environ.get("TMP", "/tmp"))
s = st.harvest_and_update(g, fake_render, iteration=100)
assert s["signal"] == "depth", "signal mode not logged"
r = st.applied_rates(g, 100)
r_bad, r_good, r_neu = float(r[:200].mean()), float(r[200:400].mean()), float(r[400:].mean())
assert r_bad > r_neu > r_good, f"DEPTH-SIGN broken: bad {r_bad:.4f} / neu {r_neu:.4f} / good {r_good:.4f}"
assert abs(float(r.mean()) - 0.1) < 0.005, f"double-renorm failed: mean {float(r.mean()):.4f}"
print(f"[ok] depth-harm SIGN: rate(bad)={r_bad:.4f} > rate(neutral)={r_neu:.4f} > rate(good)={r_good:.4f}, "
      f"mean={float(r.mean()):.4f} (pinned)")

# (2) hybrid exponent: with wgsd_beta>0 and fake _swd_scores, high-front Gaussians get higher rates
opt.hog_wgsd_beta = 2.0
st2 = HOGState(HOGConfig(opt), cams, model_path=os.environ.get("TMP", "/tmp"))
g2 = FakeG()
N = 600
g2._swd_scores = {
    "front_ratio": torch.zeros(N, device=dev), "witness": torch.zeros(N, device=dev),
    "back_ratio": torch.zeros(N, device=dev), "conf_seen": torch.ones(N, device=dev),
    "rel_mean": torch.zeros(N, device=dev), "evidence": torch.ones(N, dtype=torch.bool, device=dev),
    "contradiction": torch.zeros(N, dtype=torch.bool, device=dev),
}
g2._swd_scores["front_ratio"][400:] = 1.0          # neutral-by-harm group is high-front
def fake_render2(cam, rates):
    return fake_render.__wrapped__(cam, rates) if hasattr(fake_render, "__wrapped__") else None
def fake_render2(cam, rates):
    a = torch.sigmoid(g2._opacity).squeeze(1)
    push = (a * g2.bad).sum() * 0.001
    pull = (a * g2.good).sum() * 0.001
    return {"render": torch.full((3, 8, 8), 0.5, device=dev),
            "depth": torch.full((1, 8, 8), 0.5, device=dev) + push - pull + 0.2}
st2.harvest_and_update(g2, fake_render2, iteration=100)
r2 = st2.applied_rates(g2, 100)
assert float(r2[400:].mean()) > float(r2[200:400].mean()), "hybrid: high-front group must out-rate helpful group"
print(f"[ok] hybrid: high-front neutral group rate {float(r2[400:].mean()):.4f} > helpful {float(r2[200:400].mean()):.4f}")

# (3) holdout window: upcoming fold names = next fold to harvest, 3 names, subset of ring
names = st.upcoming_query_names()
assert len(names) == 3 and all(isinstance(n, str) and n for n in names), "upcoming_query_names broken"
print(f"[ok] upcoming holdout fold exposed: {sorted(names)}")

print("\nALL HOG-V4 UNIT TESTS PASSED")
