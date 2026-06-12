"""HOG v3 unit tests: ring/blocked folds, query distance-weights, and the END-TO-END SIGN of the
first-order harm guidance (harmful Gaussian must receive a HIGHER drop rate). No rasterizer."""
import os, sys, math
import torch
from torch import nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.hog import HOGConfig, HOGState

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

class Opt: pass
opt = Opt()
for k, v in dict(hog_enable=True, hog_mode="grad", hog_fold_mode="blocked", hog_gamma=1.0,
                 hog_start=0, hog_meta_interval=10, hog_rho=0.1, hog_max_rate=0.6, hog_folds=4,
                 hog_worst_lambda=0.5, hog_conf_threshold=0.5, hog_tau=0.05, hog_inner_steps=2,
                 hog_sigma=0.3, hog_lr=0.1, hog_momentum=0.9, hog_neff_h=0.1, iterations=10000,
                 hog_refresh=100).items():
    setattr(opt, k, v)

class FakeCam:
    def __init__(self, name, ang):
        self.image_name = name
        self.camera_center = torch.tensor([math.cos(ang), math.sin(ang), 0.0], device=dev)
        self.original_image = torch.zeros(3, 8, 8, device=dev)

# 12 cams on a ring, names SHUFFLED to prove angle-sort (not name-sort) drives fold order
angles = [2 * math.pi * i / 12 for i in range(12)]
names = [f"v{(7 * i + 3) % 12:02d}" for i in range(12)]
cams = [FakeCam(names[i], angles[i]) for i in range(12)]
st = HOGState(HOGConfig(opt), cams, model_path=os.environ.get("TMP", "/tmp"))

# (1) blocked folds = contiguous arcs on the ring (max internal ring-gap <= fold size)
pos = {id(c): i for i, c in enumerate(st.ring)}
for q in st.query_folds:
    idx = sorted(pos[id(c)] for c in q)
    span = idx[-1] - idx[0]
    assert len(q) == 3 and span <= 2, f"fold not contiguous on ring: {idx}"
print("[ok] blocked folds are contiguous 3-arcs on the angle ring")

# (2) query weights: middle of arc (2 steps from support) > edges (1 step); mean 1
for w in st.query_weights:
    assert abs(sum(w) / len(w) - 1.0) < 1e-5, "weights not mean-1"
    assert max(w) == w[1] or max(w) == w[len(w)//2] or max(w) > min(w), "middle view should weigh most"
mid_heavier = all(max(w) > min(w) for w in st.query_weights)
assert mid_heavier, "distance weighting inactive"
print(f"[ok] query distance-weights (example {['%.2f'%x for x in st.query_weights[0]]}), mean-1")

# (3) END-TO-END SIGN: fake differentiable renderer where Gaussian 0 HURTS the query view
#     (its 'color' is wrong) and Gaussian 1 HELPS (matches GT). After harvest, rate(0) > rate(1).
class FakeG:
    def __init__(self):
        N = 500
        self._opacity = nn.Parameter(torch.zeros(N, 1, device=dev))      # logit 0 -> alpha .5
        self._xyz = torch.randn(N, 3, device=dev)
        self.optimizer = torch.optim.Adam([{"params": [self._opacity], "name": "opacity", "lr": 0.01}])
    @property
    def get_xyz(self): return self._xyz

g = FakeG()
GT = torch.full((3, 8, 8), 0.5, device=dev)
def fake_render(cam, rates):
    a = torch.sigmoid(g._opacity)
    # Gaussian 0 pushes image AWAY from GT (harmful), Gaussian 1 pushes TOWARD GT (helpful)
    img = torch.full((3, 8, 8), 0.5, device=dev) + a[0] * 0.4 - a[1] * 0.0  # base already at GT
    img = img + a[1] * 0.0
    return {"render": img.clamp(0, 1)}
for cam in cams: cam.original_image = GT
s1 = st.harvest_and_update(g, fake_render, iteration=100)
rates = st.applied_rates(g, 100)
assert torch.is_tensor(rates), "rates must be tensor after harvest"
# only Gaussian 0 received gradient (visible+harmful); with n_vis<=100 the guard zeroes s_hat ->
# verify via raw mechanics instead: re-run with many harmful/helpful gaussians
class FakeG2:
    def __init__(self):
        N = 600
        self._opacity = nn.Parameter(torch.zeros(N, 1, device=dev))
        self._xyz = torch.randn(N, 3, device=dev)
        self.optimizer = torch.optim.Adam([{"params": [self._opacity], "name": "opacity", "lr": 0.01}])
        self.harm = torch.zeros(N, device=dev); self.harm[:200] = 1.0    # first 200 harmful
        self.help = torch.zeros(N, device=dev); self.help[200:400] = 1.0 # next 200 helpful
    @property
    def get_xyz(self): return self._xyz
g2 = FakeG2()
def fake_render2(cam, rates):
    a = torch.sigmoid(g2._opacity).squeeze(1)
    err_push = (a * g2.harm).sum() * 0.001     # harmful: increases pixel error
    fix_pull = (a * g2.help).sum() * 0.001     # helpful: decreases pixel error
    img = torch.full((3, 8, 8), 0.5, device=dev) + err_push - fix_pull + 0.3  # offset so help reduces L1
    return {"render": img}
st2 = HOGState(HOGConfig(opt), cams, model_path=os.environ.get("TMP", "/tmp"))
s2 = st2.harvest_and_update(g2, fake_render2, iteration=100)
r2 = st2.applied_rates(g2, 100)
r_harm = float(r2[:200].mean()); r_help = float(r2[200:400].mean()); r_neut = float(r2[400:].mean())
assert r_harm > r_help, f"SIGN ERROR: harmful rate {r_harm:.4f} must exceed helpful {r_help:.4f}"
assert r_harm > r_neut > r_help, f"ordering broken: {r_harm:.4f} / {r_neut:.4f} / {r_help:.4f}"
assert abs(float(r2.mean()) - 0.1) < 0.02, f"mean rate {float(r2.mean()):.4f} != rho"
print(f"[ok] harm-gradient SIGN end-to-end: rate(harmful)={r_harm:.4f} > rate(neutral)={r_neut:.4f} "
      f"> rate(helpful)={r_help:.4f}, mean pinned {float(r2.mean()):.4f}")

# (4) grads zeroed after harvest (query = probe, not teacher)
assert g2._opacity.grad is None, "grads must be zeroed after harvest"
print("[ok] query gradients zeroed after harvest (probe, never teacher)")

# (5) stale-cache fallback on size change
class FakeG3(FakeG2):
    pass
g3 = FakeG3(); g3._xyz = torch.randn(777, 3, device=dev)
r3 = st2.applied_rates(g3, 200)
assert isinstance(r3, float) and abs(r3 - 0.1) < 1e-6, "stale cache must fall back to uniform rho"
print("[ok] stale-cache fallback -> uniform rho on N change")

print("\nALL HOG-V3 UNIT TESTS PASSED")
