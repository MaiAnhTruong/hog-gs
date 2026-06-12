"""HOG-GS unit tests (no rasterizer): folds, n_eff math, policy renorm, snapshot/restore roundtrip,
and ES convergence on a toy objective. All must PASS before any GPU run."""
import os, sys, math
import torch
from torch import nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.hog import HOGConfig, HOGState

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

class Opt: pass
opt = Opt()
for k, v in dict(hog_enable=True, hog_start=0, hog_meta_interval=10, hog_inner_steps=2, hog_rho=0.1,
                 hog_max_rate=0.6, hog_sigma=0.3, hog_lr=0.1, hog_momentum=0.9, hog_folds=4,
                 hog_refresh=10, hog_worst_lambda=0.5, hog_neff_h=0.1, hog_conf_threshold=0.5,
                 hog_tau=0.05).items():
    setattr(opt, k, v)

class FakeCam:
    def __init__(self, name, C):
        self.image_name = name
        self.camera_center = torch.tensor(C, dtype=torch.float32, device=dev)
        self.R = torch.eye(3); self.T = torch.zeros(3)
        self.FoVx = 1.0; self.FoVy = 1.0; self.image_height = 64; self.image_width = 64
        self.depth_reliable = False; self.invdepthmap = None; self.depth_confidence = None

cams = [FakeCam(f"v{i:02d}", [math.cos(i), math.sin(i), -3.0]) for i in range(12)]
st = HOGState(HOGConfig(opt), cams, model_path=os.environ.get("TMP", "/tmp"))

# (1) folds: 4 disjoint query folds of 3, support = complement of 9, all views covered
assert len(st.query_folds) == 4 and all(len(q) == 3 for q in st.query_folds), "fold sizes"
assert all(len(s) == 9 for s in st.support_folds), "support sizes"
allq = [c.image_name for q in st.query_folds for c in q]
assert sorted(allq) == sorted(c.image_name for c in cams), "folds must cover all views exactly once"
for q, s in zip(st.query_folds, st.support_folds):
    assert not (set(id(c) for c in q) & set(id(c) for c in s)), "query/support overlap!"
print("[ok] folds: 4x(3 query / 9 support), disjoint, covering")

# (2) n_eff: identical directions -> ~1 ; orthogonal-ish directions -> ~V
class FakeG:
    def __init__(self, xyz): self._xyz = xyz
    @property
    def get_xyz(self): return self._xyz
    @property
    def get_opacity(self): return torch.full((self._xyz.shape[0], 1), 0.5, device=dev)
g1 = FakeG(torch.tensor([[0.0, 0.0, 2.0]], device=dev))
same_cams = [FakeCam(f"s{i}", [0.0, 0.0, -3.0]) for i in range(6)]          # identical viewpoints
spread = [[5, 0, 2], [-5, 0, 2], [0, 5, 2], [0, -5, 2], [0, 0, 7.0], [3.5, 3.5, 2]]
diff_cams = [FakeCam(f"d{i}", c) for i, c in enumerate(spread)]              # well-spread viewpoints
n_same = float(st._neff(g1, same_cams)[0]); n_diff = float(st._neff(g1, diff_cams)[0])
assert n_same < 1.5, f"identical cams should give n_eff~1, got {n_same}"
assert n_diff > 3.0, f"spread cams should give n_eff>>1, got {n_diff}"
print(f"[ok] n_eff: identical={n_same:.2f} (~1), spread={n_diff:.2f} (>>1)")

# (3) policy renorm: mean rate == rho regardless of phi; clamp respected
feats = torch.rand(5000, st.N_FEATURES, device=dev)
for phi in (torch.zeros(st.N_FEATURES + 1, device=dev),
            torch.randn(st.N_FEATURES + 1, device=dev) * 2):
    r = st.rates_from_policy(feats, phi)
    assert abs(float(r.mean()) - 0.1) < 0.02, f"mean rate {float(r.mean())} != rho"
    assert float(r.max()) <= 0.6 + 1e-6 and float(r.min()) >= 0.0, "clamp violated"
print(f"[ok] policy: mean rate pinned to rho (allocation-only), clamps hold")

# (4) snapshot/restore roundtrip incl. Adam state
class TinyModel:
    def __init__(self):
        self.p1 = nn.Parameter(torch.randn(10, 3, device=dev))
        self.p2 = nn.Parameter(torch.randn(10, 1, device=dev))
        self.optimizer = torch.optim.Adam(
            [{"params": [self.p1], "name": "xyz", "lr": 0.01},
             {"params": [self.p2], "name": "opacity", "lr": 0.01}], eps=1e-15)
tm = TinyModel()
(tm.p1.sum() + tm.p2.sum()).backward(); tm.optimizer.step(); tm.optimizer.zero_grad()
snap = st.snapshot(tm)
p1_0, ea_0 = tm.p1.data.clone(), tm.optimizer.state[tm.p1]["exp_avg"].clone()
for _ in range(3):
    (tm.p1.pow(2).sum() + tm.p2.pow(2).sum()).backward(); tm.optimizer.step(); tm.optimizer.zero_grad()
assert not torch.allclose(tm.p1.data, p1_0), "params should have moved"
st.restore(tm, snap)
assert torch.allclose(tm.p1.data, p1_0, atol=1e-7), "param restore failed"
assert torch.allclose(tm.optimizer.state[tm.p1]["exp_avg"], ea_0, atol=1e-7), "Adam state restore failed"
print("[ok] snapshot/restore: params + Adam moments roundtrip exactly")

# (5) ES toy convergence: hidden target weights w*; L(phi) = ||rates(phi)-rates(w*)||^2.
torch.manual_seed(1)
w_star = torch.tensor([2.0, -2.0, 1.0, 0.0, 0.0, 0.0, 1.5, 0.0], device=dev)
target = st.rates_from_policy(feats, w_star)
def toy_loss(phi): return float(((st.rates_from_policy(feats, phi) - target) ** 2).mean() * 1e3)
st.phi = torch.zeros(st.N_FEATURES + 1, device=dev); st.m = torch.zeros_like(st.phi)
l0 = toy_loss(st.phi)
for t in range(300):
    eps = torch.randn_like(st.phi)
    lp, lm = toy_loss(st.phi + 0.3 * eps), toy_loss(st.phi - 0.3 * eps)
    g = math.copysign(1.0, lp - lm)
    st.m = 0.9 * st.m + 0.1 * (g * eps)
    st.phi = st.phi - 0.1 * st.m
l1 = toy_loss(st.phi)
assert l1 < 0.5 * l0, f"sign-ES failed to reduce toy loss: {l0:.4f} -> {l1:.4f}"
print(f"[ok] sign-ES converges on toy objective: loss {l0:.4f} -> {l1:.4f}")

print("\nALL HOG UNIT TESTS PASSED")
