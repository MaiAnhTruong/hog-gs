"""HOG-P (v5) unit tests: persistent robust harm field.
(1) weighted-median kernel vs brute force; (2) SPIKE SIGN end-to-end - one query view exploding
~400x with inverted signal (the measured failure) flips the legacy weighted-sum allocation but NOT
the median allocation; (3) EMA field semantics; (4) ancestry inheritance through REAL GaussianModel
clone/split/prune; (5) applied_rates re-derive after N change (mean pinned, order preserved,
no-witness upgrade path, legacy uniform fallback). No rasterizer needed."""
import os, sys, math
import torch
from torch import nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.hog import HOGConfig, HOGState

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

class Opt: pass
def make_opt(**over):
    opt = Opt()
    base = dict(hog_enable=True, hog_mode="grad", hog_signal="depth", hog_fold_mode="blocked",
                hog_gamma=1.0, hog_wgsd_beta=0.0, hog_holdout_window=30, hog_start=0,
                hog_meta_interval=10, hog_rho=0.1, hog_max_rate=0.6, hog_folds=4,
                hog_worst_lambda=0.5, hog_conf_threshold=0.5, hog_tau=0.05, hog_inner_steps=2,
                hog_sigma=0.3, hog_lr=0.1, hog_momentum=0.9, hog_neff_h=0.1, iterations=10000,
                hog_refresh=100, hog_persist=False, hog_median=False, hog_eta=0.3)
    base.update(over)
    for k, v in base.items():
        setattr(opt, k, v)
    return opt

class FakeCam:
    def __init__(self, name, ang):
        self.image_name = name
        self.camera_center = torch.tensor([math.cos(ang), math.sin(ang), 0.0], device=dev)
        self.original_image = torch.full((3, 8, 8), 0.5, device=dev)
        self.depth_reliable = True
        self.invdepthmap = torch.full((8, 8), 0.5, device=dev)
        self.depth_mask = torch.ones(1, 8, 8, device=dev)
        self.depth_confidence = torch.ones(8, 8, device=dev)
        self.spike = False

def make_cams():
    return [FakeCam(f"v{i:02d}", 2 * math.pi * i / 12) for i in range(12)]

TMP = os.environ.get("TMP", "/tmp")

# ---------------- (1) weighted-median kernel vs brute force ----------------
V, N = 4, 50
vals = torch.randn(V, N, device=dev)
wts = torch.rand(V, N, device=dev)
wts[torch.rand(V, N, device=dev) < 0.3] = 0.0     # missing observations
wts[:, 0] = 0.0                                   # one fully unobserved column
med, total = HOGState._weighted_median(vals, wts)
for n in range(N):
    t = float(wts[:, n].sum())
    if t == 0:
        assert float(med[n]) == 0.0, "all-zero column must be neutral"
        continue
    pairs = sorted(zip(vals[:, n].tolist(), wts[:, n].tolist()))
    c, exp = 0.0, None
    for v_, w_ in pairs:
        c += w_
        if c >= 0.5 * t:
            exp = v_
            break
    assert abs(float(med[n]) - exp) < 1e-6, f"weighted median mismatch col {n}"
# equal-weight 3-view sanity: median == middle value
v3 = torch.tensor([[-3.0], [0.7], [1.2]], device=dev)
m3, _ = HOGState._weighted_median(v3, torch.ones(3, 1, device=dev))
assert abs(float(m3[0]) - 0.7) < 1e-6, "3-view equal-weight median must be the middle value"
print("[ok] weighted-median kernel matches brute force (incl. zero-weight rows/columns)")

# ---------------- (2) SPIKE SIGN end-to-end: median resists, weighted-sum flips ----------------
# Gaussian groups: bad [0:200] pushes held-out depth off the prior; good [200:400] pulls it back;
# neutral [400:600] invisible. The MID query view of fold 0 (weight 1.5 in the legacy sum) explodes
# ~400x with INVERTED signal (good looks massively harmful there) = the measured L_q-spike failure.
class FakeG:
    def __init__(self):
        n = 600
        self._opacity = nn.Parameter(torch.zeros(n, 1, device=dev))
        self._xyz = torch.randn(n, 3, device=dev)
        self.optimizer = torch.optim.Adam([{"params": [self._opacity], "name": "opacity", "lr": 0.01}])
        self.bad = torch.zeros(n, device=dev); self.bad[:200] = 1.0
        self.good = torch.zeros(n, device=dev); self.good[200:400] = 1.0
    @property
    def get_xyz(self): return self._xyz

def run_spike_harvest(median):
    opt = make_opt(hog_median=median)
    cams = make_cams()
    st = HOGState(HOGConfig(opt), cams, model_path=TMP)
    st.query_folds[0][1].spike = True            # corrupt the MID query view (hardest case)
    g = FakeG()
    def fake_render(cam, rates):
        a = torch.sigmoid(g._opacity).squeeze(1)
        push = (a * g.bad).sum() * 0.001
        pull = (a * g.good).sum() * 0.001
        if getattr(cam, "spike", False):
            # ~400x magnitude, inverted: good explodes the error, bad slightly reduces it
            # (bad coefficient 0.002 != 0.001 so the cross-view sum cannot cancel exactly)
            depth = (torch.full((1, 8, 8), 0.5, device=dev)
                     + (a * g.good).sum() * 0.4 - (a * g.bad).sum() * 0.002 + 0.2)
        else:
            depth = torch.full((1, 8, 8), 0.5, device=dev) + push - pull + 0.2
        return {"render": torch.full((3, 8, 8), 0.5, device=dev), "depth": depth}
    st.harvest_and_update(g, fake_render, iteration=100)
    r = st.applied_rates(g, 100)
    return float(r[:200].mean()), float(r[200:400].mean()), float(r[400:].mean()), float(r.mean())

rb_s, rg_s, rn_s, _ = run_spike_harvest(median=False)
assert rg_s > rn_s, (f"expected the legacy weighted sum to FLIP under the spike "
                     f"(good {rg_s:.4f} should exceed neutral {rn_s:.4f}); defect not reproduced")
rb_m, rg_m, rn_m, mean_m = run_spike_harvest(median=True)
assert rb_m > rn_m > rg_m, f"median must resist the spike: bad {rb_m:.4f} / neu {rn_m:.4f} / good {rg_m:.4f}"
assert abs(mean_m - 0.1) < 0.005, f"median-mode mean not pinned: {mean_m:.4f}"
print(f"[ok] SPIKE SIGN: weighted-sum flips (good {rg_s:.4f} > neu {rn_s:.4f}) but median resists "
      f"(bad {rb_m:.4f} > neu {rn_m:.4f} > good {rg_m:.4f}, mean {mean_m:.4f} pinned)")

# ---------------- (3) EMA field semantics: h2 = (2-eta)*h1 under identical signal ----------------
opt3 = make_opt(hog_persist=True, hog_eta=0.3)
st3 = HOGState(HOGConfig(opt3), make_cams(), model_path=TMP)
g3 = FakeG()
def clean_render(cam, rates):
    a = torch.sigmoid(g3._opacity).squeeze(1)
    push = (a * g3.bad).sum() * 0.001
    pull = (a * g3.good).sum() * 0.001
    return {"render": torch.full((3, 8, 8), 0.5, device=dev),
            "depth": torch.full((1, 8, 8), 0.5, device=dev) + push - pull + 0.2}
st3.harvest_and_update(g3, clean_render, iteration=10)
h1 = g3._hog_harm.clone()
vis = h1 != 0
assert int(vis.sum()) == 400, "bad+good groups must be visible in the field"
st3.harvest_and_update(g3, clean_render, iteration=20)     # fold 1, identical signal
h2 = g3._hog_harm
assert torch.allclose(h2[vis], (2 - 0.3) * h1[vis], rtol=1e-4), "EMA recursion broken"
assert torch.all(h2[~vis] == 0), "rows never observed must stay neutral"
print(f"[ok] EMA field: h2 = (2-eta)*h1 on visible rows (eta=0.3), unobserved rows untouched")

# ---------------- (4) ancestry inheritance through REAL GaussianModel ops ----------------
if dev == "cuda":
    from scene.gaussian_model import GaussianModel
    K = 60
    gm = GaussianModel(0)
    def P(t): return nn.Parameter(t.requires_grad_(True))
    gm._xyz = P(torch.randn(K, 3, device=dev))
    gm._features_dc = P(torch.randn(K, 1, 3, device=dev))
    gm._features_rest = P(torch.zeros(K, 0, 3, device=dev))
    gm._opacity = P(torch.zeros(K, 1, device=dev))
    gm._scaling = P(torch.full((K, 3), math.log(1e-4), device=dev))
    rot = torch.zeros(K, 4, device=dev); rot[:, 0] = 1.0
    gm._rotation = P(rot)
    gm.optimizer = torch.optim.Adam([
        {"params": [gm._xyz], "lr": 1e-3, "name": "xyz"},
        {"params": [gm._features_dc], "lr": 1e-3, "name": "f_dc"},
        {"params": [gm._features_rest], "lr": 1e-3, "name": "f_rest"},
        {"params": [gm._opacity], "lr": 1e-3, "name": "opacity"},
        {"params": [gm._scaling], "lr": 1e-3, "name": "scaling"},
        {"params": [gm._rotation], "lr": 1e-3, "name": "rotation"}])
    gm.percent_dense = 0.01
    gm.xyz_gradient_accum = torch.zeros(K, 1, device=dev)
    gm.denom = torch.zeros(K, 1, device=dev)
    gm.max_radii2D = torch.zeros(K, device=dev)
    gm.tmp_radii = torch.zeros(K, device=dev)
    gm.spatial_lr_scale = 1.0
    gm._hog_harm = torch.arange(K, dtype=torch.float32, device=dev)

    # clone: children appended, harm copied from parents
    grads = torch.zeros(K, 1, device=dev)
    clone_idx = [5, 17, 23]
    grads[clone_idx] = 1.0
    gm.densify_and_clone(grads, 0.5, 1.0)
    h = gm._hog_harm
    assert h.shape[0] == K + 3 and torch.equal(h[K:], torch.tensor([5., 17., 23.], device=dev)), \
        "clone inheritance broken"

    # split: parents 2 and 8 (made large) -> 2 children each inherit, parents pruned
    pre = gm._hog_harm.clone()
    n_now = gm.get_xyz.shape[0]
    with torch.no_grad():
        gm._scaling[[2, 8]] = 0.0                      # log-scale 0 -> scale 1.0 > percent_dense
    g2 = torch.zeros(n_now, device=dev)
    g2[[2, 8]] = 1.0
    gm.densify_and_split(g2, 0.5, 1.0)
    sel = torch.zeros(n_now, dtype=torch.bool, device=dev); sel[[2, 8]] = True
    expected = torch.cat((pre[~sel], pre[sel].repeat(2)))
    assert torch.equal(gm._hog_harm, expected), "split inheritance/prune alignment broken"

    # direct prune: random rows removed, field stays aligned
    pre2 = gm._hog_harm.clone()
    m = torch.zeros(pre2.shape[0], dtype=torch.bool, device=dev)
    m[torch.randperm(pre2.shape[0])[:10]] = True
    gm.prune_points(m)
    assert torch.equal(gm._hog_harm, pre2[~m]), "prune alignment broken"
    print(f"[ok] ancestry inheritance through real GaussianModel clone/split/prune "
          f"(N {K} -> {gm._hog_harm.shape[0]}, field aligned)")
else:
    print("[skip] GaussianModel inheritance test (no CUDA)")

# ---------------- (5) applied_rates: re-derive, no-witness upgrade, fallback, pinning ----------------
class StubG:
    def __init__(self, n):
        self._x = torch.randn(n, 3, device=dev)
    @property
    def get_xyz(self): return self._x

# (5a) persist re-derive after N change: mean pinned, order follows h, counted once
st5 = HOGState(HOGConfig(make_opt(hog_persist=True)), make_cams(), model_path=TMP)
stub = StubG(300)
stub._hog_harm = 0.1 * torch.randn(300, device=dev)
st5._rates_cache = torch.full((200,), 0.1, device=dev)     # stale (pre-densify N)
r = st5.applied_rates(stub, 0)
assert r.shape[0] == 300 and st5.rederive_count == 1, "re-derive did not trigger"
assert abs(float(r.mean()) - 0.1) < 1e-4, f"re-derived mean not pinned: {float(r.mean()):.5f}"
assert torch.equal(torch.argsort(r), torch.argsort(stub._hog_harm)), "re-derived order must follow h"
_ = st5.applied_rates(stub, 1)
assert st5.rederive_count == 1, "must serve from cache after one re-derive"

# (5b) no-witness upgrade: derive without SWD, then re-derive once scores appear
st6 = HOGState(HOGConfig(make_opt(hog_persist=True, hog_wgsd_beta=2.0)), make_cams(), model_path=TMP)
stub6 = StubG(300)
stub6._hog_harm = 0.1 * torch.randn(300, device=dev)
ra = st6.applied_rates(stub6, 0)
assert st6._rates_no_witness and st6.rederive_count == 1, "no-witness derivation not flagged"
fr = torch.zeros(300, device=dev); fr[:150] = 1.0
stub6._swd_scores = {"front_ratio": fr, "witness": torch.zeros(300, device=dev)}
rb = st6.applied_rates(stub6, 1)
assert st6.rederive_count == 2 and not st6._rates_no_witness, "witness upgrade did not re-derive"
assert not torch.allclose(ra, rb), "witness term must change the rates"
assert float(rb[:150].mean()) > float(rb[150:].mean()), "high-front group must out-rate after upgrade"
_ = st6.applied_rates(stub6, 2)
assert st6.rederive_count == 2, "must cache after the witness upgrade"

# (5c) legacy fallback unchanged when persist is off
st7 = HOGState(HOGConfig(make_opt()), make_cams(), model_path=TMP)
r7 = st7.applied_rates(StubG(123), 0)
assert isinstance(r7, float) and abs(r7 - 0.1) < 1e-9 and st7.fallback_count == 1, \
    "legacy uniform fallback broken"

# (5d) adversarial pinning: extreme field +-3 with witness +-1*beta=2 stays mean-rho, max<=0.6
st8 = HOGState(HOGConfig(make_opt(hog_persist=True, hog_wgsd_beta=2.0)), make_cams(), model_path=TMP)
stub8 = StubG(400)
h8 = torch.full((400,), -3.0, device=dev); h8[:200] = 3.0
stub8._hog_harm = h8
fr8 = torch.zeros(400, device=dev); fr8[:200] = 1.0
stub8._swd_scores = {"front_ratio": fr8, "witness": 1.0 - fr8}
r8 = st8.applied_rates(stub8, 0)
assert abs(float(r8.mean()) - 0.1) < 1e-3 and float(r8.max()) <= 0.6 + 1e-6, \
    f"adversarial pinning broken: mean {float(r8.mean()):.4f} max {float(r8.max()):.4f}"
print("[ok] applied_rates: re-derive (mean pinned, h-ordered), no-witness upgrade, "
      "legacy fallback, adversarial pinning")

print("\nALL HOG-P UNIT TESTS PASSED")
