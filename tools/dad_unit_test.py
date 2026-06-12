"""Fast synthetic unit test of Depth-Anchored Densification target math (no rasterizer)."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.depthguide import compute_depth_targets

dev = "cuda" if torch.cuda.is_available() else "cpu"

class FakeG:
    def __init__(self, xyz): self._xyz = xyz
    @property
    def get_xyz(self): return self._xyz

class FakeCam:
    def __init__(self, inv_val, R=None, T=None, C=None):
        self.R = (R if R is not None else torch.eye(3)).to(dev)
        self.T = (T if T is not None else torch.zeros(3)).to(dev)
        self.camera_center = (C if C is not None else torch.zeros(3)).to(dev)
        self.FoVx = 1.0; self.FoVy = 1.0
        self.image_height = 100; self.image_width = 100
        self.depth_reliable = True
        self.invdepthmap = torch.full((100, 100), float(inv_val), device=dev)

# Gaussian on the optical axis at camera-depth 2; prior says surface at depth 3 (invdepth 1/3).
xyz = torch.tensor([[0.0, 0.0, 2.0]], device=dev)
g = FakeG(xyz)

# (1) two agreeing cameras (both prior 1/3) -> reliable, target depth ~3
tgt, rel = compute_depth_targets(g, [FakeCam(1/3), FakeCam(1/3)], min_views=2, agree_rel=0.15)
assert bool(rel[0]), "should be reliable when views agree"
assert abs(float(tgt[0, 2]) - 3.0) < 1e-3, f"target z should be 3, got {float(tgt[0,2])}"
print(f"[ok] agree: reliable=True, target z={float(tgt[0,2]):.4f} (expect 3.0)")

# (2) disagreeing cameras (1/3 -> t=1.5 vs 2/3 -> t=0.75, both within clip) -> NOT reliable, fallback z=2
tgt2, rel2 = compute_depth_targets(g, [FakeCam(1/3), FakeCam(2/3)], min_views=2, agree_rel=0.15)
assert not bool(rel2[0]), "should be UNreliable when views disagree"
assert abs(float(tgt2[0, 2]) - 2.0) < 1e-3, f"fallback target z should be 2 (original), got {float(tgt2[0,2])}"
print(f"[ok] disagree: reliable=False, target z={float(tgt2[0,2]):.4f} (fallback to original 2.0)")

# (3) single view < min_views=2 -> not reliable
tgt3, rel3 = compute_depth_targets(g, [FakeCam(1/3)], min_views=2, agree_rel=0.15)
assert not bool(rel3[0]) and abs(float(tgt3[0, 2]) - 2.0) < 1e-3, "single view should be unreliable+fallback"
print(f"[ok] single-view: reliable=False, fallback z={float(tgt3[0,2]):.4f}")

# (4) batch + finiteness
xyzB = torch.randn(500, 3, device=dev) + torch.tensor([0., 0., 3.], device=dev)
tB, rB = compute_depth_targets(FakeG(xyzB), [FakeCam(1/3), FakeCam(1/3.2)], min_views=1, agree_rel=0.3)
assert torch.isfinite(tB).all(), "non-finite targets"
print(f"[ok] batch: N=500 finite, reliable frac={float(rB.float().mean()):.2f}")
print("\nALL DAD UNIT TESTS PASSED")
