#
# HOG-GS v3: Held-Out-Guided Gaussian Regularization for sparse-view 3DGS.
#
# v1/v2 POST-MORTEM (measured on kitchen):
#   v1: antithetic arms used different RNG -> sign(dL) ~ coin flip -> policy random-walked. Fixed by
#       Common Random Numbers (kept below for the "es" ablation mode).
#   v2 (CRN): the ES signal was real (above the measured noise floor) but optimized the WRONG target:
#       with INTERLEAVED folds the query views sit BETWEEN support views, so the held-out task is pure
#       INTERPOLATION - floaters that fit the train views also fit nearby query views, and the learned
#       policy ended up PROTECTING front-floaters (w_front=-3.24), the exact opposite of the measured
#       winning direction (WGSD beta=+2). Classic spatially-autocorrelated-CV failure.
#
# v3 fixes both diagnosed causes:
#   (1) BLOCKED FOLDS: cameras are ring-ordered by angle (PCA plane around the scene), folds are
#       CONTIGUOUS ARCS, and each query view is weighted by its ring-distance to the nearest support
#       view -> the held-out task becomes semi-EXTRAPOLATIVE (blocked cross-validation for spatially
#       autocorrelated data).
#   (2) FIRST-ORDER HARM GUIDANCE ("grad" mode, default): instead of zeroth-order 8-step ES probes
#       (myopic, 1 bit / ~22 renders), render the held-out views WITHOUT training on them and read the
#       per-Gaussian opacity gradient of the held-out loss: s_i = dL_Q/d(opacity_i) > 0 means Gaussian
#       i HURTS the held-out views -> drop it more. One harvest = N-dimensional signal for ~3 renders.
#       Query views are probes, never teachers: their gradients are read, then ZEROED - model
#       parameters are never updated from them.
#   Rates: r_i = rho * exp(gamma * s_hat_i) / mean(.), s_hat = robust-standardized harm, mean pinned
#   to rho (allocation-only; renderer's per-Gaussian inverted dropout keeps E[opacity] unbiased).
#
# Eval parity: rates apply to TRAINING renders only; all modes evaluate the plain model.
#
import os
import json
import math
import torch

from utils.depthguide import compute_surface_witness_scores
from utils.pgdr_utils import _project_points_to_camera
from utils.loss_utils import l1_loss, ssim


class HOGConfig:
    def __init__(self, opt):
        self.enable = bool(getattr(opt, "hog_enable", False))
        self.mode = str(getattr(opt, "hog_mode", "grad"))            # "grad" (v3) | "es" (ablation)
        self.fold_mode = str(getattr(opt, "hog_fold_mode", "blocked"))  # "blocked" | "interleaved"
        self.start = int(getattr(opt, "hog_start", 1500))
        self.meta_interval = int(getattr(opt, "hog_meta_interval", 100))
        self.rho = float(getattr(opt, "hog_rho", 0.10))              # target MEAN drop rate
        self.max_rate = float(getattr(opt, "hog_max_rate", 0.6))
        self.gamma = float(getattr(opt, "hog_gamma", 1.0))           # harm->rate sharpness (grad mode)
        # v4: harm signal space. "rgb" (v3) ties the signal to the train pool's GT (interpolative bias,
        # measured: corr_front<0). "depth" anchors it to the EXTERNAL depth prior at held-out views =
        # render-aware, contribution-weighted depth-violation (the project's only winning signal class).
        self.signal = str(getattr(opt, "hog_signal", "depth"))       # "depth" | "rgb"
        self.wgsd_beta = float(getattr(opt, "hog_wgsd_beta", 0.0))   # hybrid: + beta*(front-witness) in exponent
        # v5 HOG-P: persistent robust harm field. persist = EMA field carried through densification
        # (children inherit the parent's score), so rates are RE-DERIVED when N changes instead of
        # falling back to uniform rho (legacy: end-of-iteration densify at t=0 mod 100 invalidated the
        # same-iteration harvest -> ~49% of iterations ran uniform). median = per-view standardized
        # unweighted median across query views (breakdown point 1/3; one exploding query view -
        # measured L_q spikes up to ~400x - cannot flip the allocation, unlike the weighted sum).
        self.persist = bool(getattr(opt, "hog_persist", False))
        self.eta = float(getattr(opt, "hog_eta", 0.3))
        self.median = bool(getattr(opt, "hog_median", False))
        self.holdout_window = int(getattr(opt, "hog_holdout_window", 30))  # iters before harvest: queries excluded from training
        self.folds = int(getattr(opt, "hog_folds", 4))
        self.worst_lambda = float(getattr(opt, "hog_worst_lambda", 0.5))
        self.conf_threshold = float(getattr(opt, "hog_conf_threshold", 0.5))
        self.tau = float(getattr(opt, "hog_tau", 0.05))
        # ES-mode (ablation) knobs
        self.inner_steps = int(getattr(opt, "hog_inner_steps", 8))
        self.sigma = float(getattr(opt, "hog_sigma", 0.3))
        self.lr = float(getattr(opt, "hog_lr", 0.1))
        self.momentum = float(getattr(opt, "hog_momentum", 0.9))
        self.neff_h = float(getattr(opt, "hog_neff_h", 0.1))
        self.total_iters = int(getattr(opt, "iterations", 10000))
        self.refresh = int(getattr(opt, "hog_refresh", 100))

    def __repr__(self):
        return ("HOGConfig(mode={} fold_mode={} start={} interval={} rho={} gamma={} folds={} "
                "worst_lambda={} persist={} median={} eta={})".format(
                self.mode, self.fold_mode, self.start, self.meta_interval,
                self.rho, self.gamma, self.folds, self.worst_lambda,
                self.persist, self.median, self.eta))


class HOGState:
    N_FEATURES = 7   # (es-mode policy features)

    def __init__(self, config: HOGConfig, train_cameras, model_path):
        self.cfg = config
        self.model_path = model_path
        self.log_path = os.path.join(model_path, "hog_meta_log.jsonl")
        ring = self._ring_order(train_cameras)
        F = max(2, min(self.cfg.folds, len(ring)))
        if self.cfg.fold_mode == "blocked":
            # contiguous arcs on the camera ring -> semi-extrapolative held-out task
            arc = max(1, len(ring) // F)
            self.query_folds = [ring[k * arc:(k + 1) * arc] for k in range(F)]
        else:
            self.query_folds = [[ring[i] for i in range(k, len(ring), F)] for k in range(F)]
        self.support_folds = [[c for c in ring if c not in q] for q in self.query_folds]
        self.ring = ring
        self.query_weights = self._query_weights()
        self.all_cams = ring
        # es-mode policy state
        self.phi = torch.zeros(self.N_FEATURES + 1, device="cuda")
        self.m = torch.zeros_like(self.phi)
        self.meta_count = 0
        self._rates_cache = None
        self._rates_iter = -1
        # HOG-P coverage bookkeeping (counted between harvests, logged per harvest)
        self.rederive_count = 0    # rates re-derived from the inherited field after N changed
        self.fallback_count = 0    # iterations that fell back to uniform rho (no field available)
        self._rates_no_witness = False   # cached rates were derived without the witness term (stale SWD)

    # ---------- camera ring + folds ----------
    @staticmethod
    def _ring_order(cameras):
        """Order cameras by angle in the PCA plane of camera centers (ring order for 360-ish scenes)."""
        cams = sorted(cameras, key=lambda c: getattr(c, "image_name", ""))
        C = torch.stack([torch.as_tensor(c.camera_center, dtype=torch.float32).cpu() for c in cams])
        X = C - C.mean(0, keepdim=True)
        try:
            _, _, V = torch.linalg.svd(X, full_matrices=False)
            uv = X @ V[:2].T                                   # [V,2] coords in dominant plane
            ang = torch.atan2(uv[:, 1], uv[:, 0])
            order = torch.argsort(ang).tolist()
            return [cams[i] for i in order]
        except Exception:
            return cams

    def _query_weights(self):
        """Per-fold per-query weight = ring-distance (in view steps, circular) to nearest support view,
        normalized to mean 1 within the fold. Pushes the held-out risk toward the most extrapolative
        view of each fold."""
        n = len(self.ring)
        pos = {id(c): i for i, c in enumerate(self.ring)}
        weights = []
        for q_fold, s_fold in zip(self.query_folds, self.support_folds):
            s_pos = [pos[id(c)] for c in s_fold]
            w = []
            for qc in q_fold:
                qp = pos[id(qc)]
                d = min(min(abs(qp - sp), n - abs(qp - sp)) for sp in s_pos)
                w.append(float(max(d, 1)))
            w = torch.tensor(w, dtype=torch.float32)
            weights.append((w / w.mean().clamp(min=1e-6)).tolist())
        return weights

    # ---------- v3 core: first-order held-out harm guidance ----------
    def harvest_and_update(self, gaussians, render_fn, iteration):
        """Render the current fold's query views (NO training on them), accumulate the distance-
        weighted held-out loss, and read the per-Gaussian opacity gradient = held-out HARM. Then map
        harm -> per-Gaussian drop rates (exp family, mean pinned to rho). Gradients are zeroed after
        harvesting: query views are probes, never teachers."""
        fold = self.meta_count % len(self.query_folds)
        query = self.query_folds[fold]
        wq = self.query_weights[fold]
        # make sure no stale grads pollute the harvest
        for group in gaussians.optimizer.param_groups:
            if group["params"][0].grad is not None:
                group["params"][0].grad = None
        N = gaussians.get_xyz.shape[0]
        per_view = []
        s_views = []                       # median mode: one harm vector per query view
        for cam, w in zip(query, wq):
            pkg = render_fn(cam, None)                          # plain model render (no dropout)
            if self.cfg.signal == "depth" and getattr(cam, "depth_reliable", False) and cam.invdepthmap is not None:
                # held-out GEOMETRY violation against the EXTERNAL depth prior (confidence-gated):
                # render-aware version of the depth-witness signal, measured where not being fitted.
                mask = cam.depth_mask.cuda()
                conf = getattr(cam, "depth_confidence", None)
                if conf is not None:
                    c = (conf if torch.is_tensor(conf) else torch.as_tensor(conf)).cuda().float()
                    if c.dim() == 2:
                        c = c[None]
                    mask = mask * c.clamp(0.0, 1.0)
                num = (torch.abs(pkg["depth"] - cam.invdepthmap.cuda()) * mask).sum()
                lq = num / mask.sum().clamp(min=1.0)
            else:
                lq = l1_loss(pkg["render"], cam.original_image.cuda())
            per_view.append(float(lq.item()))
            if self.cfg.median:
                # read this view's harm alone; per-view standardization makes any scalar backward
                # weight irrelevant (wq stays in the legacy sum path and the log only)
                lq.backward()
                g = gaussians._opacity.grad
                s_views.append(g.detach().squeeze(1).clone() if g is not None
                               else torch.zeros(N, device="cuda"))
                for group in gaussians.optimizer.param_groups:
                    group["params"][0].grad = None
            else:
                (w * lq).backward()
        if self.cfg.median:
            s_hat, vis = self._median_harm(s_views, wq)
            s = s_hat
        else:
            s = gaussians._opacity.grad
            s = s.detach().squeeze(1).clone() if s is not None else torch.zeros(N, device="cuda")
            vis = s != 0
        # query views are probes, never teachers: zero ALL parameter grads now
        for group in gaussians.optimizer.param_groups:
            group["params"][0].grad = None
        n_vis = int(vis.sum())
        if not self.cfg.median:
            # robust standardization over Gaussians that actually received signal (visible in fold)
            if n_vis > 100:
                mu = s[vis].mean()
                sd = s[vis].std().clamp(min=1e-12)
                s_hat = torch.zeros_like(s)
                s_hat[vis] = ((s[vis] - mu) / sd).clamp(-3.0, 3.0)
            else:
                s_hat = torch.zeros_like(s)
        elif n_vis <= 100:
            s_hat = torch.zeros_like(s_hat)     # too few observations to trust this fold
        # HOG-P persistent field: EMA of standardized harm. gaussian_model carries it through
        # densification (clone/split children inherit the parent's value, prune trims rows), so
        # applied_rates can re-derive instead of falling back to uniform when N changes.
        if self.cfg.persist:
            h = getattr(gaussians, "_hog_harm", None)
            if h is None or h.shape[0] != N:
                h = torch.zeros(N, device=s_hat.device)
            else:
                h = h.clone()
            if n_vis > 100:
                h[vis] = (1.0 - self.cfg.eta) * h[vis] + self.cfg.eta * s_hat[vis]
            gaussians._hog_harm = h
            drive = h
        else:
            drive = s_hat
        exponent = self.cfg.gamma * drive
        term, witness_ok = self._witness_term(gaussians, N)
        if term is not None:
            # hybrid: external witness-state term (the proven WGSD axis) + functional harm term
            exponent = exponent + term
        rates = self._rates_from_exponent(exponent)
        self._rates_cache = rates
        self._rates_no_witness = not witness_ok
        self._rates_iter = iteration
        self.meta_count += 1
        summary = {
            "iteration": int(iteration), "harvest": int(self.meta_count), "fold": int(fold),
            "mode": "grad", "signal": self.cfg.signal, "n_visible": n_vis,
            "L_q_per_view": [round(v, 5) for v in per_view],
            "mean_rate": float(rates.mean()), "max_rate": float(rates.max()),
            "frac_harmful": float((s > 0).float().mean()),
            "median": bool(self.cfg.median), "persist": bool(self.cfg.persist),
            # coverage diagnostics since the previous harvest: legacy would have logged 49 fallbacks
            # per densify cycle; HOG-P should log rederives instead and (near-)zero fallbacks
            "rederives": int(self.rederive_count), "fallbacks": int(self.fallback_count),
        }
        self.rederive_count = 0
        self.fallback_count = 0
        # diagnostic: does held-out harm align with the known-winning direction (front-floaters)?
        sc = getattr(gaussians, "_swd_scores", None)
        if sc is not None and sc["front_ratio"].shape[0] == s.shape[0] and n_vis > 100:
            summary["corr_front"] = self._corr(s_hat[vis], sc["front_ratio"][vis])
            summary["corr_witness"] = self._corr(s_hat[vis], sc["witness"][vis])
            if self.cfg.persist:
                summary["corr_front_h"] = self._corr(gaussians._hog_harm[vis], sc["front_ratio"][vis])
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary) + "\n")
        except Exception:
            pass
        return summary

    def upcoming_query_names(self):
        """Image names of the NEXT fold to be harvested (for the training-sampler holdout window)."""
        fold = self.meta_count % len(self.query_folds)
        return {getattr(c, "image_name", "") for c in self.query_folds[fold]}

    @staticmethod
    def _corr(a, b):
        a = a.float() - a.float().mean()
        b = b.float() - b.float().mean()
        d = (a.norm() * b.norm()).clamp(min=1e-12)
        return round(float((a * b).sum() / d), 4)

    def _witness_term(self, gaussians, N):
        """Static witness-state exponent term. witness_ok=False only when the term is wanted
        (beta != 0) but the SWD scores are missing or stale-sized (the 1 iteration right after a
        densify, before train.py's size-mismatch refresh runs)."""
        if self.cfg.wgsd_beta == 0.0:
            return None, True
        sc = getattr(gaussians, "_swd_scores", None)
        if sc is not None and sc["front_ratio"].shape[0] == N:
            return self.cfg.wgsd_beta * (sc["front_ratio"] - sc["witness"]).clamp(-1, 1), True
        return None, False

    def _rates_from_exponent(self, exponent):
        rates = self.cfg.rho * torch.exp(exponent)
        rates = (rates * (self.cfg.rho / rates.mean().clamp(min=1e-9))).clamp(0.0, self.cfg.max_rate)
        # second renorm pass: clamping pulls the mean below rho; re-pin (allocation-only invariant)
        rates = (rates * (self.cfg.rho / rates.mean().clamp(min=1e-9))).clamp(0.0, self.cfg.max_rate)
        return rates

    @staticmethod
    def _weighted_median(vals, wts):
        """Per-column weighted median over rows ([V,N] -> [N]). Zero-weight rows cannot be selected
        (cumsum only crosses the half-total threshold on a positive-weight row, and the first
        crossing is taken); columns with total weight 0 return 0 (neutral)."""
        total = wts.sum(0)
        order = torch.argsort(vals, dim=0)
        v_sorted = torch.gather(vals, 0, order)
        w_sorted = torch.gather(wts, 0, order)
        cum = torch.cumsum(w_sorted, dim=0)
        sel = (cum >= 0.5 * total.unsqueeze(0)).float().argmax(dim=0)
        med = torch.gather(v_sorted, 0, sel.unsqueeze(0)).squeeze(0)
        return torch.where(total > 0, med, torch.zeros_like(med)), total

    def _median_harm(self, s_views, wq):
        """Robust cross-view aggregation: standardize each view's harm over ITS visible set, then
        take the UNWEIGHTED median across visible views per Gaussian. Unweighted on purpose: with
        the wq weights (0.75, 1.5, 0.75) the mid view owns exactly half the mass, so the lower
        weighted median would select a downward-corrupted mid view at the tie - emphasis weighting
        is fundamentally at odds with single-view robustness. Equal weights give breakdown point
        1/3 in both directions: one corrupted query view (measured L_q explosions up to ~400x from
        a transient floater in front of that camera) cannot flip the allocation."""
        V, N = len(s_views), s_views[0].shape[0]
        vals = torch.zeros((V, N), device=s_views[0].device)
        wts = torch.zeros((V, N), device=s_views[0].device)
        for j, s_q in enumerate(s_views):
            vis_q = s_q != 0
            if int(vis_q.sum()) > 100:
                mu = s_q[vis_q].mean()
                sd = s_q[vis_q].std().clamp(min=1e-12)
                z = torch.zeros_like(s_q)
                z[vis_q] = ((s_q[vis_q] - mu) / sd).clamp(-3.0, 3.0)
                vals[j] = z
                wts[j] = vis_q.float()
            # else: view contributes nothing (too few visible for a reliable standardization)
        med, total = self._weighted_median(vals, wts)
        return med, total > 0

    @torch.no_grad()
    def applied_rates(self, gaussians, iteration):
        """Rates applied to REAL training renders. If densification changed N since the last
        harvest: with persist, RE-DERIVE rates from the inherited harm field (allocation coverage
        stays 100%); otherwise fall back to uniform rho (legacy v3/v4 behavior: the end-of-iteration
        densify at t = 0 mod 100 invalidated the harvest from that same t, so ~49% of iterations
        ran uniform)."""
        N = gaussians.get_xyz.shape[0]
        cache_ok = self._rates_cache is not None and self._rates_cache.shape[0] == N
        if cache_ok and not self._rates_no_witness:
            return self._rates_cache
        if self.cfg.persist:
            h = getattr(gaussians, "_hog_harm", None)
            if h is not None and h.shape[0] == N:
                term, witness_ok = self._witness_term(gaussians, N)
                if cache_ok and not witness_ok:
                    return self._rates_cache        # witness still stale; nothing new to fold in
                exponent = self.cfg.gamma * h
                if term is not None:
                    exponent = exponent + term
                self._rates_cache = self._rates_from_exponent(exponent)
                self._rates_no_witness = not witness_ok
                self.rederive_count += 1
                return self._rates_cache
        if cache_ok:
            return self._rates_cache
        self.fallback_count += 1
        return float(self.cfg.rho)

    # =====================================================================================
    # "es" ablation mode below (v2 machinery, kept verbatim for controlled comparison)
    # =====================================================================================
    @torch.no_grad()
    def _neff(self, gaussians, cams):
        xyz = gaussians.get_xyz.detach()
        N = xyz.shape[0]
        V = len(cams)
        device = xyz.device
        W = torch.zeros(N, V, device=device)
        D = torch.zeros(N, V, 3, device=device)
        for j, cam in enumerate(cams):
            proj = _project_points_to_camera(xyz, cam)
            w = proj["valid"].float()
            conf = getattr(cam, "depth_confidence", None)
            if conf is not None:
                c = conf if torch.is_tensor(conf) else torch.as_tensor(conf)
                c = c.to(device).float().squeeze()
                if c.dim() == 2:
                    h, wd = c.shape
                    col = proj["u"].round().long().clamp(0, wd - 1)
                    row = proj["v"].round().long().clamp(0, h - 1)
                    w = w * (c[row, col] >= self.cfg.conf_threshold).float()
            d = xyz - cam.camera_center.to(device)
            D[:, j, :] = d / d.norm(dim=1, keepdim=True).clamp(min=1e-9)
            W[:, j] = w
        neff = torch.zeros(N, device=device)
        for s in range(0, N, 65536):
            e = min(N, s + 65536)
            cos = torch.einsum('nvc,nuc->nvu', D[s:e], D[s:e])
            K = torch.exp((cos - 1.0) / max(self.cfg.neff_h, 1e-3))
            denom = torch.einsum('nv,nu,nvu->n', W[s:e], W[s:e], K)
            neff[s:e] = (W[s:e].sum(1) ** 2) / denom.clamp(min=1e-6)
        return neff.clamp(min=0.0, max=float(V))

    @torch.no_grad()
    def features(self, gaussians, cams):
        sc = compute_surface_witness_scores(
            gaussians, cams, tau=self.cfg.tau, conf_threshold=self.cfg.conf_threshold,
            min_seen=2, front_ratio_threshold=0.5)
        conf_norm = (sc["conf_seen"] / sc["conf_seen"].max().clamp(min=1.0)).clamp(0, 1)
        neff = self._neff(gaussians, cams)
        return torch.stack([
            sc["witness"].clamp(0, 1), sc["front_ratio"].clamp(0, 1), sc["back_ratio"].clamp(0, 1),
            conf_norm, sc["rel_mean"].clamp(0, 2) * 0.5,
            gaussians.get_opacity.detach().squeeze(1).clamp(0, 1), neff / max(len(cams), 1),
        ], dim=1)

    @torch.no_grad()
    def rates_from_policy(self, feats, phi):
        a = torch.sigmoid(feats @ phi[:-1] + phi[-1])
        r = self.cfg.rho * a / a.mean().clamp(min=1e-6)
        return r.clamp(min=0.0, max=self.cfg.max_rate)

    @torch.no_grad()
    def snapshot(self, gaussians):
        snap = {"params": {}, "opt": []}
        for group in gaussians.optimizer.param_groups:
            p = group["params"][0]
            snap["params"][group["name"]] = p.data.clone()
            st = gaussians.optimizer.state.get(p, None)
            if st is not None and "exp_avg" in st:
                snap["opt"].append((group["name"], st["exp_avg"].clone(), st["exp_avg_sq"].clone(),
                                    st.get("step", None)))
        return snap

    @torch.no_grad()
    def restore(self, gaussians, snap):
        for group in gaussians.optimizer.param_groups:
            p = group["params"][0]
            if group["name"] in snap["params"]:
                p.data.copy_(snap["params"][group["name"]])
            if p.grad is not None:
                p.grad = None
        for name, ea, eas, step in snap["opt"]:
            for group in gaussians.optimizer.param_groups:
                if group["name"] == name:
                    st = gaussians.optimizer.state.get(group["params"][0], None)
                    if st is not None and "exp_avg" in st:
                        st["exp_avg"].copy_(ea)
                        st["exp_avg_sq"].copy_(eas)
                        if step is not None and "step" in st:
                            if torch.is_tensor(st["step"]):
                                st["step"].copy_(step)
                            else:
                                st["step"] = step

    def _probe(self, gaussians, support, query, wq, rates, render_fn, depth_weight):
        K = self.cfg.inner_steps
        for k in range(K):
            cam = support[k % len(support)]
            pkg = render_fn(cam, rates)
            image = pkg["render"]
            gt = cam.original_image.cuda()
            loss = 0.8 * l1_loss(image, gt) + 0.2 * (1.0 - ssim(image, gt))
            if depth_weight > 0 and getattr(cam, "depth_reliable", False) and cam.invdepthmap is not None:
                loss = loss + depth_weight * torch.abs(
                    (pkg["depth"] - cam.invdepthmap.cuda()) * cam.depth_mask.cuda()).mean()
            loss.backward()
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            per_view = []
            for cam, w in zip(query, wq):
                img = render_fn(cam, None)["render"].clamp(0, 1)
                per_view.append(w * l1_loss(img, cam.original_image.cuda()).item())
            per_view = torch.tensor(per_view)
            return float(per_view.mean() + self.cfg.worst_lambda * per_view.max())

    def meta_step(self, gaussians, render_fn, iteration, depth_weight):
        fold = self.meta_count % len(self.query_folds)
        support = self.support_folds[fold]
        query = self.query_folds[fold]
        wq = self.query_weights[fold]
        feats_S = self.features(gaussians, support)
        eps = torch.randn_like(self.phi)
        snap = self.snapshot(gaussians)
        gaussians.optimizer.zero_grad(set_to_none=True)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        probe_seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        losses = []
        for sgn in (+1.0, -1.0):
            torch.manual_seed(probe_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(probe_seed)
            phi_p = self.phi + sgn * self.cfg.sigma * eps
            rates = self.rates_from_policy(feats_S, phi_p)
            losses.append(self._probe(gaussians, support, query, wq, rates, render_fn, depth_weight))
            self.restore(gaussians, snap)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        l_plus, l_minus = losses
        g = math.copysign(1.0, l_plus - l_minus)
        lr_t = self.cfg.lr * max(0.1, 1.0 - float(iteration) / max(self.cfg.total_iters, 1))
        self.m = self.cfg.momentum * self.m + (1.0 - self.cfg.momentum) * (g * eps)
        self.phi = self.phi - lr_t * self.m
        self.meta_count += 1
        self._rates_cache = None
        summary = {
            "iteration": int(iteration), "meta_step": int(self.meta_count), "fold": int(fold),
            "mode": "es", "L_plus": float(l_plus), "L_minus": float(l_minus),
            "dL": float(l_plus - l_minus), "lr_t": round(lr_t, 5),
            "phi": [round(float(x), 5) for x in self.phi.tolist()],
            "mean_rate": float(self.rates_from_policy(feats_S, self.phi).mean()),
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary) + "\n")
        except Exception:
            pass
        return summary

    @torch.no_grad()
    def applied_rates_es(self, gaussians, iteration):
        N = gaussians.get_xyz.shape[0]
        if (self._rates_cache is None or self._rates_cache.shape[0] != N
                or iteration - self._rates_iter >= self.cfg.refresh):
            feats = self.features(gaussians, self.all_cams)
            self._rates_cache = self.rates_from_policy(feats, self.phi)
            self._rates_iter = iteration
        return self._rates_cache
