#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                    if value is True:
                        group.add_argument("--no_" + key, dest=key, action="store_false")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                    if value is True:
                        group.add_argument("--no_" + key, dest=key, action="store_false")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        self.llffhold = 8
        self.full_eval_path = ""
        self.full_eval_images = "images"
        self.full_eval_sparse = "sparse/0"
        self.eval_hold = 8
        self.eval_overlap_shift = "backward"
        self.eval_boundary_forward_fallback = True
        self.eval_strict_backward_shift = False
        self.split_report_enable = True
        self.split_only = False
        self.sparse_train_images = ""
        self.sparse_train_indices = ""
        self.sparse_train_count = 0
        self.full_test_source_path = ""
        self.full_test_images = ""
        self.dpcr_eval_source_path = ""
        self.dpcr_eval_images = ""
        self.dpcr_eval_split_mode = "llffhold"
        self.dpcr_eval_llffhold = 8
        self.dpcr_train_view_list = ""
        self.dpcr_eval_test_view_list = ""
        self.dpcr_write_split_manifest = True
        self.dpcr_eval_require_disjoint = True
        self.dpcr_eval_frame_mode = "strict"
        self.dpcr_eval_alignment_min_common = 4
        self.dpcr_eval_frame_check_tol = 1e-3
        self.split_train_views = "off"
        self.split_hold = 8
        self.split_output_root = ""
        self.split_name = ""
        self.split_copy_mode = "copy"
        self.split_force = False
        self.split_validate_only = False
        self.split_train_sample_mode = "paper_even"
        self.split_strict_no_overlap = True
        self.split_init_policy = "sparsegs_triangulate"
        self.split_colmap_exe = "colmap"
        self.split_colmap_matcher = "exhaustive"
        self.split_require_all_train_registered = True
        self.split_min_train_points = 100
        self.split_min_triangulated_points = 100
        self.split_strict_sparsegs = True
        self.use_existing_split = False
        self.external_test_source_path = ""
        self.auto_split_report_path = ""
        self.auto_split_validation_report_path = ""
        self.source_path_original = ""
        self.pgdr_enable = False
        self.pgdr_depth_cache = ""
        self.pgdr_strict_precheck = True
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        if getattr(g, "model_path", ""):
            g.model_path = os.path.abspath(g.model_path)
        if getattr(g, "full_eval_path", ""):
            g.full_eval_path = os.path.abspath(g.full_eval_path)
        if g.full_test_source_path:
            g.full_test_source_path = os.path.abspath(g.full_test_source_path)
        if getattr(g, "dpcr_eval_source_path", ""):
            g.dpcr_eval_source_path = os.path.abspath(g.dpcr_eval_source_path)
        if getattr(g, "dpcr_train_view_list", ""):
            g.dpcr_train_view_list = os.path.abspath(g.dpcr_train_view_list)
        if getattr(g, "dpcr_eval_test_view_list", ""):
            g.dpcr_eval_test_view_list = os.path.abspath(g.dpcr_eval_test_view_list)
        if getattr(g, "split_output_root", ""):
            g.split_output_root = os.path.abspath(g.split_output_root)
        if getattr(g, "external_test_source_path", ""):
            g.external_test_source_path = os.path.abspath(g.external_test_source_path)
        if getattr(g, "auto_split_report_path", ""):
            g.auto_split_report_path = os.path.abspath(g.auto_split_report_path)
        if getattr(g, "auto_split_validation_report_path", ""):
            g.auto_split_validation_report_path = os.path.abspath(g.auto_split_validation_report_path)
        if getattr(g, "pgdr_depth_cache", ""):
            g.pgdr_depth_cache = os.path.abspath(g.pgdr_depth_cache)
        if not getattr(g, "source_path_original", ""):
            g.source_path_original = g.source_path
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        self.dropgaussian_enable = False
        self.dropgaussian_start = 0
        self.dropgaussian_end = 10_000
        self.dropgaussian_max_rate = 0.2
        self.dropgaussian_schedule = "linear"
        self.pgdr_update_interval = 100
        self.pgdr_sample_views = 2
        self.pgdr_pixels_per_view = 256
        self.pgdr_max_candidates_per_pixel = 24
        self.pgdr_support_scale = 1.5
        self.pgdr_min_screen_radius = 1.5
        self.pgdr_max_screen_radius = 64.0
        self.pgdr_sigma_scale = 0.5
        self.pgdr_alpha_scale = 1.0
        self.pgdr_alpha_cap = 0.95
        self.pgdr_min_pixel_contrib = 0.00001
        self.pgdr_responsibility_threshold = 0.04
        self.pgdr_gate_start = 700
        self.pgdr_certify_min_p = 0.02
        self.pgdr_certify_min_neff = 1.1
        self.pgdr_certify_max_residual = 0.12
        self.pgdr_pull_residual = 0.08
        self.pgdr_death_front_residual = 0.16
        self.pgdr_lambda_pull = 0.02
        self.pgdr_lambda_death = 0.01
        # ---- Depth-Anchored Densification (DAD): seed split children at the multi-view-consistent
        # depth-prior surface (no increase of INITIAL points; densification-time placement only). ----
        self.dad_enable = False
        self.dad_alpha = 0.8        # pull strength of children toward the depth target (0..1)
        self.dad_agree = 0.15       # max per-Gaussian relative std of multi-view target depth to trust it
        self.dad_min_views = 2      # min views observing a Gaussian for a reliable depth target
        # load the split's aligned inverse-depth cache (enables depth-reg loss + DAD)
        self.use_depth_cache = False
        self.depth_cache = ""       # default: <source_path>/pgdr_depth_cache_aligned
        # CDW: complementary (texture-aware) soft reweighting of the depth-reg loss
        self.cdw_enable = False
        self.cdw_gamma = 2.0        # sharpness: higher -> more emphasis on textureless regions
        # CFDC: cross-fitted depth confidence, a soft multi-view-consistency weight for depth loss
        self.cfdc_enable = False
        self.cfdc_cache = ""        # directory with cfdc_confidence.pt, or the tensor path
        self.cfdc_power = 1.0       # confidence exponent before renormalization
        self.cfdc_floor = 0.05      # nonzero floor so low-confidence pixels still receive weak depth
        # SWD: Surface-Witnessed Depth; soft opacity penalty / birth gate for high-confidence front floaters
        self.swd_enable = False
        self.swd_start = 1000
        self.swd_update_interval = 500
        self.swd_tau = 0.05
        self.swd_conf_threshold = 0.5
        self.swd_min_seen = 2
        self.swd_front_ratio = 0.5
        self.swd_lambda_opacity = 0.001
        self.swd_birth_gate = False
        # WGSD: witness-guided structural dropout (per-Gaussian rates from SWD cross-view witness;
        # requires --dropgaussian_enable AND --swd_enable to have scores). Mean stays at base rate.
        self.wgsd_enable = False
        self.wgsd_beta = 1.0        # modulation sharpness: rate *= exp(beta*(front_ratio - witness))
        self.wgsd_max_rate = 0.6    # per-Gaussian rate cap
        # ECU: learned evidence-conditioned uncertainty head (UGOD-style differentiable soft dropout,
        # but conditioned on cross-view witness evidence). Replaces uniform/WGSD dropout when enabled.
        self.ecu_enable = False
        self.ecu_inputs = "evidence"   # "evidence" (ours) | "shape" (UGOD-style baseline) | "both"
        self.ecu_hidden = 64
        self.ecu_lr = 0.001
        self.ecu_tau = 0.5             # concrete-distribution temperature
        self.ecu_u_target = 0.1        # rate anchor: mean suppression ~ DropGaussian level
        self.ecu_reg = 1.0             # weight of the (mean(u)-u_target)^2 anchor
        self.ecu_start = 1500          # net starts influencing after SWD scores exist
        self.ecu_freeze_iter = 9000    # stop training the net (keep applying) near the end
        # HOG-GS: held-out-guided regularization policy (tiny linear policy, antithetic-ES outer loop
        # on rotating support/query folds of the TRAIN views; allocation-only suppression).
        self.hog_enable = False
        self.hog_mode = "grad"         # "grad" = v3 first-order heldout-harm | "es" = v2 ES ablation
        self.hog_fold_mode = "blocked" # "blocked" = contiguous camera arcs (anti-interpolation) | "interleaved"
        self.hog_gamma = 1.0           # harm->rate sharpness: r = rho*exp(gamma*s_hat)/mean
        self.hog_signal = "depth"      # "depth" = harm vs EXTERNAL depth prior at held-out views (v4)
                                       # "rgb" = v3 GT-photometric harm (measured: interpolative bias)
        self.hog_wgsd_beta = 0.0       # hybrid: + beta*(front_ratio - witness) in the rate exponent
        self.hog_holdout_window = 30   # iters before each harvest with upcoming queries excluded from training
        self.hog_start = 1500          # begin after geometry roughly formed (and evidence is loaded)
        self.hog_meta_interval = 100   # harvest (grad) / outer ES step (es) every M iterations
        self.hog_inner_steps = 8       # K real update steps per probe
        self.hog_rho = 0.10            # target MEAN drop rate (allocation-only; matches Drop 0.1 level)
        self.hog_max_rate = 0.6
        self.hog_sigma = 0.3           # ES perturbation scale
        self.hog_lr = 0.1              # sign-ES step size
        self.hog_momentum = 0.9
        self.hog_folds = 4             # 12 views -> 4 folds of 3 query / 9 support
        self.hog_refresh = 100         # applied-rate refresh interval
        self.hog_worst_lambda = 0.5    # weight of worst-query-view term in the held-out risk
        self.hog_neff_h = 0.1          # angular decorrelation bandwidth for n_eff
        self.hog_conf_threshold = 0.5
        self.hog_tau = 0.05
        # v5 HOG-P: persistent robust harm field (all OFF by default -> hog_hyb behavior unchanged)
        self.hog_persist = False       # EMA harm field inherited through densification; rates re-derived
                                       # when N changes instead of uniform fallback (coverage 51% -> 100%)
        self.hog_eta = 0.3             # EMA coefficient of the persistent harm field
        self.hog_median = False        # per-view standardize + unweighted MEDIAN across query views
                                       # (breakdown point 1/3: one corrupted query view cannot flip allocation)
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
