import math

import torch

from utils.sh_utils import eval_sh


def camera_direction(xyz, camera_center, eps=1e-8):
    direction = xyz - camera_center.reshape(1, 3)
    return direction / torch.clamp(direction.norm(dim=1, keepdim=True), min=eps)


@torch.no_grad()
def leave_one_out_angular_support(
    xyz,
    target_camera,
    source_cameras,
    temperature,
    exclude_matching_name=True,
):
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    target_direction = camera_direction(xyz, target_camera.camera_center)
    max_cosine = torch.full(
        (xyz.shape[0],),
        -1.0,
        dtype=xyz.dtype,
        device=xyz.device,
    )
    source_count = 0

    for source_camera in source_cameras:
        if (
            exclude_matching_name
            and source_camera.image_name == target_camera.image_name
        ):
            continue
        source_direction = camera_direction(xyz, source_camera.camera_center)
        cosine = torch.sum(target_direction * source_direction, dim=1)
        max_cosine = torch.maximum(max_cosine, cosine)
        source_count += 1

    if source_count == 0:
        raise ValueError("angular support requires at least one source camera")

    angular_distance = torch.clamp(1.0 - max_cosine, min=0.0)
    support = torch.exp(-angular_distance / float(temperature))
    return support, max_cosine


def sh_degree_components(features, directions, active_degree):
    if active_degree < 0 or active_degree > 3:
        raise ValueError("active_degree must be in [0, 3]")

    cumulative = []
    for degree in range(active_degree + 1):
        cumulative.append(eval_sh(degree, features, directions))

    components = [cumulative[0]]
    for degree in range(1, active_degree + 1):
        components.append(cumulative[degree] - cumulative[degree - 1])
    return components


def compose_support_gated_color(
    components,
    support,
    floor,
    order_power,
):
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must be in [0, 1]")
    if order_power < 0.0:
        raise ValueError("order_power must be non-negative")

    color = components[0]
    for degree, component in enumerate(components[1:], start=1):
        exponent = order_power * degree
        degree_gate = floor + (1.0 - floor) * support.pow(exponent)
        color = color + degree_gate[:, None] * component
    return torch.clamp_min(color + 0.5, 0.0)


def compose_fixed_degree_color(components, degree):
    if degree < 0 or degree >= len(components):
        raise ValueError("requested degree is unavailable")
    color = torch.stack(components[: degree + 1], dim=0).sum(dim=0)
    return torch.clamp_min(color + 0.5, 0.0)


def support_quantiles(support):
    quantile_levels = torch.tensor(
        [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        dtype=support.dtype,
        device=support.device,
    )
    values = torch.quantile(support, quantile_levels)
    return {
        "q{:02d}".format(int(round(float(level) * 100.0))): float(value.item())
        for level, value in zip(quantile_levels, values)
    }


def angular_gap_degrees(max_cosine):
    cosine = torch.clamp(max_cosine, min=-1.0, max=1.0)
    return torch.rad2deg(torch.acos(cosine))

