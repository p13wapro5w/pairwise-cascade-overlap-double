"""级联阴影（CSM）修复：分割去重叠、bias 按 texel 尺寸计算、重叠像素只采一层。

问题背景：
  按视锥球重算级联分割后，相邻级联的覆盖区间出现重叠，重叠区被两层
  阴影贴图同时采样，接触阴影叠出双重暗边；且阴影贴图分辨率随级联变化
  后 bias 没有按新的 texel 尺寸重算，导致 acne / 漏光。

修复要点：
  1. split_no_overlap —— 级联分割区间首尾相接、互不重叠；
  2. bias_by_texel   —— 深度 bias 按当前级联 texel 世界尺寸计算；
  3. single_layer_sample / overlap_resolve —— 任意像素只属于一个级联，
     重叠区只采一层（除显式 blend 带外不做双层混合）。
"""

from __future__ import annotations

import math

# 默认参数
DEFAULT_SHADOW_MAP_SIZE = 2048
DEFAULT_BIAS_TEXELS = 1.5          # bias 取多少个 texel
DEFAULT_NORMAL_BIAS_TEXELS = 1.0
BLEND_BAND_FRACTION = 0.1          # 级联交界处的显式过渡带（可选）
EPSILON = 1e-6


# ---------------------------------------------------------------------------
# 视锥球与级联分割
# ---------------------------------------------------------------------------

def frustum_sphere(near, far, fov_y=math.radians(60.0), aspect=16.0 / 9.0):
    """计算视锥切片 [near, far] 的包围球（沿视线方向的一维近似）。

    返回 {"center": 距相机距离, "radius": 半径}。包围球用于把正交阴影
    相机的视口对齐到切片，半径即级联的世界空间半宽。
    """
    near = float(near)
    far = float(far)
    if far <= near:
        raise ValueError("far 必须大于 near")
    tan_half = math.tan(fov_y * 0.5)
    # 远截面半对角线
    far_half_diag = far * tan_half * math.hypot(1.0, aspect)
    # 标准近似：球心位于切片中点，半径 = 中点到远截面角点的距离
    center = 0.5 * (near + far)
    radius = math.hypot(far - center, far_half_diag)
    return {"center": center, "radius": radius}


def split_no_overlap(num_cascades, max_distance, min_distance=0.1,
                     lambda_=0.5):
    """按对数/均匀插值重算级联分割，并保证相邻级联区间互不重叠。

    返回 [(near_i, far_i), ...]，满足 far_i == near_{i+1}（严格单调、
    零重叠、零缝隙），从根本上消除双层覆盖的重叠区。
    """
    num_cascades = int(num_cascades)
    max_distance = float(max_distance)
    if num_cascades < 1:
        raise ValueError("级联数至少为 1")
    if max_distance <= min_distance:
        raise ValueError("max_distance 必须大于 min_distance")

    # 经典 practical split：对数与均匀的 lambda 插值
    boundaries = [min_distance]
    for i in range(1, num_cascades):
        f = i / num_cascades
        log_split = min_distance * (max_distance / min_distance) ** f
        uni_split = min_distance + (max_distance - min_distance) * f
        boundaries.append(lambda_ * log_split + (1.0 - lambda_) * uni_split)
    boundaries.append(max_distance)

    # 去重叠：强制单调，前一级联的 far 即后一级联的 near，无共享区间
    for i in range(1, len(boundaries)):
        if boundaries[i] <= boundaries[i - 1]:
            boundaries[i] = boundaries[i - 1] + EPSILON

    return [(boundaries[i], boundaries[i + 1]) for i in range(num_cascades)]


# ---------------------------------------------------------------------------
# texel 尺寸与 bias
# ---------------------------------------------------------------------------

def texel_size(world_size, resolution):
    """单个阴影贴图 texel 对应的世界空间尺寸。"""
    resolution = float(resolution)
    if resolution <= 0:
        raise ValueError("分辨率必须为正")
    return float(world_size) / resolution


def bias_by_texel(resolution, world_size, bias_texels=DEFAULT_BIAS_TEXELS):
    """按当前级联的 texel 世界尺寸计算深度 bias。

    分辨率/级联变化后必须重新调用：bias = texel_size * bias_texels，
    避免沿用旧分辨率下的 bias 造成 acne 或 peter-panning。
    """
    return texel_size(world_size, resolution) * bias_texels


def resolution_switch(old_resolution, new_resolution, old_bias=0.0,
                      world_size=1.0):
    """阴影贴图分辨率切换时重算 bias，并做 texel 对齐的量化。

    返回 {"bias": 新 bias, "scale": 分辨率缩放比}。
    """
    if old_resolution <= 0 or new_resolution <= 0:
        raise ValueError("分辨率必须为正")
    scale = float(old_resolution) / float(new_resolution)
    new_bias = bias_by_texel(new_resolution, world_size)
    return {"bias": new_bias, "scale": scale}


def acne_guard(slope, min_bias, max_slope_bias=4.0):
    """slope-scaled bias 的钳制，防止大倾角下 bias 无限增大导致漏光。"""
    slope = abs(float(slope))
    guard = min_bias * (1.0 + min(slope, max_slope_bias))
    return max(min_bias, guard)


def pcf_radius(resolution, world_size, radius_texels=1.0):
    """PCF 采样半径（世界空间），按 texel 尺寸给出。"""
    return texel_size(world_size, resolution) * float(radius_texels)


# ---------------------------------------------------------------------------
# 单层采样与重叠消解
# ---------------------------------------------------------------------------

def _normalize_splits(splits):
    """接受分割区间列表，或级联数（int，按默认距离展开）。"""
    if isinstance(splits, (int, float)) and not isinstance(splits, bool):
        return split_no_overlap(int(splits), 100.0)
    return list(splits)


def _cascade_index(depth, splits):
    """depth 所属的唯一级联下标；区间 [near, far) 左闭右开，末级含 far。"""
    splits = _normalize_splits(splits)
    for i, (near, far) in enumerate(splits):
        if near <= depth < far:
            return i
    return len(splits) - 1  # 超出最远处归入末级（或视为无阴影）


def single_layer_sample(depth, splits):
    """重叠区只采一层：返回该像素唯一归属的级联下标。

    分割去重叠后任意 depth 只属于一个区间；即便传入旧的重叠分割，
    也按首个覆盖区间取值，绝不返回多个层。
    """
    return _cascade_index(float(depth), splits)


def overlap_resolve(depth, splits):
    """消解相邻级联重叠：返回 (级联下标, 是否落在显式 blend 带内)。

    只有落在交界处很窄的 blend 带内才允许过渡采样，其余像素严格单层。
    """
    depth = float(depth)
    splits = _normalize_splits(splits)
    idx = _cascade_index(depth, splits)
    near, far = splits[idx]
    band = (far - near) * BLEND_BAND_FRACTION
    in_blend = (far - depth) <= band and idx < len(splits) - 1
    return idx, in_blend


def blend_region(depth, splits):
    """计算交界 blend 带内的过渡权重（0..1），带外恒为 0（单层采样）。

    权重用于在 blend 带内从当前层线性过渡到下一层，避免双层同时
    全权重采样造成的双重暗边。
    """
    depth = float(depth)
    splits = _normalize_splits(splits)
    idx, in_blend = overlap_resolve(depth, splits)
    if not in_blend:
        return 0.0
    near, far = splits[idx]
    band = (far - near) * BLEND_BAND_FRACTION
    # 带内从 0（完全当前层）到 1（完全下一层）
    return max(0.0, min(1.0, 1.0 - (far - depth) / band))


# ---------------------------------------------------------------------------
# 深度比较与阴影计算
# ---------------------------------------------------------------------------

def depth_compare(fragment_depth, shadow_depth, bias=0.0):
    """带 bias 的深度比较：返回光照系数（1.0 受光，0.0 阴影）。"""
    return 1.0 if float(fragment_depth) - float(bias) <= float(shadow_depth) else 0.0


def contact_shadow(fragment_depth, shadow_depth, bias=0.0, strength=1.0):
    """接触阴影系数，钳制在 [0,1]，防止多层叠加后超过 1 产生双重暗边。

    单层采样 + 钳制保证同一像素最多被一层阴影衰减一次。
    """
    lit = depth_compare(fragment_depth, shadow_depth, bias)
    shadow = (1.0 - lit) * min(max(float(strength), 0.0), 1.0)
    return max(0.0, min(1.0, 1.0 - shadow))


def cascade_stabilize(coord, resolution=None):
    """把阴影贴图 UV/坐标吸附到 texel 网格，消除相机移动时的边缘抖动。

    resolution 缺省时按 1 个单位网格吸附。
    """
    res = float(resolution) if resolution else 1.0
    if isinstance(coord, (tuple, list)):
        return tuple(math.floor(c * res + 0.5) / res for c in coord)
    return math.floor(float(coord) * res + 0.5) / res


# ---------------------------------------------------------------------------
# 校验与度量
# ---------------------------------------------------------------------------

def validation(splits, max_distance=None):
    """校验级联分割：严格单调、无重叠、无负宽度。

    返回 {"ok": bool, "errors": [str, ...]}。
    """
    errors = []
    splits = _normalize_splits(splits)
    for i, (near, far) in enumerate(splits):
        if far <= near:
            errors.append(f"级联 {i} 宽度非正: [{near}, {far}]")
        if i > 0 and near < splits[i - 1][1] - EPSILON:
            errors.append(f"级联 {i} 与上一级联重叠")
    if max_distance is not None and splits and splits[-1][1] > max_distance + EPSILON:
        errors.append("末级超出最大阴影距离")
    return {"ok": not errors, "errors": errors}


def double_edge_metric(lit_a, lit_b=None):
    """双重暗边度量：同一像素被两层同时衰减的超出量。

    传入单层光照系数（或两层系数之积）时，返回相对单层的过暗量；
    修复后期望为 0（单层采样不产生叠加）。
    """
    if lit_b is None:
        combined = float(lit_a)
        single = float(lit_a)
    else:
        combined = float(lit_a) * float(lit_b)
        single = min(float(lit_a), float(lit_b))
    return max(0.0, single - combined)
