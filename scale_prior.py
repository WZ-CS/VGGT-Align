import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict


# ============================================================================
# Scale prior configuration
# ============================================================================
@dataclass
class ScalePriorConfig:
    """Scale prior parameters injected from argparse."""
    ground_prior: bool = True
    road_width_prior: bool = False

    # Ground height prior
    camera_height: float = 1.65
    ground_row_ratio: float = 0.3
    ransac_iters: int = 500
    ransac_thresh_ratio: float = 0.05
    conf_percentile: float = 40.0
    min_ground_points: int = 200

    # Road width prior
    lane_width: float = 3.75
    num_lanes: float = 2.0
    road_width_row_ratio: float = 0.25
    road_width_ransac_iters: int = 200
    width_conf_percentile: float = 50.0

    # Fusion
    blend_alpha: float = 1.0
    adaptive_blend: bool = False
    adaptive_min_alpha: float = 0.3
    adaptive_max_alpha: float = 1.0
    height_weight: float = 0.7
    width_weight: float = 0.3

    # Diagnostics
    diagnostics: bool = True


# ============================================================================
# RANSAC plane fitting
# ============================================================================
def ransac_fit_plane(points: np.ndarray,
                     n_iter: int = 500,
                     thresh: float = 0.05
                     ) -> Optional[Tuple[np.ndarray, float, int]]:
    """
    Fit a plane ax+by+cz+d=0 with RANSAC.
    
    Returns:
        (normal, d, n_inliers) or None.
    """
    n = points.shape[0]
    if n < 3:
        return None

    best_n_inliers = 0
    best_result = None

    for _ in range(n_iter):
        idx = np.random.choice(n, 3, replace=False)
        p1, p2, p3 = points[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal /= norm_len
        d = -np.dot(normal, p1)

        dists = np.abs(points @ normal + d)
        n_inliers = np.sum(dists < thresh)

        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            # Refit with inliers using SVD.
            inlier_pts = points[dists < thresh]
            centroid = inlier_pts.mean(axis=0)
            cov = (inlier_pts - centroid).T @ (inlier_pts - centroid)
            _, _, Vt = np.linalg.svd(cov)
            normal = Vt[-1]
            normal /= np.linalg.norm(normal)
            d = -np.dot(normal, centroid)
            best_result = (normal, d, best_n_inliers)

    return best_result


# ============================================================================
# Ground candidate extraction
# ============================================================================
def _extract_ground_candidates(world_points: np.ndarray,
                                confs: np.ndarray,
                                cfg: ScalePriorConfig
                                ) -> Optional[np.ndarray]:
    """
    Extract ground candidate points from world_points.
    
    Args:
        world_points: (N_frames, H, W, 3) or (N_frames, N_pts, 3).
    
    Returns:
        ground_pts: (M, 3) or None.
    """
    row_ratio = cfg.ground_row_ratio
    conf_pct = cfg.conf_percentile
    n_frames = world_points.shape[0]
    ground_pts_list = []

    if world_points.ndim == 4:
        H = world_points.shape[1]
        bottom_start = int(H * (1.0 - row_ratio))

        for f in range(n_frames):
            pts = world_points[f, bottom_start:, :, :].reshape(-1, 3)

            if confs.ndim == 4:
                cf = confs[f].squeeze()[bottom_start:].reshape(-1)
            elif confs.ndim == 3:
                cf = confs[f, bottom_start:].reshape(-1)
            else:
                cf = np.ones(pts.shape[0])

            valid = np.isfinite(pts).all(axis=1) & (cf > 0)
            if valid.sum() < 10:
                continue
            pts_v, cf_v = pts[valid], cf[valid]

            if cf_v.max() > 0:
                thr = np.percentile(cf_v, conf_pct)
                good = cf_v > thr
                if good.sum() > 0:
                    ground_pts_list.append(pts_v[good])

    elif world_points.ndim == 3:
        for f in range(n_frames):
            pts = world_points[f]
            cf = confs[f].reshape(-1) if confs.ndim >= 2 else np.ones(pts.shape[0])
            valid = np.isfinite(pts).all(axis=1) & (cf > 0)
            pts_v = pts[valid]
            cf_v = cf[valid] if valid.sum() > 0 else np.array([])
            if pts_v.shape[0] < 50:
                continue
            y_vals = pts_v[:, 1]
            y_thr = np.percentile(y_vals, 100 * (1 - row_ratio))
            ground_mask = y_vals > y_thr
            if cf_v.max() > 0:
                conf_thr = np.percentile(cf_v, conf_pct)
                ground_mask = ground_mask & (cf_v > conf_thr)
            if ground_mask.sum() > 0:
                ground_pts_list.append(pts_v[ground_mask])

    if not ground_pts_list:
        return None

    ground_pts = np.concatenate(ground_pts_list, axis=0)
    if ground_pts.shape[0] < cfg.min_ground_points:
        return None

    # Downsample.
    if ground_pts.shape[0] > 10000:
        idx = np.random.choice(ground_pts.shape[0], 10000, replace=False)
        ground_pts = ground_pts[idx]

    return ground_pts


# ============================================================================
# Ground height estimation
# ============================================================================
def estimate_ground_height(world_points: np.ndarray,
                            confs: np.ndarray,
                            extrinsics: np.ndarray,
                            cfg: ScalePriorConfig
                            ) -> Optional[Tuple[float, float]]:
    """
    Estimate the camera-to-ground-plane distance for a chunk.
    
    Returns:
        (height, confidence_score) or None.
    """
    ground_pts = _extract_ground_candidates(world_points, confs, cfg)
    if ground_pts is None:
        return None

    # Rough height estimate for the RANSAC threshold.
    cam_positions = extrinsics[:, :3, 3]
    mean_cam = cam_positions.mean(axis=0)
    ground_center = ground_pts.mean(axis=0)
    rough_height = np.linalg.norm(mean_cam - ground_center)
    if rough_height < 1e-3:
        rough_height = 1.0

    ransac_thresh = cfg.ransac_thresh_ratio * rough_height

    plane = ransac_fit_plane(ground_pts, n_iter=cfg.ransac_iters, thresh=ransac_thresh)
    if plane is None:
        return None

    normal, d, n_inliers = plane

    # Orient the normal toward the camera.
    if np.dot(normal, mean_cam) + d < 0:
        normal = -normal
        d = -d

    # Per-frame distance from camera to plane.
    n_frames = extrinsics.shape[0]
    heights = []
    for f in range(n_frames):
        cam_pos = extrinsics[f, :3, 3]
        h = np.dot(normal, cam_pos) + d
        if h > 0:
            heights.append(h)

    if len(heights) < n_frames * 0.5:
        return None

    median_h = np.median(heights)
    std_h = np.std(heights)

    # Confidence combines inlier ratio and height consistency.
    inlier_ratio = n_inliers / ground_pts.shape[0]
    consistency = 1.0 - min(std_h / (median_h + 1e-6), 1.0)
    frame_ratio = len(heights) / n_frames

    confidence = inlier_ratio * 0.4 + consistency * 0.4 + frame_ratio * 0.2

    return median_h, confidence


# ============================================================================
# Road width estimation
# ============================================================================
def estimate_road_width(world_points: np.ndarray,
                        confs: np.ndarray,
                        extrinsics: np.ndarray,
                        cfg: ScalePriorConfig,
                        ground_normal: Optional[np.ndarray] = None,
                        ground_d: Optional[float] = None
                        ) -> Optional[Tuple[float, float]]:
    """
    Estimate the lateral width of the visible road surface in a chunk.
    
    Returns:
        (width, confidence_score) or None.
    """
    # Use a narrower bottom region with a higher confidence threshold.
    cfg_local = ScalePriorConfig(
        ground_row_ratio=cfg.road_width_row_ratio,
        conf_percentile=cfg.width_conf_percentile,
        min_ground_points=cfg.min_ground_points,
    )
    ground_pts = _extract_ground_candidates(world_points, confs, cfg_local)
    if ground_pts is None:
        return None

    # Estimate travel direction from the camera trajectory.
    cam_positions = extrinsics[:, :3, 3]  # (N, 3)
    if cam_positions.shape[0] < 2:
        return None

    # Use the first-to-last displacement as the primary travel direction.
    travel_vec = cam_positions[-1] - cam_positions[0]
    travel_norm = np.linalg.norm(travel_vec)
    if travel_norm < 1e-6:
        return None
    travel_dir = travel_vec / travel_norm

    # Project travel direction onto the ground plane if available.
    if ground_normal is not None:
        travel_dir = travel_dir - np.dot(travel_dir, ground_normal) * ground_normal
        tn = np.linalg.norm(travel_dir)
        if tn < 1e-6:
            return None
        travel_dir /= tn

    # Lateral direction is travel direction crossed with the up vector.
    up_vec = ground_normal if ground_normal is not None else np.array([0, 1, 0], dtype=np.float64)
    lateral_dir = np.cross(travel_dir, up_vec)
    ln = np.linalg.norm(lateral_dir)
    if ln < 1e-6:
        return None
    lateral_dir /= ln

    # Project ground points onto the lateral direction.
    # Center by the mean camera position for numerical stability.
    mean_cam = cam_positions.mean(axis=0)
    pts_centered = ground_pts - mean_cam

    lateral_proj = pts_centered @ lateral_dir  # (M,)

    # Estimate width with robust percentiles.
    w_low = np.percentile(lateral_proj, 5)
    w_high = np.percentile(lateral_proj, 95)
    estimated_width = w_high - w_low

    if estimated_width < 0.1:
        return None

    # Confidence combines point count and distribution symmetry.
    n_pts = ground_pts.shape[0]
    pts_score = min(n_pts / 2000.0, 1.0)

    # Check whether the lateral distribution is roughly balanced.
    median_proj = np.median(lateral_proj)
    left_count = np.sum(lateral_proj < median_proj)
    right_count = np.sum(lateral_proj >= median_proj)
    symmetry = min(left_count, right_count) / max(left_count, right_count)

    confidence = pts_score * 0.5 + symmetry * 0.5

    return estimated_width, confidence


# ============================================================================
# Multi-source scale fusion
# ============================================================================
class ScalePriorEstimator:
    """
    Manage per-chunk scale prior estimates and fused scale correction.
    """

    def __init__(self, cfg: ScalePriorConfig):
        self.cfg = cfg
        self.chunk_heights: Dict[int, Optional[Tuple[float, float]]] = {}
        self.chunk_widths: Dict[int, Optional[Tuple[float, float]]] = {}
        self.chunk_ground_planes: Dict[int, Optional[Tuple[np.ndarray, float]]] = {}

    def estimate_chunk(self,
                       chunk_idx: int,
                       world_points: np.ndarray,
                       confs: np.ndarray,
                       extrinsics: np.ndarray):
        """Estimate all priors for a single chunk."""

        # Ground height
        if self.cfg.ground_prior:
            result = estimate_ground_height(world_points, confs, extrinsics, self.cfg)
            if result is not None:
                self.chunk_heights[chunk_idx] = result
                # Save the ground plane for the road-width prior.
                ground_pts = _extract_ground_candidates(world_points, confs, self.cfg)
                if ground_pts is not None:
                    cam_mean = extrinsics[:, :3, 3].mean(axis=0)
                    rough_h = np.linalg.norm(cam_mean - ground_pts.mean(axis=0))
                    plane = ransac_fit_plane(ground_pts,
                                             n_iter=self.cfg.ransac_iters,
                                             thresh=self.cfg.ransac_thresh_ratio * max(rough_h, 1.0))
                    if plane is not None:
                        normal, d, _ = plane
                        if np.dot(normal, cam_mean) + d < 0:
                            normal, d = -normal, -d
                        self.chunk_ground_planes[chunk_idx] = (normal, d)
            else:
                self.chunk_heights[chunk_idx] = None
        
        # Road width
        if self.cfg.road_width_prior:
            gp = self.chunk_ground_planes.get(chunk_idx)
            gn = gp[0] if gp else None
            gd = gp[1] if gp else None
            result = estimate_road_width(world_points, confs, extrinsics, self.cfg,
                                          ground_normal=gn, ground_d=gd)
            self.chunk_widths[chunk_idx] = result

    def get_corrected_scale(self,
                            chunk_idx: int,
                            s_irls: float,
                            R: np.ndarray,
                            t_irls: np.ndarray,
                            point_map2: np.ndarray
                            ) -> Tuple[float, np.ndarray, dict]:
        """
        Correct the IRLS alignment scale for a neighboring chunk pair.
        
        Args:
            chunk_idx: Left chunk index in the pair.
            s_irls: Scale estimated by IRLS.
            R: Rotation estimated by IRLS.
            t_irls: Translation estimated by IRLS.
            point_map2: Overlap points from chunk_{k+1}.
        
        Returns:
            Corrected scale, corrected translation, and diagnostics.
        """
        info = {'s_irls': s_irls, 's_ground': None, 's_width': None,
                'h_k': None, 'h_k1': None, 'w_k': None, 'w_k1': None,
                'alpha_used': 0.0, 'source': 'irls_only'}

        if not self.cfg.ground_prior and not self.cfg.road_width_prior:
            return s_irls, t_irls, info

        # Collect scale estimates from all enabled sources.
        scale_estimates = []  # [(scale, weight, source_name), ...]

        # Ground height
        h_k_result = self.chunk_heights.get(chunk_idx)
        h_k1_result = self.chunk_heights.get(chunk_idx + 1)

        if (self.cfg.ground_prior and
            h_k_result is not None and h_k1_result is not None):
            h_k, conf_k = h_k_result
            h_k1, conf_k1 = h_k1_result
            info['h_k'] = h_k
            info['h_k1'] = h_k1

            if h_k > 0.01 and h_k1 > 0.01:
                s_ground = h_k / h_k1
                # Weight is the geometric mean of both confidence scores.
                w = np.sqrt(conf_k * conf_k1) * self.cfg.height_weight
                scale_estimates.append((s_ground, w, 'ground'))
                info['s_ground'] = s_ground

        # Road width
        w_k_result = self.chunk_widths.get(chunk_idx)
        w_k1_result = self.chunk_widths.get(chunk_idx + 1)

        if (self.cfg.road_width_prior and
            w_k_result is not None and w_k1_result is not None):
            w_k, conf_k = w_k_result
            w_k1, conf_k1 = w_k1_result
            info['w_k'] = w_k
            info['w_k1'] = w_k1

            if w_k > 0.1 and w_k1 > 0.1:
                s_width = w_k / w_k1
                w = np.sqrt(conf_k * conf_k1) * self.cfg.width_weight
                scale_estimates.append((s_width, w, 'width'))
                info['s_width'] = s_width

        if not scale_estimates:
            info['source'] = 'irls_fallback'
            return s_irls, t_irls, info

        # Weighted fusion.
        total_w = sum(w for _, w, _ in scale_estimates)
        s_prior = sum(s * w for s, w, _ in scale_estimates) / total_w
        avg_conf = total_w / len(scale_estimates)

        # Determine blend alpha.
        if self.cfg.adaptive_blend:
            drift_magnitude = abs(s_irls - 1.0)
            drift_factor = min(drift_magnitude / 0.1, 1.0)
            
            alpha = (self.cfg.adaptive_min_alpha +
                     (self.cfg.adaptive_max_alpha - self.cfg.adaptive_min_alpha) *
                     avg_conf * drift_factor)
            alpha = np.clip(alpha, self.cfg.adaptive_min_alpha, self.cfg.adaptive_max_alpha)
        else:
            alpha = self.cfg.blend_alpha

        info['alpha_used'] = alpha
        info['source'] = '+'.join(name for _, _, name in scale_estimates)

        # Blend prior scale with IRLS scale.
        s_final = alpha * s_prior + (1.0 - alpha) * s_irls

        # Adjust translation to preserve overlap centroid alignment.
        centroid_2 = point_map2.reshape(-1, 3).mean(axis=0)
        t_final = t_irls + (s_irls - s_final) * (R @ centroid_2)

        info['s_final'] = s_final

        return s_final, t_final, info

    def print_summary(self):
        """Print the estimated prior summary for all chunks."""
        print("\n" + "=" * 60)
        print("Scale Prior Summary")
        print("=" * 60)
        for ci in sorted(set(list(self.chunk_heights.keys()) + list(self.chunk_widths.keys()))):
            h_str = "N/A"
            w_str = "N/A"
            if ci in self.chunk_heights and self.chunk_heights[ci] is not None:
                h, hc = self.chunk_heights[ci]
                abs_s = self.cfg.camera_height / h
                h_str = f"h={h:.4f} (conf={hc:.2f}, abs_scale={abs_s:.4f})"
            if ci in self.chunk_widths and self.chunk_widths[ci] is not None:
                w, wc = self.chunk_widths[ci]
                known_w = self.cfg.lane_width * self.cfg.num_lanes
                abs_s = known_w / w
                w_str = f"w={w:.4f} (conf={wc:.2f}, abs_scale={abs_s:.4f})"
            print(f"  Chunk {ci:3d}: {h_str} | {w_str}")
        print("=" * 60 + "\n")
