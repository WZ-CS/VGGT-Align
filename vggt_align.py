"""VGGT-Align entrypoint."""

import numpy as np
import argparse
import os
import glob
import torch
from tqdm.auto import tqdm
import cv2
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
base_models_path = os.path.join(current_dir, 'base_models')
if base_models_path not in sys.path:
    sys.path.append(base_models_path)

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from LoopModels.LoopModel import LoopDetector
from LoopModelDBoW.retrieval.retrieval_dbow import RetrievalDBOW
from base_models.base_model import VGGTAdapter, Pi3Adapter, MapAnythingAdapter

from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import *
from datetime import datetime
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from loop_utils.config_utils import load_config
from pathlib import Path

from scale_prior import ScalePriorConfig, ScalePriorEstimator
from tta_module import TTAConfig, TestTimeAdapter


# ============================================================================
# Utility functions
# ============================================================================
def remove_duplicates(data_list):
    seen = {}
    result = []
    for item in data_list:
        if item[0] == item[2]:
            continue
        key = (item[0], item[2])
        if key not in seen:
            seen[key] = True
            result.append(item)
    return result


def extract_p2_k_matrix(calib_path):
    calib_path = Path(calib_path)
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('P2:'):
                values = line.split(':')[1].split()
                values = [float(v) for v in values]
                p2_matrix = np.array(values).reshape(3, 4)
                k_matrix = p2_matrix[:3, :3]
                return k_matrix, p2_matrix
    raise ValueError("P2 not found in calibration file")


# ============================================================================
# Main class
# ============================================================================
class VGGTAlign:
    def __init__(self, image_dir, save_dir, config, 
                 scale_cfg: ScalePriorConfig, tta_cfg: TTAConfig):
        self.config = config
        self.scale_cfg = scale_cfg
        self.tta_cfg = tta_cfg

        self.chunk_size = self.config['Model']['chunk_size']
        self.overlap = self.config['Model']['overlap']
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = (torch.bfloat16 
                      if torch.cuda.get_device_capability()[0] >= 8 
                      else torch.float16)
        self.sky_mask = False
        self.useDBoW = self.config['Model']['useDBoW']

        self.img_dir = image_dir
        self.img_list = None
        self.output_dir = save_dir

        self.result_unaligned_dir = os.path.join(save_dir, '_tmp_results_unaligned')
        self.result_aligned_dir = os.path.join(save_dir, '_tmp_results_aligned')
        self.result_loop_dir = os.path.join(save_dir, '_tmp_results_loop')
        self.pcd_dir = os.path.join(save_dir, 'pcd')
        for d in [self.result_unaligned_dir, self.result_aligned_dir, 
                  self.result_loop_dir, self.pcd_dir]:
            os.makedirs(d, exist_ok=True)

        self.all_camera_poses = []
        self.all_camera_intrinsics = []
        self.delete_temp_files = self.config['Model']['delete_temp_files']

        # Model
        if self.config['Weights']['model'] == 'VGGT':
            self.model = VGGTAdapter(self.config)
        elif self.config['Weights']['model'] == 'Pi3':
            self.model = Pi3Adapter(self.config)
        elif self.config['Weights']['model'] == 'Mapanything':
            self.model = MapAnythingAdapter(self.config)
        else:
            raise ValueError(f"Unsupported model: {self.config['Weights']['model']}")

        self.skyseg_session = None
        self.chunk_indices = None
        self.loop_list = []
        self.loop_optimizer = Sim3LoopOptimizer(self.config)
        self.sim3_list = []
        self.loop_sim3_list = []
        self.loop_predict_list = []
        self.loop_enable = self.config['Model']['loop_enable']

        if self.loop_enable:
            if self.useDBoW:
                self.retrieval = RetrievalDBOW(config=self.config)
            else:
                loop_info_save_path = os.path.join(save_dir, "loop_closures.txt")
                self.loop_detector = LoopDetector(
                    image_dir=image_dir,
                    output=loop_info_save_path,
                    config=self.config
                )

        # Scale priors
        self.scale_estimator = ScalePriorEstimator(self.scale_cfg)

        # TTA
        self.tta_adapter = TestTimeAdapter(self.model, self.tta_cfg) if self.tta_cfg.tta_enable else None

        # Configuration summary
        print("=" * 60)
        print("VGGT-Align Configuration")
        print("=" * 60)
        if self.scale_cfg.ground_prior:
            print(f"  [GroundPrior] ON  camera_height={self.scale_cfg.camera_height}m "
                  f"blend_alpha={self.scale_cfg.blend_alpha} "
                  f"adaptive={self.scale_cfg.adaptive_blend}")
        else:
            print(f"  [GroundPrior] OFF")
        if self.scale_cfg.road_width_prior:
            print(f"  [RoadWidth]   ON  lane_width={self.scale_cfg.lane_width}m "
                  f"x{self.scale_cfg.num_lanes} lanes "
                  f"weight={self.scale_cfg.width_weight}")
        else:
            print(f"  [RoadWidth]   OFF")
        if self.tta_cfg.tta_enable:
            print(f"  [TTA]         ON  steps={self.tta_cfg.tta_steps} "
                  f"lr={self.tta_cfg.tta_lr} "
                  f"frames={self.tta_cfg.tta_n_frames}@{self.tta_cfg.tta_resolution}px "
                  f"accumulate={self.tta_cfg.tta_accumulate}")
        else:
            print(f"  [TTA]         OFF")
        print("=" * 60)
        print('Init done.')

    # ------------------------------------------------------------------
    # Loop detection
    # ------------------------------------------------------------------
    def get_loop_pairs(self):
        if self.useDBoW:
            for frame_id, img_path in tqdm(enumerate(self.img_list)):
                image_ori = np.array(Image.open(img_path))
                if len(image_ori.shape) == 2:
                    image_ori = cv2.cvtColor(image_ori, cv2.COLOR_GRAY2RGB)
                frame = cv2.resize(image_ori, None, fx=0.5, fy=0.5, 
                                   interpolation=cv2.INTER_AREA)
                self.retrieval(frame, frame_id)
                cands = self.retrieval.detect_loop(
                    thresh=self.config['Loop']['DBoW']['thresh'],
                    num_repeat=self.config['Loop']['DBoW']['num_repeat'])
                if cands is not None:
                    (i, j) = cands
                    self.retrieval.confirm_loop(i, j)
                    self.retrieval.found.clear()
                    self.loop_list.append(cands)
                self.retrieval.save_up_to(frame_id)
        else:
            self.loop_detector.run()
            self.loop_list = self.loop_detector.get_loop_list()

    # ------------------------------------------------------------------
    # Single chunk processing
    # ------------------------------------------------------------------
    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            s2, e2 = range_2
            chunk_image_paths += self.img_list[s2:e2]

        # TTA before inference
        if (self.tta_adapter is not None and not is_loop and range_2 is None
            and chunk_idx is not None):
            self.tta_adapter.adapt(chunk_image_paths, chunk_idx)
            torch.cuda.empty_cache()

        predictions = self.model.infer_chunk(chunk_image_paths)
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)

        # TTA reset after chunk inference
        if self.tta_adapter is not None and not is_loop:
            self.tta_adapter.post_chunk_reset()

        if is_loop:
            save_dir = self.result_loop_dir
            filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"

        save_path = os.path.join(save_dir, filename)

        if not is_loop and range_2 is None:
            extrinsics = predictions['extrinsic']
            intrinsics = predictions['intrinsic']
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, extrinsics))
            self.all_camera_intrinsics.append((chunk_range, intrinsics))

        predictions['depth'] = np.squeeze(predictions['depth'])
        np.save(save_path, predictions)

        return predictions if is_loop or range_2 is not None else None

    # ------------------------------------------------------------------
    # Main processing pipeline
    # ------------------------------------------------------------------
    def process_long_sequence(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"Overlap ({self.overlap}) must be < chunk size ({self.chunk_size})")

        if len(self.img_list) <= self.chunk_size:
            num_chunks = 1
            self.chunk_indices = [(0, len(self.img_list))]
        else:
            step = self.chunk_size - self.overlap
            num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
            self.chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + self.chunk_size, len(self.img_list))
                self.chunk_indices.append((start_idx, end_idx))

        # Chunk inference
        for chunk_idx in range(len(self.chunk_indices)):
            print(f'[Progress]: {chunk_idx}/{len(self.chunk_indices)-1}')
            self.process_single_chunk(self.chunk_indices[chunk_idx], chunk_idx=chunk_idx)
            torch.cuda.empty_cache()

        # ---- Loop closure ----
        if self.loop_enable:
            print('Loop SIM(3) estimating...')
            loop_results = process_loop_list(
                self.chunk_indices, self.loop_list,
                half_window=int(self.config['Model']['loop_chunk_size'] / 2))
            loop_results = remove_duplicates(loop_results)
            print(loop_results)
            for item in loop_results:
                single_chunk_predictions = self.process_single_chunk(
                    item[1], range_2=item[3], is_loop=True)
                self.loop_predict_list.append((item, single_chunk_predictions))
                print(item)

        print(f"Processing {len(self.img_list)} images in {num_chunks} chunks "
              f"of size {self.chunk_size} with {self.overlap} overlap")

        del self.model
        torch.cuda.empty_cache()

        # ================================================================
        # Precompute scale priors for every chunk.
        # ================================================================
        use_prior = self.scale_cfg.ground_prior or self.scale_cfg.road_width_prior

        if use_prior:
            print("[ScalePrior] Estimating scale priors for all chunks...")
            for ci in range(len(self.chunk_indices)):
                cd = np.load(
                    os.path.join(self.result_unaligned_dir, f"chunk_{ci}.npy"),
                    allow_pickle=True).item()
                _, ext = self.all_camera_poses[ci]
                self.scale_estimator.estimate_chunk(
                    ci, cd['world_points'], cd['world_points_conf'], ext)

            if self.scale_cfg.diagnostics:
                self.scale_estimator.print_summary()

        # ================================================================
        # Neighboring chunk alignment
        # ================================================================
        print("Aligning all the chunks...")
        diag_records = []

        for chunk_idx in range(len(self.chunk_indices) - 1):
            print(f"Aligning {chunk_idx} and {chunk_idx+1} "
                  f"(Total {len(self.chunk_indices)-1})")

            chunk_data1 = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"),
                allow_pickle=True).item()
            chunk_data2 = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"),
                allow_pickle=True).item()

            point_map1 = chunk_data1['world_points'][-self.overlap:]
            point_map2 = chunk_data2['world_points'][:self.overlap]
            conf1 = chunk_data1['world_points_conf'][-self.overlap:]
            conf2 = chunk_data2['world_points_conf'][:self.overlap]

            mask = None
            if chunk_data1["mask"] is not None:
                mask1 = chunk_data1["mask"][-self.overlap:]
                mask2 = chunk_data2["mask"][:self.overlap]
                mask = mask1.squeeze() & mask2.squeeze()

            if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True):
                conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1
            else:
                conf_threshold = -1.0

            s_irls, R, t_irls = weighted_align_point_maps(
                point_map1, conf1, point_map2, conf2, mask,
                conf_threshold=conf_threshold, config=self.config)

            if use_prior:
                s, t, info = self.scale_estimator.get_corrected_scale(
                    chunk_idx, s_irls, R, t_irls, point_map2)

                if self.scale_cfg.diagnostics:
                    print(f"  [Prior] source={info['source']} "
                          f"alpha={info['alpha_used']:.3f} "
                          f"s_irls={s_irls:.4f} s_final={s:.4f}")
            else:
                s, t = s_irls, t_irls
                info = {'s_irls': s_irls, 's_final': s_irls, 'source': 'irls_only',
                        'alpha_used': 0.0}

            cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
            rot_deg = np.degrees(np.arccos(cos_a))
            diag_records.append({
                'pair': chunk_idx,
                's_irls': s_irls,
                's_final': s,
                'rot_deg': rot_deg,
                'trans_norm': np.linalg.norm(t),
                'source': info.get('source', ''),
                'alpha': info.get('alpha_used', 0.0),
            })

            print(f"  Scale: {s:.6f}  Rotation: {rot_deg:.2f}°  "
                  f"Translation: {np.linalg.norm(t):.4f}")

            self.sim3_list.append((s, R, t))

        if self.scale_cfg.diagnostics and diag_records:
            self._save_diagnostics(diag_records)

        # ================================================================
        # Loop closure alignment
        # ================================================================
        if self.loop_enable:
            for item in self.loop_predict_list:
                chunk_idx_a = item[0][0]
                chunk_idx_b = item[0][2]
                chunk_a_range = item[0][1]
                chunk_b_range = item[0][3]

                # chunk_a align
                point_map_loop = item[1]['world_points'][:chunk_a_range[1] - chunk_a_range[0]]
                conf_loop = item[1]['world_points_conf'][:chunk_a_range[1] - chunk_a_range[0]]
                chunk_a_rela_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
                chunk_a_rela_end = chunk_a_rela_begin + chunk_a_range[1] - chunk_a_range[0]
                chunk_data_a = np.load(
                    os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"),
                    allow_pickle=True).item()
                point_map_a = chunk_data_a['world_points'][chunk_a_rela_begin:chunk_a_rela_end]
                conf_a = chunk_data_a['world_points_conf'][chunk_a_rela_begin:chunk_a_rela_end]
                if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True):
                    conf_threshold = min(np.median(conf_a), np.median(conf_loop)) * 0.1
                else:
                    conf_threshold = -1.0
                mask = None
                if item[1]['mask'] is not None:
                    mask_loop = item[1]['mask'][:chunk_a_range[1] - chunk_a_range[0]]
                    mask_a = chunk_data_a['mask'][chunk_a_rela_begin:chunk_a_rela_end]
                    mask = mask_loop.squeeze() & mask_a.squeeze()
                s_a, R_a, t_a = weighted_align_point_maps(
                    point_map_a, conf_a, point_map_loop, conf_loop,
                    mask, conf_threshold=conf_threshold, config=self.config)

                # chunk_b align
                point_map_loop = item[1]['world_points'][-chunk_b_range[1] + chunk_b_range[0]:]
                conf_loop = item[1]['world_points_conf'][-chunk_b_range[1] + chunk_b_range[0]:]
                chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
                chunk_b_rela_end = chunk_b_rela_begin + chunk_b_range[1] - chunk_b_range[0]
                chunk_data_b = np.load(
                    os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"),
                    allow_pickle=True).item()
                point_map_b = chunk_data_b['world_points'][chunk_b_rela_begin:chunk_b_rela_end]
                conf_b = chunk_data_b['world_points_conf'][chunk_b_rela_begin:chunk_b_rela_end]
                if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True):
                    conf_threshold = min(np.median(conf_b), np.median(conf_loop)) * 0.1
                else:
                    conf_threshold = -1.0
                mask = None
                if item[1]['mask'] is not None:
                    mask_loop = item[1]['mask'][-chunk_b_range[1] + chunk_b_range[0]:]
                    mask_b = chunk_data_b['mask'][chunk_b_rela_begin:chunk_b_rela_end]
                    mask = mask_loop.squeeze() & mask_b.squeeze()
                s_b, R_b, t_b = weighted_align_point_maps(
                    point_map_b, conf_b, point_map_loop, conf_loop,
                    mask, conf_threshold=conf_threshold, config=self.config)

                # a -> b SIM3
                s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
                self.loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))

        # ---- Loop optimization ----
        if self.loop_enable:
            input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)
            self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
            optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)

            def extract_xyz(pose_tensor):
                poses = pose_tensor.cpu().numpy()
                return poses[:, 0], poses[:, 1], poses[:, 2]

            x0, _, y0 = extract_xyz(input_abs_poses)
            x1, _, y1 = extract_xyz(optimized_abs_poses)

            plt.figure(figsize=(8, 6))
            plt.plot(x0, y0, 'o--', alpha=0.45, label='Before Optimization')
            plt.plot(x1, y1, 'o-', label='After Optimization')
            for i, j, _ in self.loop_sim3_list:
                plt.plot([x0[i], x0[j]], [y0[i], y0[j]], 'r--', alpha=0.25,
                         label='Loop (Before)' if i == 5 else "")
                plt.plot([x1[i], x1[j]], [y1[i], y1[j]], 'g-', alpha=0.35,
                         label='Loop (After)' if i == 5 else "")
            plt.gca().set_aspect('equal')
            plt.title("Sim3 Loop Closure Optimization")
            plt.xlabel("x"); plt.ylabel("z")
            plt.legend(); plt.grid(True); plt.axis("equal")
            plt.savefig(os.path.join(self.output_dir, 'sim3_opt_result.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        # ================================================================
        # Apply alignment and generate point clouds.
        # ================================================================
        print('Apply alignment')
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)

        for chunk_idx in range(len(self.chunk_indices) - 1):
            print(f'Applying {chunk_idx+1} -> {chunk_idx} '
                  f'(Total {len(self.chunk_indices)-1})')
            s, R, t = self.sim3_list[chunk_idx]

            chunk_data = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"),
                allow_pickle=True).item()
            chunk_data['world_points'] = apply_sim3_direct(
                chunk_data['world_points'], s, R, t)

            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy")
            np.save(aligned_path, chunk_data)

            if chunk_idx == 0:
                chunk_data_first = np.load(
                    os.path.join(self.result_unaligned_dir, "chunk_0.npy"),
                    allow_pickle=True).item()
                np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"),
                        chunk_data_first)
                self._save_chunk_ply(chunk_data_first, 0)

            aligned_cd = (np.load(
                os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy"),
                allow_pickle=True).item() if chunk_idx > 0 else chunk_data_first)
            self._save_chunk_ply(aligned_cd, chunk_idx + 1)

        self.save_camera_poses()
        print('Done.')

    def _save_chunk_ply(self, chunk_data, chunk_idx):
        """Save a point cloud for a single chunk."""
        points = chunk_data['world_points'].reshape(-1, 3)
        colors = (chunk_data['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255
                  ).astype(np.uint8)
        confs = chunk_data['world_points_conf'].reshape(-1)
        ply_path = os.path.join(self.pcd_dir, f'{chunk_idx}_pcd.ply')
        save_confident_pointcloud_batch(
            points=points, colors=colors, confs=confs,
            output_path=ply_path,
            conf_threshold=(
                np.mean(confs) * self.config['Model']['Pointcloud_Save']['conf_threshold_coef']
                if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True)
                else -1.0),
            sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio'])

    def _save_diagnostics(self, diag_records):
        """Save alignment diagnostics."""
        rp = os.path.join(self.output_dir, 'alignment_diagnostics.txt')
        with open(rp, 'w') as f:
            f.write(f"{'Pair':>6} {'s_IRLS':>10} {'s_final':>10} {'RotDeg':>10} "
                    f"{'TransNorm':>12} {'Source':>16} {'Alpha':>8}\n")
            f.write('-' * 75 + '\n')
            for r in diag_records:
                f.write(f"{r['pair']:>6} {r['s_irls']:>10.4f} {r['s_final']:>10.4f} "
                        f"{r['rot_deg']:>10.4f} {r['trans_norm']:>12.4f} "
                        f"{r['source']:>16} {r['alpha']:>8.3f}\n")
            scales_f = [r['s_final'] for r in diag_records]
            scales_i = [r['s_irls'] for r in diag_records]
            f.write(f"\n--- IRLS scale: mean={np.mean(scales_i):.4f} "
                    f"std={np.std(scales_i):.4f} ---\n")
            f.write(f"--- Final scale: mean={np.mean(scales_f):.4f} "
                    f"std={np.std(scales_f):.4f} ---\n")
        print(f"[Diag] → {rp}")

        n = len(diag_records)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        scales_i = [r['s_irls'] for r in diag_records]
        scales_f = [r['s_final'] for r in diag_records]
        rots = [r['rot_deg'] for r in diag_records]

        axes[0].plot(range(n), scales_i, 'r.--', alpha=0.5, ms=3, label='IRLS')
        axes[0].plot(range(n), scales_f, 'b.-', ms=3, label='Prior-corrected')
        axes[0].axhline(1.0, color='k', ls=':', alpha=0.3)
        axes[0].set_title('Scale per pair'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(range(n), np.cumprod(scales_i), 'r.--', alpha=0.5, ms=3, label='IRLS')
        axes[1].plot(range(n), np.cumprod(scales_f), 'b.-', ms=3, label='Prior-corrected')
        axes[1].axhline(1.0, color='k', ls=':', alpha=0.3)
        axes[1].set_title('Cumulative Scale'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(range(n), rots, 'g.-', ms=3)
        axes[2].set_title('Rotation (deg)'); axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'alignment_diagnostics.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self):
        print(f"Loading images from {self.img_dir}...")
        self.img_list = sorted(
            glob.glob(os.path.join(self.img_dir, "*.jpg")) +
            glob.glob(os.path.join(self.img_dir, "*.png")))
        if len(self.img_list) == 0:
            raise ValueError(f"No images found in {self.img_dir}!")
        print(f"Found {len(self.img_list)} images")

        if self.loop_enable:
            self.get_loop_pairs()
            if self.useDBoW:
                self.retrieval.close()
                gc.collect()
            else:
                del self.loop_detector
        torch.cuda.empty_cache()

        print('Loading model...')
        self.model.load()

        # TTA setup
        if self.tta_adapter is not None:
            self.tta_adapter.setup()

        if self.config['Model']['calib']:
            calib_path = Path(self.img_dir).parent / 'calib.txt'
            k, p2_matrix = extract_p2_k_matrix(calib_path)
            self.model.k = k

        self.process_long_sequence()

    # ------------------------------------------------------------------
    # Camera pose saving (unchanged from original)
    # ------------------------------------------------------------------
    def save_camera_poses(self):
        chunk_colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
            [0, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
        ]
        print("Saving all camera poses to txt file...")
        all_poses = [None] * len(self.img_list)
        all_intrinsics = [None] * len(self.img_list)

        first_chunk_range, first_chunk_extrinsics = self.all_camera_poses[0]
        _, first_chunk_intrinsics = self.all_camera_intrinsics[0]
        for i, idx in enumerate(range(first_chunk_range[0], first_chunk_range[1])):
            all_poses[idx] = first_chunk_extrinsics[i]
            if first_chunk_intrinsics is not None:
                all_intrinsics[idx] = first_chunk_intrinsics[i]

        for chunk_idx in range(1, len(self.all_camera_poses)):
            chunk_range, chunk_extrinsics = self.all_camera_poses[chunk_idx]
            _, chunk_intrinsics = self.all_camera_intrinsics[chunk_idx]
            s, R, t = self.sim3_list[chunk_idx - 1]
            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t
            for i, idx in enumerate(range(chunk_range[0], chunk_range[1])):
                c2w = chunk_extrinsics[i]
                transformed_c2w = S @ c2w
                transformed_c2w[:3, :3] /= s
                all_poses[idx] = transformed_c2w
                if chunk_intrinsics is not None:
                    all_intrinsics[idx] = chunk_intrinsics[i]

        poses_path = os.path.join(self.output_dir, 'camera_poses.txt')
        with open(poses_path, 'w') as f:
            for pose in all_poses:
                f.write(' '.join([str(x) for x in pose.flatten()]) + '\n')
        print(f"Camera poses saved to {poses_path}")

        if all_intrinsics[0] is not None:
            intrinsics_path = os.path.join(self.output_dir, 'intrinsic.txt')
            with open(intrinsics_path, 'w') as f:
                for intrinsic in all_intrinsics:
                    f.write(f'{intrinsic[0,0]} {intrinsic[1,1]} '
                            f'{intrinsic[0,2]} {intrinsic[1,2]}\n')
            print(f"Camera intrinsics saved to {intrinsics_path}")

        ply_path = os.path.join(self.output_dir, 'camera_poses.ply')
        with open(ply_path, 'w') as f:
            f.write('ply\nformat ascii 1.0\n')
            f.write(f'element vertex {len(all_poses)}\n')
            f.write('property float x\nproperty float y\nproperty float z\n')
            f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
            f.write('end_header\n')
            for pose in all_poses:
                pos = pose[:3, 3]
                f.write(f'{pos[0]} {pos[1]} {pos[2]} 255 0 0\n')
        print(f"Camera poses visualization saved to {ply_path}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self):
        if not self.delete_temp_files:
            return
        total_space = 0
        for d in [self.result_unaligned_dir, self.result_aligned_dir, 
                  self.result_loop_dir]:
            print(f'Deleting temp files under {d}')
            for fn in os.listdir(d):
                fp = os.path.join(d, fn)
                if os.path.isfile(fp):
                    total_space += os.path.getsize(fp)
                    os.remove(fp)
        print(f'Temp files deleted. Saved {total_space/1024/1024/1024:.4f} GiB')


# ============================================================================
# argparse + main
# ============================================================================
import shutil

def copy_file(src_path, dst_dir):
    try:
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst)
        print(f"Config copied to: {dst}")
        return dst
    except Exception as e:
        print(f"Copy error: {e}")


def build_argparser():
    parser = argparse.ArgumentParser(
        description='VGGT-Align: Multi-source Scale Prior + Test-Time Adaptation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline (no improvements)
  python vggt_align.py --image_dir ./data/kitti/00/image_2

  # Ground prior only (KITTI)
  python vggt_align.py --image_dir ./data/kitti/00/image_2 --ground_prior --camera_height 1.65

  # Ground + Road width + Adaptive (Waymo)
  python vggt_align.py --image_dir ./data/waymo/seg1/images \\
      --ground_prior --camera_height 2.05 \\
      --road_width_prior --lane_width 3.7 --num_lanes 2 \\
      --adaptive_blend

  # All on
  python vggt_align.py --image_dir ./data/kitti/00/image_2 \\
      --ground_prior --road_width_prior --tta --tta_steps 3
        """)

    # Basic arguments
    parser.add_argument('--image_dir', type=str, required=True, help='Image directory')
    parser.add_argument('--config', type=str, default='./configs/base_config.yaml',
                        help='YAML config path')
    parser.add_argument('--exp_dir', type=str, default='./exps', help='Experiment root dir')

    # Ground height prior
    parser.add_argument('--ground_prior', action='store_true',
                        help='Enable ground plane height prior')
    parser.add_argument('--camera_height', type=float, default=1.65,
                        help='Known camera height in meters (KITTI=1.65, Waymo≈2.05)')
    parser.add_argument('--ground_row_ratio', type=float, default=0.3,
                        help='Bottom fraction of image rows for ground candidates')
    parser.add_argument('--ransac_iters', type=int, default=500,
                        help='RANSAC iterations for ground plane fitting')

    # Road width prior
    parser.add_argument('--road_width_prior', action='store_true',
                        help='Enable road width prior')
    parser.add_argument('--lane_width', type=float, default=3.75,
                        help='Known single lane width in meters')
    parser.add_argument('--num_lanes', type=float, default=2.0,
                        help='Expected number of visible lanes')

    # Fusion arguments
    parser.add_argument('--blend_alpha', type=float, default=1.0,
                        help='Prior vs IRLS blend ratio (1.0=full prior, 0.0=full IRLS)')
    parser.add_argument('--adaptive_blend', action='store_true',
                        help='Enable adaptive blending (recommended for Waymo)')
    parser.add_argument('--adaptive_min_alpha', type=float, default=0.3,
                        help='Minimum alpha in adaptive mode')
    parser.add_argument('--adaptive_max_alpha', type=float, default=1.0,
                        help='Maximum alpha in adaptive mode')
    parser.add_argument('--height_weight', type=float, default=0.7,
                        help='Weight of height prior in multi-source fusion')
    parser.add_argument('--width_weight', type=float, default=0.3,
                        help='Weight of width prior in multi-source fusion')

    # ---- TTA ----
    parser.add_argument('--tta', action='store_true',
                        help='Enable test-time adaptation')
    parser.add_argument('--tta_steps', type=int, default=3,
                        help='Gradient steps per chunk')
    parser.add_argument('--tta_lr', type=float, default=1e-4,
                        help='TTA learning rate')
    parser.add_argument('--tta_photo_weight', type=float, default=1.0,
                        help='Photometric consistency loss weight')
    parser.add_argument('--tta_geo_weight', type=float, default=0.5,
                        help='Geometric consistency loss weight')
    parser.add_argument('--tta_smooth_weight', type=float, default=0.1,
                        help='Temporal smoothness loss weight')
    parser.add_argument('--tta_n_frames', type=int, default=3,
                        help='Number of frames for TTA (fewer = less VRAM, 3 is enough)')
    parser.add_argument('--tta_resolution', type=int, default=336,
                        help='TTA input resolution (lower than inference 518, saves ~70%% VRAM)')
    parser.add_argument('--tta_warmup_chunks', type=int, default=0,
                        help='Skip TTA for first N chunks')
    parser.add_argument('--tta_accumulate', action='store_true', default=True,
                        help='Accumulate norm updates across chunks')
    parser.add_argument('--no_tta_accumulate', action='store_false', dest='tta_accumulate',
                        help='Reset norm params after each chunk')

    # Diagnostics
    parser.add_argument('--no_diagnostics', action='store_true',
                        help='Disable diagnostic outputs')

    return parser


def args_to_configs(args):
    """Convert argparse values to ScalePriorConfig and TTAConfig."""

    scale_cfg = ScalePriorConfig(
        ground_prior=args.ground_prior,
        road_width_prior=args.road_width_prior,
        camera_height=args.camera_height,
        ground_row_ratio=args.ground_row_ratio,
        ransac_iters=args.ransac_iters,
        lane_width=args.lane_width,
        num_lanes=args.num_lanes,
        blend_alpha=args.blend_alpha,
        adaptive_blend=args.adaptive_blend,
        adaptive_min_alpha=args.adaptive_min_alpha,
        adaptive_max_alpha=args.adaptive_max_alpha,
        height_weight=args.height_weight,
        width_weight=args.width_weight,
        diagnostics=not args.no_diagnostics,
    )

    tta_cfg = TTAConfig(
        tta_enable=args.tta,
        tta_steps=args.tta_steps,
        tta_lr=args.tta_lr,
        photo_weight=args.tta_photo_weight,
        geo_weight=args.tta_geo_weight,
        smooth_weight=args.tta_smooth_weight,
        tta_n_frames=args.tta_n_frames,
        tta_resolution=args.tta_resolution,
        tta_warmup_chunks=args.tta_warmup_chunks,
        tta_accumulate=args.tta_accumulate,
    )

    return scale_cfg, tta_cfg


if __name__ == '__main__':
    parser = build_argparser()
    args = parser.parse_args()

    config = load_config(args.config)
    scale_cfg, tta_cfg = args_to_configs(args)

    image_dir = args.image_dir
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Build an experiment directory name with method tags.
    method_tag = "baseline"
    tags = []
    if args.ground_prior:
        tags.append("gp")
    if args.road_width_prior:
        tags.append("rw")
    if args.adaptive_blend:
        tags.append("ab")
    if args.tta:
        tags.append("tta")
    if tags:
        method_tag = "_".join(tags)

    save_dir = os.path.join(
        args.exp_dir,
        image_dir.replace("/", "_"),
        f"{method_tag}_{current_datetime}"
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f'Experiment dir: {save_dir}')
        copy_file(args.config, save_dir)

    # Save full command-line arguments.
    with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write(f'{k}: {v}\n')

    if config['Model']['align_method'] == 'numba':
        warmup_numba()

    vggt_align = VGGTAlign(image_dir, save_dir, config, scale_cfg, tta_cfg)
    vggt_align.run()
    vggt_align.close()

    del vggt_align
    torch.cuda.empty_cache()
    gc.collect()

    all_ply_path = os.path.join(save_dir, 'pcd/combined_pcd.ply')
    input_dir = os.path.join(save_dir, 'pcd')
    print("Merging all point clouds...")
    merge_ply_files(input_dir, all_ply_path)
    print('All done.')
    sys.exit()
