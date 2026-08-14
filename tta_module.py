"""Lightweight test-time adaptation utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import copy


@dataclass
class TTAConfig:
    """TTA parameters injected from argparse."""
    tta_enable: bool = False
    tta_steps: int = 3
    tta_lr: float = 1e-4
    tta_momentum: float = 0.9         # SGD momentum

    # Loss weights
    photo_weight: float = 1.0
    geo_weight: float = 0.5
    smooth_weight: float = 0.1

    # Photometric loss settings
    photo_ssim_weight: float = 0.85
    photo_n_pairs: int = 2

    # Memory controls
    tta_n_frames: int = 3
    tta_resolution: int = 336
    tta_use_checkpoint: bool = True

    # Misc
    tta_warmup_chunks: int = 0
    tta_accumulate: bool = True


class TestTimeAdapter:
    """Test-time adaptation manager."""

    def __init__(self, model, cfg: TTAConfig):
        self.cfg = cfg
        self.base_model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._original_norm_state = None
        self._norm_params = []
        self._optimizer = None
        self.prev_chunk_predictions = None

        self._oom_fallback_level = 0

    def setup(self):
        """Find norm layers and freeze all other parameters."""
        inner_model = self._get_inner_model()
        if inner_model is None:
            print("[TTA] WARNING: Cannot access inner model, TTA disabled.")
            self.cfg.tta_enable = False
            return

        self._original_norm_state = {}
        self._norm_params = []

        for param in inner_model.parameters():
            param.requires_grad = False

        norm_layer_count = 0
        norm_param_count = 0
        for name, module in inner_model.named_modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                                   nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d)):
                norm_layer_count += 1
                for pname, param in module.named_parameters():
                    full_name = f"{name}.{pname}"
                    self._original_norm_state[full_name] = param.data.clone()
                    param.requires_grad = True
                    self._norm_params.append(param)
                    norm_param_count += param.numel()

        total_params = sum(p.numel() for p in inner_model.parameters())
        print(f"[TTA] {norm_layer_count} norm layers, "
              f"{norm_param_count:,} trainable / {total_params:,} total params "
              f"({norm_param_count/total_params*100:.2f}%)")
        print(f"[TTA] Memory budget: {self.cfg.tta_n_frames} frames @ "
              f"{self.cfg.tta_resolution}x{self.cfg.tta_resolution}")

        if self._norm_params:
            self._optimizer = torch.optim.SGD(
                self._norm_params,
                lr=self.cfg.tta_lr,
                momentum=self.cfg.tta_momentum
            )
        else:
            print("[TTA] WARNING: No norm parameters found, TTA will be no-op.")

    def _get_inner_model(self):
        """Get the underlying PyTorch model."""
        if hasattr(self.base_model, 'model'):
            return self.base_model.model
        if hasattr(self.base_model, 'net'):
            return self.base_model.net
        if isinstance(self.base_model, nn.Module):
            return self.base_model
        return None

    def reset(self):
        """Restore norm parameters to their original values."""
        inner_model = self._get_inner_model()
        if inner_model is None or not self._original_norm_state:
            return
        for name, module in inner_model.named_modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                                   nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d)):
                for pname, param in module.named_parameters():
                    full_name = f"{name}.{pname}"
                    if full_name in self._original_norm_state:
                        param.data.copy_(self._original_norm_state[full_name])

    @torch.enable_grad()
    def adapt(self,
              chunk_image_paths: List[str],
              chunk_idx: int,
              prev_predictions: Optional[Dict] = None):
        """Adapt the model on the current chunk."""
        if not self.cfg.tta_enable or not self._norm_params:
            return

        if chunk_idx < self.cfg.tta_warmup_chunks:
            return

        inner_model = self._get_inner_model()
        if inner_model is None:
            return

        # Clear cached allocations from the previous inference pass.
        torch.cuda.empty_cache()

        was_training = inner_model.training
        inner_model.train()

        # Choose frame count and resolution from the OOM fallback level.
        n_frames, resolution = self._get_tta_budget()
        if n_frames == 0:
            inner_model.train(was_training)
            return

        images = self._load_images(chunk_image_paths, n_frames, resolution)
        if images is None:
            inner_model.train(was_training)
            return

        print(f"[TTA] Adapting chunk {chunk_idx} "
              f"({self.cfg.tta_steps} steps, {n_frames} frames @ {resolution}px"
              f"{', fallback=' + str(self._oom_fallback_level) if self._oom_fallback_level > 0 else ''})")

        oom_hit = False
        completed_steps = 0

        for step in range(self.cfg.tta_steps):
            self._optimizer.zero_grad(set_to_none=True)

            try:
                predictions = self._forward_pass(inner_model, images)

                if predictions is None:
                    break

                loss = self._compute_tta_loss(predictions, images, prev_predictions)

                if loss is None or not torch.isfinite(loss):
                    del predictions
                    if loss is not None:
                        del loss
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self._norm_params, max_norm=1.0)
                self._optimizer.step()

                loss_val = loss.item()
                completed_steps += 1

                del predictions, loss
                torch.cuda.empty_cache()

                if self.cfg.tta_steps <= 5 or step == self.cfg.tta_steps - 1:
                    print(f"  [TTA] Step {step}: loss={loss_val:.6f}")

            except torch.cuda.OutOfMemoryError:
                print(f"  [TTA] OOM at step {step}, stopping this chunk.")
                oom_hit = True
                self._optimizer.zero_grad(set_to_none=True)
                try:
                    del predictions
                except NameError:
                    pass
                try:
                    del loss
                except NameError:
                    pass
                torch.cuda.empty_cache()
                break

            except Exception as e:
                print(f"  [TTA] Error at step {step}: {e}")
                break

        # Release TTA inputs before the main inference pass.
        del images
        torch.cuda.empty_cache()

        inner_model.train(was_training)

        if oom_hit:
            self._oom_fallback_level = min(self._oom_fallback_level + 1, 3)
            print(f"  [TTA] Fallback level → {self._oom_fallback_level} for future chunks. "
                  f"(Completed {completed_steps}/{self.cfg.tta_steps} steps before OOM)")
        elif self._oom_fallback_level > 0 and completed_steps == self.cfg.tta_steps:
            self._oom_fallback_level = max(self._oom_fallback_level - 1, 0)

    def _get_tta_budget(self) -> Tuple[int, int]:
        """Return (n_frames, resolution) for the current OOM fallback level."""
        level = self._oom_fallback_level
        base_n = self.cfg.tta_n_frames
        base_res = self.cfg.tta_resolution

        if level == 0:
            return base_n, base_res
        elif level == 1:
            return max(base_n - 1, 2), base_res
        elif level == 2:
            return 2, min(base_res, 256)
        else:
            return 0, 0

    def post_chunk_reset(self):
        """Reset after chunk inference when tta_accumulate is disabled."""
        if not self.cfg.tta_accumulate:
            self.reset()

    def _load_images(self, image_paths: List[str],
                     n_frames: int, resolution: int
                     ) -> Optional[torch.Tensor]:
        """Load a small low-resolution image batch for TTA."""
        from PIL import Image
        import torchvision.transforms as T

        try:
            transform = T.Compose([
                T.Resize((resolution, resolution)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

            n = len(image_paths)
            n_frames = min(n_frames, n)
            if n_frames >= n:
                indices = list(range(n))
            else:
                indices = np.linspace(0, n - 1, n_frames, dtype=int).tolist()

            images = []
            for i in indices:
                img = Image.open(image_paths[i]).convert('RGB')
                images.append(transform(img))

            images = torch.stack(images, dim=0)        # (N, 3, H, W)
            return images.unsqueeze(0).to(self.device)  # (1, N, 3, H, W)

        except Exception as e:
            print(f"[TTA] Image loading failed: {e}")
            return None

    def _forward_pass(self, model, images):
        """Run a forward pass with gradients enabled."""
        try:
            if hasattr(model, 'forward'):
                checkpoint_enabled = False
                if (self.cfg.tta_use_checkpoint and
                    hasattr(model, 'set_grad_checkpointing')):
                    model.set_grad_checkpointing(True)
                    checkpoint_enabled = True

                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    output = model(images)

                if checkpoint_enabled:
                    model.set_grad_checkpointing(False)

                if isinstance(output, dict):
                    return output
                elif isinstance(output, (tuple, list)):
                    return {
                        'depth': output[0] if len(output) > 0 else None,
                        'extrinsic': output[1] if len(output) > 1 else None,
                        'world_points': output[3] if len(output) > 3 else None,
                    }
            return None

        except torch.cuda.OutOfMemoryError:
            raise

        except Exception as e:
            print(f"[TTA] Forward pass error: {e}")
            return None

    def _compute_tta_loss(self,
                          predictions: Dict,
                          images: torch.Tensor,
                          prev_predictions: Optional[Dict] = None
                          ) -> Optional[torch.Tensor]:
        """Compute the weighted self-supervised TTA loss."""
        losses = []

        if self.cfg.photo_weight > 0:
            photo_loss = self._photometric_loss(predictions, images)
            if photo_loss is not None:
                losses.append(self.cfg.photo_weight * photo_loss)

        if self.cfg.geo_weight > 0:
            geo_loss = self._geometric_loss(predictions)
            if geo_loss is not None:
                losses.append(self.cfg.geo_weight * geo_loss)

        if self.cfg.smooth_weight > 0:
            smooth_loss = self._temporal_smooth_loss(predictions, prev_predictions)
            if smooth_loss is not None:
                losses.append(self.cfg.smooth_weight * smooth_loss)

        if not losses:
            return None
        return sum(losses)

    def _photometric_loss(self, predictions, images):
        """Photometric consistency proxy based on adjacent depth changes."""
        depth = predictions.get('depth')
        if depth is None:
            return None

        try:
            if isinstance(depth, np.ndarray):
                depth = torch.from_numpy(depth).to(self.device)

            if depth.ndim == 3:
                depth = depth.unsqueeze(0)

            N = depth.shape[1] if depth.ndim == 4 else depth.shape[0]
            if N < 2:
                return None

            loss = torch.tensor(0.0, device=self.device)
            count = 0
            n_pairs = min(self.cfg.photo_n_pairs, N - 1)
            pair_indices = np.linspace(0, N - 2, n_pairs, dtype=int)

            for idx in pair_indices:
                d1 = depth[0, idx] if depth.ndim == 4 else depth[idx]
                d2 = depth[0, idx + 1] if depth.ndim == 4 else depth[idx + 1]

                valid = (d1 > 1e-6) & (d2 > 1e-6)
                if valid.sum() < 100:
                    continue

                rel_diff = torch.abs(d1[valid] - d2[valid]) / (d1[valid] + 1e-6)
                loss = loss + rel_diff.mean()
                count += 1

            return loss / max(count, 1) if count > 0 else None

        except Exception:
            return None

    def _geometric_loss(self, predictions):
        """Geometric consistency proxy based on 3D displacement variance."""
        wp = predictions.get('world_points')
        if wp is None:
            return None

        try:
            if isinstance(wp, np.ndarray):
                wp = torch.from_numpy(wp).to(self.device)

            if wp.ndim < 3:
                return None

            N = wp.shape[0]
            if N < 3:
                return None

            displacements = []
            for i in range(min(N - 1, 4)):
                p1 = wp[i].reshape(-1, 3)
                p2 = wp[i + 1].reshape(-1, 3)
                c1 = p1[torch.isfinite(p1).all(dim=1)].mean(dim=0)
                c2 = p2[torch.isfinite(p2).all(dim=1)].mean(dim=0)
                if torch.isfinite(c1).all() and torch.isfinite(c2).all():
                    displacements.append((c2 - c1).norm())

            if len(displacements) < 2:
                return None

            d_tensor = torch.stack(displacements)
            return d_tensor.var()

        except Exception:
            return None

    def _temporal_smooth_loss(self, predictions, prev_predictions):
        """Temporal smoothness proxy based on camera acceleration."""
        ext = predictions.get('extrinsic')
        if ext is None:
            return None

        try:
            if isinstance(ext, np.ndarray):
                ext = torch.from_numpy(ext).to(self.device)

            if ext.ndim == 4:
                ext = ext[0]
            if ext.ndim != 3 or ext.shape[0] < 3:
                return None

            N = ext.shape[0]

            translations = ext[:, :3, 3]
            velocities = translations[1:] - translations[:-1]
            if velocities.shape[0] < 2:
                return None
            accelerations = velocities[1:] - velocities[:-1]
            accel_loss = (accelerations ** 2).sum(dim=1).mean()

            rotations = ext[:, :3, :3]
            rot_diffs = []
            for i in range(N - 1):
                R_rel = rotations[i].T @ rotations[i + 1]
                trace_val = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
                cos_angle = torch.clamp((trace_val - 1.0) / 2.0, -1.0, 1.0)
                rot_diffs.append(torch.acos(cos_angle))

            rot_loss = torch.tensor(0.0, device=self.device)
            if len(rot_diffs) >= 2:
                rot_tensor = torch.stack(rot_diffs)
                rot_loss = (rot_tensor[1:] - rot_tensor[:-1]).pow(2).mean()

            return accel_loss + 0.5 * rot_loss

        except Exception:
            return None
