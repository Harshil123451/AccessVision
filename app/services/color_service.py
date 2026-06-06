import numpy as np
import colorsys
from PIL import Image
from app.services.base import BaseService
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger("accessvision")

class ColorService(BaseService):
    """Service to extract dominant colors using KMeans clustering and map to HSV-aware color names with confidence assessment."""

    def kmeans_numpy(self, pixels: np.ndarray, k: int = 3, max_iters: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Performs simple KMeans clustering on the input pixels in RGB space."""
        if len(pixels) == 0:
            return np.zeros((0, 3)), np.array([])
            
        n_samples = len(pixels)
        k = min(k, n_samples)
        
        # Initialize centroids randomly from existing pixels
        np.random.seed(42)
        idx = np.random.choice(n_samples, k, replace=False)
        centroids = pixels[idx].astype(float)
        
        labels = np.zeros(n_samples, dtype=int)
        for _ in range(max_iters):
            # Compute Euclidean distances to all centroids
            dists = np.linalg.norm(pixels[:, None, :] - centroids[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            
            # Recompute centroids
            for i in range(k):
                mask = (labels == i)
                if np.any(mask):
                    centroids[i] = pixels[mask].mean(axis=0)
                    
        counts = np.bincount(labels, minlength=k)
        return centroids, counts

    def rgb_to_hsv_numpy(self, rgb: np.ndarray) -> np.ndarray:
        """Vectorized RGB to HSV conversion for NumPy arrays."""
        rgb_normalized = rgb / 255.0
        r, g, b = rgb_normalized[:, 0], rgb_normalized[:, 1], rgb_normalized[:, 2]
        
        max_c = np.max(rgb_normalized, axis=1)
        min_c = np.min(rgb_normalized, axis=1)
        delta = max_c - min_c
        
        v = max_c
        
        s = np.zeros_like(max_c)
        non_zero_max = max_c > 0
        s[non_zero_max] = delta[non_zero_max] / max_c[non_zero_max]
        
        h = np.zeros_like(max_c)
        non_zero_delta = delta > 0
        
        idx_r = (max_c == r) & non_zero_delta
        h[idx_r] = ((g[idx_r] - b[idx_r]) / delta[idx_r]) % 6
        
        idx_g = (max_c == g) & non_zero_delta
        h[idx_g] = ((b[idx_g] - r[idx_g]) / delta[idx_g]) + 2
        
        idx_b = (max_c == b) & non_zero_delta
        h[idx_b] = ((r[idx_b] - g[idx_b]) / delta[idx_b]) + 4
        
        h = h * 60.0
        return np.stack([h, s, v], axis=1)

    def classify_hsv(self, h: float, s: float, v: float) -> str:
        """Classifies HSV color value to a human-readable string with saturation/brightness modifiers."""
        # 1. Grayscale checks
        if v < 0.08 or (v < 0.20 and s < 0.25):
            return "black"
        if s < 0.08:
            if v > 0.85:
                return "white"
            return "gray"
        if s < 0.15 and v < 0.5:
            return "dark gray"
        if s < 0.15 and v >= 0.5:
            return "light gray"
            
        # 2. Hue classifications
        if h < 15 or h >= 345:
            base_color = "red"
        elif h < 45:
            base_color = "orange"
        elif h < 75:
            base_color = "yellow"
        elif h < 165:
            base_color = "green"
        elif h < 255:
            base_color = "blue"
        elif h < 310:
            base_color = "purple"
        else:
            base_color = "pink"
            
        # 3. Apply modifiers based on saturation and value
        if s < 0.28:
            # Grayish tint
            if v < 0.4:
                return f"dark {base_color}-gray"
            elif v > 0.7:
                return f"light {base_color}-gray"
            return f"{base_color}-gray"
            
        if v < 0.35:
            return f"dark {base_color}"
        if v > 0.8 and s < 0.4:
            return f"light {base_color}"
        if v > 0.85 and s > 0.75:
            return f"bright {base_color}"
            
        return base_color

    def analyze_color(self, crop_image: Image.Image, detection_confidence: float = 1.0, fast_mode: bool = False) -> Dict[str, Any]:
        """Analyzes a cropped object region and returns color name, confidence, and dominant HSV."""
        from app.core.telemetry import trace_stage
        import time
        
        t_start = time.perf_counter()
        
        with trace_stage("COLOR"):
            t0 = time.perf_counter()
            if crop_image.mode != "RGB":
                crop_image = crop_image.convert("RGB")
            t_prep = (time.perf_counter() - t0) * 1000
            
            if fast_mode:
                t_ds_start = time.perf_counter()
                with trace_stage("COLOR_DOWNSAMPLE"):
                    resized_crop = crop_image.resize((32, 32))
                t_downsample = (time.perf_counter() - t_ds_start) * 1000
                
                t_rgb_start = time.perf_counter()
                pixels = np.array(resized_crop)
                mean_rgb = pixels.mean(axis=(0, 1))
                t_rgb = (time.perf_counter() - t_rgb_start) * 1000
                
                t_hsv_start = time.perf_counter()
                with trace_stage("COLOR_HSV"):
                    dom_h, dom_s, dom_v = colorsys.rgb_to_hsv(mean_rgb[0] / 255.0, mean_rgb[1] / 255.0, mean_rgb[2] / 255.0)
                    dom_h_deg = dom_h * 360.0
                t_hsv = (time.perf_counter() - t_hsv_start) * 1000
                
                t_cls_start = time.perf_counter()
                with trace_stage("COLOR_CLASSIFY"):
                    color_name = self.classify_hsv(dom_h_deg, dom_s, dom_v)
                t_classify = (time.perf_counter() - t_cls_start) * 1000
                
                confidence = "MEDIUM" if detection_confidence >= 0.5 else "LOW"
                
                logger.info(
                    f"[COLOR_PROFILE] fast=True prep={t_prep:.2f}ms downsample={t_downsample:.2f}ms "
                    f"rgb_extract={t_rgb:.2f}ms hsv_conv={t_hsv:.2f}ms classify={t_classify:.2f}ms"
                )
                logger.info(f"[COLOR] Dominant HSV: ({int(dom_h_deg)}, {int(dom_s * 100)}, {int(dom_v * 100)})")
                logger.info(f"[COLOR] Classified as {color_name.upper().replace('-', '_').replace(' ', '_')}")
                logger.info(f"[COLOR] Confidence: {confidence}")
                
                return {
                    "color_name": color_name,
                    "confidence": confidence,
                    "hsv": (dom_h_deg, dom_s, dom_v)
                }
            else:
                t_ds_start = time.perf_counter()
                with trace_stage("COLOR_DOWNSAMPLE"):
                    resized_crop = crop_image.resize((64, 64))
                t_downsample = (time.perf_counter() - t_ds_start) * 1000
                
                t_rgb_start = time.perf_counter()
                pixels = np.array(resized_crop).reshape(-1, 3)
                t_rgb = (time.perf_counter() - t_rgb_start) * 1000
                
                t_hsv_start = time.perf_counter()
                with trace_stage("COLOR_HSV"):
                    hsv_pixels = self.rgb_to_hsv_numpy(pixels)
                    h, s, v = hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2]
                t_hsv = (time.perf_counter() - t_hsv_start) * 1000
                
                t_cluster_start = time.perf_counter()
                valid_mask = (v >= 0.12) & ~((v > 0.95) & (s < 0.08))
                filtered_rgb = pixels[valid_mask]
                filtered_hsv = hsv_pixels[valid_mask]
                
                if len(filtered_rgb) == 0:
                    filtered_rgb = pixels
                    filtered_hsv = hsv_pixels
                    
                centroids, counts = self.kmeans_numpy(filtered_rgb, k=3)
                t_clustering = (time.perf_counter() - t_cluster_start) * 1000
                
                t_sel_start = time.perf_counter()
                sorted_indices = np.argsort(counts)[::-1]
                centroids = centroids[sorted_indices]
                counts = counts[sorted_indices]
                
                dom_rgb = centroids[0]
                dom_h, dom_s, dom_v = colorsys.rgb_to_hsv(dom_rgb[0] / 255.0, dom_rgb[1] / 255.0, dom_rgb[2] / 255.0)
                dom_h_deg = dom_h * 360.0
                t_selection = (time.perf_counter() - t_sel_start) * 1000
                
                t_cls_start = time.perf_counter()
                with trace_stage("COLOR_CLASSIFY"):
                    color_name = self.classify_hsv(dom_h_deg, dom_s, dom_v)
                t_classify = (time.perf_counter() - t_cls_start) * 1000
                
                total_valid_pixels = len(filtered_rgb)
                dominance = counts[0] / total_valid_pixels if total_valid_pixels > 0 else 0.0
                
                if dom_v > 0.30 and dom_v < 0.85:
                    v_score = 1.0
                elif dom_v > 0.20 and dom_v < 0.92:
                    v_score = 0.7
                else:
                    v_score = 0.4
                    
                is_grayish = (dom_s < 0.15)
                if is_grayish:
                    s_score = 1.0
                else:
                    if dom_s > 0.30:
                        s_score = 1.0
                    elif dom_s > 0.20:
                        s_score = 0.8
                    else:
                        s_score = 0.5
                        
                color_score = dominance * v_score * s_score
                overall_score = color_score * detection_confidence
                
                if overall_score >= 0.55:
                    confidence = "HIGH"
                elif overall_score >= 0.25:
                    confidence = "MEDIUM"
                else:
                    confidence = "LOW"
                    
                logger.info(
                    f"[COLOR_PROFILE] fast=False prep={t_prep:.2f}ms downsample={t_downsample:.2f}ms "
                    f"rgb_extract={t_rgb:.2f}ms hsv_conv={t_hsv:.2f}ms clustering={t_clustering:.2f}ms "
                    f"dominant_sel={t_selection:.2f}ms classify={t_classify:.2f}ms"
                )
                logger.info(f"[COLOR] Dominant HSV: ({int(dom_h_deg)}, {int(dom_s * 100)}, {int(dom_v * 100)})")
                logger.info(f"[COLOR] Classified as {color_name.upper().replace('-', '_').replace(' ', '_')}")
                logger.info(f"[COLOR] Confidence: {confidence}")
                
                return {
                    "color_name": color_name,
                    "confidence": confidence,
                    "hsv": (dom_h_deg, dom_s, dom_v)
                }

    def get_dominant_color(self, crop_image: Image.Image) -> str:
        """Returns the classified color name for backward compatibility."""
        res = self.analyze_color(crop_image)
        return res["color_name"]
