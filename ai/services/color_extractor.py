# ============================================================
# ai/services/color_extractor.py
# يطابق: AIClassifier.extractColors(image): List[Color]
# ============================================================

import colorsys
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from typing import Dict, List


class ColorExtractor:

    def extract(self, image: Image.Image, n_clusters: int = 5) -> List[Dict]:

        arr = np.array(image)
        mask = arr[:, :, 3] > 128

        pixels_rgb = arr[mask][:, :3]

        if len(pixels_rgb) < 50:
            print("  ⚠ لا يكفي بكسلات")
            return []

        sample_size = min(3000, len(pixels_rgb))
        idx = np.random.choice(len(pixels_rgb), sample_size, replace=False)
        sample = pixels_rgb[idx]

        pixels_lab = []
        for px in sample:
            try:
                rgb = sRGBColor(px[0]/255, px[1]/255, px[2]/255)
                lab = convert_color(rgb, LabColor)
                pixels_lab.append([lab.lab_l, lab.lab_a, lab.lab_b])
            except:
                continue

        if len(pixels_lab) < 10:
            return []

        km = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=15,
            max_iter=300
        )

        km.fit(np.array(pixels_lab))

        colors = []
        total = len(km.labels_)

        for i, center in enumerate(km.cluster_centers_):

            weight = (km.labels_ == i).sum() / total

            if weight < 0.02:
                continue

            try:
                lab = LabColor(center[0], center[1], center[2])
                rgb = convert_color(lab, sRGBColor)

                r = int(np.clip(rgb.rgb_r * 255, 0, 255))
                g = int(np.clip(rgb.rgb_g * 255, 0, 255))
                b = int(np.clip(rgb.rgb_b * 255, 0, 255))

                colors.append({
                    "hex": f"#{r:02x}{g:02x}{b:02x}",
                    "weight": round(weight * 100, 1),
                    "lab": [
                        round(float(center[0]), 1),
                        round(float(center[1]), 1),
                        round(float(center[2]), 1)
                    ]
                })

            except:
                continue

        colors = self._merge_similar(colors)
        colors.sort(key=lambda x: x["weight"], reverse=True)

        print(f"  ✓ {len(colors[:5])} لون مستخرج")
        return colors[:5]

    def classify_groups(self, colors: List[Dict]) -> List[Dict]:

        for color in colors:

            hex_c = color["hex"].lstrip("#")

            r = int(hex_c[0:2], 16) / 255.0
            g = int(hex_c[2:4], 16) / 255.0
            b = int(hex_c[4:6], 16) / 255.0

            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            h_deg = h * 360

            if s < 0.10:
                group = "Neutrals"
            elif v < 0.22:
                group = "Darks"
            elif 20 < h_deg < 60 and s < 0.50 and v > 0.50:
                group = "Metallics"
            elif s < 0.32 and v > 0.72:
                group = "Pastels"
            elif s > 0.55 and v > 0.40:
                group = "Brights"
            elif v < 0.38:
                group = "Darks"
            elif s < 0.20:
                group = "Neutrals"
            else:
                group = "Neutrals"

            color["group"] = group

        return colors

    def _delta_e(self, lab1: List, lab2: List) -> float:
        return float(np.sqrt(
            (lab1[0] - lab2[0])**2 +
            (lab1[1] - lab2[1])**2 +
            (lab1[2] - lab2[2])**2
        ))

    def _merge_similar(self, colors: List[Dict], delta_e: float = 8.0) -> List[Dict]:

        merged = []
        used = set()

        for i, c1 in enumerate(colors):

            if i in used:
                continue

            group = [c1]

            for j, c2 in enumerate(colors):
                if j <= i or j in used:
                    continue

                if self._delta_e(c1["lab"], c2["lab"]) < delta_e:
                    group.append(c2)
                    used.add(j)

            total_w = sum(c["weight"] for c in group)
            best = max(group, key=lambda c: c["weight"])

            merged.append({
                "hex": best["hex"],
                "weight": round(total_w, 1),
                "lab": best["lab"]
            })

            used.add(i)

        return merged
