# ============================================================
# ai/classifiers/skin_analyzer.py
# يطابق: UserProfile.getSuitableColors()
#         UserProfile.updateMeasurements()
# ============================================================

import cv2
import numpy as np
from typing import Dict, List


SKIN_RULES: Dict[str, Dict] = {
    "Type I — فاتح جداً": {
        "suitable": [
            "#FFFFFF","#F5F5DC","#FFB6C1","#B0E0E6",
            "#98FB98","#DDA0DD","#E6E6FA","#FFDAB9",
            "#E0FFFF","#FFC0CB","#B0C4DE","#FFFACD",
        ],
        "unsuitable": [
            "#FF0000","#FF6600","#FFD700","#FF4500",
        ],
        "description": "البشرة الفاتحة جداً تناسب الألوان الباردة والباستيل"
    },
    "Type II — فاتح": {
        "suitable": [
            "#FFFFFF","#F5F5DC","#E6E6FA","#B0C4DE",
            "#90EE90","#FFB6C1","#FFFACD","#E0FFFF",
        ],
        "unsuitable": [
            "#FF6600","#8B0000","#FF4500","#A52A2A",
        ],
        "description": "البشرة الفاتحة تناسب الألوان الهادئة والمتوسطة"
    },
    "Type III — متوسط فاتح": {
        "suitable": [
            "#FFFFFF","#000080","#006400","#8B0000",
            "#4B0082","#2F4F4F","#191970","#003366",
        ],
        "unsuitable": [
            "#FF6600","#FFFF00","#FFD700","#F0E68C",
        ],
        "description": "البشرة الزيتونية الفاتحة تناسب الألوان الداكنة والعميقة"
    },
    "Type IV — زيتوني": {
        "suitable": [
            "#FFFFFF","#000000","#00008B","#8B0000",
            "#006400","#4B0082","#FF6600","#DC143C",
        ],
        "unsuitable": [
            "#FFD700","#8B4513","#D2691E","#DEB887",
        ],
        "description": "البشرة الزيتونية تناسب الألوان الزاهية والداكنة"
    },
    "Type V — بني": {
        "suitable": [
            "#FFFFFF","#000000","#FF0000","#FFD700",
            "#006400","#00008B","#FF6600","#4169E1",
        ],
        "unsuitable": [
            "#D2691E","#8B4513","#A0522D","#CD853F",
        ],
        "description": "البشرة البنية تناسب الألوان الجريئة والزاهية"
    },
    "Type VI — داكن جداً": {
        "suitable": [
            "#FFFFFF","#FFD700","#FF0000","#FF6600",
            "#00CED1","#FF1493","#7FFF00","#EE82EE",
        ],
        "unsuitable": [
            "#000000","#1C1C1C","#2F2F2F","#191919",
        ],
        "description": "البشرة الداكنة تناسب الألوان الفاقعة والمضيئة"
    },
}


class SkinAnalyzer:
    """
    يطابق Class Diagram:
    - UserProfile.getSuitableColors()
    - UserProfile.updateMeasurements() (جزء منها)
    """

    def analyze(self, skin_hex: str) -> Dict:
        """
        المدخل:  hex color للبشرة
        المخرج:  skin_type + suitable/unsuitable colors
        """
        hex_c = skin_hex.lstrip('#')
        r = int(hex_c[0:2], 16) / 255.0
        g = int(hex_c[2:4], 16) / 255.0
        b = int(hex_c[4:6], 16) / 255.0

        # RGB → Lab
        rgb_arr = np.uint8([[[
            int(r*255), int(g*255), int(b*255)
        ]]])
        lab_arr = cv2.cvtColor(
            rgb_arr, cv2.COLOR_RGB2LAB
        )[0][0]

        L     = float(lab_arr[0]) * 100.0 / 255.0
        b_val = float(lab_arr[2]) - 128.0

        ITA = np.arctan(
            (L - 50.0) / (b_val + 1e-8)
        ) * (180.0 / np.pi)

        if   ITA > 55:  skin_type = "Type I — فاتح جداً"
        elif ITA > 41:  skin_type = "Type II — فاتح"
        elif ITA > 28:  skin_type = "Type III — متوسط فاتح"
        elif ITA > 10:  skin_type = "Type IV — زيتوني"
        elif ITA > -30: skin_type = "Type V — بني"
        else:           skin_type = "Type VI — داكن جداً"

        rules = SKIN_RULES[skin_type]

        return {
            "skin_hex":          skin_hex,
            "ita_value":         round(float(ITA), 2),
            "skin_type":         skin_type,
            "description":       rules["description"],
            "suitable_colors":   rules["suitable"],
            "unsuitable_colors": rules["unsuitable"],
        }

    def is_color_suitable(
            self,
            clothing_hex: str,
            skin_result:  Dict
    ) -> bool:
        """
        يتحقق هل لون القطعة مناسب للبشرة
        """
        unsuitable = [
            c.upper()
            for c in skin_result["unsuitable_colors"]
        ]
        return clothing_hex.upper() not in unsuitable
