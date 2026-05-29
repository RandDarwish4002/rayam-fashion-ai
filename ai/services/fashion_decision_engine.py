
# ============================================================
# ai/services/fashion_decision_engine.py
# طبقة القرار الذكي فوق Florence-2 + CLIP
# ============================================================

from typing import Dict, List, Optional, Tuple

# ════════════════════════════════════════════════════════════
# 1 — قواعد الأزياء المنطقية
# ════════════════════════════════════════════════════════════

# قواعد الإلزام: لو X إذن Y لازم يكون كذا
FASHION_RULES = {
    "sleeve_from_neckline": {
        "off shoulder":           "sleeveless",
        "off shoulder neckline":  "sleeveless",
        "strapless":              "sleeveless",
    },
    "sleeve_from_category": {
        "a dress":                None,   # حر
        "a skirt":                None,   # لا علاقة
        "trousers or dress pants":None,
        "jeans or denim pants":   None,
        "shorts":                 None,
        "shoes or leather shoes": None,
        "sneakers or athletic shoes": None,
        "boots":                  None,
        "sandals or flip flops":  None,
    },
    "neckline_conflicts": {
        # turtleneck لا يتوافق مع
        "turtleneck or high neck": [
            "off shoulder",
            "v-neck",
            "scoop neck",
            "square neckline",
        ],
        "high turtleneck covering the neck": [
            "off shoulder",
            "v-neck",
        ]
    },
    "category_sleeve_map": {
        # فئات لا تحتاج sleeve
        "no_sleeve_needed": {
            "shoes or leather shoes",
            "sneakers or athletic shoes",
            "boots", "sandals or flip flops",
            "trousers or dress pants",
            "jeans or denim pants",
            "shorts", "a skirt",
        }
    }
}

# عتبات الثقة
CONFIDENCE_THRESHOLDS = {
    "category":     40.0,   # أهم attribute
    "sleeve":       35.0,
    "neckline":     30.0,
    "fit":          25.0,
    "pattern":      25.0,
    "occasion":     30.0,
    "season":       30.0,
    "material_look":20.0,
}

# أوزان Florence مقابل CLIP
FLORENCE_WEIGHT = 0.60
CLIP_WEIGHT     = 0.40


# ════════════════════════════════════════════════════════════
# 2 — مستخلص Florence
# ════════════════════════════════════════════════════════════

FLORENCE_KEYWORDS = {
    "sleeve": {
        "sleeveless":    ["sleeveless","no sleeve","without sleeve","strapless","tank"],
        "short sleeve":  ["short sleeve","short-sleeve","cap sleeve","half sleeve"],
        "long sleeve":   ["long sleeve","long-sleeve","full sleeve","full-length sleeve"],
        "3/4 sleeve":    ["three-quarter","3/4","elbow length"],
    },
    "neckline": {
        "turtleneck or high neck": ["turtleneck","high neck","polo neck","funnel neck"],
        "v-neck":                  ["v-neck","v neck","v-shaped"],
        "off shoulder":            ["off shoulder","off-shoulder","bardot"],
        "crew neck or round neck": ["crew neck","round neck","crew-neck"],
        "collared or polo collar": ["collar","collared","polo"],
        "square neckline":         ["square neck","square neckline"],
        "scoop neck":              ["scoop","scoop neck"],
    },
    "category": {
        "a dress":                      ["dress","gown","maxi dress","mini dress"],
        "a t-shirt or tee":             ["t-shirt","tee","tshirt"],
        "a dress shirt or button-up":   ["shirt","button","dress shirt"],
        "a sweater or knitwear":        ["sweater","knitwear","knit","pullover","jumper"],
        "a jacket or blazer":           ["jacket","blazer","suit jacket"],
        "a coat or overcoat":           ["coat","overcoat","trench"],
        "jeans or denim pants":         ["jeans","denim"],
        "trousers or dress pants":      ["trousers","pants","slacks"],
        "a skirt":                      ["skirt"],
        "a hoodie or sweatshirt":       ["hoodie","sweatshirt"],
        "boots":                        ["boots","boot"],
        "sneakers or athletic shoes":   ["sneakers","sneaker","trainers"],
        "shoes or leather shoes":       ["shoes","loafers","oxfords"],
    },
    "fit": {
        "slim fit":      ["slim","fitted","tailored","skinny"],
        "oversized":     ["oversized","baggy","loose","wide"],
        "regular fit":   ["regular","standard","classic fit"],
        "tight":         ["tight","bodycon","form-fitting"],
    },
    "pattern": {
        "solid color":   ["solid","plain","single color","monochrome"],
        "striped":       ["stripe","striped","stripes"],
        "floral":        ["floral","flower","flowers","botanical"],
        "plaid":         ["plaid","checkered","tartan","check"],
        "graphic print": ["graphic","print","logo","printed"],
    }
}

def extract_from_florence(
        description: str
) -> Dict[str, Optional[str]]:
    """
    يستخرج attributes من وصف Florence النصي
    """
    desc  = description.lower()
    found = {}

    for attr, keyword_map in FLORENCE_KEYWORDS.items():
        for value, keywords in keyword_map.items():
            if any(kw in desc for kw in keywords):
                found[attr] = value
                break

    return found


# ════════════════════════════════════════════════════════════
# 3 — Conflict Detector
# ════════════════════════════════════════════════════════════

def detect_conflicts(
        attrs: Dict[str, str]
) -> List[str]:
    """
    يكتشف التناقضات في الـ attributes
    """
    conflicts = []
    neckline  = attrs.get("neckline", "")
    sleeve    = attrs.get("sleeve", "")
    category  = attrs.get("category", "")

    # تضارب: off-shoulder مع أي كم
    if ("off shoulder" in neckline
            and "sleeveless" not in sleeve):
        conflicts.append(
            f"off-shoulder ⇒ يجب أن يكون sleeveless"
            f" لكن وجدنا: {sleeve}"
        )

    # تضارب: turtleneck مع off-shoulder
    if ("turtleneck" in neckline
            and "off shoulder" in neckline):
        conflicts.append("turtleneck لا يمكن مع off-shoulder")

    return conflicts


# ════════════════════════════════════════════════════════════
# 4 — Rule Fixer
# ════════════════════════════════════════════════════════════

def apply_fashion_rules(
        attrs: Dict[str, str]
) -> Dict[str, str]:
    """
    يصحح التناقضات تلقائياً
    """
    fixed = attrs.copy()

    # Rule 1: off-shoulder ⇒ sleeveless
    neckline = fixed.get("neckline", "")
    if "off shoulder" in neckline:
        fixed["sleeve"] = "sleeveless clothing, no sleeves"

    # Rule 2: turtleneck ⇒ لا v-neck
    if "turtleneck" in neckline:
        if "v-neck" in neckline:
            fixed["neckline"] = "turtleneck or high neck"

    # Rule 3: فئات لا تحتاج sleeve
    category = fixed.get("category", "")
    no_sleeve_cats = {
        "shoes", "sneakers", "boots", "sandals",
        "trousers", "jeans", "shorts", "skirt"
    }
    if any(c in category for c in no_sleeve_cats):
        fixed["sleeve"] = "not applicable"

    return fixed


# ════════════════════════════════════════════════════════════
# 5 — Confidence Gate
# ════════════════════════════════════════════════════════════

def gate_by_confidence(
        clip_result: Dict
) -> Tuple[Dict[str, str], Dict[str, float], List[str]]:
    """
    يفلتر النتائج بناءً على عتبة الثقة
    """
    accepted  = {}
    rejected  = []
    conf_vals = {}

    for attr, data in clip_result.items():
        value = data.get("value", "")
        conf  = data.get("confidence", 0.0)
        thr   = CONFIDENCE_THRESHOLDS.get(attr, 25.0)

        conf_vals[attr] = conf

        if conf >= thr:
            accepted[attr] = value
        else:
            rejected.append(
                f"{attr}: {conf:.1f}% < {thr}% threshold"
            )

    return accepted, conf_vals, rejected


# ════════════════════════════════════════════════════════════
# 6 — الدالة الرئيسية — Fusion + Decision
# ════════════════════════════════════════════════════════════

class FashionDecisionEngine:
    """
    يطابق الفكرة: Fashion Decision Engine
    يجمع Florence + CLIP ويقرر النهائي
    """

    def decide(
            self,
            florence_desc: str,
            clip_result:   Dict,
    ) -> Dict:
        """
        المدخل:  وصف Florence + نتائج CLIP
        المخرج:  attributes نهائية + تقرير القرارات
        """

        decisions_log = []

        # ① استخراج من Florence
        florence_attrs = extract_from_florence(florence_desc)
        decisions_log.append(
            f"Florence استخرج: {list(florence_attrs.keys())}"
        )

        # ② فلترة CLIP بالثقة
        clip_accepted, conf_vals, rejected = \
            gate_by_confidence(clip_result)

        if rejected:
            decisions_log.append(
                f"CLIP مرفوض لضعف الثقة: {rejected}"
            )

        # ③ دمج Florence + CLIP (Florence أولوية)
        final_attrs = {}

        all_attrs = set(
            list(florence_attrs.keys()) +
            list(clip_accepted.keys())
        )

        for attr in all_attrs:
            f_val = florence_attrs.get(attr)
            c_val = clip_accepted.get(attr)
            c_conf = conf_vals.get(attr, 0.0)

            if f_val and c_val:
                # الاثنان موجودان — قارن
                if f_val == c_val:
                    final_attrs[attr] = f_val
                    decisions_log.append(
                        f"{attr}: اتفاق ✓ → {f_val}"
                    )
                else:
                    # Florence أولوية لو الـ CLIP ضعيف
                    if c_conf < 50.0:
                        final_attrs[attr] = f_val
                        decisions_log.append(
                            f"{attr}: تعارض → "
                            f"Florence ({f_val}) "
                            f"فاز على CLIP ({c_val}, "
                            f"{c_conf:.1f}%)"
                        )
                    else:
                        # CLIP قوي — خذ الاثنين وقرر
                        final_attrs[attr] = c_val
                        decisions_log.append(
                            f"{attr}: تعارض → "
                            f"CLIP ({c_val}, {c_conf:.1f}%) "
                            f"فاز على Florence ({f_val})"
                        )

            elif f_val:
                final_attrs[attr] = f_val
                decisions_log.append(
                    f"{attr}: Florence فقط → {f_val}"
                )

            elif c_val:
                final_attrs[attr] = c_val
                decisions_log.append(
                    f"{attr}: CLIP فقط → "
                    f"{c_val} ({c_conf:.1f}%)"
                )

        # ④ اكتشاف التناقضات
        conflicts = detect_conflicts(final_attrs)
        if conflicts:
            decisions_log.append(
                f"تناقضات مكتشفة: {conflicts}"
            )

        # ⑤ تصحيح بقواعد الأزياء
        final_attrs = apply_fashion_rules(final_attrs)
        decisions_log.append("✓ تطبيق قواعد الأزياء")

        # ⑥ أضف ما تبقى من CLIP المقبول
        for attr, val in clip_accepted.items():
            if attr not in final_attrs:
                final_attrs[attr] = val

        return {
            "final_attributes": final_attrs,
            "confidence":       conf_vals,
            "decisions_log":    decisions_log,
            "florence_found":   florence_attrs,
            "clip_accepted":    clip_accepted,
            "clip_rejected":    rejected,
        }
