
# ============================================================
# ai/services/fashion_decision_engine.py
# نسخة متوازنة — Florence + CLIP + NLP بأوزان صحيحة
# ============================================================

from typing import Dict, List, Optional, Tuple

# ════════════════════════════════════════════════════════════
# 1 — أوزان المصادر
# ════════════════════════════════════════════════════════════

SOURCE_WEIGHTS = {
    "florence": 0.40,   # الوصف الغني — الأساس
    "clip":     0.35,   # التصنيف البصري المباشر
    "nlp":      0.25,   # تكملة فقط — لا يتحكم
}

# عتبات الثقة لكل مصدر
CLIP_THRESHOLDS = {
    "category":     40.0,
    "sleeve":       30.0,
    "neckline":     28.0,
    "fit":          25.0,
    "pattern":      25.0,
    "occasion":     30.0,
    "season":       28.0,
    "material_look":20.0,
}

# الـ attributes اللي Florence جيد فيها
FLORENCE_STRONG = {
    "category", "neckline", "sleeve",
    "pattern", "length", "fit"
}

# الـ attributes اللي CLIP جيد فيها
CLIP_STRONG = {
    "category", "pattern", "occasion",
    "season", "material_look"
}

# الـ attributes اللي NLP يكملها فقط
NLP_FILL_ONLY = {
    "length", "silhouette", "detail"
}

# ════════════════════════════════════════════════════════════
# 2 — قواعد الأزياء
# ════════════════════════════════════════════════════════════

FASHION_RULES = {
    "off shoulder":  {"sleeve": "sleeveless"},
    "strapless":     {"sleeve": "sleeveless"},
    "turtleneck":    {"neckline_block": ["v-neck","off shoulder","scoop neck"]},
}

# قيم NLP المسموحة فقط (لمنع الهلوسة)
NLP_ALLOWED_VALUES = {
    "sleeve":   {
        "sleeveless","short sleeve",
        "long sleeve","3/4 sleeve","cap sleeve"
    },
    "neckline": {
        "crew neck","v-neck","turtleneck",
        "off shoulder","collared","scoop neck",
        "boat neck","square neck","sweetheart","strapless"
    },
    "fit": {
        "slim fit","regular fit","oversized",
        "tight","relaxed"
    },
    "pattern": {
        "solid","striped","floral","plaid",
        "graphic","polka dots","geometric",
        "animal print","tie-dye"
    },
    "occasion": {
        "formal","casual","sport","party",
        "outdoor","beach"
    },
    "season": {
        "summer","winter","spring-autumn"
    },
    "length": {
        "mini","midi","maxi","cropped","regular"
    },
}

# ════════════════════════════════════════════════════════════
# 3 — Florence Keyword Extractor
# ════════════════════════════════════════════════════════════

FLORENCE_KEYWORDS = {
    "sleeve": {
        "sleeveless": ["sleeveless","no sleeve","strapless",
                       "tank","without sleeve","bare arm"
        ],
        "short sleeve": [
            "short sleeve","short-sleeve","cap sleeve", "half sleeve"],
        "long sleeve": ["long sleeve","long-sleeve", "full sleeve", "long sleeves"
        ],
        "3/4 sleeve": ["3/4","three-quarter","elbow length"],
    },

    "neckline": {
        "turtleneck": [
            "turtleneck","polo neck",
            "high neck","funnel neck","roll neck"],
        "v-neck": ["v-neck", "v neck","v-shaped"],
        "off shoulder": ["off-shoulder","off shoulder","bardot"],
        "sweetheart": [
            "sweetheart","sweetheart neckline","sweetheart-shaped neckline",
            "heart-shaped neckline", "heart neckline"],
        "strapless": ["strapless","tube top"],
        "crew neck": ["crew neck","round neck","crew-neck"],
        "collared": [ "collar", "collared"],
        "scoop neck": ["scoop neck", "scoop"],
        "boat neck": ["boat neck","bateau" ],
        "square neck": ["square neck","square neckline"],
    },

    "category": {
        "dress": ["dress","gown","frock"],
        "t-shirt": ["t-shirt","tee","tshirt"],
        "shirt": ["shirt","button-up","blouse"],
        "sweater": ["sweater","knitwear","pullover","jumper","knit"],
        "jacket": ["jacket","blazer"],
        "coat": ["coat","overcoat","trench"],
        "jeans": ["jeans","denim"],
        "pants": ["trousers","pants","slacks"],
        "skirt": ["skirt"],
        "hoodie": ["hoodie","sweatshirt"],
        "shoes": ["shoes","heels","pumps","loafers"],
        "boots": ["boots"],
        "sneakers": ["sneakers","trainers"],
        "shorts": ["shorts"],
    },

    "pattern": {
        "solid": ["solid","plain","single color"],
        "floral": ["floral","flower","flowers"],
        "striped": ["stripe", "striped"],
        "plaid": ["plaid","checkered","tartan"],
        "graphic": ["graphic","print","logo"],
    },

    "length": {
        "mini": ["mini","short dress","micro"],
        "midi": ["midi","knee","mid-length"],
        "maxi": ["maxi","floor length","full length"],
        "cropped": ["crop","cropped"],
    },

    "fit": {
        "slim fit": ["slim fit","tailored fit","skinny fit"],
        "oversized": ["oversized","baggy","loose","boxy"],
        "tight": ["bodycon","tight","form-fitting"],
    },

    "silhouette": {
        "ball gown": ["ball gown","princess gown"],
        "a-line": ["a-line","aline" ],
        "mermaid": ["mermaid","trumpet dress"],
        "sheath": ["sheath"],
        "bodycon": ["bodycon"],
        "empire": ["empire waist"],
        "fit and flare": ["fit and flare"]
    }
}

def extract_florence_attrs(description: str) -> Dict:
    desc   = description.lower()
    result = {}
    for attr, keyword_map in FLORENCE_KEYWORDS.items():
        for value, keywords in keyword_map.items():
            if any(kw in desc for kw in keywords):
                result[attr] = value
                break
    return result


# ════════════════════════════════════════════════════════════
# 4 — NLP Validator
# ════════════════════════════════════════════════════════════

def validate_nlp(nlp_attrs: Dict) -> Dict:
    """
    يفلتر NLP ويحذف القيم الغير standard أو الهلوسة
    """
    validated = {}
    for attr, val in nlp_attrs.items():
        if attr in NLP_ALLOWED_VALUES:
            allowed = NLP_ALLOWED_VALUES[attr]
            # تحقق تطابق مباشر أو جزئي
            matched = None
            for allowed_val in allowed:
                if (allowed_val in val.lower()
                        or val.lower() in allowed_val):
                    matched = allowed_val
                    break
            if matched:
                validated[attr] = matched
            # لو ما تطابق — تجاهل (لا تضيف)
        elif attr in {"category", "material_look"}:
            # هذه حر — قبلها كما هي
            validated[attr] = val

    return validated


# ════════════════════════════════════════════════════════════
# 5 — Decision Engine
# ════════════════════════════════════════════════════════════

class FashionDecisionEngine:
    """
    موزون: Florence (40%) + CLIP (35%) + NLP (25%)
    """

    def decide(
            self,
            florence_desc: str,
            clip_result:   Dict,
            nlp_attrs:     Optional[Dict] = None,
    ) -> Dict:

        log = []

        # ① استخرج من Florence
        f_attrs = extract_florence_attrs(florence_desc)
        log.append(f"Florence استخرج: {list(f_attrs.keys())}")

        # ② فلتر NLP (لا تقبل هلوسة)
        n_attrs = {}
        if nlp_attrs:
            n_attrs = validate_nlp(nlp_attrs)
            removed = set(nlp_attrs) - set(n_attrs)
            if removed:
                log.append(f"NLP: حُذف (غير standard): {removed}")
            log.append(f"NLP validated: {list(n_attrs.keys())}")

        # ③ فلتر CLIP بالثقة
        c_attrs = {}
        for attr, data in clip_result.items():
            conf = data.get("confidence", 0.0)
            thr  = CLIP_THRESHOLDS.get(attr, 25.0)
            if conf >= thr:
                c_attrs[attr] = {
                    "value": data["value"],
                    "conf":  conf
                }

        log.append(f"CLIP قبل: {list(c_attrs.keys())}")

        # ④ دمج بالأولويات
        final = {}
        all_attrs = set(
            list(f_attrs.keys()) +
            list(c_attrs.keys()) +
            list(n_attrs.keys())
        )

        for attr in all_attrs:
            f_val = f_attrs.get(attr)
            c_data = c_attrs.get(attr)
            c_val  = c_data["value"] if c_data else None
            c_conf = c_data["conf"]  if c_data else 0.0
            n_val  = n_attrs.get(attr)

            # ── منطق الأولوية ──────────────────────────

            # NLP يملأ فقط ما لم يجده Florence وCLIP
            if attr in NLP_FILL_ONLY:
                if not f_val and not c_val and n_val:
                    final[attr] = n_val
                    log.append(f"{attr}: NLP fill → {n_val}")
                elif f_val:
                    final[attr] = f_val
                elif c_val:
                    final[attr] = c_val
                continue

            # Florence قوي في هذا الـ attribute
            if attr in FLORENCE_STRONG and f_val:
                if c_val and c_conf >= 60.0 and c_val != f_val:
                    # CLIP واثق جداً ويختلف — خذ CLIP
                    final[attr] = c_val
                    log.append(
                        f"{attr}: CLIP قوي ({c_conf:.0f}%)"
                        f" فاز على Florence"
                        f" ({f_val} → {c_val})"
                    )
                else:
                    # Florence يفوز
                    final[attr] = f_val
                    log.append(
                        f"{attr}: Florence → {f_val}"
                    )

            # CLIP قوي في هذا الـ attribute
            elif attr in CLIP_STRONG and c_val:
                if f_val and f_val != c_val:
                    # Florence موجود — قارن
                    if c_conf >= 55.0:
                        final[attr] = c_val
                        log.append(
                            f"{attr}: CLIP ({c_conf:.0f}%)"
                            f" → {c_val}"
                        )
                    else:
                        final[attr] = f_val
                        log.append(
                            f"{attr}: Florence → {f_val}"
                            f" (CLIP ضعيف {c_conf:.0f}%)"
                        )
                else:
                    final[attr] = c_val
                    log.append(
                        f"{attr}: CLIP ({c_conf:.0f}%)"
                        f" → {c_val}"
                    )

            # NLP كآخر خيار
            elif n_val and attr not in final:
                final[attr] = n_val
                log.append(
                    f"{attr}: NLP (fallback) → {n_val}"
                )

        # ⑤ تطبيق قواعد الأزياء
        final = self._apply_rules(final, log)

        return {
            "final_attributes": final,
            "confidence": {
                k: v.get("confidence", 0.0)
                for k, v in clip_result.items()
            },
            "decisions_log": log,
        }

    def _apply_rules(
            self,
            attrs: Dict,
            log:   List
    ) -> Dict:
        fixed    = attrs.copy()
        neckline = fixed.get("neckline", "")

        # off-shoulder ⇒ sleeveless
        if "off shoulder" in neckline:
            if fixed.get("sleeve") != "sleeveless":
                fixed["sleeve"] = "sleeveless"
                log.append(
                    "Rule: off-shoulder ⇒ sleeve=sleeveless"
                )

        # strapless ⇒ sleeveless
        if "strapless" in neckline:
            fixed["sleeve"] = "sleeveless"
            log.append("Rule: strapless ⇒ sleeve=sleeveless")

        # turtleneck يحذف necklines متعارضة
        if "turtleneck" in neckline:
            blocked = ["v-neck","off shoulder","scoop"]
            for b in blocked:
                if b in neckline:
                    fixed["neckline"] = "turtleneck"
                    log.append(
                        f"Rule: turtleneck ⇒ حذف {b}"
                    )

        return fixed
