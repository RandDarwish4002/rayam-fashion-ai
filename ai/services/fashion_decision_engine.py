
# ============================================================
# ai/services/fashion_decision_engine.py
# النسخة النهائية المحسّنة - مع إضافة sleeve إلى CLIP_STRONG
# ============================================================

from typing import Dict, List, Optional, Tuple
import re

# ════════════════════════════════════════════════════════════
# 1 — أوزان المصادر (نهائية)
# ════════════════════════════════════════════════════════════

SOURCE_WEIGHTS = {
    "florence": 0.35,
    "clip":     0.45,
    "nlp":      0.20,
}

# عتبات الثقة لكل مصدر
CLIP_THRESHOLDS = {
    "category":     10.0,   # تم تخفيضها من 12 إلى 10
    "sleeve":       12.0,
    "neckline":     12.0,
    "fit":          12.0,
    "pattern":      12.0,
    "occasion":     12.0,
    "season":       12.0,
    "material_look":10.0,
}

# الـ attributes اللي Florence جيد فيها
FLORENCE_STRONG = {
    "neckline", "length", "silhouette", "fit"
}

# الـ attributes اللي CLIP جيد فيها (تمت إضافة sleeve)
CLIP_STRONG = {
    "category", "pattern", "occasion", "season", "material_look", "fit", "sleeve"
}

# الـ attributes اللي NLP يكملها فقط
NLP_FILL_ONLY = {
    "detail", "color"
}

# ════════════════════════════════════════════════════════════
# 2 — قواعد الأزياء (محسّنة)
# ════════════════════════════════════════════════════════════

NLP_ALLOWED_VALUES = {
    "sleeve":   {
        "sleeveless", "short sleeve", "short sleeves",
        "long sleeve", "long sleeves", "3/4 sleeve", "cap sleeves", "cap sleeve"
    },
    "neckline": {
        "crew neck", "v-neck", "turtleneck", "turtle neck",
        "off shoulder", "off-shoulder", "collared", "scoop neck",
        "boat neck", "square neck", "sweetheart", "strapless", "high neck"
    },
    "fit": {
        "slim fit", "regular fit", "oversized", "oversize",
        "tight", "relaxed", "bodycon", "loose", "baggy"
    },
    "pattern": {
        "solid", "striped", "stripes", "floral", "flower",
        "plaid", "checkered", "graphic", "print", "polka dots",
        "geometric", "animal print", "tie-dye", "beaded", "embellished"
    },
    "occasion": {
        "formal", "casual", "sport", "athletic", "party", "evening",
        "outdoor", "beach", "wedding", "cocktail", "business"
    },
    "season": {
        "summer", "winter", "spring", "autumn", "fall", "spring autumn", "spring-autumn", "transitional", "all season"
    },
    "length": {
        "mini", "midi", "maxi", "cropped", "regular", "floor length", "full length", "knee length"
    },
    "silhouette": {
        "ball gown", "a-line", "aline", "mermaid", "trumpet",
        "sheath", "bodycon", "empire", "fit and flare", "princess"
    },
    "color": {
        "black", "white", "red", "blue", "green", "navy",
        "purple", "pink", "yellow", "brown", "beige", 
        "gray", "grey", "gold", "silver", "cream", "ivory"
    }
}

# ════════════════════════════════════════════════════════════
# 3 — Florence Keyword Extractor
# ════════════════════════════════════════════════════════════

FLORENCE_KEYWORDS = {
    "sleeve": {
        "sleeveless": ["sleeveless","no sleeve","strapless","off shoulder","off-the-shoulder",
                       "tank","without sleeve","bare arm", "sweetheart"],
        "short sleeve": ["short sleeve","short-sleeve","cap sleeve","half sleeve","short sleeves", "cap sleeves"],
        "long sleeve": ["long sleeve","long-sleeve","full sleeve","long sleeves","long sleeved"],
        "3/4 sleeve": ["3/4","three-quarter","elbow length","three quarters"],
    },

    "neckline": {
        "turtleneck": ["turtleneck","turtle neck","polo neck","high neck","funnel neck","roll neck"],
        "v-neck": ["v-neck","v neck","v-shaped","deep v"],
        "off shoulder": ["off-shoulder","off shoulder","bardot","off the shoulder"],
        "sweetheart": ["sweetheart","sweetheart neckline","heart-shaped","heart neckline"],
        "strapless": ["strapless","tube top","straight neck"],
        "crew neck": ["crew neck","round neck","crew-neck","round neckline"],
        "collared": ["collar","collared","point collar","button down collar"],
        "scoop neck": ["scoop neck","scoop","deep scoop"],
        "boat neck": ["boat neck","bateau","boatneck"],
        "square neck": ["square neck","square neckline"],
    },

    "category": {
        "dress": ["dress","gown","frock","evening gown","ball gown","maxi dress","mini dress"],
        "t-shirt": ["t-shirt","tee","tshirt","t shirt"],
        "shirt": ["shirt","button-up","blouse","button down","dress shirt"],
        "sweater": ["sweater","knitwear","pullover","jumper","knit","cardigan"],
        "jacket": ["jacket","blazer","bomber","leather jacket"],
        "coat": ["coat","overcoat","trench","winter coat","parka"],
        "jeans": ["jeans","denim","blue jeans"],
        "pants": ["trousers","pants","slacks","dress pants","cargo pants"],
        "skirt": ["skirt","mini skirt","maxi skirt","midi skirt"],
        "hoodie": ["hoodie","sweatshirt","hooded"],
        "shoes": ["shoes","heels","pumps","loafers","flats"],
        "boots": ["boots","ankle boots","knee boots","combat boots"],
        "sneakers": ["sneakers","trainers","athletic shoes","running shoes"],
        "shorts": ["shorts","bermuda shorts"],
    },

    "pattern": {
        "solid": ["solid","plain","single color","solid color","monochrome"],
        "floral": ["floral","flower","flowers","botanical","bloom"],
        "striped": ["stripe","striped","stripes","pinstripe"],
        "plaid": ["plaid","checkered","tartan","check","gingham"],
        "graphic": ["graphic","print","logo","graphic print"],
        "polka dots": ["polka","polka dot","dotted","dots","spot"],
        "geometric": ["geometric","abstract","shapes"],
        "animal print": ["leopard","zebra","snake","animal print","cheetah","tiger"],
        "tie-dye": ["tie-dye","tiedye","tie dye"],
        "beaded": ["beaded","beading","embellished","sequin","sparkle"],
    },

    "length": {
        "mini": ["mini","short","micro","above knee","short dress","mini skirt"],
        "midi": ["midi","mid-length","below knee","midi dress","midi skirt","knee length"],
        "maxi": ["maxi","floor length","full length","long","long dress","long skirt","ankle length"],
        "cropped": ["crop","cropped","cut-off","belly","cropped top"],
    },

    "fit": {
        "slim fit": ["slim fit","tailored fit","skinny fit","fitted","close fitting"],
        "oversized": ["oversized","oversize","baggy","loose","boxy","relaxed fit","wide"],
        "tight": ["bodycon","tight","form-fitting","body hugging","figure hugging"],
        "regular fit": ["regular","classic fit","standard","normal fit"],
    },

    "silhouette": {
        "ball gown": ["ball gown","ballgown","princess gown","princess dress","full skirt"],
        "a-line": ["a-line","aline","a line","fit and flare"],
        "mermaid": ["mermaid","trumpet dress","trumpet","fishtail"],
        "sheath": ["sheath","sheath dress","column"],
        "bodycon": ["bodycon","body con","bandage"],
        "empire": ["empire waist","empire line","high waist"],
        "fit and flare": ["fit and flare","skater","skater dress"],
    }
}


def normalize_text(text: str) -> str:
    """توحيد النص قبل المطابقة"""
    text = text.lower()
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def extract_florence_attrs(description: str) -> Dict:
    """استخراج attributes من وصف Florence"""
    desc = normalize_text(description)
    result = {}

    for attr, keyword_map in FLORENCE_KEYWORDS.items():
        for value, keywords in keyword_map.items():
            if any(normalize_text(kw) in desc for kw in keywords):
                result[attr] = value
                break

    return result


# ════════════════════════════════════════════════════════════
# 4 — NLP Validator
# ════════════════════════════════════════════════════════════

def validate_nlp(nlp_attrs: Dict) -> Dict:
    """يفلتر NLP - أقل صرامة ويحافظ على المعلومات"""
    validated = {}
    
    for attr, val in nlp_attrs.items():
        if not val:
            continue
            
        val_lower = val.lower().strip()
        
        if attr in NLP_ALLOWED_VALUES:
            allowed = NLP_ALLOWED_VALUES[attr]
            matched = None
            
            # محاولة مطابقة تامة أولاً
            if val_lower in allowed:
                matched = val_lower
            else:
                # مطابقة جزئية
                for allowed_val in allowed:
                    if (allowed_val in val_lower or val_lower in allowed_val):
                        matched = allowed_val
                        break
            
            if matched:
                validated[attr] = matched
            else:
                # نحتفظ بالقيمة مع تحذير خفيف
                validated[attr] = val_lower
                
        elif attr in {"category", "material_look", "color"}:
            validated[attr] = val_lower
    
    return validated


# ════════════════════════════════════════════════════════════
# Season Normalization
# ════════════════════════════════════════════════════════════

SEASON_NORMALIZATION = {
    "medium weight spring or autumn clothing": "spring-autumn",
    "spring or autumn mid-weight clothing": "spring-autumn",
    "summer lightweight thin clothing": "summer",
    "winter warm thick clothing": "winter",
    "spring": "spring-autumn",
    "autumn": "spring-autumn",
    "fall": "spring-autumn",
    "spring autumn": "spring-autumn",
    "transitional": "spring-autumn",
    "all season": "all-season",
}


def normalize_season(value):
    if not value:
        return value
    
    val_lower = value.lower()
    for key, mapped in SEASON_NORMALIZATION.items():
        if key in val_lower or val_lower in key:
            return mapped
    
    if "summer" in val_lower or "light" in val_lower:
        return "summer"
    if "winter" in val_lower or "warm" in val_lower or "cold" in val_lower:
        return "winter"
    
    return value


# ════════════════════════════════════════════════════════════
# 5 — Decision Engine (النسخة النهائية)
# ════════════════════════════════════════════════════════════

class FashionDecisionEngine:

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

        # ② فلتر NLP
        n_attrs = {}
        if nlp_attrs:
            n_attrs = validate_nlp(nlp_attrs)
            removed = set(nlp_attrs.keys()) - set(n_attrs.keys())
            if removed:
                log.append(f"NLP: حُذف {removed}")
            log.append(f"NLP validated: {list(n_attrs.keys())}")

        # ③ فلتر CLIP بالثقة
        c_attrs = {}
        for attr, data in clip_result.items():
            conf = data.get("confidence", 0.0)
            thr  = CLIP_THRESHOLDS.get(attr, 12.0)
            if conf >= thr:
                c_attrs[attr] = {
                    "value": data["value"],
                    "conf":  conf
                }
            else:
                log.append(f"CLIP {attr}: {conf:.1f}% < {thr}% threshold")
        
        log.append(f"CLIP مقبول: {list(c_attrs.keys())}")

        # ④ دمج بالأولويات
        final = {}
        all_attrs = set(f_attrs.keys()) | set(c_attrs.keys()) | set(n_attrs.keys())

        for attr in all_attrs:
            f_val = f_attrs.get(attr)
            c_data = c_attrs.get(attr)
            c_val  = c_data["value"] if c_data else None
            c_conf = c_data["conf"]  if c_data else 0.0
            n_val  = n_attrs.get(attr)

            # NLP يملأ فقط
            if attr in NLP_FILL_ONLY:
                if n_val and not f_val and not c_val:
                    final[attr] = n_val
                    log.append(f"{attr}: NLP fill → {n_val}")
                elif f_val:
                    final[attr] = f_val
                elif c_val:
                    final[attr] = c_val
                continue

            # CLIP قوي (العتبة 18% بدل 25%)
            if attr in CLIP_STRONG and c_val:
                if f_val and f_val != c_val:
                    # عتبة أقل - 15% بدل 25%
                    if c_conf >= 15.0:
                        final[attr] = c_val
                        log.append(f"{attr}: CLIP ({c_conf:.1f}%) > Florence ({f_val}) → {c_val}")
                    else:
                        final[attr] = f_val
                        log.append(f"{attr}: Florence → {f_val} (CLIP weak {c_conf:.1f}%)")
                else:
                    final[attr] = c_val
                    log.append(f"{attr}: CLIP ({c_conf:.1f}%) → {c_val}")
            
            # Florence قوي
            elif attr in FLORENCE_STRONG and f_val:
                if c_val and c_val != f_val and c_conf >= 30.0:
                    final[attr] = c_val
                    log.append(f"{attr}: CLIP قوي ({c_conf:.1f}%) → {c_val} (كان {f_val})")
                else:
                    final[attr] = f_val
                    log.append(f"{attr}: Florence → {f_val}")
            
            # NLP كحل أخير
            elif n_val and attr not in final:
                final[attr] = n_val
                log.append(f"{attr}: NLP (fallback) → {n_val}")

        # ⑤ تطبيق قواعد الأزياء
        final = self._apply_rules(final, log)
        
        # Normalize season
        if "season" in final:
            old_season = final["season"]
            final["season"] = normalize_season(final["season"])
            if old_season != final["season"]:
                log.append(f"Season normalized: {old_season} → {final['season']}")

        # ⑥ إضافة color إذا موجود
        if "color" in n_attrs and "color" not in final:
            final["color"] = n_attrs["color"]
            log.append(f"color: added from NLP → {n_attrs['color']}")

        return {
            "final_attributes": final,
            "confidence": {
                k: v.get("confidence", 0.0)
                for k, v in clip_result.items()
            },
            "decisions_log": log,
        }

    def _apply_rules(self, attrs: Dict, log: List) -> Dict:
        """
        تطبيق قواعد الأزياء
        المعدل: لا يفرض sleeve إذا كان موجوداً بالفعل
        """
        fixed = attrs.copy()
        neckline = fixed.get("neckline", "").lower()

        # off-shoulder ⇒ sleeveless (فقط إذا لم يكن sleeve محدد)
        if "off shoulder" in neckline or "off-the-shoulder" in neckline:
            if "sleeve" not in fixed:
                fixed["sleeve"] = "sleeveless"
                log.append("Rule: off-shoulder → sleeveless (no sleeve defined)")

        # strapless ⇒ sleeveless (فقط إذا لم يكن sleeve محدد)
        if "strapless" in neckline:
            if "sleeve" not in fixed:
                fixed["sleeve"] = "sleeveless"
                log.append("Rule: strapless → sleeveless (no sleeve defined)")

        # sweetheart ⇒ sleeveless (فقط إذا لم يكن sleeve محدد)
        if "sweetheart" in neckline:
            if "sleeve" not in fixed:
                fixed["sleeve"] = "sleeveless"
                log.append("Rule: sweetheart → sleeveless (no sleeve defined)")

        return fixed
