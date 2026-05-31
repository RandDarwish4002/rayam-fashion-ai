
# ============================================================
# ai/services/fashion_nlp_extractor.py
# NLP Extractor محسّن - بين Florence وCLIP
# ============================================================

import json
import re
import torch
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


# ════════════════════════════════════════════════════════════
# 1 — Prompt محسّن للنموذج
# ════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """You are a fashion attribute extractor.
Extract these attributes from the clothing description.
Return ONLY valid JSON.

Attributes:
- category: (dress/evening gown/t-shirt/shirt/blouse/pants/jeans/skirt/jacket/coat/sweater/hoodie/shoes/boots/sneakers/shorts/cardigan/jumpsuit)
- sleeve: (sleeveless/short sleeve/long sleeve/3/4 sleeve/cap sleeves)
- neckline: (crew neck/v-neck/turtleneck/off shoulder/collared/scoop neck/boat neck/square neck/sweetheart/strapless)
- fit: (slim fit/regular fit/oversized/bodycon/relaxed)
- pattern: (solid/floral/striped/plaid/graphic/polka dots/geometric/animal print/beaded)
- occasion: (formal/casual/party/sport/outdoor/beach/wedding)
- season: (summer/winter/spring-autumn/all-season)
- length: (mini/midi/maxi/cropped/regular)
- silhouette: (ball gown/a-line/mermaid/sheath/bodycon/empire/fit and flare)
- color: (black/white/red/blue/green/purple/pink/yellow/brown/beige/gray/gold/silver)

Description: {description}

JSON:"""


# ════════════════════════════════════════════════════════════
# 2 — Rule-Based (محسّن مع مرادفات أكثر)
# ════════════════════════════════════════════════════════════

RULE_KEYWORDS = {
    "category": {
        "evening gown": ["evening gown","ball gown","gala dress","red carpet","formal gown"],
        "dress":        ["dress","frock","maxi dress","mini dress","midi dress"],
        "t-shirt":      ["t-shirt","tee","tshirt","t shirt"],
        "shirt":        ["shirt","button-up","dress shirt","blouse"],
        "blouse":       ["blouse","silk blouse"],
        "pants":        ["pants","trousers","slacks","dress pants"],
        "jeans":        ["jeans","denim","blue jeans"],
        "skirt":        ["skirt","mini skirt","maxi skirt"],
        "jacket":       ["jacket","blazer","suit jacket"],
        "coat":         ["coat","overcoat","trench","winter coat"],
        "sweater":      ["sweater","knitwear","pullover","jumper","knit"],
        "hoodie":       ["hoodie","sweatshirt","hooded"],
        "shoes":        ["shoes","loafers","oxfords","heels","pumps"],
        "boots":        ["boots","ankle boots","knee boots"],
        "sneakers":     ["sneakers","trainers","athletic shoes","running shoes"],
        "shorts":       ["shorts","bermuda shorts"],
        "cardigan":     ["cardigan","knit cardigan"],
        "jumpsuit":     ["jumpsuit","romper","onesie"],
    },
    "sleeve": {
        "sleeveless":   ["sleeveless","no sleeve","strapless","tank","without sleeve","bare arms"],
        "short sleeve": ["short sleeve","short-sleeve","half sleeve","short sleeves"],
        "long sleeve":  ["long sleeve","long-sleeve","full sleeve","long sleeves","full length sleeve"],
        "3/4 sleeve":   ["3/4","three-quarter","three quarter","elbow length"],
        "cap sleeves":  ["cap sleeve","cap sleeves","short cap"],
    },
    "neckline": {
        "turtleneck":   ["turtleneck","turtle neck","polo neck","high neck","funnel neck","roll neck"],
        "v-neck":       ["v-neck","v neck","v-shaped","deep v"],
        "off shoulder": ["off-shoulder","off shoulder","bardot","off the shoulder","cold shoulder"],
        "sweetheart":   ["sweetheart","sweetheart neckline","heart-shaped","heart neckline"],
        "strapless":    ["strapless","tube top","straight neck"],
        "crew neck":    ["crew neck","round neck","crew-neck","round neckline"],
        "collared":     ["collar","collared","point collar","button down collar","shirt collar"],
        "scoop neck":   ["scoop","scoop neck","deep scoop"],
        "boat neck":    ["boat neck","bateau","boatneck"],
        "square neck":  ["square neck","square neckline"],
    },
    "fit": {
        "slim fit":     ["slim fit","tailored fit","skinny fit","fitted","close fitting"],
        "regular fit":  ["regular","classic fit","standard","normal fit"],
        "oversized":    ["oversized","oversize","baggy","loose","boxy","wide","relaxed"],
        "bodycon":      ["bodycon","tight","form-fitting","body hugging","figure hugging"],
        "relaxed":      ["relaxed","easy fit","comfortable"],
    },
    "pattern": {
        "solid":        ["solid","plain","single color","monochrome","solid color"],
        "floral":       ["floral","flower","flowers","botanical","bloom","floral print"],
        "striped":      ["stripe","striped","stripes","pinstripe","horizontal stripe"],
        "plaid":        ["plaid","checkered","tartan","check","gingham","buffalo check"],
        "graphic":      ["graphic","print","logo","printed","text","graphic print"],
        "polka dots":   ["polka","polka dot","dotted","dots","spot"],
        "geometric":    ["geometric","abstract","shapes","patterned"],
        "animal print": ["leopard","zebra","snake","animal print","cheetah","tiger"],
        "beaded":       ["beaded","beading","embellished","sequin","sequins","sparkle","jeweled"],
        "tie-dye":      ["tie-dye","tiedye","tie dye"],
    },
    "occasion": {
        "formal":       ["formal","business","office","professional","work","gala","evening","wedding","bridal","red carpet"],
        "casual":       ["casual","everyday","relaxed","weekend","daily"],
        "party":        ["party","night out","cocktail","club","festive","celebration"],
        "sport":        ["sport","athletic","gym","workout","activewear","running","sports"],
        "outdoor":      ["outdoor","hiking","camping","adventure","travel"],
        "beach":        ["beach","swim","swimwear","tropical","resort","vacation"],
    },
    "season": {
        "summer":       ["summer","lightweight","light","thin","hot weather","tropical","beach"],
        "winter":       ["winter","warm","thick","heavy","cold","wool","fleece","cozy"],
        "spring-autumn":["spring","autumn","fall","mid-weight","transitional","mild"],
        "all-season":   ["all season","all year","versatile","every season"],
    },
    "length": {
        "mini":         ["mini","short dress","short skirt","micro","above knee"],
        "midi":         ["midi","mid-length","below knee","mid-calf","knee length"],
        "maxi":         ["maxi","floor length","full length","long dress","long skirt","ankle length","floor-length"],
        "cropped":      ["crop","cropped","cut-off","belly","cropped top"],
        "regular":      ["regular","standard","normal length"],
    },
    "silhouette": {
        "ball gown":    ["ball gown","ballgown","princess gown","princess dress","full skirt","ballroom"],
        "a-line":       ["a-line","aline","a line","fit and flare","skater"],
        "mermaid":      ["mermaid","trumpet dress","trumpet","fishtail","mermaid style"],
        "sheath":       ["sheath","sheath dress","column","straight cut"],
        "bodycon":      ["bodycon","body con","bandage","hugging"],
        "empire":       ["empire waist","empire line","high waist","empire cut"],
        "fit and flare":["fit and flare","skater dress","fit & flare"],
    },
    "color": {
        "black":        ["black","dark"],
        "white":        ["white","cream","ivory","off white"],
        "red":          ["red","crimson","scarlet","burgundy","maroon"],
        "blue":         ["blue","navy","cobalt","sky blue","baby blue","royal blue"],
        "green":        ["green","emerald","olive","forest green","lime"],
        "purple":       ["purple","violet","lavender","lilac","magenta"],
        "pink":         ["pink","fuchsia","hot pink","rose","blush"],
        "yellow":       ["yellow","gold","mustard","sunflower"],
        "brown":        ["brown","chocolate","tan","camel","khaki"],
        "beige":        ["beige","nude","taupe","sand","ecru"],
        "gray":         ["gray","grey","charcoal","slate","silver"],
        "gold":         ["gold","golden","metallic gold"],
        "silver":       ["silver","metallic silver","platinum"],
    }
}


def rule_based_extract(description: str) -> Dict:
    """
    Fallback سريع بالقواعد - محسّن
    """
    desc = description.lower()
    result = {}

    # استخراج كل الـ attributes من النص
    for attr, keyword_map in RULE_KEYWORDS.items():
        for value, keywords in keyword_map.items():
            if any(kw in desc for kw in keywords):
                result[attr] = value
                # نكسر الحلقة عشان ناخذ أول تطابق
                break

    # تطبيق قواعد إضافية ذكية
    
    # إذا كان ball gown، نضبط الـ occasion و length
    if "ball gown" in desc or "evening gown" in desc:
        result["occasion"] = "formal"
        result["length"] = "maxi"
        if "silhouette" not in result:
            result["silhouette"] = "ball gown"
    
    # off shoulder يعني sleeveless (إذا ما كان sleeve محدد)
    if "off shoulder" in desc or "off-the-shoulder" in desc:
        if "sleeve" not in result:
            result["sleeve"] = "sleeveless"
    
    # strapless يعني sleeveless
    if "strapless" in desc:
        if "sleeve" not in result:
            result["sleeve"] = "sleeveless"
    
    # sweetheart يعني sleeveless غالباً
    if "sweetheart" in desc:
        if "sleeve" not in result:
            result["sleeve"] = "sleeveless"
    
    # beaded أو embellished يعني pattern beaded
    if "beaded" in desc or "embellished" in desc or "sequin" in desc:
        result["pattern"] = "beaded"
    
    # floor length أو long dress يعني maxi
    if "floor length" in desc or "full length" in desc:
        result["length"] = "maxi"
    
    # knee length يعني midi
    if "knee length" in desc or "knee-length" in desc:
        if "length" not in result:
            result["length"] = "midi"

    return result


# ════════════════════════════════════════════════════════════
# 3 — LLM Extractor (Qwen2.5-0.5B)
# ════════════════════════════════════════════════════════════

class FashionNLPExtractor:
    """
    NLP Extractor محسّن - يحوّل وصف Florence إلى attributes
    """

    def __init__(
            self,
            use_llm: bool = True,
            model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    ):
        self.use_llm = use_llm
        self._tok = None
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        if use_llm:
            self._load_model(model_id)
        else:
            print("✓ FashionNLPExtractor — Rule-Based mode")

    def _load_model(self, model_id: str):
        try:
            print(f"تحميل {model_id}...")
            self._tok = AutoTokenizer.from_pretrained(model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                device_map="auto"
            )
            self._model.eval()
            print(f"✓ FashionNLPExtractor جاهز ({self._device})")
        except Exception as e:
            print(f"⚠ فشل تحميل LLM: {e}")
            print("⚠ الانتقال لـ Rule-Based mode")
            self.use_llm = False

    def _llm_extract(self, description: str) -> Dict:
        """استخراج بالـ LLM"""
        prompt = EXTRACTION_PROMPT.format(description=description)
        messages = [
            {"role": "system", "content": "You extract fashion attributes. Return only JSON."},
            {"role": "user", "content": prompt}
        ]

        text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tok([text], return_tensors="pt").to(self._device)

        with torch.no_grad():
            ids = self._model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self._tok.eos_token_id
            )

        output = self._tok.decode(ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # استخراج JSON
        try:
            json_match = re.search(r'\{.*?\}', output, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {k: v for k, v in parsed.items() if v and v != "null"}
        except Exception:
            pass

        # Fallback للقواعد
        return rule_based_extract(description)

    def extract(self, description: str) -> Dict:
        """
        المدخل: وصف نصي من Florence
        المخرج: attributes منظمة
        """
        if not description or len(description) < 10:
            return {}

        if self.use_llm and self._model:
            result = self._llm_extract(description)
        else:
            result = rule_based_extract(description)

        print(f"  ✓ NLP استخرج: {list(result.keys())}")
        return result
