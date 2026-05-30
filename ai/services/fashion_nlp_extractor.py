
# ============================================================
# ai/services/fashion_nlp_extractor.py
# NLP Extractor بين Florence وCLIP
# يحوّل الوصف النصي إلى attributes منظمة
# ============================================================

import json
import re
import torch
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


# ════════════════════════════════════════════════════════════
# 1 — Prompt للنموذج
# ════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """You are a fashion attribute extractor.
Extract ONLY these attributes from the clothing description.
Return ONLY valid JSON, nothing else.

Attributes to extract:
- category: (t-shirt/shirt/dress/pants/skirt/jacket/coat/sweater/shoes/boots/shorts/blouse/hoodie/cardigan/jumpsuit)
- sleeve: (sleeveless/short sleeve/long sleeve/3/4 sleeve/cap sleeve)
- neckline: (crew neck/v-neck/turtleneck/off shoulder/collared/scoop neck/boat neck/square neck/sweetheart/strapless)
- fit: (slim fit/regular fit/oversized/tight/relaxed)
- pattern: (solid/striped/floral/plaid/graphic/polka dots/geometric/animal print/tie-dye)
- occasion: (formal/casual/sport/party/outdoor/beach)
- season: (summer/winter/spring-autumn)
- length: (mini/midi/maxi/cropped/regular) — for dresses and skirts only
- silhouette: (ball gown/a-line/mermaid/sheath/bodycon/empire/fit and flare)
- color: (black/white/red/blue/green/purple/pink/yellow/brown/beige/gray/gold/silver)

Rules:
- If not mentioned, set to null
- Use ONLY the values listed above
- Return JSON only

Description: {description}

JSON:"""


# ════════════════════════════════════════════════════════════
# 2 — Rule-Based Fallback (بدون LLM)
# ════════════════════════════════════════════════════════════

RULE_KEYWORDS = {
    "category": {
        "t-shirt":      ["t-shirt","tee","tshirt"],
        "shirt":        ["shirt","button-up","dress shirt"],
        "dress":        ["dress","gown","frock"],
        "pants":        ["pants","trousers","slacks"],
        "jeans":        ["jeans","denim"],
        "skirt":        ["skirt"],
        "jacket":       ["jacket","blazer"],
        "coat":         ["coat","overcoat","trench"],
        "sweater":      ["sweater","knitwear","pullover","jumper","knit"],
        "hoodie":       ["hoodie","sweatshirt"],
        "shoes":        ["shoes","loafers","oxfords","heels","pumps"],
        "boots":        ["boots","boot"],
        "sneakers":     ["sneakers","trainers","athletic shoes"],
        "shorts":       ["shorts"],
        "blouse":       ["blouse"],
        "cardigan":     ["cardigan"],
        "jumpsuit":     ["jumpsuit","romper"],
    },
    "sleeve": {
        "sleeveless":   ["sleeveless","no sleeve","strapless","tank","without sleeve"],
        "short sleeve": ["short sleeve","short-sleeve","cap sleeve","half sleeve"],
        "long sleeve":  ["long sleeve","long-sleeve","full sleeve","full length sleeve","long sleeves"],
        "3/4 sleeve":   ["3/4","three-quarter","three quarter","elbow"],
    },
    "neckline": {
        "turtleneck":   ["turtleneck","polo neck","high neck","funnel neck","roll neck"],
        "v-neck":       ["v-neck","v neck","v-shaped neckline"],
        "off shoulder": ["off-shoulder","off shoulder","bardot","cold shoulder"],
        "sweetheart":   ["sweetheart","sweetheart neckline","sweetheart-shaped neckline","heart-shaped neckline"],
        "strapless":    ["strapless","tube top"],
        "crew neck":    ["crew neck","round neck","crew-neck"],
        "collared":     ["collar","collared","point collar"],
        "scoop neck":   ["scoop","scoop neck"],
        "boat neck":    ["boat neck","bateau"],
        "square neck":  ["square neck","square neckline"],
    },
    "fit": {
        "slim fit":     ["slim fit","tailored fit","skinny fit"],
        "oversized":    ["oversized","baggy","loose","boxy","wide"],
        "tight":        ["tight","bodycon","form-fitting","fitted"],
        "regular fit":  ["regular","classic fit","standard"],
        "relaxed":      ["relaxed","easy fit","comfortable"],
    },
    "pattern": {
        "solid":        ["solid","plain","single color","monochrome","unicolor"],
        "striped":      ["stripe","striped","stripes","pinstripe"],
        "floral":       ["floral","flower","flowers","botanical","bloom"],
        "plaid":        ["plaid","checkered","tartan","check","gingham"],
        "graphic":      ["graphic","print","logo","printed","text"],
        "polka dots":   ["polka","dotted","dots"],
        "geometric":    ["geometric","abstract","shapes"],
        "animal print": ["leopard","zebra","snake","animal print"],
        "tie-dye":      ["tie-dye","tiedye"],
    },
    "occasion": {
        "formal":       ["formal","business","office","professional","work","gala","evening"],
        "casual":       ["casual","everyday","relaxed","weekend"],
        "sport":        ["sport","athletic","gym","workout","activewear","running"],
        "party":        ["party","night out","cocktail","club","festive"],
        "outdoor":      ["outdoor","hiking","camping","adventure"],
        "beach":        ["beach","swim","swimwear","tropical","resort"],
    },
    "season": {
        "summer":       ["summer","lightweight","light","thin","hot weather","tropical"],
        "winter":       ["winter","warm","thick","heavy","cold","wool","fleece"],
        "spring-autumn":["spring","autumn","fall","mid-weight","transitional"],
    },
    "length": {
        "mini":         ["mini","short dress","short skirt","micro"],
        "midi":         ["midi","knee","below knee","mid-length"],
        "maxi":         ["maxi","floor length","long dress","full length"],
        "cropped":      ["crop","cropped","cut-off","belly"],
    },
    "silhouette": {
    "ball gown": [
        "ball gown",
        "princess gown"
    ],
    "a-line": [
        "a-line",
        "aline"
    ],
    "mermaid": [
        "mermaid",
        "trumpet dress"
    ],
    "sheath": [
        "sheath"
    ],
    "bodycon": [
        "bodycon"
    ],
    "empire": [
        "empire waist"
    ],
    "fit and flare": [
        "fit and flare"
    ]
},"color": {
    "black": ["black"],
    "white": ["white"],
    "red": ["red"],
    "blue": ["blue"],
    "green": ["green"],
    "purple": ["purple","violet","lavender"],
    "pink": ["pink","fuchsia"],
    "yellow": ["yellow"],
    "brown": ["brown"],
    "beige": ["beige","cream","ivory"],
    "gray": ["gray","grey"],
    "gold": ["gold","golden"],
    "silver": ["silver"]
},
}

def rule_based_extract(description: str) -> Dict:
    """
    Fallback سريع بالقواعد
    """
    desc = description.lower()
    result = {}

    for attr, keyword_map in RULE_KEYWORDS.items():
        for value, keywords in keyword_map.items():
            if any(kw in desc for kw in keywords):
                result[attr] = value
                break

    # ball gown لا نعتبره slim fit
    if result.get("silhouette") == "ball gown":
        result.pop("fit", None)

    formal_keywords = [
        "ball gown",
        "evening gown",
        "gala",
        "formal dress",
        "beaded bodice",
        "red carpet",
        "bridal",
        "wedding gown",
    ]

    if any(k in desc for k in formal_keywords):
        result["occasion"] = "formal"

    if result.get("neckline") == "off shoulder":
        result["sleeve"] = "sleeveless"

    if "ball gown" in desc:
        result["length"] = "maxi"

    if "floor length" in desc:
        result["length"] = "maxi"       

    return result

# ════════════════════════════════════════════════════════════
# 3 — LLM Extractor (Qwen2.5-0.5B)
# ════════════════════════════════════════════════════════════

class FashionNLPExtractor:
    """
    يحوّل وصف Florence النصي إلى attributes منظمة
    يُوضع بين Florence وCLIP في الـ pipeline
    """

    def __init__(
            self,
            use_llm: bool = True,
            model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    ):
        self.use_llm  = use_llm
        self._tok     = None
        self._model   = None
        self._device  = "cuda" if torch.cuda.is_available() \
                        else "cpu"

        if use_llm:
            self._load_model(model_id)
        else:
            print("✓ FashionNLPExtractor — Rule-Based mode")

    def _load_model(self, model_id: str):
        try:
            print(f"تحميل {model_id}...")
            self._tok = AutoTokenizer.from_pretrained(
                model_id
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16
                            if self._device == "cuda"
                            else torch.float32,
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
        prompt = EXTRACTION_PROMPT.format(
            description=description
        )
        messages = [
            {"role": "system",
             "content": "You extract fashion attributes. Return only JSON."},
            {"role": "user",
             "content": prompt}
        ]

        text = self._tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self._tok(
            [text], return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            ids = self._model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self._tok.eos_token_id
            )

        output = self._tok.decode(
            ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        # استخرج JSON
        try:
            json_match = re.search(
                r'\{.*?\}', output, re.DOTALL
            )
            if json_match:
                parsed = json.loads(json_match.group())
                # احذف null values
                return {
                    k: v for k, v in parsed.items()
                    if v and v != "null"
                }
        except Exception:
            pass

        # fallback
        return rule_based_extract(description)

    def extract(self, description: str) -> Dict:
        """
        المدخل:  وصف نصي من Florence
        المخرج:  attributes منظمة
        """
        if not description or len(description) < 10:
            return {}

        if self.use_llm and self._model:
            result = self._llm_extract(description)
        else:
            result = rule_based_extract(description)

        print(f"  ✓ NLP استخرج: {list(result.keys())}")
        return result
