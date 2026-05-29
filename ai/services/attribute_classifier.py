
# ============================================================
# ai/services/attribute_classifier.py
# Improved CLIP Attribute Classifier
# ============================================================

import torch
from PIL import Image
from typing import Dict, List

from ai.services.model_loader import ModelLoader


# ============================================================
# Labels
# Short + CLIP-friendly prompts
# ============================================================

ATTRIBUTES: Dict[str, List[str]] = {

    "category": [
        "turtleneck sweater",
        "knit sweater",
        "t-shirt",
        "dress shirt",
        "blouse",
        "trousers",
        "jeans",
        "shorts",
        "skirt",
        "dress",
        "blazer",
        "coat",
        "hoodie",
        "sneakers",
        "cardigan",
        "vest",
        "jumpsuit",
        "boots",
        "sandals",
        "polo shirt",
    ],

    "sleeve": [
        "sleeveless",
        "short sleeves",
        "long sleeves",
        "3/4 sleeves",
        "cap sleeves",
    ],

    "fit": [
        "slim fit",
        "regular fit",
        "oversized fit",
        "tight fit",
        "relaxed fit",
    ],

    "neckline": [
        "crew neck",
        "v-neck",
        "turtleneck",
        "collared",
        "off shoulder",
        "scoop neck",
        "boat neck",
        "square neck",
    ],

    "pattern": [
        "solid color",
        "striped",
        "plaid",
        "floral",
        "graphic print",
        "polka dots",
        "geometric pattern",
        "animal print",
        "camouflage",
        "tie dye",
    ],

    "occasion": [
        "formal wear",
        "casual wear",
        "sportswear",
        "party wear",
        "outdoor wear",
        "beachwear",
    ],

    "season": [
        "summer clothing",
        "winter clothing",
        "spring autumn clothing",
    ],

    "material_look": [
        "denim fabric",
        "leather fabric",
        "knit fabric",
        "cotton fabric",
        "silk fabric",
        "linen fabric",
        "polyester fabric",
        "wool fabric",
        "velvet fabric",
        "mesh fabric",
    ]
}


# ============================================================
# Attribute Classifier
# ============================================================

class AttributeClassifier:

    def classify(
            self,
            image: Image.Image,
            description: str = ""
    ) -> Dict:

        clip_model, clip_preprocess = ModelLoader.clip()

        DEVICE = ModelLoader.get_device()

        # ====================================================
        # preprocess image
        # ====================================================

        image_rgb = image.convert("RGB")

        image_tensor = clip_preprocess(
            image_rgb
        ).unsqueeze(0).to(DEVICE)

        result = {}

        # ====================================================
        # encode image
        # ====================================================

        with torch.no_grad():

            image_features = clip_model.encode_image(
                image_tensor
            )

            image_features = image_features / image_features.norm(
                dim=-1,
                keepdim=True
            )

            # =================================================
            # classify each attribute
            # =================================================

            for attr_name, labels in ATTRIBUTES.items():

                text_features = ModelLoader.get_clip_text_features(
                    attr_name,
                    labels
                )

                similarities = (
                    image_features @ text_features.T
                )[0]

                # IMPORTANT:
                # OpenAI CLIP logits scaling
                similarities = similarities * 100

                scores = similarities.softmax(dim=0)

                best_idx = scores.argmax().item()

                confidence = round(
                    scores[best_idx].item() * 100,
                    1
                )

                result[attr_name] = {
                    "value": labels[best_idx],
                    "confidence": confidence
                }

        # ====================================================
        # Rule-based corrections from Florence caption
        # ====================================================

        desc = description.lower()

        # ----------------------------
        # neckline
        # ----------------------------

        if "turtleneck" in desc:
            result["neckline"] = {
                "value": "turtleneck",
                "confidence": 99.0
            }

        # ----------------------------
        # sleeve
        # ----------------------------

        if "long sleeves" in desc:
            result["sleeve"] = {
                "value": "long sleeves",
                "confidence": 98.0
            }

        # ----------------------------
        # category
        # ----------------------------

        if "sweater" in desc:
            result["category"] = {
                "value": "knit sweater",
                "confidence": 96.0
            }

        # ----------------------------
        # material
        # ----------------------------

        if "knit" in desc or "sweater" in desc:
            result["material_look"] = {
                "value": "knit fabric",
                "confidence": 95.0
            }

        # ----------------------------
        # pattern
        # ----------------------------

        if "solid" in desc or "black" in desc:
            result["pattern"] = {
                "value": "solid color",
                "confidence": 90.0
            }

        # ====================================================
        # debug prints
        # ====================================================

        print(
            f"  ✓ category: "
            f"{result['category']['value']} "
            f"({result['category']['confidence']}%)"
        )

        print(
            f"  ✓ sleeve: "
            f"{result['sleeve']['value']} "
            f"({result['sleeve']['confidence']}%)"
        )

        print(
            f"  ✓ neckline: "
            f"{result['neckline']['value']} "
            f"({result['neckline']['confidence']}%)"
        )

        return result
