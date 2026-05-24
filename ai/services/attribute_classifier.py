# ============================================================
# ai/services/attribute_classifier.py
# يطابق: AIClassifier.classifyAttributes(image): Map
# ============================================================

import torch
import clip
from PIL import Image
from typing import Dict, List
from ai.services.model_loader import ModelLoader


# ── Labels لكل attribute ─────────────────────────────────
ATTRIBUTES: Dict[str, List[str]] = {
    "category": [
        "a t-shirt or tee",
        "a dress shirt or button-up shirt",
        "a blouse",
        "trousers or dress pants",
        "jeans or denim pants",
        "shorts",
        "a skirt",
        "a dress",
        "a jacket or blazer",
        "a coat or overcoat",
        "a hoodie or sweatshirt",
        "a sweater or knitwear",
        "shoes or leather shoes",
        "sneakers or athletic shoes",
        "a cardigan",
        "a vest or waistcoat",
        "a jumpsuit or romper",
        "boots",
        "sandals or flip flops",
        "a polo shirt",
    ],
    "sleeve": [
        "clothing with no sleeves, sleeveless",
        "clothing with short sleeves",
        "clothing with long sleeves",
        "clothing with 3/4 length sleeves",
        "clothing with cap sleeves",
    ],
    "fit": [
        "slim fit or fitted clothing",
        "regular or standard fit clothing",
        "oversized or loose fit clothing",
        "tight or bodycon clothing",
        "relaxed or comfortable fit clothing",
    ],
    "neckline": [
        "crew neck or round neck",
        "v-neck",
        "turtleneck or high neck",
        "collared or polo collar",
        "off shoulder",
        "scoop neck",
        "boat neck or bateau neck",
        "square neck",
    ],
    "pattern": [
        "solid color, no pattern",
        "horizontal or vertical stripes",
        "plaid or tartan or checkered",
        "floral or flower print",
        "graphic print or logo print",
        "polka dots",
        "geometric or abstract pattern",
        "animal print like leopard or zebra",
        "camouflage pattern",
        "tie-dye pattern",
    ],
    "occasion": [
        "formal business or office wear",
        "casual everyday clothing",
        "sportswear or athletic clothing",
        "party or evening wear",
        "outdoor or activewear",
        "beachwear or summer casual",
    ],
    "season": [
        "summer lightweight thin clothing",
        "winter warm thick clothing",
        "spring or autumn mid-weight clothing",
    ],
    "material_look": [
        "denim or jean fabric",
        "leather or faux leather",
        "knit or knitwear texture",
        "cotton or plain fabric",
        "silk or satin shiny fabric",
        "linen or linen-look fabric",
        "synthetic or polyester fabric",
        "wool or woolen fabric",
        "velvet fabric",
        "mesh or sheer fabric",
    ]
}


class AttributeClassifier:
    """
    يطابق Class Diagram:
    AIClassifier.classifyAttributes(image): Map
    """

    def classify(self, image: Image.Image) -> Dict:

        clip_m, clip_p = ModelLoader.clip()

        img_rgb = image.convert("RGB")
        img_tensor = clip_p(img_rgb).unsqueeze(0)

        device = ModelLoader.get_device()
        img_tensor = img_tensor.to(device)

        result = {}

        with torch.no_grad():

            img_feat = clip_m.encode_image(img_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            for attr, labels in ATTRIBUTES.items():

                all_scores = []

                for i in range(0, len(labels), 10):
                    batch = labels[i:i+10]

                    tokens = clip.tokenize(batch).to(device)

                    text_feat = clip_m.encode_text(tokens)
                    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

                    sims = (img_feat @ text_feat.T)[0]
                    all_scores.extend(sims.cpu().tolist())

                scores = torch.softmax(torch.tensor(all_scores), dim=0)

                best_idx = scores.argmax().item()

                result[attr] = {
                    "value": labels[best_idx],
                    "confidence": round(scores[best_idx].item() * 100, 1)
                }

        print(f"  ✓ category: {result['category']['value']}")
        return result
