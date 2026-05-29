
import torch
import clip
from PIL import Image
from typing import Dict, List
from ai.services.model_loader import ModelLoader

ATTRIBUTES: Dict[str, List[str]] = {
    "category": [
        # أكثر تفصيلاً وأوضح
        "a black or colored turtleneck sweater",
        "a knit sweater or pullover",
        "a t-shirt or casual tee shirt",
        "a formal dress shirt with buttons",
        "a women's blouse",
        "dress pants or formal trousers",
        "blue jeans or denim pants",
        "shorts or bermuda shorts",
        "a women's skirt",
        "a women's dress or gown",
        "a blazer or suit jacket",
        "a winter coat or long coat",
        "a zip-up hoodie or sweatshirt",
        "leather shoes or oxford shoes",
        "running sneakers or sports shoes",
        "a knit cardigan with buttons",
        "a suit vest or formal waistcoat",
        "a jumpsuit or one-piece outfit",
        "ankle boots or knee-high boots",
        "sandals or open-toe shoes",
        "a polo shirt with collar",
    ],
    "sleeve": [
        # أوضح وأكثر تمييزاً
        "sleeveless clothing, tank top, no sleeves at all",
        "short sleeves ending above the elbow",
        "full long sleeves covering the entire arm to the wrist",
        "three-quarter sleeves ending below the elbow",
        "very short cap sleeves on the shoulder only",
    ],
    "fit": [
        "very tight slim fit clothing hugging the body",
        "normal regular fit clothing",
        "very loose oversized baggy clothing",
        "bodycon tight fitting clothing",
        "relaxed comfortable loose fit clothing",
    ],
    "neckline": [
        "crew neck round neckline",
        "v-shaped neckline",
        "high turtleneck covering the neck",
        "shirt collar or polo collar",
        "off shoulder neckline",
        "wide scoop neckline",
        "wide boat neck neckline",
        "square neckline",
    ],
    "pattern": [
        "plain solid single color no pattern",
        "striped pattern with lines",
        "plaid checkered tartan pattern",
        "floral pattern with flowers",
        "graphic design logo or print pattern",
        "polka dots pattern",
        "geometric shapes pattern",
        "animal print leopard zebra pattern",
        "camouflage military pattern",
        "tie-dye swirl pattern",
    ],
    "occasion": [
        "business formal office professional wear",
        "casual everyday relaxed wear",
        "sports gym athletic activewear",
        "party night out evening wear",
        "outdoor hiking camping wear",
        "beach summer vacation wear",
    ],
    "season": [
        "light thin summer clothing for hot weather",
        "thick warm winter clothing for cold weather",
        "medium weight spring or autumn clothing",
    ],
    "material_look": [
        "denim jean material",
        "leather or faux leather material",
        "knitted wool or knitwear texture",
        "plain cotton casual fabric",
        "shiny silk or satin fabric",
        "linen natural fabric",
        "synthetic polyester fabric",
        "heavy wool woolen fabric",
        "soft velvet fabric",
        "transparent mesh or sheer fabric",
    ]
}


class AttributeClassifier:

    def classify(self, image: Image.Image) -> Dict:

        clip_model, clip_preprocess = ModelLoader.clip()
        DEVICE = ModelLoader.get_device()

        image_rgb    = image.convert("RGB")
        image_tensor = clip_preprocess(image_rgb)\
            .unsqueeze(0).to(DEVICE)

        result = {}

        with torch.no_grad():
            image_features = clip_model.encode_image(
                image_tensor
            )
            image_features = image_features / \
                image_features.norm(dim=-1, keepdim=True)

            for attr_name, labels in ATTRIBUTES.items():
                batch_size = 10
                all_scores = []

                for i in range(0, len(labels), batch_size):
                    batch  = labels[i:i+batch_size]
                    tokens = clip.tokenize(batch).to(DEVICE)
                    tf     = clip_model.encode_text(tokens)
                    tf     = tf / tf.norm(
                        dim=-1, keepdim=True
                    )
                    sims   = (image_features @ tf.T)[0]
                    all_scores.extend(sims.cpu().tolist())

                scores   = torch.softmax(
                    torch.FloatTensor(all_scores), dim=0
                )
                best_idx = scores.argmax().item()
                conf     = round(
                    scores[best_idx].item() * 100, 1
                )

                result[attr_name] = {
                    "value":      labels[best_idx],
                    "confidence": conf
                }

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
        return result
