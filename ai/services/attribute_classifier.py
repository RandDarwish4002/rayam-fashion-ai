
import torch
import clip
from PIL import Image
from typing import Dict, List
from ai.services.model_loader import ModelLoader

# ============================================================
# ATTRIBUTES — PROMPTS محسّنة وقصيرة
# ============================================================

ATTRIBUTES: Dict[str, List[str]] = {
    "category": [
        # فئات أوضح وأقصر للـ CLIP
        "turtleneck sweater",
        "knit sweater", 
        "t-shirt",
        "dress shirt",
        "blouse",
        "dress pants",
        "jeans",
        "shorts",
        "skirt",
        "dress",
        "evening gown",
        "blazer",
        "winter coat",
        "hoodie",
        "leather shoes",
        "sneakers",
        "cardigan",
        "vest",
        "jumpsuit",
        "boots",
        "sandals",
        "polo shirt",
        "jacket",
        "suit",
        "trench coat",
        "pajamas",
        "swimsuit",
        "leggings",
    ],
    
    "sleeve": [
        "sleeveless",
        "short sleeves",
        "long sleeves",
        "three quarter sleeves",
        "cap sleeves",
    ],
    
    "fit": [
        "slim fit",
        "regular fit",
        "oversized",
        "bodycon",
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
        "sweetheart",
        "strapless",
    ],
    
    "pattern": [
        "solid",
        "striped",
        "plaid",
        "floral",
        "graphic",
        "polka dots",
        "geometric",
        "animal print",
        "camouflage",
        "tie dye",
        "checkered",
        "argyle",
    ],
    
    "occasion": [
        "formal",
        "casual",
        "sportswear",
        "party wear",
        "outdoor wear",
        "beach wear",
        "business wear",
        "club wear",
        "wedding attire",
    ],
    
    "season": [
        "summer",
        "winter", 
        "spring autumn",
        "all season",
    ],
    
    "material_look": [
        "denim",
        "leather",
        "knit",
        "cotton",
        "silk",
        "linen",
        "polyester",
        "wool",
        "velvet",
        "mesh",
        "lace",
        "satin",
        "fur",
        "corduroy",
        "chiffon",
    ]
}

# ============================================================
# Temperature Scaling لتحسين توزيع الثقة
# ============================================================

class AttributeClassifier:
    
    def __init__(self, temperature: float = 0.07):
        """
        temperature: أقل = توزيع أكثر حدة (أفضل للتصنيف)
        """
        self.temperature = temperature
        print(f"✓ AttributeClassifier (temperature={temperature})")
    
    def classify(self, image: Image.Image) -> Dict:
        
        clip_model, clip_preprocess = ModelLoader.clip()
        DEVICE = ModelLoader.get_device()
        
        image_rgb = image.convert("RGB")
        image_tensor = clip_preprocess(image_rgb).unsqueeze(0).to(DEVICE)
        
        result = {}
        
        with torch.no_grad():
            # Normalize image features
            image_features = clip_model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            for attr_name, labels in ATTRIBUTES.items():
                batch_size = 20  # أكبر قليلاً
                all_scores = []
                
                for i in range(0, len(labels), batch_size):
                    batch = labels[i:i+batch_size]
                    tokens = clip.tokenize(batch).to(DEVICE)
                    text_features = clip_model.encode_text(tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # حساب التشابه
                    similarities = (image_features @ text_features.T).squeeze(0)
                    all_scores.extend(similarities.cpu().tolist())
                
                # تطبيق temperature scaling قبل softmax
                scores_tensor = torch.FloatTensor(all_scores) / self.temperature
                scores = torch.softmax(scores_tensor, dim=0)
                
                best_idx = scores.argmax().item()
                confidence = round(scores[best_idx].item() * 100, 1)
                
                result[attr_name] = {
                    "value": labels[best_idx],
                    "confidence": confidence
                }
        
        # طباعة النتائج
        print(f"  ✓ category: {result['category']['value']} ({result['category']['confidence']}%)")
        print(f"  ✓ sleeve: {result['sleeve']['value']} ({result['sleeve']['confidence']}%)")
        print(f"  ✓ fit: {result['fit']['value']} ({result['fit']['confidence']}%)")
        print(f"  ✓ neckline: {result['neckline']['value']} ({result['neckline']['confidence']}%)")
        print(f"  ✓ pattern: {result['pattern']['value']} ({result['pattern']['confidence']}%)")
        print(f"  ✓ occasion: {result['occasion']['value']} ({result['occasion']['confidence']}%)")
        print(f"  ✓ season: {result['season']['value']} ({result['season']['confidence']}%)")
        
        return result
