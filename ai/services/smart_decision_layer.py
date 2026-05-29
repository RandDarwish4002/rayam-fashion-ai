
from typing import Dict, Any

class SmartDecisionLayer:
    """
    يدمج CLIP + Florence description
    ويحل التناقضات بين النتائج
    """

    def __init__(self):
        self.threshold = 50  # confidence threshold

    def resolve(self, attributes: Dict[str, Dict], description: str) -> Dict[str, Any]:

        final = {}

        desc = description.lower()

        for key, value in attributes.items():

            label = value["value"]
            conf  = value["confidence"]

            # إذا الثقة ضعيفة → لا نعتمد CLIP
            if conf < self.threshold:
                final[key] = None
            else:
                final[key] = label

        # -------------------------
        # RULES from Florence text
        # -------------------------

        # sleeve correction
        if "long sleeve" in desc or "long sleeves" in desc:
            final["sleeve"] = "long sleeves"

        elif "short sleeve" in desc:
            final["sleeve"] = "short sleeves"

        elif "sleeveless" in desc:
            final["sleeve"] = "sleeveless"

        # neckline correction
        if "turtleneck" in desc or "high neck" in desc:
            final["neckline"] = "turtleneck"

        elif "v-neck" in desc:
            final["neckline"] = "v-neck"

        elif "off shoulder" in desc:
            final["neckline"] = "off shoulder"

        # pattern correction
        if "solid" in desc or "plain" in desc:
            final["pattern"] = "solid color"

        elif "striped" in desc:
            final["pattern"] = "striped"

        # fallback fit
        if final.get("fit") is None:
            final["fit"] = "regular fit"

        return final
