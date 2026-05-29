from typing import Dict

class SmartDecisionLayer:
    """
    يدمج CLIP + Florence description
    ويحل التناقضات (sleeve / neckline / fit ...)
    """

    def __init__(self):
        self.threshold = 50  # confidence threshold

    def resolve(self, attributes: Dict, description: str) -> Dict:

        final = {}

        for key, value in attributes.items():

            label = value["value"]
            conf  = value["confidence"]

            # default acceptance
            final[key] = label

            # 1. low confidence rejection
            if conf < self.threshold:
                final[key] = None

        # -----------------------------
        # 2. RULES from Florence text
        # -----------------------------

        desc = description.lower()

        # sleeve fix
        if "long sleeve" in desc or "long sleeves" in desc:
            final["sleeve"] = "long sleeves"

        elif "short sleeve" in desc:
            final["sleeve"] = "short sleeves"

        elif "sleeveless" in desc:
            final["sleeve"] = "sleeveless"

        # neckline fix
        if "turtleneck" in desc or "high neck" in desc:
            final["neckline"] = "turtleneck"

        elif "v-neck" in desc:
            final["neckline"] = "v-neck"

        elif "off shoulder" in desc:
            final["neckline"] = "off shoulder"

        # pattern fix
        if "solid" in desc or "plain" in desc:
            final["pattern"] = "solid color"

        elif "striped" in desc:
            final["pattern"] = "striped"

        # fit fallback
        if final.get("fit") is None:
            final["fit"] = "regular fit"

        return final
