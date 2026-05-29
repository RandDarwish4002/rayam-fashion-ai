class SmartDecisionLayer:
    """
    يحل تناقضات CLIP + Florence
    """

    def resolve(self, clip_result: dict, florence_text: str = "") -> dict:

        final = {}

        for attr, data in clip_result.items():

            value = data["value"]
            conf  = data["confidence"]

            # تجاهل النتائج الضعيفة
            if conf < 40:
                continue

            # =========================
            # sleeve fix
            # =========================
            if attr == "sleeve":
                text = florence_text.lower()

                if "long sleeve" in text or "long sleeves" in text:
                    if conf < 80:
                        final[attr] = {
                            "value": "long sleeves",
                            "confidence": 100.0
                        }
                        continue

                if "sleeveless" in value and conf < 70:
                    continue

            # =========================
            # neckline fix
            # =========================
            if attr == "neckline":
                text = florence_text.lower()

                if "turtleneck" in text:
                    final[attr] = {
                        "value": "turtleneck",
                        "confidence": 100.0
                    }
                    continue

            # =========================
            # default
            # =========================
            final[attr] = data

        return final
