# ============================================================
# ai/pipeline/onboarding_pipeline.py
# يربط: SkinAnalyzer + ColorClassifier + MannequinGenerator
# يطابق: RegularUser.setupProfile()
# ============================================================

from ai.classifiers.skin_analyzer import SkinAnalyzer
from ai.classifiers.color_classifier import ColorClassifier
from ai.generators.mannequin_generator import MannequinGenerator
from typing import Dict, Optional


class OnboardingPipeline:
    """
    يطابق Class Diagram:
    RegularUser.setupProfile() تستدعي هذا الـ pipeline
    """

    def __init__(
            self,
            caesar_path: Optional[str] = None,
            models_dir: str = "models/"
    ):
        self.skin_analyzer = SkinAnalyzer()

        self.color_clf = ColorClassifier(
            model_path=f"{models_dir}color_classifier.pth"
        )

        self.mannequin_gen = MannequinGenerator(
            model_path=f"{models_dir}measurement_regressor.pth",
            scaler_path=f"{models_dir}measurement_scaler.pkl",
            smplx_dir=models_dir,
            auto_train=True,
            caesar_path=caesar_path
        )

    def run(
            self,
            user_id: str,
            skin_hex: str,
            height: float,
            weight: float,
            chest: float,
            waist: float,
            hip: float,
            shoulder: float,
            gender: str = "neutral"
    ) -> Dict:
        """
        Pipeline كامل من البيانات حتى المانيكان
        """

        print(f"\n{'═'*50}")
        print(f"OnboardingPipeline — user: {user_id}")
        print(f"{'═'*50}")

        # ① تحليل البشرة
        print("\n① تحليل لون البشرة...")
        skin_result = self.skin_analyzer.analyze(skin_hex)
        print(f"  النوع: {skin_result['skin_type']}")

        # ② معلومات المقاس
        print("\n② حساب المقاسات...")
        size_info = self.mannequin_gen.get_size_info(
            height,
            weight,
            chest,
            waist,
            hip,
            shoulder
        )

        print(
            f"  المقاس: {size_info['size']} "
            f"| BMI: {size_info['bmi']}"
        )

        # ③ توليد beta vector
        print("\n③ توليد beta vector...")

        beta = self.mannequin_gen.measurementsToVector(
            height,
            weight,
            chest,
            waist,
            hip,
            shoulder
        )

        print(
            f"  Beta: {[round(b,3) for b in beta[:4]]}..."
        )

        # ④ توليد المانيكان
        print("\n④ توليد المانيكان...")

        mesh_path = self.mannequin_gen.generateMesh(
            beta_vector=beta,
            gender=gender,
            skin_hex=skin_hex,
            output_path=f"models/mannequin_{user_id}.obj"
        )

        # ⑤ تجهيز بيانات العرض
        print("\n⑤ تجهيز بيانات العرض...")

        render_data = self.mannequin_gen.render(
            mesh_path,
            skin_hex=skin_hex
        )

        result = {
            "status": "success",
            "user_id": user_id,

            "skin": skin_result,

            "size": size_info,

            "mannequin": {
                "beta_vector": beta,
                "mesh_path": mesh_path,
                "render_data": render_data
            },

            "profile_colors": {
                "suitable": skin_result["suitable_colors"],
                "unsuitable": skin_result["unsuitable_colors"]
            }
        }

        print("\n✓ OnboardingPipeline اكتمل")

        return result
