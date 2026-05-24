# ============================================================
# ai/classifiers/color_classifier.py
# يطابق: AIClassifier.classifyColorGroups()
# ============================================================

import colorsys
import numpy as np
import torch
import torch.nn as nn
import os
from typing import Dict, List, Tuple


GROUPS   = ["Neutrals","Pastels","Brights","Darks","Metallics"]
G2I      = {g: i for i, g in enumerate(GROUPS)}
I2G      = {i: g for i, g in enumerate(GROUPS)}

COLOR_DATA: List[Tuple[str,str]] = [
    # Neutrals
    ("#000000","Neutrals"), ("#FFFFFF","Neutrals"),
    ("#808080","Neutrals"), ("#A9A9A9","Neutrals"),
    ("#D3D3D3","Neutrals"), ("#F5F5DC","Neutrals"),
    ("#C2B280","Neutrals"), ("#E8DCC8","Neutrals"),
    ("#D2B48C","Neutrals"), ("#DCDCDC","Neutrals"),
    ("#F5F5F5","Neutrals"), ("#C0C0C0","Neutrals"),
    ("#696969","Neutrals"), ("#778899","Neutrals"),
    ("#B0C4DE","Neutrals"), ("#DEB887","Neutrals"),
    ("#D4C5A9","Neutrals"), ("#C8B99A","Neutrals"),
    ("#BDB5A6","Neutrals"), ("#A8A090","Neutrals"),
    # Pastels
    ("#FFB6C1","Pastels"),  ("#B0E0E6","Pastels"),
    ("#98FB98","Pastels"),  ("#DDA0DD","Pastels"),
    ("#FFFACD","Pastels"),  ("#E6E6FA","Pastels"),
    ("#F0E68C","Pastels"),  ("#FFDAB9","Pastels"),
    ("#E0FFFF","Pastels"),  ("#FFC0CB","Pastels"),
    ("#FFE4E1","Pastels"),  ("#F0FFF0","Pastels"),
    ("#E6F3FF","Pastels"),  ("#F0FFFF","Pastels"),
    ("#E8F8F5","Pastels"),  ("#F9F0FF","Pastels"),
    ("#FFF0E8","Pastels"),  ("#E8FFF0","Pastels"),
    ("#FFE8F0","Pastels"),  ("#E8F0FF","Pastels"),
    # Brights
    ("#FF0000","Brights"),  ("#00FF00","Brights"),
    ("#0000FF","Brights"),  ("#FFD700","Brights"),
    ("#FF6600","Brights"),  ("#FF1493","Brights"),
    ("#00CED1","Brights"),  ("#7FFF00","Brights"),
    ("#FF4500","Brights"),  ("#1E90FF","Brights"),
    ("#FF69B4","Brights"),  ("#00FA9A","Brights"),
    ("#FF00FF","Brights"),  ("#00FFFF","Brights"),
    ("#DC143C","Brights"),  ("#00BFFF","Brights"),
    ("#32CD32","Brights"),  ("#FF8C00","Brights"),
    ("#9400D3","Brights"),  ("#4169E1","Brights"),
    # Darks
    ("#003366","Darks"),    ("#8B0000","Darks"),
    ("#2F4F4F","Darks"),    ("#4B0082","Darks"),
    ("#8B4513","Darks"),    ("#191970","Darks"),
    ("#006400","Darks"),    ("#1C1C1C","Darks"),
    ("#0D0D0D","Darks"),    ("#002B36","Darks"),
    ("#000033","Darks"),    ("#330000","Darks"),
    ("#111111","Darks"),    ("#222222","Darks"),
    ("#333333","Darks"),    ("#1A3300","Darks"),
    ("#001A33","Darks"),    ("#330033","Darks"),
    ("#1B0000","Darks"),    ("#36013F","Darks"),
    # Metallics
    ("#FFD700","Metallics"), ("#C0C0C0","Metallics"),
    ("#CD7F32","Metallics"), ("#B87333","Metallics"),
    ("#D4AF37","Metallics"), ("#AA98A9","Metallics"),
    ("#8C7853","Metallics"), ("#CFB53B","Metallics"),
    ("#D1B000","Metallics"), ("#C5A028","Metallics"),
    ("#B5A642","Metallics"), ("#967117","Metallics"),
    ("#85754E","Metallics"), ("#A99A86","Metallics"),
    ("#C9AE5D","Metallics"), ("#D4C5A9","Metallics"),
    ("#A57C52","Metallics"), ("#8E8E8E","Metallics"),
    ("#C8B89A","Metallics"), ("#B8A898","Metallics"),
]


class _ColorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
    def forward(self, x):
        return self.net(x)


def _hex_to_features(hex_color: str) -> List[float]:
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    L = 0.2126*r + 0.7152*g + 0.0722*b
    ita = np.arctan(
        (L - 0.5) / (0.5 - v + 1e-8)
    ) * (180.0/np.pi) / 180.0
    return [
        float(np.sin(h * 2 * np.pi)),
        float(np.cos(h * 2 * np.pi)),
        float(s), float(v), float(ita)
    ]


class ColorClassifier:
    """
    يطابق Class Diagram:
    AIClassifier.classifyColorGroups(colors)
    """

    def __init__(
            self,
            model_path: str = "models/color_classifier.pth"
    ):
        self.model_path = model_path
        self._model     = None
        self._load_or_train()

    def _load_or_train(self):
        if os.path.exists(self.model_path):
            self._model = _ColorNet()
            self._model.load_state_dict(
                torch.load(
                    self.model_path,
                    map_location="cpu"
                )
            )
            self._model.eval()
            print("✓ Color Classifier محمّل")
        else:
            print("تدريب Color Classifier...")
            self._train()

    def _train(self):
        os.makedirs(
            os.path.dirname(self.model_path),
            exist_ok=True
        )
        X = torch.FloatTensor(
            [_hex_to_features(c[0]) for c in COLOR_DATA]
        )
        y = torch.LongTensor(
            [G2I[c[1]] for c in COLOR_DATA]
        )

        model     = _ColorNet()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.003,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler\
            .CosineAnnealingLR(optimizer, T_max=3000)
        criterion = nn.CrossEntropyLoss()
        best_acc  = 0.0
        best_state = None

        for epoch in range(3000):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 500 == 0:
                model.eval()
                with torch.no_grad():
                    acc = (
                        model(X).argmax(1) == y
                    ).float().mean().item()
                if acc > best_acc:
                    best_acc   = acc
                    best_state = {
                        k: v.clone()
                        for k, v in model.state_dict().items()
                    }
                print(f"  Epoch {epoch} | Acc: {acc:.1%}")

        model.load_state_dict(best_state)
        torch.save(model.state_dict(), self.model_path)
        self._model = model
        self._model.eval()
        print(f"✓ محفوظ | دقة: {best_acc:.1%}")

    def classify(self, hex_color: str) -> Dict:
        """
        المدخل:  hex color
        المخرج:  group + confidence
        """
        feat = torch.FloatTensor(
            [_hex_to_features(hex_color)]
        )
        with torch.no_grad():
            logits = self._model(feat)
            probs  = torch.softmax(logits, dim=-1)[0]
            pred   = probs.argmax().item()

        return {
            "hex":        hex_color,
            "group":      I2G[pred],
            "confidence": round(
                probs[pred].item() * 100, 1
            )
        }

    def classify_list(
            self,
            colors: List[Dict]
    ) -> List[Dict]:
        """
        يصنف قائمة ألوان دفعة واحدة
        يُستدعى من: AIClassifier.classifyColorGroups()
        """
        for color in colors:
            result = self.classify(color["hex"])
            color["group"] = result["group"]
        return colors
