# ============================================================
# ai/generators/mannequin_generator.py
# ============================================================

import os
import torch
import torch.nn as nn
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, Tuple


# ── بنية النموذج ─────────────────────────────────────────
class _MeasurementNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.net(x)


# ── داتا اصطناعية ─────────────────────────────────────────
def _synthetic_data(n: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)

    heights   = np.clip(np.random.normal(170,10,n),145,200)
    weights   = np.clip(np.random.normal(70,15,n),40,130)
    chests    = np.clip(np.random.normal(95,10,n),70,135)
    waists    = np.clip(np.random.normal(80,12,n),55,115)
    hips      = np.clip(np.random.normal(98,10,n),75,130)
    shoulders = np.clip(np.random.normal(42,4,n),30,55)

    meas = np.column_stack([
        heights, weights, chests,
        waists, hips, shoulders
    ]).astype(np.float32)

    betas = np.zeros((n,10), dtype=np.float32)
    betas[:,0] = (heights   - 170)/20.0
    betas[:,1] = (weights   - 70)/30.0
    betas[:,2] = (chests    - 95)/20.0
    betas[:,3] = (waists    - 80)/24.0
    betas[:,4] = (hips      - 98)/20.0
    betas[:,5] = (shoulders - 42)/8.0
    betas[:,6:] = np.random.randn(n,4)*0.1

    return meas, betas


# ════════════════════════════════════════════════════════════
# الكلاس الرئيسي
# ════════════════════════════════════════════════════════════

class MannequinGenerator:

    def __init__(
            self,
            model_path="models/measurement_regressor.pth",
            scaler_path="models/measurement_scaler.pkl",
            smplx_dir="models/",
            auto_train=True,
            caesar_path=None
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.smplx_dir = smplx_dir
        self._net = None
        self._scaler = None

        os.makedirs("models", exist_ok=True)

        if auto_train and not os.path.exists(model_path):
            self._train(caesar_path)
        else:
            self._load()

    # ─────────────────────────────────────────────
    def _load(self):
        self._net = _MeasurementNet()
        self._net.load_state_dict(
            torch.load(self.model_path, map_location="cpu")
        )
        self._net.eval()
        self._scaler = joblib.load(self.scaler_path)
        print("✓ MannequinGenerator محمّل")

    # ─────────────────────────────────────────────
    def _train(
            self,
            caesar_path=None,
            epochs=600,
            batch_size=64,
            lr=0.001
    ):
        if caesar_path and os.path.exists(caesar_path):
            import pandas as pd
            print("تحميل CAESAR Dataset...")
            df = pd.read_csv(caesar_path)

            meas = df[[
                'height','weight','chest',
                'waist','hip','shoulder'
            ]].values.astype(np.float32)

            betas = df[
                [f'beta_{i}' for i in range(10)]
            ].values.astype(np.float32)

        else:
            print("⚠ CAESAR غير متوفرة — داتا اصطناعية")
            meas, betas = _synthetic_data(2000)

        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(meas).astype(np.float32)
        joblib.dump(self._scaler, self.scaler_path)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, betas, test_size=0.2, random_state=42
        )

        tr_ld = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_tr),
                torch.FloatTensor(y_tr)
            ),
            batch_size=batch_size,
            shuffle=True
        )

        val_ld = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            ),
            batch_size=batch_size
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        net = _MeasurementNet().to(device)

        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=lr,
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=30,
            factor=0.5
        )

        criterion = nn.MSELoss()

        best_val = float('inf')
        best_state = None
        no_improve = 0

        print(f"تدريب على {device}...")

        for epoch in range(epochs):
            net.train()

            for xb, yb in tr_ld:
                xb, yb = xb.to(device), yb.to(device)

                optimizer.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()

                nn.utils.clip_grad_norm_(
                    net.parameters(), 1.0
                )

                optimizer.step()

            net.eval()
            val_loss = 0.0

            with torch.no_grad():
                for xb, yb in val_ld:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += criterion(
                        net(xb), yb
                    ).item()

            scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {
                    k: v.clone().cpu()
                    for k, v in net.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Val: {val_loss:.4f}")

            if no_improve >= 80:
                print(f"إيقاف مبكر عند Epoch {epoch}")
                break

        net.load_state_dict(best_state)
        torch.save(net.state_dict(), self.model_path)

        net.to("cpu")
        self._net = net
        self._net.eval()

        print(f"✓ محفوظ | Val Loss: {best_val:.4f}")

    # ─────────────────────────────────────────────
    def measurementsToVector(
            self,
            height,
            weight,
            chest,
            waist,
            hip,
            shoulder
    ):

        raw = np.array([[
            height, weight, chest,
            waist, hip, shoulder
        ]], dtype=np.float32)

        scaled = self._scaler.transform(raw)
        tensor = torch.FloatTensor(scaled)

        with torch.no_grad():
            beta = self._net(tensor)

        return beta.numpy()[0].tolist()

    # ─────────────────────────────────────────────
    def generateMesh(
            self,
            beta_vector,
            gender="neutral",
            skin_hex="#D2B48C",
            output_path="models/mannequin.obj"
    ):
        try:
            import smplx
            import trimesh
        except ImportError:
            raise ImportError("ثبّت: pip install smplx trimesh")

        smplx_model = smplx.create(
            self.smplx_dir,
            model_type='smplx',
            gender=gender,      # ← هون صار gender فعلي
            use_face_contour=False,
            num_betas=10,
            num_expression_coeffs=10
        )

        beta_tensor = torch.FloatTensor([beta_vector])

        output = smplx_model(
            betas=beta_tensor,
            return_verts=True
        )

        vertices = output.vertices.detach().numpy()[0]
        faces = smplx_model.faces

        # تحويل HEX → RGB
        skin_hex = skin_hex.lstrip("#")
        rgb = tuple(
            int(skin_hex[i:i+2], 16)
            for i in (0, 2, 4)
        )

        colors = np.tile(
            [rgb[0], rgb[1], rgb[2], 255],
            (len(vertices), 1)
        )

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=colors
        )

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        mesh.export(output_path)

        print(f"✓ Mesh: {output_path}")
        print(f"✓ Gender: {gender}")
        print(f"✓ Skin: {skin_hex}")

        return output_path

    # ─────────────────────────────────────────────
    def render(
            self,
            mesh_file,
            skin_hex="#D2B48C"
    ):
        import trimesh

        mesh = trimesh.load(mesh_file)

        return {
            "mesh_path": mesh_file,
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),

            "bounds": {
                "min": mesh.bounds[0].tolist(),
                "max": mesh.bounds[1].tolist(),
            },

            "center": mesh.centroid.tolist(),

            "skin_color": skin_hex,   # ← مهم للـ frontend
            "ready_for_threejs": True
        }

    # ─────────────────────────────────────────────
    def get_size_info(
            self,
            height,
            weight,
            chest,
            waist,
            hip,
            shoulder
    ):
        bmi = weight / ((height/100)**2)

        build = (
            "نحيف" if bmi < 18.5 else
            "طبيعي" if bmi < 25 else
            "ممتلئ" if bmi < 30 else
            "سمين"
        )

        size = (
            "XS" if chest < 82 else
            "S" if chest < 88 else
            "M" if chest < 96 else
            "L" if chest < 104 else
            "XL" if chest < 112 else
            "XXL"
        )

        return {
            "bmi": round(bmi, 1),
            "build": build,
            "size": size
        }
