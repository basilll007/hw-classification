# =========================================
# src/inference.py
# =========================================
import os
import json
import numpy as np
import joblib

# --- Optional torch fallback (if ONNX not available) ---
try:
    import onnxruntime as ort
    _HAS_ONNX = True
except Exception:
    _HAS_ONNX = False

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# -------- Paths --------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "scalers", "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "scalers", "label_encoder.pkl")
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "models", "cancer_classifier.onnx")
PTH_MODEL_PATH  = os.path.join(BASE_DIR, "models", "cancer_classifier.pth")

# -------- Artifacts --------
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# ======== TRAIN-TIME FEATURE CONTRACT ========
# This is the exact feature order used for training (after one-hot for Disease_Status)
# If your train step produced a different set/order, update this list accordingly.
FEATURE_ORDER = [
    "Gene_E_Housekeeping",
    "Gene_A_Oncogene",
    "Gene_B_Immune",
    "Gene_C_Stromal",
    "Gene_D_Therapy",
    "Pathway_Score_Inflam",
    "UMAP_1",
    "Disease_Status_Tumor"  # one-hot column (0 or 1)
]

def _coerce_disease_status(d):
    """
    Accepts: 'Tumor' | 'Normal' | 'Healthy' | 1 | 0 | True | False
    Returns the one-hot value for Disease_Status_Tumor.
    """
    if isinstance(d, (int, float, bool)):
        return 1.0 if float(d) != 0.0 else 0.0
    if isinstance(d, str):
        s = d.strip().lower()
        if s in ("tumor", "tumour", "1", "true", "yes"):
            return 1.0
        return 0.0
    return 0.0

def preprocess_input(raw: dict) -> np.ndarray:
    """
    Convert raw dict -> ordered numpy row (1, n_features),
    handle one-hot for Disease_Status, and apply the saved scaler.
    """
    # Accept either Disease_Status or Disease_Status_Tumor in input
    if "Disease_Status" in raw and "Disease_Status_Tumor" not in raw:
        raw["Disease_Status_Tumor"] = _coerce_disease_status(raw["Disease_Status"])
    elif "Disease_Status_Tumor" in raw:
        raw["Disease_Status_Tumor"] = _coerce_disease_status(raw["Disease_Status_Tumor"])
    else:
        # default to 1.0 if not provided (your earlier demo)
        raw["Disease_Status_Tumor"] = 1.0

    # Build feature vector in the exact FEATURE_ORDER
    try:
        row = [float(raw[k]) for k in FEATURE_ORDER]
    except KeyError as e:
        missing = str(e).strip("'")
        raise ValueError(f"Missing required feature: {missing}. "
                         f"Expected keys: {FEATURE_ORDER}")
    except ValueError as e:
        raise ValueError(f"Non-numeric value encountered: {e}")

    X = np.array([row], dtype=np.float32)               # (1, n_features)
    X_scaled = scaler.transform(X).astype(np.float32)   # apply train-time scaler
    return X_scaled

# ======== ONNX primary path ========
class ONNXPredictor:
    def __init__(self, model_path: str):
        if not _HAS_ONNX:
            raise RuntimeError("onnxruntime not available.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
    def predict_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        out = self.session.run(None, {self.input_name: X_scaled})[0]
        # out is logits. softmax:
        e = np.exp(out - np.max(out, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X_scaled)
        return np.argmax(proba, axis=1)

# ======== PyTorch fallback (architecture included) ========
class DeepCancerClassifier(nn.Module):
    """
    Mirrors the deep MLP you trained:
    512 -> 256 -> 128 -> 64 -> 32 -> out
    with BatchNorm, LeakyReLU, Dropout(0.3)
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Linear(i, o),
                nn.BatchNorm1d(o),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.3),
            )
        self.backbone = nn.Sequential(
            block(input_dim, 512),
            block(512, 256),
            block(256, 128),
            block(128, 64),
            block(64, 32),
        )
        self.head = nn.Linear(32, output_dim)

    def forward(self, x):
        return self.head(self.backbone(x))

class TorchPredictor:
    def __init__(self, pth_path: str, input_dim: int, output_dim: int):
        if not _HAS_TORCH:
            raise RuntimeError("torch not available.")
        if not os.path.exists(pth_path):
            raise FileNotFoundError(f"PyTorch model not found: {pth_path}")
        self.device = torch.device("cpu")
        self.model = DeepCancerClassifier(input_dim, output_dim).to(self.device)
        state = torch.load(pth_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            logits = self.model(x).cpu().numpy()
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X_scaled)
        return np.argmax(proba, axis=1)

# ======== Resolver: ONNX preferred, Torch fallback ========
def _build_predictor():
    if _HAS_ONNX and os.path.exists(ONNX_MODEL_PATH):
        return ONNXPredictor(ONNX_MODEL_PATH), "onnx"
    if _HAS_TORCH and os.path.exists(PTH_MODEL_PATH):
        # output_dim from label encoder
        return TorchPredictor(PTH_MODEL_PATH, input_dim=len(FEATURE_ORDER),
                              output_dim=len(label_encoder.classes_)), "torch"
    raise RuntimeError(
        "No inference backend available. "
        "Ensure 'models/cancer_classifier.onnx' or 'models/cancer_classifier.pth' exists, "
        "and install 'onnxruntime' and/or 'torch'."
    )

_predictor, _backend = _build_predictor()

def predict_one(raw_features: dict):
    """
    Public API: accepts raw dict, returns label + probabilities.
    """
    X_scaled = preprocess_input(raw_features)
    cls_idx = _predictor.predict(X_scaled)[0]
    label = label_encoder.inverse_transform([cls_idx])[0]
    proba = _predictor.predict_proba(X_scaled)[0].tolist()

    return {
        "backend": _backend,
        "prediction": label,
        "class_index": int(cls_idx),
        "classes": label_encoder.classes_.tolist(),
        "probabilities": proba
    }
