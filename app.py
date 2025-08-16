# app.py
# QML Playground — Streamlit + PennyLane + PyTorch
# Binary classification on toy or real tabular data, with Quantum vs Classical comparison.

import os, math, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml

from sklearn.datasets import make_moons, make_circles, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc, confusion_matrix, average_precision_score, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression

# -----------------------------
# App config & seeds
# -----------------------------
st.set_page_config(page_title="QML Playground | Streamlit + PennyLane", layout="wide")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("⚙️ Settings")

ds = st.sidebar.selectbox("Dataset", ["make_moons (toy)", "make_circles (toy)", "Breast Cancer (UCI)"], index=0)
if "moons" in ds:
    noise = st.sidebar.slider("Noise", 0.00, 0.50, 0.25, 0.01)
    samples = st.sidebar.slider("Samples", 100, 2000, 600, 50)
elif "circles" in ds:
    noise = st.sidebar.slider("Noise", 0.00, 0.50, 0.08, 0.01)
    factor = st.sidebar.slider("Factor (gap)", 0.1, 0.9, 0.5, 0.05)
    samples = st.sidebar.slider("Samples", 100, 2000, 600, 50)
else:
    samples = None

test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
n_qubits = st.sidebar.slider("Qubits (PCA dims)", 2, 8, 3, 1)
circuit_type = st.sidebar.selectbox("Quantum circuit", ["StronglyEntanglingLayers", "BasicEntanglerLayers"], index=0)
layers = st.sidebar.slider("Layers", 1, 8, 3, 1)
shots_choice = st.sidebar.selectbox("Shots", ["analytic (None)", "256", "1024", "4096"], index=0)
shots = None if shots_choice.startswith("analytic") else int(shots_choice)
device_backend = st.sidebar.selectbox("Backend", ["default.qubit", "lightning.qubit"], index=0)

epochs = st.sidebar.slider("Epochs", 5, 300, 60, 5)
batch = st.sidebar.slider("Batch size", 8, 128, 32, 8)
lr = st.sidebar.select_slider("Learning rate", options=[0.1,0.05,0.02,0.01,0.005,0.002], value=0.02)
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.5, 0.01)
decision_grid = st.sidebar.slider("Decision grid (NxN)", 20, 100, 50, 10)
early_patience = st.sidebar.slider("Early stopping patience", 3, 20, 8, 1)

st.title("QML Playground | Variational Quantum Classifier")
st.caption("Train a PennyLane + PyTorch VQC, compare with a classical baseline, and explore decision boundaries & metrics.")

# -----------------------------
# Data helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_dataset(name, noise=None, samples=None, factor=None):
    if "moons" in name:
        X, y = make_moons(n_samples=samples, noise=noise, random_state=SEED)
        feature_names = ["x1", "x2"]
    elif "circles" in name:
        X, y = make_circles(n_samples=samples, noise=noise, factor=factor, random_state=SEED)
        feature_names = ["x1", "x2"]
    else:
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = list(data.feature_names)
    return X.astype(np.float32), y.astype(int), feature_names

X, y, feat_names = load_dataset(ds, noise=noise if 'noise' in locals() else None,
                                samples=samples, factor=factor if 'factor' in locals() else None)

# Scale + PCA
scaler = StandardScaler()
Xz = scaler.fit_transform(X)
n_qubits = min(n_qubits, Xz.shape[1])
pca = PCA(n_components=n_qubits, random_state=SEED)
Xp = pca.fit_transform(Xz)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(Xp, y, test_size=test_size, random_state=SEED, stratify=y)

def to_angles(X):
    Xz_ = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    Xz_ = np.clip(Xz_, -3, 3)
    return Xz_ * (math.pi/3.0)

# -----------------------------
# Quantum model
# -----------------------------
class VQC(nn.Module):
    def __init__(self, n_qubits, n_layers, circuit="StronglyEntanglingLayers", backend="default.qubit", shots=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.circuit = circuit
        self.dev = qml.device(backend, wires=n_qubits, shots=shots)

        if circuit == "StronglyEntanglingLayers":
            self.weights = nn.Parameter(0.01*torch.randn(n_layers, n_qubits, 3))
        else:
            # BasicEntanglerLayers has shape (n_layers, n_qubits)
            self.weights = nn.Parameter(0.01*torch.randn(n_layers, n_qubits))

        @qml.qnode(self.dev, interface="torch")
        def _qnode(x, weights):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
            if circuit == "StronglyEntanglingLayers":
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            else:
                qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        self.qnode = _qnode
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward_one(self, x_vec: torch.Tensor):
        expval = self.qnode(x_vec, self.weights)         # scalar
        logit = self.scale * expval + self.bias          # scalar
        return logit.unsqueeze(0).unsqueeze(1)           # (1,1)

    def forward(self, x):
        if x.dim() == 1:
            return self.forward_one(x)
        outs = []
        for i in range(x.shape[0]):
            outs.append(self.forward_one(x[i]))
        return torch.cat(outs, dim=0)                    # (B,1)

@st.cache_resource(show_spinner=False)
def make_model(n_qubits, n_layers, circuit, backend, shots):
    return VQC(n_qubits, n_layers, circuit=circuit, backend=backend, shots=shots)

def train_vqc(model, Xp_train, y_train, Xp_val, y_val, epochs=60, batch=32, lr=0.02, patience=8):
    Xtr = torch.tensor(to_angles(Xp_train).astype(np.float32))
    ytr = torch.tensor(y_train.astype(np.float32).reshape(-1,1))
    Xva = torch.tensor(to_angles(Xp_val).astype(np.float32))
    yva = y_val

    pos_weight = torch.tensor(((y_train==0).sum()+1e-9) / ((y_train==1).sum()+1e-9), dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"loss": [], "val_acc": []}
    best_acc, best_state, bad = -1.0, None, 0

    def batches():
        idx = np.arange(len(Xtr)); np.random.shuffle(idx)
        for i in range(0, len(idx), batch):
            j = idx[i:i+batch]
            yield Xtr[j], ytr[j]

    for ep in range(1, epochs+1):
        model.train()
        run = 0.0
        for xb, yb in batches():
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            run += loss.item()*len(xb)

        # validation
        model.eval()
        with torch.no_grad():
            logits = model(Xva)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            preds = (probs >= 0.5).astype(int)
            acc = accuracy_score(yva, preds)

        history["loss"].append(run/len(Xtr))
        history["val_acc"].append(float(acc))

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state: model.load_state_dict(best_state)
    return model, history

# -----------------------------
# Classical baseline
# -----------------------------
def train_baseline(Xp_train, y_train):
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xp_train, y_train)
    return clf

def evaluate_quantum(model, Xp, y, thr=0.5):
    X_angles = torch.tensor(to_angles(Xp).astype(np.float32))
    with torch.no_grad():
        logits = model(X_angles)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        preds = (probs >= thr).astype(int)
    return probs, preds

def evaluate_classical(clf, Xp, y, thr=0.5):
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(Xp)[:,1]
    else:
        s = clf.decision_function(Xp)
        probs = (s - s.min()) / (s.max()-s.min()+1e-9)
    preds = (probs >= thr).astype(int)
    return probs, preds

# -----------------------------
# Layout: tabs
# -----------------------------
tab_data, tab_train, tab_eval, tab_play = st.tabs(["📄 Dataset", "🧠 Train", "📊 Evaluate", "🎛️ Playground"])

# ----- Dataset preview -----
with tab_data:
    st.subheader("Dataset preview")
    st.write(f"Samples: **{len(X)}**, Features: **{X.shape[1]}** → PCA dims: **{n_qubits}**")
    df_prev = pd.DataFrame(Xp, columns=[f"PC{i+1}" for i in range(n_qubits)])
    df_prev["label"] = y
    st.dataframe(df_prev.head(20), use_container_width=True)

    if n_qubits >= 2:
        fig = px.scatter(
            df_prev, x="PC1", y="PC2", color=df_prev["label"].astype(str),
            title="Data in PCA space (first two components)", opacity=0.85
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Increase qubits (≥2) to see 2D scatter.")

# ----- Training -----
with tab_train:
    st.subheader("Train models")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚀 Train Quantum VQC"):
            model = make_model(n_qubits, layers, circuit_type, device_backend, shots)
            with st.spinner("Training VQC..."):
                model, hist = train_vqc(model, X_train, y_train, X_test, y_test,
                                        epochs=epochs, batch=batch, lr=lr, patience=early_patience)
            st.session_state["vqc"] = model.state_dict()
            st.session_state["vqc_meta"] = dict(n_qubits=n_qubits, layers=layers,
                                                circuit=circuit_type, backend=device_backend, shots=shots)
            st.session_state["hist"] = hist
            st.success("VQC trained and saved in session.")

    with c2:
        if st.button("⚡ Train Classical Baseline"):
            with st.spinner("Training Logistic Regression..."):
                clf = train_baseline(X_train, y_train)
            st.session_state["clf"] = joblib.dumps(clf)
            st.success("Baseline trained and saved in session.")

    # Show training curves if available
    if "hist" in st.session_state:
        h = st.session_state["hist"]
        fig1 = go.Figure(data=[go.Scatter(y=h["loss"], mode="lines", name="Loss")])
        fig1.update_layout(title="Training Loss", xaxis_title="Epoch", yaxis_title="Loss", height=280)
        fig2 = go.Figure(data=[go.Scatter(y=h["val_acc"], mode="lines", name="Val Acc")])
        fig2.update_layout(title="Validation Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy", height=280)
        colA, colB = st.columns(2)
        colA.plotly_chart(fig1, use_container_width=True)
        colB.plotly_chart(fig2, use_container_width=True)

# ----- Evaluation -----
with tab_eval:
    st.subheader("Evaluate & compare")

    # Restore models
    if "vqc" in st.session_state:
        meta = st.session_state["vqc_meta"]
        vqc = make_model(meta["n_qubits"], meta["layers"], meta["circuit"], meta["backend"], meta["shots"])
        vqc.load_state_dict(st.session_state["vqc"])
    else:
        vqc = None

    if "clf" in st.session_state:
        clf = joblib.loads(st.session_state["clf"])
    else:
        clf = None

    if vqc is None and clf is None:
        st.info("Train at least one model in the **Train** tab.")
    else:
        cols = st.columns(2)
        if vqc is not None:
            q_probs, q_preds = evaluate_quantum(vqc, X_test, y_test, thr=threshold)
            q_acc = accuracy_score(y_test, q_preds)
            q_prec, q_rec, q_f1, _ = precision_recall_fscore_support(y_test, q_preds, average="binary", zero_division=0)
            fpr, tpr, _ = roc_curve(y_test, q_probs)
            q_auc = auc(fpr, tpr)
            cols[0].metric("VQC Accuracy", f"{q_acc:.3f}")
            cols[0].metric("VQC F1", f"{q_f1:.3f}")
            cols[0].metric("VQC ROC-AUC", f"{q_auc:.3f}")
        if clf is not None:
            c_probs, c_preds = evaluate_classical(clf, X_test, y_test, thr=threshold)
            c_acc = accuracy_score(y_test, c_preds)
            c_prec, c_rec, c_f1, _ = precision_recall_fscore_support(y_test, c_preds, average="binary", zero_division=0)
            fpr_c, tpr_c, _ = roc_curve(y_test, c_probs)
            c_auc = auc(fpr_c, tpr_c)
            cols[1].metric("Baseline Accuracy", f"{c_acc:.3f}")
            cols[1].metric("Baseline F1", f"{c_f1:.3f}")
            cols[1].metric("Baseline ROC-AUC", f"{c_auc:.3f}")

        # Plots
        pp = []
        if vqc is not None:
            fpr, tpr, _ = roc_curve(y_test, q_probs)
            pp.append(go.Scatter(x=fpr, y=tpr, mode="lines", name="VQC"))
        if clf is not None:
            fpr_c, tpr_c, _ = roc_curve(y_test, c_probs)
            pp.append(go.Scatter(x=fpr_c, y=tpr_c, mode="lines", name="Baseline"))
        if pp:
            roc_fig = go.Figure(data=pp + [go.Scatter(x=[0,1], y=[0,1], mode="lines", name="chance", line=dict(dash="dash"))])
            roc_fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", height=320)
            st.plotly_chart(roc_fig, use_container_width=True)

        # Confusion matrices
        cm_cols = st.columns(2)
        if vqc is not None:
            cm = confusion_matrix(y_test, q_preds, labels=[0,1])
            cm_fig = px.imshow(cm, text_auto=True, aspect="equal", color_continuous_scale="Blues",
                               labels=dict(x="Pred", y="True", color="Count"),
                               x=["0","1"], y=["0","1"], title="VQC Confusion Matrix")
            cm_cols[0].plotly_chart(cm_fig, use_container_width=True)
        if clf is not None:
            cm = confusion_matrix(y_test, c_preds, labels=[0,1])
            cm_fig = px.imshow(cm, text_auto=True, aspect="equal", color_continuous_scale="Greens",
                               labels=dict(x="Pred", y="True", color="Count"),
                               x=["0","1"], y=["0","1"], title="Baseline Confusion Matrix")
            cm_cols[1].plotly_chart(cm_fig, use_container_width=True)

        # Decision boundary (first two PCs)
        if n_qubits >= 2:
            db_cols = st.columns(2)
            x_min, x_max = Xp[:,0].min()-0.5, Xp[:,0].max()+0.5
            y_min, y_max = Xp[:,1].min()-0.5, Xp[:,1].max()+0.5
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, decision_grid),
                np.linspace(y_min, y_max, decision_grid)
            )
            grid = np.c_[xx.ravel(), yy.ravel()]
            # pad remaining PCs with zeros if n_qubits>2
            if n_qubits > 2:
                pad = np.zeros((grid.shape[0], n_qubits-2))
                grid_full = np.hstack([grid, pad])
            else:
                grid_full = grid

            def grid_probs_quantum(model, Xgrid):
                Xa = torch.tensor(to_angles(Xgrid).astype(np.float32))
                with torch.no_grad():
                    logits = model(Xa)
                    return torch.sigmoid(logits).cpu().numpy().ravel()

            def grid_probs_classical(clf, Xgrid):
                if hasattr(clf, "predict_proba"):
                    return clf.predict_proba(Xgrid)[:,1]
                s = clf.decision_function(Xgrid)
                return (s - s.min())/(s.max()-s.min()+1e-9)

            if vqc is not None:
                qz = grid_probs_quantum(vqc, grid_full).reshape(xx.shape)
                fig_q = go.Figure(data=go.Contour(x=np.linspace(x_min, x_max, decision_grid),
                                                  y=np.linspace(y_min, y_max, decision_grid),
                                                  z=qz, contours_coloring="heatmap",
                                                  colorbar=dict(title="P(class=1)")))
                fig_q.add_trace(go.Scatter(x=Xp[:,0], y=Xp[:,1], mode="markers",
                                           marker=dict(color=y, colorscale="Viridis"),
                                           name="Data", opacity=0.7))
                fig_q.update_layout(title="VQC Decision Landscape (PC1, PC2)", height=420)
                db_cols[0].plotly_chart(fig_q, use_container_width=True)

            if clf is not None:
                cz = grid_probs_classical(clf, grid_full).reshape(xx.shape)
                fig_c = go.Figure(data=go.Contour(x=np.linspace(x_min, x_max, decision_grid),
                                                  y=np.linspace(y_min, y_max, decision_grid),
                                                  z=cz, contours_coloring="heatmap",
                                                  colorbar=dict(title="P(class=1)")))
                fig_c.add_trace(go.Scatter(x=Xp[:,0], y=Xp[:,1], mode="markers",
                                           marker=dict(color=y, colorscale="Viridis"),
                                           name="Data", opacity=0.7))
                fig_c.update_layout(title="Baseline Decision Landscape (PC1, PC2)", height=420)
                db_cols[1].plotly_chart(fig_c, use_container_width=True)
        else:
            st.info("Decision boundary requires at least 2 PCs (qubits).")

# ----- Playground -----
with tab_play:
    st.subheader("Interactive real-time inference")
    # Restore models if present
    vqc = make_model(n_qubits, layers, circuit_type, device_backend, shots)
    if "vqc" in st.session_state:
        vqc.load_state_dict(st.session_state["vqc"])
    else:
        vqc = None

    clf = joblib.loads(st.session_state["clf"]) if "clf" in st.session_state else None

    if n_qubits >= 2:
        st.write("Adjust the first two PCA components; others (if any) fixed at 0.")
        s1, s2 = st.columns(2)
        pc1 = s1.slider("PC1", float(Xp[:,0].min()-1), float(Xp[:,0].max()+1), float(Xp[:,0].mean()), 0.01)
        pc2 = s2.slider("PC2", float(Xp[:,1].min()-1), float(Xp[:,1].max()+1), float(Xp[:,1].mean()), 0.01)
    else:
        st.write("Adjust the single PCA component.")
        pc1 = st.slider("PC1", float(Xp[:,0].min()-1), float(Xp[:,0].max()+1), float(Xp[:,0].mean()), 0.01)
        pc2 = 0.0

    # Build input vector
    x_vec = np.zeros((1, n_qubits), dtype=np.float32)
    x_vec[0,0] = pc1
    if n_qubits >= 2:
        x_vec[0,1] = pc2

    cols = st.columns(3)
    if vqc is not None:
        with torch.no_grad():
            xv = torch.tensor(to_angles(x_vec).astype(np.float32)[0])
            p_q = float(torch.sigmoid(vqc(xv)).cpu().numpy().ravel()[0])
        cols[0].metric("VQC P(class=1)", f"{p_q:.3f}")
        cols[0].progress(min(max(p_q,0.0), 1.0))
    else:
        cols[0].info("Train VQC to enable.")

    if clf is not None:
        if hasattr(clf, "predict_proba"):
            p_c = float(clf.predict_proba(x_vec)[:,1][0])
        else:
            s = float(clf.decision_function(x_vec)[0])
            p_c = (s - s)  # trivial, but logistic is default so ignore
        cols[1].metric("Baseline P(class=1)", f"{p_c:.3f}")
        cols[1].progress(min(max(p_c,0.0), 1.0))
    else:
        cols[1].info("Train baseline to enable.")

    # Show the point on scatter if 2D
    if n_qubits >= 2:
        figp = px.scatter(x=Xp[:,0], y=Xp[:,1], color=y.astype(str), opacity=0.75,
                          labels={"x":"PC1","y":"PC2"}, title="Your point vs data (PCA space)")
        figp.add_trace(go.Scatter(x=[pc1], y=[pc2], mode="markers", marker=dict(size=12, symbol="x", line=dict(width=2)),
                                  name="Your point"))
        st.plotly_chart(figp, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.caption("Built with ❤️ using Streamlit, PennyLane, PyTorch, and scikit-learn.")
