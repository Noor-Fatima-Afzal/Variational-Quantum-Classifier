# 🧪 QML Playground | Variational Quantum Classifier

An **interactive Streamlit app** that lets you experiment with **Quantum Machine Learning (QML)** for binary classification.  
It uses **PennyLane + PyTorch** to implement a **Variational Quantum Classifier (VQC)** and compares it with a **classical baseline (Logistic Regression)**.  

You can:
- Choose toy or real datasets
- Configure quantum circuit parameters
- Train and evaluate a VQC vs a classical model
- Visualize training curves, ROC, confusion matrices, and decision boundaries
- Do **real-time inference** with sliders

---

## ✨ Features

- **Datasets**
  - 🌓 `make_moons` (toy)
  - ⭕ `make_circles` (toy)
  - 🫁 Breast Cancer (UCI, sklearn)

- **Quantum model**
  - Angle embedding → StronglyEntanglingLayers / BasicEntanglerLayers
  - Configurable **qubits, layers, backend, shots**
  - Trained with **PyTorch + PennyLane**

- **Classical baseline**
  - Logistic Regression for comparison

- **Visualizations**
  - Training curves (loss, validation accuracy)
  - ROC curves, confusion matrices
  - 2D decision landscapes (PC1 vs PC2)
  - Real-time inference with interactive sliders

---

## 🧭 How to Use

1. **Choose dataset** in the sidebar  
   - `moons`  
   - `circles`  
   - `breast cancer`  

2. **Adjust circuit hyperparameters**  
   - **Qubits** (via PCA)  
   - **Layers**  
   - **Backend** (`default.qubit` or `lightning.qubit`)  
   - **Shots** (analytic or finite)  

3. **Train**  
   - Quantum **VQC**  
   - Classical **baseline**  

4. **Evaluate results**  
   - Accuracy, F1, ROC-AUC  
   - Confusion matrices  
   - Decision boundary plots  

5. **Playground tab**  
   - Move sliders for PCA components  
   - See real-time quantum vs classical predictions  

---

## 📖 Notes

- Decision boundary plots require **≥2 qubits (PCA components)**.  
- Logistic regression is used as a simple baseline — you can replace with **SVM/NN** if desired.  
- VQC uses **early stopping** to prevent overfitting.  
- Works with **analytic simulation** (`shots=None`) or **finite shots** (e.g., `1024`).  

---

## 🙏 Acknowledgments

- [PennyLane](https://pennylane.ai) — Hybrid quantum-classical ML  
- [Streamlit](https://streamlit.io) — Fast interactive apps  
- [PyTorch](https://pytorch.org) — Deep learning engine  
- [scikit-learn](https://scikit-learn.org) — Classical ML baselines  

