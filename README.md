# DAF-MoE: Distribution-Aware Feature-level Mixture of Experts

This repository contains the official implementation of DAF-MoE Transformer for Tabular Data.

## 📌 Features
- **Symmetric Rareness Expansion**: Handles imbalanced categorical data.
- **Induction-based Routing**: Routes tokens based on statistical distribution ($P, \gamma$).
- **Dual-Gated Expert**: Separates reasoning and preservation paths.

## 🚀 Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt