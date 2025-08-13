# Star–Galaxy Image Classification (CNN vs Random Forest)

**Author:** Prachi Diwan 
📄 **Full Dissertation (PDF):** [`docs/PrachiDiwan_MSc_FinalProject.pdf`](docs/FinalProjectWork_PrachiDiwan.pdf)  

> Binary classification of astronomical images into **stars** vs **galaxies**, comparing a from-scratch **Convolutional Neural Network (CNN)** with a **Random Forest (RF)** baseline.  
> Achieved **~92% accuracy** (CNN) and **~75% accuracy** (RF), demonstrating the advantages of deep learning for image-based astronomy.

---

## Project Overview
This project automates **star vs galaxy** classification for large-scale sky surveys, reducing manual work for astronomers.  
It compares:
- **Deep Learning (CNN)** – learns image features directly.
- **Classical Machine Learning (Random Forest)** – uses engineered features.

Dataset: **3,986 images** (64×64) – **3,044 stars**, **942 galaxies** (CC0 License). Class imbalance handled via augmentation.

---

## My Key Contributions
- Designed & implemented **CNN architecture** with **data augmentation** & **Dropout regularization**.
- Built **Random Forest baseline** for benchmarking.
- Addressed **class imbalance** and tuned hyperparameters for optimal results.
- Created **evaluation suite** with ROC curves, confusion matrices, and precision/recall/F1 scores.
- Wrote a comprehensive dissertation detailing methods, results, and future improvements.

---

## Results Summary

| Model        | Accuracy | ROC-AUC | Notes |
|--------------|----------|---------|-------|
| **CNN**      | ~92%     | ~0.98   | Strong separation between classes |
| **Random Forest** | ~75%     | ~0.81   | Good baseline; less robust for galaxies |
