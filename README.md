# Android Spyware Detection using Support Vector Machines (SVM)

This repository contains a complete and reproducible machine learning pipeline for detecting Android spyware using network traffic analysis.  
The project follows rigorous academic machine learning practices and is designed to be fully reproducible.

The system supports:
- **Binary classification**: Normal traffic vs Spyware traffic
- **Multiclass classification**: Identification of specific spyware families

---

## Project Structure

```text
android-spyware-svm/
├── data/
│   ├── Normal.csv
│   ├── FlexiSpy.csv
│   ├── MobileSpy.csv
│   ├── UMobix.csv
│   ├── TheWispy.csv
│   ├── Mspy1.csv
│   ├── Mspy2.csv
│   ├── FlexiSpy_Installation.csv
│   ├── MobileSpy_Installation.csv
│   ├── UMobix_Installation.csv
│   ├── TheWispy_Installation.csv
│   └── Mspy_Installation.csv
│
├── src/
│   └── train_svm_spyware.py
│
├── requirements.txt
├── README.md
└── .gitignore

Dataset Description

The dataset consists of network traffic captured from Android devices and exported as CSV files.
It includes both benign traffic and multiple spyware families, covering background activity and installation phases.

Classes

Normal (benign traffic)

Spyware families

FlexiSpy

MobileSpy

UMobix

TheWispy

Mspy

Due to privacy, size, and licensing constraints, the dataset is not included in this repository.

Methodology

The project follows a strict machine learning methodology to ensure valid and reproducible results:

Stratified train/test split

Stratified K-Fold cross-validation

Feature preprocessing using pipelines

Proper handling of categorical and numerical features

SMOTE applied only on training folds to avoid data leakage

Feature standardization

Linear Support Vector Machine (SVM) classifier

Separate pipelines for binary and multiclass classification

Installation

Clone the repository and install dependencies:

git clone https://github.com/YOUR_GITHUB_USERNAME/android-spyware-svm.git
cd android-spyware-svm
pip install -r requirements.txt

Usage

Place all CSV dataset files inside the data/ directory.

Run the training and evaluation script:

python src/train_svm_spyware.py --outdir artifacts

Outputs

After execution, the following artifacts are generated:

Trained models:

model_binary.joblib

model_multiclass.joblib

Confusion matrices (PNG format)

Detailed evaluation metrics (JSON and CSV)

Cross-validation performance scores

All outputs are saved inside the specified output directory (default: artifacts/).

Evaluation Metrics

The system reports:

Accuracy

Precision

Recall

F1-score

Confusion matrices for both binary and multiclass tasks

Per-class performance analysis

This allows a detailed understanding of spyware detection performance and class separability.

Reproducibility

Reproducibility is ensured through:

Fixed random seeds

Pipeline-based preprocessing

Proper cross-validation strategy

Model serialization using joblib

No manual data manipulation outside the pipeline

Technologies Used

Python 3

NumPy

Pandas

Scikit-learn

Imbalanced-learn

Matplotlib

Author

Moussamb Mohamed Oussein Dahalani
Master 2 – Distributed Artificial Intelligence
Université Paris Cité
