# Android Spyware Detection using Support Vector Machines (SVM)

This repository contains a complete and reproducible machine learning pipeline for detecting Android spyware based on network traffic analysis. The project was developed as part of an academic research work and follows rigorous evaluation and preprocessing practices.

The system supports:
- Binary classification: Normal traffic vs Spyware traffic
- Multiclass classification: Identification of specific spyware families

## Project Structure

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
├── src/
│   └── train_svm_spyware.py
├── requirements.txt
├── README.md
└── .gitignore

## Dataset Description

The dataset consists of network traffic captured from Android devices and exported as CSV files. It includes both benign traffic and multiple spyware families, covering background activity and installation phases.

Classes included:
- Normal (benign traffic)
- Spyware families: FlexiSpy, MobileSpy, UMobix, TheWispy, Mspy

Due to privacy, size, and licensing constraints, the dataset is not included in this repository.

## Methodology

The project follows a strict machine learning methodology to ensure valid and reproducible results:
- Stratified train/test split
- Stratified K-Fold cross-validation
- Feature preprocessing using pipelines
- Proper handling of categorical and numerical features
- SMOTE applied only on training folds to avoid data leakage
- Feature standardization
- Linear Support Vector Machine classifier
- Separate pipelines for binary and multiclass classification

## Installation

Clone the repository and install dependencies:

git clone https://github.com/YOUR_GITHUB_USERNAME/android-spyware-svm.git
cd android-spyware-svm
pip install -r requirements.txt

## Usage

Place all CSV dataset files inside the data/ directory and run:

python src/train_svm_spyware.py --outdir artifacts

## Outputs

The execution produces:
- Trained binary and multiclass SVM models
- Confusion matrices in image format
- Detailed evaluation metrics in JSON and CSV formats
- Cross-validation performance scores

All outputs are saved in the specified output directory.

## Evaluation Metrics

The system reports accuracy, precision, recall, F1-score, and confusion matrices for both classification tasks, allowing detailed analysis of spyware detection performance.

## Reproducibility

Reproducibility is ensured through fixed random seeds, pipeline-based preprocessing, model serialization using joblib, and the absence of manual data manipulation.

## Technologies Used

Python, NumPy, Pandas, Scikit-learn, Imbalanced-learn, Matplotlib

## Author

Moussamb Mohamed Oussein Dahalani  
Master 2 – Distributed Artificial Intelligence  
Université Paris Cité

## License

This project is intended for academic and research purposes only.
