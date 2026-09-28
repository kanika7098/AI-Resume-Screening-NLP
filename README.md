# AI Resume Screening & Job Category Classification

An end-to-end NLP and machine learning project that analyzes resume text and predicts the most suitable job category. The project uses TF-IDF feature extraction with a Linear SVM classifier and provides a Streamlit interface for testing resumes.

## Project Overview

The model was trained on 9,000 resumes across 9 job categories. The workflow covers data exploration, text preprocessing, feature extraction, model training, evaluation, model saving, and deployment through Streamlit.

## Features

- Resume text preprocessing
- Exploratory data analysis
- TF-IDF feature extraction using unigrams and bigrams
- Linear SVM multiclass classification
- Accuracy, precision, recall and F1-score evaluation
- Confusion matrix
- Saved scikit-learn pipeline using Joblib
- PDF, DOCX and TXT resume upload
- Resume text preview
- Top predicted job categories with SVM decision scores
- Interactive Streamlit interface

## Dataset

The training dataset contains 9,000 resumes and 9 job categories. The public repository does not include the raw CSV because it may contain personal information and is larger than GitHub's standard per-file upload limit.

For local training, place `Resume dataset.csv` inside `data/` with these columns:

- `category`
- `job_title`
- `Text`

## Machine Learning Workflow

```text
Resume Dataset
      ↓
Text Cleaning
      ↓
Train/Test Split (80/20, stratified)
      ↓
TF-IDF (unigrams + bigrams)
      ↓
Linear SVM
      ↓
Model Evaluation
      ↓
Saved Joblib Pipeline
      ↓
Streamlit Application
```

## Model Performance

Test accuracy: **87.11%** on 1,800 held-out resumes.

The model is a multiclass Linear SVM trained on TF-IDF features. SVM decision scores shown in the application are ranking scores, not calibrated probabilities.

## Project Structure

```text
AI-Resume-Screening-NLP/
│
├── data/
│   └── README.md
├── models/
│   ├── resume_classifier.joblib
│   └── metrics.json
├── outputs/
│   └── training_summary.txt
├── Resume_Screening_NLP_Project.ipynb
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
├── Project_Report.txt
└── .gitignore
```

## Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit application

```bash
streamlit run app.py
```

### 3. Retrain the model (optional)

Place `Resume dataset.csv` in `data/` and run:

```bash
python train_model.py
```

## Technologies

Python, Pandas, NumPy, Scikit-learn, TF-IDF, Linear SVM, Matplotlib, Seaborn, Joblib, Streamlit, pypdf and python-docx.

## Limitations

The classifier predicts job categories based on patterns learned from the training dataset. The displayed SVM decision scores are not hiring probabilities. The application is intended as an educational/decision-support project and should not be used as the sole basis for recruitment decisions.

## Future Improvements

- Skill extraction
- Education and experience extraction
- Job description matching
- Semantic embeddings
- Transformer-based models
- Recruiter dashboard
