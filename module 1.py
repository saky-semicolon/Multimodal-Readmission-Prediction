"""
# Module 1

Module 1 focused on building a robust, reproducible pipeline for extracting, cleaning, and engineering features from both structured and unstructured MIMIC-III ICU data. Structured data processing included demographic attributes (age, gender, ethnicity) and temporal summaries of vital signs (mean, min, max, standard deviation, slope) derived from CHARTEVENTS. These were normalized, encoded, and saved as baseline features. For unstructured data, the module filtered and cleaned discharge summaries from NOTEEVENTS, applied ClinicalBERT to extract contextual embeddings from each note, and saved the resulting [CLS] token representations. This module served as a foundational step to prepare high-quality inputs for single-modality baselines and multimodal fusion experiments.
"""

import os
import pandas as pd
import numpy as np
import torch
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import re
from sklearn import __version__ as sklearn_version
from packaging import version

# === CONFIG ===
DATA_RAW_DIR = '/kaggle/input/mmic-dataset/mimic-iii-clinical-database-1.4'
# ✅ This is writable
DATA_PROCESSED_DIR = '/kaggle/working/output_data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
MIMIC_FILES = {
    "admissions": os.path.join(DATA_RAW_DIR, "ADMISSIONS.csv/ADMISSIONS.csv"),
    "patients": os.path.join(DATA_RAW_DIR, "PATIENTS.csv/PATIENTS.csv"),
    "chartevents": os.path.join(DATA_RAW_DIR, "CHARTEVENTS.csv/CHARTEVENTS.csv"),  # gzipped
    "labevents": os.path.join(DATA_RAW_DIR, "LABEVENTS.csv/LABEVENTS.csv"),
    "noteevents": os.path.join(DATA_RAW_DIR, "NOTEEVENTS.csv/NOTEEVENTS.csv")
}

STRUCTURED_SAVE_PATH = os.path.join(DATA_PROCESSED_DIR, "structured_features.csv")
EMBEDDING_SAVE_PATH = os.path.join(DATA_PROCESSED_DIR, "bert_embeddings.pt")

# Common ITEMIDs for ICU vitals (MIMIC-III specific)
COMMON_VITALS = {
    'HeartRate': 211,
    'SysBP': 51,
    'DiasBP': 8368,
    'MeanBP': 52,
    'RespRate': 618,
    'Temp': 223761,
    'SpO2': 646
}

# === FUNCTIONS ===

def load_structured_data(adm_path, pat_path, chart_path, lab_path, sample_size=1_000_000):
    admissions = pd.read_csv(adm_path)
    patients = pd.read_csv(pat_path)
    chart = pd.read_csv(chart_path, nrows=sample_size)
    lab = pd.read_csv(lab_path, nrows=sample_size)
    df = pd.merge(admissions, patients, on='SUBJECT_ID', how='inner')
    return df, chart, lab

def handle_missing_and_normalize(df, categorical_cols, numeric_cols, impute_strategy='knn'):
    if impute_strategy == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif impute_strategy == 'median':
        imputer = SimpleImputer(strategy='median')
    else:
        imputer = KNNImputer(n_neighbors=5)

    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    if version.parse(sklearn_version) >= version.parse("1.2"):
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    else:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    cat_encoded = encoder.fit_transform(df[categorical_cols])
    cat_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(categorical_cols))

    df.reset_index(drop=True, inplace=True)
    cat_df.reset_index(drop=True, inplace=True)

    return pd.concat([df[['SUBJECT_ID', 'HADM_ID']], df[numeric_cols], cat_df], axis=1)

def extract_vital_trends(chart_df, vitals_list, max_rows=1_000_000):
    chart_df = chart_df[chart_df['ITEMID'].isin(vitals_list)]
    chart_df = chart_df[['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM']].dropna()
    chart_df = chart_df.head(max_rows)

    chart_df['CHARTTIME'] = pd.to_datetime(chart_df['CHARTTIME'])

    trend_features = []

    for (hadm_id, item_id), group in chart_df.groupby(['HADM_ID', 'ITEMID']):
        group = group.sort_values('CHARTTIME')
        times = (group['CHARTTIME'] - group['CHARTTIME'].min()).dt.total_seconds().values.reshape(-1, 1)
        values = group['VALUENUM'].values

        if len(values) < 3:
            continue

        try:
            lr = LinearRegression()
            lr.fit(times, values)
            slope = lr.coef_[0]
        except:
            slope = np.nan

        summary = {
            'HADM_ID': hadm_id,
            'ITEMID': item_id,
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'slope': slope
        }
        trend_features.append(summary)

    trend_df = pd.DataFrame(trend_features)
    pivot = trend_df.pivot(index='HADM_ID', columns='ITEMID')
    pivot.columns = ['{}_{}'.format(stat, item) for stat, item in pivot.columns]
    pivot.reset_index(inplace=True)
    return pivot

def clean_discharge_summaries(df):
    df = df[df['CATEGORY'] == 'Discharge summary']
    df['TEXT'] = df['TEXT'].str.replace(r"\[.*?\]", "", regex=True)
    df['TEXT'] = df['TEXT'].str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
    df['TEXT'] = df['TEXT'].str.lower()
    return df[['SUBJECT_ID', 'HADM_ID', 'TEXT']]

def embed_texts(texts, model_name='emilyalsentzer/Bio_ClinicalBERT', batch_size=512):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Batches"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        embeddings.extend(cls_embeddings)

    return torch.stack(embeddings)


# === EXECUTION ===

def run_module1():
    # Load data
    adm, chart, lab = load_structured_data(
        MIMIC_FILES['admissions'],
        MIMIC_FILES['patients'],
        MIMIC_FILES['chartevents'],
        MIMIC_FILES['labevents']
    )

    print("✅ Loaded admissions, patients, chart, and lab data.")

    # Age calculation
    adm['DOB'] = pd.to_datetime(adm['DOB'], errors='coerce')
    adm['ADMITTIME'] = pd.to_datetime(adm['ADMITTIME'], errors='coerce')
    adm['AGE'] = adm['ADMITTIME'].dt.year - adm['DOB'].dt.year

    # Prepare basic structured features
    features = adm[['SUBJECT_ID', 'HADM_ID', 'GENDER', 'ETHNICITY', 'AGE']]
    basic_struct = handle_missing_and_normalize(features, ['GENDER', 'ETHNICITY'], ['AGE'])
    print("✅ Processed demographic structured features.")

    # Vital trends
    vital_trends = extract_vital_trends(chart_df=chart, vitals_list=list(COMMON_VITALS.values()))
    print("✅ Extracted and computed vital sign trends.")

    # Merge all structured features
    merged = pd.merge(basic_struct, vital_trends, on='HADM_ID', how='left')
    merged.fillna(0, inplace=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    merged.to_csv(STRUCTURED_SAVE_PATH, index=False)
    print(f"✅ Final structured data saved to: {STRUCTURED_SAVE_PATH}")

    # Process discharge summaries
    notes = pd.read_csv(MIMIC_FILES['noteevents'])
    clean_notes = clean_discharge_summaries(notes)
    texts = clean_notes['TEXT'].tolist()

    # Generate BERT embeddings
    embeddings = embed_texts(texts)
    torch.save(embeddings, EMBEDDING_SAVE_PATH)
    print(f"✅ BERT text embeddings saved to: {EMBEDDING_SAVE_PATH}")

    # Save HADM_IDs used for embedding alignment
    embedding_index = clean_notes[['HADM_ID']].reset_index(drop=True)
    embedding_index_path = os.path.join(DATA_PROCESSED_DIR, "embedding_index.csv")
    embedding_index.to_csv(embedding_index_path, index=False)
    print(f"✅ Embedding index (HADM_ID) saved to: {embedding_index_path}")


if __name__ == "__main__":
    run_module1()
