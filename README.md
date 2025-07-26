# 🎬 Movie Recommendation System

This project is a hybrid Movie Recommendation System built using **Python**, **Pandas**, and **Scikit-learn**. It combines **Collaborative Filtering** and **Content-Based Filtering** to provide intelligent movie suggestions based on user ratings and metadata.

---

## 📌 Features

- 📊 Content-Based Filtering using genres and overviews
- 🤝 Collaborative Filtering using user rating patterns
- 🧠 Cosine Similarity for measuring similarity
- 📈 Model training and results in Jupyter Notebook
- 📄 Project report included (`Report.docx`)

---

## 📁 Project Structure

```
📦movie-recommendation-system/
├── notebook.ipynb           # Main notebook for model training
├── movies_metadata.csv      # Movie metadata
├── ratings_small.csv        # User ratings
├── Report.docx              # Final report (2 pages)
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Open `notebook.ipynb` in Jupyter Notebook or Google Colab.

3. Run the cells step-by-step to train and test the recommendation system.

---

## 🧠 Model Overview

- **Content-Based Filtering**:
  - Uses `CountVectorizer` to encode movie genres
  - Computes cosine similarity between movie features

- **Collaborative Filtering**:
  - Builds user-item matrix
  - Uses cosine similarity to suggest movies based on similar user preferences

---

## 📚 Datasets Used

- **movies_metadata.csv** – Metadata like title, genres, etc.
- **ratings_small.csv** – User-movie rating dataset

---

## ✅ Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📄 Report

A 2-page Word report is included as `Report.docx`. It summarizes the objectives, methodology, datasets, models used, and evaluation.

---

