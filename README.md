# 📚✨ Hybrid Book Recommendation System  
_Combining NLP-based Content Filtering with Collaborative Filtering (SVD)_

![Python](https://img.shields.io/badge/Python-3.x-blue)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-orange)
![Recommender](https://img.shields.io/badge/System-Hybrid-green)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

## 📌 Overview  
This project implements a **Hybrid Book Recommendation System** that combines:

- **Content-Based Filtering** using Natural Language Processing (NLP) on book metadata (titles, descriptions, genres).  
- **Collaborative Filtering** using **matrix factorization (SVD)** on user–item rating data.  

The goal is to generate **personalized, context-aware book recommendations** by leveraging both what a user has liked in the past and how similar books are in terms of content.

---


## 🎯 Project Objectives  

- Build a **hybrid recommendation engine** for books using:  
  - TF–IDF + cosine similarity for **content-based recommendations**.  
  - SVD-based collaborative filtering for **rating prediction**.  
- Support **top-N recommendations** for a given user or book.  
- Provide a clean, reproducible **notebook workflow** (`book-recommender-system.ipynb`).  
- Visualize the ecosystem of **RecSys algorithms** using the diagrams above.

---

## 🧰 Tech Stack  

### Languages & Tools  
- Python  
- Jupyter Notebook  

### Python Libraries  
- Pandas, NumPy  
- Scikit-learn  
- Surprise  
- Matplotlib, Seaborn  

---

## 📂 Project Structure  

```bash
📦 Hybrid-Book-recommendation-system/
│
├── book-recommender-system.ipynb    # Main notebook: EDA + models + hybrid logic
├── data/                            # (Books, ratings & metadata files)
│   ├── ...
│
├── images/                          
│   ├── Deep learnng recmd algorithm.png
│   ├── classic recmdn algorithm.png
│   └── rec_systm_flowchart.png
│
├── README.md                        
├── LICENSE                          
└── requirements.txt                 
````

---

## 🔍 Exploratory Data Analysis (EDA)

Performed inside `book-recommender-system.ipynb`:

* Data shape, missing values, distributions
* Ratings per user/book
* Popularity vs niche analysis
* Text field inspection (titles, descriptions, genres)
* Building lookup dictionaries

---

## 🤖 Modeling Approach

### 1️⃣ Content-Based Filtering (NLP)

Uses TF–IDF + Cosine Similarity to match books based on text features.

```python
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(books["description"].fillna(""))

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
```

---

### 2️⃣ Collaborative Filtering (SVD)

Uses matrix factorization via the Surprise library.

```python
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[["user_id", "book_id", "rating"]], reader)

algo = SVD()
trainset = data.build_full_trainset()
algo.fit(trainset)

def predict_rating(user_id, book_id):
    return algo.predict(user_id, book_id).est
```

---

### 3️⃣ Hybrid Recommendation Logic

Combines CF + Content similarities:

```
HybridScore = 0.6 × PredictedRating + 0.4 × ContentSimilarity
```

Produces accurate, context-aware recommendations.

---

## 📊 Example Outputs

* Top-N similar books (content-based)
* Top-N hybrid recommended books (user-specific)
* Metrics such as **RMSE**, **MAE** from SVD
* Comparison with baseline recommenders

---

## 🔮 Possible Extensions

* Use **BERT / Sentence Transformers** instead of TF–IDF
* Implement **Neural CF (NeuMF)** or **DeepFM**
* Add a Streamlit UI
* Integrate Knowledge Graphs
* Improve cold-start handling

---

## 📬 Author

**Nandith Burla**
B.Tech — Data Science & Engineering

#### GitHub: [https://github.com/nandithburla](https://github.com/nandithburla)

#### LinkedIn: [https://www.linkedin.com/in/nandithburla/](https://www.linkedin.com/in/nandithburla/)

---

