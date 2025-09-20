# Amazon Food Reviews - Topic Modeling with LDA

## Project Overview
This project explores **unsupervised topic modeling** on Amazon Food Reviews using **Latent Dirichlet Allocation (LDA)**. The goal is to identify the main themes discussed by customers across hundreds of thousands of reviews, without manually reading each review. 

By uncovering topics like *delivery issues*, *taste*, or *packaging problems*, businesses can gain actionable insights into customer feedback.

---

## Dataset
- **Source:** [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Size:** ~500,000 reviews
- **Columns used:** 
  - `reviewText` → main review content  
  - `summary` → short headline of the review (optional for insights)

---

## Step 1: Data Cleaning & Preprocessing
- **Handle Missing Values:** Dropped reviews where `reviewText` is null; filled missing `summary` with empty strings.  
- **Remove Duplicates:** Ensured no duplicate reviews skew topics.  
- **Text Normalization:** Lowercased text, removed HTML tags, punctuation, and non-alphabetic characters.  
- **Tokenization & Lemmatization:** Converted text into meaningful tokens using **spaCy**, keeping only words longer than 2 characters.  
- **Stopword Removal:** Removed common English stopwords and domain-specific words.  
- **Rare & Frequent Word Filtering:** Excluded tokens appearing in fewer than 5 reviews or more than 50% of the corpus.

---

## Step 2: Text Vectorization
- **Bag-of-Words (BoW):** Created a dictionary and corpus using Gensim for LDA input.  
- **TF-IDF (Optional):** Can be used for NMF or LSA models.  
- **Embeddings (Future Work):** Sentence-BERT or transformer embeddings for semantic clustering.

---

## Step 3: LDA Topic Modeling
- **Model:** Gensim LDA with `num_topics=10`, `passes=10`, `alpha='auto'`, `eta='auto'`.  
- **Output:**  
  - Top words per topic (e.g., `"battery, charge, drain"` → "Battery Life")  
  - Dominant topic per review for downstream analysis.  
- **Evaluation:**  
  - **Coherence Score:** Measures semantic interpretability of topics.  
  - **Topic Diversity:** Checks overlap between topics.  
  - **Human Inspection:** Manual labeling of topics based on top words.

---

## Step 4: Insights
- Identified main customer discussion areas such as:
  - **Taste & Flavor**
  - **Packaging & Delivery**
  - **Price & Value**
  - **Product Quality**
- Can combine with sentiment analysis to highlight **negative vs positive topics**.

---

## Step 5: Next Steps / Future Work
- Experiment with **Mallet LDA** for improved topic coherence.  
- Compare with **NMF** and **LSA** on TF-IDF representations.  
- Apply **embedding-based clustering (Sentence-BERT + HDBSCAN)** for semantic topic discovery.  
- Visualize topics using **pyLDAvis** for interactive exploration.  

---

## Libraries & Tools
- Python 3.x  
- Pandas, NumPy  
- NLTK, spaCy  
- Gensim (Dictionary, Corpus, LDA)  
- Sentence-Transformers (optional, future work)  

---

## Conclusion
This project demonstrates a **professional pipeline for unsupervised topic modeling** on large-scale Amazon reviews. It provides insights into what customers talk about most frequently and can inform product improvements, marketing strategies, and customer service enhancements.

---

## Usage
1. Clone the repo.  
2. Install dependencies: `pip install -r requirements.txt`  
3. Run preprocessing and LDA scripts.  
4. Explore topics via `lda_model.print_topics()` or visualize with pyLDAvis.

---

## License
This project is for educational and professional demonstration purposes.
