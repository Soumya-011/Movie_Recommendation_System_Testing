# 🎬 Smart Hybrid Movie Recommendation System

An intelligent, hybrid movie recommendation engine built with Python and Streamlit. This application goes beyond standard keyword matching by using state-of-the-art NLP Transformer models to understand the semantic meaning of movie plots, while also factoring in real-world popularity and critical acclaim.

## 🚀 Features

* **Semantic Search:** Utilizes Hugging Face's `all-MiniLM-L6-v2` to generate dense vector embeddings, allowing the system to recommend movies based on plot vibe and meaning rather than just exact word matches.
* **Hybrid Scoring Engine:** Blends qualitative data (semantic similarity) with quantitative data (vote average, vote count, and popularity) to generate dynamic recommendations.
* **Smart Search Autocomplete:** Users can type partial movie titles (e.g., "Aveng") to instantly filter and find their target movie.
* **Customized Recommendation Profiles:** * 🧠 *Similar to your liking:* Weighted heavily towards plot and semantic similarities.
  * 🔥 *Currently Trending:* Weighted towards what is currently popular with audiences.
  * ⭐ *Critically Acclaimed:* Weighted towards highly-rated movies with similar themes.
* **Collection Priority:** Automatically recognizes and prioritizes movies within the same franchise/collection (e.g., recommending *Toy Story 2* if you search for *Toy Story*).

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **Machine Learning / NLP:** Sentence-Transformers (`all-MiniLM-L6-v2`), Scikit-learn (Cosine Similarity, MinMaxScaler)
* **Data Manipulation:** Pandas, NumPy

## ⚙️ Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Soumya-011/Movie_Recommendation_System_Testing.git](https://github.com/Soumya-011/Movie_Recommendation_System_Testing.git)
   cd Movie_Recommendation_System_Testing