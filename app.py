import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration
st.set_page_config(page_title="Semantic Movie Recommender", layout="centered")
st.title("🎬 Pramanik's Movie Recommendation System")
st.write("Find your next favorite film using popularity and recommendation metrics.")

# 2. Cached Data Loading (Runs once, keeping the app lightning-fast)
@st.cache_resource
def load_data():
    with open('movie_recommendation_artifacts.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['df'], data['embeddings'], data['indices']

df, embeddings, indices = load_data()

# 3. Core Logic Function
def get_hybrid_recommendations(title, df, embeddings, indices, top_n=10, nlp_weight=0.5, quality_weight=0.3, pop_weight=0.2):
    idx = indices[title]
    if isinstance(idx, p_series := type(df['title'])): # Handle duplicate title series cases safely
        idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx[0]
        
    # Collection Logic
    movie_collection = df.iloc[idx]['belongs_to_collection']
    collection_recs = []
    if movie_collection and movie_collection != 'None':
        collection_df = df[(df['belongs_to_collection'] == movie_collection) & (df['title'] != title)]
        collection_recs = collection_df['title'].tolist()
        
    # Semantic Scoring
    source_vector = embeddings[idx].reshape(1, -1)
    semantic_scores = cosine_similarity(source_vector, embeddings).flatten()
    
    # Hybrid Combination
    hybrid_scores = (
        (semantic_scores * nlp_weight) + 
        (df['quality_score'] * quality_weight) + 
        (df['norm_popularity'] * pop_weight)
    )
    
    # Extract & Filter
    pool_size = top_n + len(collection_recs) + 10
    top_indices = hybrid_scores.argsort()[-pool_size:][::-1]
    
    nlp_recs = []
    for i in top_indices:
        rec_title = df.iloc[i]['title']
        if i != idx and rec_title not in collection_recs and rec_title != title:
            nlp_recs.append(rec_title)
            
    final_recs = collection_recs + nlp_recs
    return final_recs[:top_n]

# --- 4: New Interactive Search Feature ---
st.markdown("### 🔍 Search for a Movie")
search_query = st.text_input("Start typing a movie title (e.g., 'Aveng', 'Star Wars', 'Toy'):")

# Only proceed if the user has typed something
if search_query:
    # Filter the dataframe for titles containing the search query (case-insensitive)
    search_results = df[df['title'].str.contains(search_query, case=False, na=False)]['title'].unique()
    
    if len(search_results) > 0:
        # Let the user select the exact movie from the filtered results
        selected_movie = st.selectbox("Select your movie from the matches:", options=search_results)
        
        if st.button("Generate Recommendations", type="primary"):
            st.markdown("---")
            st.subheader(f"Because you selected **{selected_movie}**...")
            
            # Create 3 visual columns for our distinct profiles
            col1, col2, col3 = st.columns(3)
            
            # Profile 1: Semantic Focus (Similar Vibe)
            with col1:
                st.markdown("#### 🧠 Similar to your liking")
                st.caption("Matches the plot, genre, and overall vibe.")
                recs_semantic = get_hybrid_recommendations(
                    selected_movie, df, embeddings, indices, 
                    top_n=10, nlp_weight=0.7, quality_weight=0.2, pop_weight=0.1
                )
                for i, movie in enumerate(recs_semantic, 1):
                    st.write(f"{i}. {movie}")
                    
            # Profile 2: Popularity Focus (Currently Trending)
            with col2:
                st.markdown("#### 🔥 Currently Trending")
                st.caption("Similar movies that the masses are watching.")
                recs_trending = get_hybrid_recommendations(
                    selected_movie, df, embeddings, indices, 
                    top_n=10, nlp_weight=0.3, quality_weight=0.1, pop_weight=0.6
                )
                for i, movie in enumerate(recs_trending, 1):
                    st.write(f"{i}. {movie}")
                    
            # Profile 3: Quality Focus (High Rated)
            with col3:
                st.markdown("#### ⭐ Critically Acclaimed")
                st.caption("Highly rated movies with similar themes.")
                recs_quality = get_hybrid_recommendations(
                    selected_movie, df, embeddings, indices, 
                    top_n=10, nlp_weight=0.4, quality_weight=0.6, pop_weight=0.0
                )
                for i, movie in enumerate(recs_quality, 1):
                    st.write(f"{i}. {movie}")
                    
    else:
        st.warning("No movies found matching that text. Try another search!")