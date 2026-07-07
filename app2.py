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

# 4. User Interface Sidebar Settings
st.sidebar.header("🎛️ Advanced filtering")
nlp_w = st.sidebar.slider("Semantic Meaning", 0.0, 1.0, 0.5, 0.1)
qual_w = st.sidebar.slider("Rating & Vote Quality", 0.0, 1.0, 0.3, 0.1)
pop_w = st.sidebar.slider("Trending Popularity", 0.0, 1.0, 0.2, 0.1)

# Ensure normal weights configuration
total_w = nlp_w + qual_w + pop_w
if total_w > 0:
    nlp_w, qual_w, pop_w = nlp_w/total_w, qual_w/total_w, pop_w/total_w

# 5. Native Autocomplete Search Box
# st.selectbox naturally filters down choices dynamically as the user types
movie_list = sorted(list(indices.keys()))
selected_movie = st.selectbox(
    "Type or select a movie to get started:",
    options=movie_list,
    index=0 if "Toy Story" not in movie_list else movie_list.index("Toy Story")
)

# 6. Trigger and Render Recommendations
if st.button("Generate Recommendations", type="primary"):
    with st.spinner("Analyzing semantics and trends..."):
        recommendations = get_hybrid_recommendations(
            selected_movie, df, embeddings, indices, 
            top_n=10, nlp_weight=nlp_w, quality_weight=qual_w, pop_weight=pop_w
        )
        
    st.subheader(f"Because you watched **{selected_movie}**:")
    
    # Display recommendations nicely in styled cards
    for idx, movie in enumerate(recommendations, 1):
        st.markdown(f"**{idx}. {movie}**")