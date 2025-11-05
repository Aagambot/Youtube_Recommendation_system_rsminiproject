import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import warnings
import gensim.downloader as api

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="YouTube Recommender",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for YouTube-like styling
st.markdown("""
<style>
    .main {
        background-color: #0f0f0f;
    }
    .stApp {
        background-color: #0f0f0f;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    .video-card {
        background-color: #1f1f1f;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        transition: transform 0.2s;
        border: 1px solid #303030;
    }
    .video-card:hover {
        transform: translateY(-5px);
        background-color: #2a2a2a;
    }
    .video-title {
        color: #ffffff;
        font-size: 16px;
        font-weight: 500;
        margin: 10px 0 5px 0;
        line-height: 1.4;
    }
    .video-stats {
        color: #aaaaaa;
        font-size: 13px;
        margin: 5px 0;
    }
    .keyword-badge {
        background-color: #3ea6ff;
        color: white;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 12px;
        display: inline-block;
        margin: 5px 0;
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .search-container {
        background-color: #1f1f1f;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #3ea6ff;
        color: white;
        border-radius: 20px;
        padding: 8px 24px;
        border: none;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #2d8fd8;
    }
    .sidebar .sidebar-content {
        background-color: #1f1f1f;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# YouTubeRecommender Class
class YouTubeRecommender:
    def __init__(self, df):
        self.df = df.copy()
        self.word2vec_model = None
        self.content_embeddings = None
        self.content_similarity = None
        
    @st.cache_resource
    def load_word2vec_model(_self):
        try:
            model = api.load("glove-wiki-gigaword-50")
            return model
        except:
            return None
    
    def get_word_embedding(self, text):
        if self.word2vec_model is None:
            return None
        words = text.lower().split()
        embeddings = [self.word2vec_model[word] for word in words if word in self.word2vec_model]
        if not embeddings:
            return np.zeros(self.word2vec_model.vector_size)
        return np.mean(embeddings, axis=0)
    
    def prepare_features(self):
        for col in ['Likes', 'Comments', 'Views']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int)
        
        self.df['engagement_rate'] = (
            (self.df['Likes'] + self.df['Comments']) / (self.df['Views'] + 1)
        ) * 100
        
        scaler = MinMaxScaler()
        cols = [c for c in ['Views', 'Likes', 'Comments'] if c in self.df.columns]
        if cols:
            self.df['popularity_score'] = scaler.fit_transform(self.df[cols]).mean(axis=1)
        
        self.df['content'] = self.df['Title'].fillna('') + ' ' + self.df['Keyword'].fillna('')
    
    def build_content_similarity(self):
        if self.word2vec_model is None:
            return
        self.content_embeddings = np.array([
            self.get_word_embedding(text) for text in self.df['content'].fillna('')
        ])
        self.content_similarity = cosine_similarity(self.content_embeddings, self.content_embeddings)
    
    def get_hybrid_recommendations(self, video_title, n=10):
        if self.content_similarity is None:
            return None
        idx = self.df[self.df['Title'] == video_title].index
        if len(idx) == 0:
            return None
        idx = idx[0]
        sim_scores = list(enumerate(self.content_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n*2+1]
        video_indices = [i[0] for i in sim_scores]
        recs = self.df.iloc[video_indices][
            ['Title', 'Keyword', 'Views', 'Likes', 'engagement_rate', 'Video ID']
        ].copy()
        recs['similarity_score'] = [i[1] for i in sim_scores]
        recs['hybrid_score'] = (
            recs['similarity_score'] * 0.7 + recs['engagement_rate'] * 0.3
        )
        return recs.nlargest(n, 'hybrid_score')[['Title', 'Views', 'Likes', 'Keyword', 'Video ID']]
    
    def get_recommendations_by_keyword(self, keyword, n=10):
        if self.word2vec_model is None or self.content_embeddings is None:
            return None
        keyword_embedding = self.get_word_embedding(keyword)
        if keyword_embedding is None or np.all(keyword_embedding == 0):
            return None
        keyword_sim = cosine_similarity([keyword_embedding], self.content_embeddings).flatten()
        idx = keyword_sim.argsort()[-1]
        source_title = self.df.iloc[idx]['Title']
        return self.get_hybrid_recommendations(source_title, n)
    
    def get_trending_videos(self, n=10):
        trending = self.df.copy()
        trending['trending_score'] = (
            trending['Views'] * 0.4 + trending['Likes'] * 0.3 + trending['Comments'] * 0.3
        )
        return trending.nlargest(n, 'trending_score')[
            ['Title', 'Keyword', 'Views', 'Likes', 'Comments', 'Video ID']
        ]
    
    def get_category_recommendations(self, category, n=10):
        cat_vids = self.df[self.df['Keyword'] == category].copy()
        if cat_vids.empty:
            return None
        cat_vids['combined_score'] = (
            cat_vids['popularity_score'] * 0.6 + cat_vids['engagement_rate'] * 0.4
        )
        return cat_vids.nlargest(n, 'combined_score')[
            ['Title', 'Keyword', 'Views', 'Likes', 'Comments', 'Video ID']
        ]

# Helper function to format numbers
def format_number(num):
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(num)

# Display video card
def display_video_card(title, views, likes, keyword, video_id, col):
    with col:
        st.markdown(f"""
        <div class="video-card">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        height: 180px; border-radius: 8px; display: flex; 
                        align-items: center; justify-content: center; margin-bottom: 10px;">
                <h2 style="color: white; text-align: center; padding: 20px;">🎥</h2>
            </div>
            <div class="video-title">{title[:80]}{'...' if len(title) > 80 else ''}</div>
            <div class="video-stats">👁️ {format_number(views)} views  •  👍 {format_number(likes)} likes</div>
            <span class="keyword-badge">{keyword}</span>
        </div>
        """, unsafe_allow_html=True)

# Initialize session state
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
if 'df' not in st.session_state:
    st.session_state.df = None

# Main App
def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<h1 style='color: #3ea6ff;'>🎥 YouTube Recommendation System</h1>", 
                   unsafe_allow_html=True)
    
    # Load dataset automatically
    if st.session_state.df is None:
        with st.spinner("Loading dataset..."):
            try:
                st.session_state.df = pd.read_csv('videos-stats.csv')
                # Data cleaning
                for col in ['Likes', 'Comments', 'Views']:
                    if col in st.session_state.df.columns:
                        st.session_state.df[col] = pd.to_numeric(
                            st.session_state.df[col], errors='coerce'
                        ).fillna(0).astype(int)
                        st.session_state.df[col] = st.session_state.df[col].apply(lambda x: max(x, 0))
            except FileNotFoundError:
                st.error("❌ videos-stats.csv not found! Please ensure the file is in the same directory.")
                st.stop()
    
    # Initialize recommender automatically
    if st.session_state.df is not None and st.session_state.recommender is None:
        with st.spinner("Initializing recommendation system..."):
            recommender = YouTubeRecommender(st.session_state.df)
            recommender.word2vec_model = recommender.load_word2vec_model()
            recommender.prepare_features()
            recommender.build_content_similarity()
            st.session_state.recommender = recommender
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='color: #3ea6ff;'>⚙️ System Info</h2>", unsafe_allow_html=True)
        
        if st.session_state.df is not None:
            st.success("✅ Recommender ready!")
            
            st.markdown("---")
            st.markdown("<h3 style='color: #3ea6ff;'>📊 Dataset Stats</h3>", 
                       unsafe_allow_html=True)
            st.metric("Total Videos", len(st.session_state.df))
            st.metric("Categories", st.session_state.df['Keyword'].nunique())
            st.metric("Avg Views", format_number(int(st.session_state.df['Views'].mean())))
    
    # Main content
    if st.session_state.recommender is None:
        st.markdown("""
        <div class="search-container">
            <h2 style='text-align: center;'>Welcome to YouTube Recommender! 🎬</h2>
            <p style='text-align: center; color: #aaaaaa;'>
                Upload your video dataset and initialize the recommender to get started.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    recommender = st.session_state.recommender
    
    # Search and recommendation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "🔥 Trending", "📂 Categories", "🎯 Similar Videos"])
    
    with tab1:
        st.markdown("<div class='search-container'>", unsafe_allow_html=True)
        search_query = st.text_input("Search", placeholder="Search for videos...", label_visibility="collapsed")
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            search_btn = st.button("🔍 Search", use_container_width=True)
        with col2:
            num_results = st.selectbox("Results", [5, 10, 15, 20], index=1)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if search_btn and search_query:
            with st.spinner("Finding videos..."):
                results = recommender.get_recommendations_by_keyword(search_query, num_results)
                
            if results is not None and not results.empty:
                st.markdown(f"<h3>Found {len(results)} videos for '{search_query}'</h3>", 
                           unsafe_allow_html=True)
                
                for i in range(0, len(results), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(results):
                            row = results.iloc[i + j]
                            display_video_card(
                                row['Title'], row['Views'], row['Likes'],
                                row['Keyword'], row.get('Video ID', ''), col
                            )
            else:
                st.warning("No videos found. Try a different search term!")
    
    with tab2:
        num_trending = st.slider("Number of trending videos", 5, 20, 10)
        trending = recommender.get_trending_videos(num_trending)
        
        st.markdown("<h2>🔥 Trending Now</h2>", unsafe_allow_html=True)
        
        for i in range(0, len(trending), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(trending):
                    row = trending.iloc[i + j]
                    display_video_card(
                        row['Title'], row['Views'], row['Likes'],
                        row['Keyword'], row.get('Video ID', ''), col
                    )
    
    with tab3:
        categories = sorted(st.session_state.df['Keyword'].unique())
        selected_category = st.selectbox("Select Category", categories)
        num_cat = st.slider("Number of videos", 5, 20, 10, key="cat_slider")
        
        if st.button("Show Videos", key="cat_btn"):
            cat_videos = recommender.get_category_recommendations(selected_category, num_cat)
            
            if cat_videos is not None and not cat_videos.empty:
                st.markdown(f"<h2>📂 {selected_category.title()} Videos</h2>", 
                           unsafe_allow_html=True)
                
                for i in range(0, len(cat_videos), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(cat_videos):
                            row = cat_videos.iloc[i + j]
                            display_video_card(
                                row['Title'], row['Views'], row['Likes'],
                                row['Keyword'], row.get('Video ID', ''), col
                            )
            else:
                st.warning("No videos found in this category!")
    
    with tab4:
        all_titles = st.session_state.df['Title'].tolist()
        selected_video = st.selectbox("Select a video to find similar ones", all_titles)
        num_similar = st.slider("Number of recommendations", 5, 20, 10, key="similar_slider")
        
        if st.button("Find Similar Videos", key="similar_btn"):
            similar = recommender.get_hybrid_recommendations(selected_video, num_similar)
            
            if similar is not None and not similar.empty:
                st.markdown("<h2>🎯 Videos Similar To:</h2>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color: #3ea6ff;'>{selected_video}</h3>", 
                           unsafe_allow_html=True)
                
                for i in range(0, len(similar), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(similar):
                            row = similar.iloc[i + j]
                            display_video_card(
                                row['Title'], row['Views'], row['Likes'],
                                row['Keyword'], row.get('Video ID', ''), col
                            )
            else:
                st.warning("Could not find similar videos!")

if __name__ == "__main__":
    main()