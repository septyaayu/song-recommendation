import streamlit as st
import numpy as np
import pandas as pd
import gdown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Custom CSS untuk mempercantik UI
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .title {
        color: #FF5733;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #FF5733;
        color: white;
        font-size: 18px;
        border-radius: 10px;
    }
    .stSelectbox {
        background-color: #f9f9f9;
        color: #333;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        border-radius: 8px;
    }
    .result-card {
        background-color: #fffae6;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
        .result-card h4 {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 5px;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# 1. Load Dataset
@st.cache_data
def load_data():
    url = 'https://drive.google.com/uc?id=17chqY8L-f6e-0_MGgJOtvCOAGyLnDjGQ'
    output = 'lagu.csv'
    gdown.download(url, output, quiet=False)
    df = pd.read_csv(output, sep=";", encoding="utf-8", on_bad_lines="skip")
    df = df.dropna()
    df['processed_judul'] = df['judul_lagu'].str.lower().str.strip()
    df['processed_artist'] = df['artist'].str.lower().str.strip()
    df['processed_mood'] = df['mood'].str.lower().str.strip()
    df['all'] = df['processed_judul'] + ' ' + df['processed_artist'] + ' ' + df['processed_mood']
    return df

df = load_data()

# 2. TF-IDF Vectorization
tfidf_judul = TfidfVectorizer()
tfidf_artist = TfidfVectorizer()
tfidf_mood = TfidfVectorizer()
tfidf_all = TfidfVectorizer()

tfidf_matrix_judul = tfidf_judul.fit_transform(df['processed_judul'])
tfidf_matrix_artist = tfidf_artist.fit_transform(df['processed_artist'])
tfidf_matrix_mood = tfidf_mood.fit_transform(df['processed_mood'])
tfidf_matrix_all = tfidf_all.fit_transform(df['all'])

# 3. Sistem Rekomendasi
def recommend_songs(query, feature='judul', top_n=5):
    vectorizers = {
        'judul': tfidf_judul,
        'artist': tfidf_artist,
        'mood': tfidf_mood,
        'all': tfidf_all
    }
    if feature not in vectorizers:
        return [], "Error: Fitur tidak valid!"
    try:
        processed_query = query.lower().strip()
        query_vector = vectorizers[feature].transform([processed_query])
        cosine_sim = cosine_similarity(query_vector, eval(f'tfidf_matrix_{feature}'))
        top_indices = np.argsort(cosine_sim[0])[::-1][:top_n]
        results = []
        for idx in top_indices:
            if cosine_sim[0][idx] > 0:
                song = df.iloc[idx]
                results.append({
                    'Judul': song['judul_lagu'],
                    'Artist': song['artist'],
                    'Mood': song['mood'],
                    'Link': song['link_play'],
                    'Similarity': f"{cosine_sim[0][idx]:.4f}"
                })
        return results, ""
    except Exception as e:
        return [], f"Error: {str(e)}"

# 4. Streamlit UI
st.title("🎵 Sistem Rekomendasi Lagu")
st.write("Masukkan kata kunci untuk menemukan lagu yang cocok dengan selera Anda!")

# Menggunakan st.form agar bisa menekan Enter tanpa tombol eksplisit
with st.form(key='search_form'):
    feature = st.selectbox("Pilih filter pencarian:", ["judul", "artist", "mood", "all"], index=0)
    query = st.text_input("Masukkan pencarian lagu, artis, atau mood:", 
                          placeholder="mood yang tersedia: happy | calm | energetic | sad")
    submit = st.form_submit_button("🔍 Cari Rekomendasi")

if submit:
    if query:
        results, error = recommend_songs(query, feature)
        if error:
            st.error(error)
        elif not results:
            st.warning("⚠️ Tidak menemukan rekomendasi untuk input tersebut.")
        else:
            st.success("🎶 Berikut rekomendasi lagu untuk Anda:")
            for i, res in enumerate(results, 1):
                st.markdown(f"""
                    <div class='result-card'>
                        <h4>{i}. {res['Judul']} - {res['Artist']}</h4>
                        <p>🎵 Mood: <b>{res['Mood']}</b></p>
                        <p>🔗 <a href="{res['Link']}" target="_blank">Dengarkan di sini</a></p>
                        <p>📊 Skor Similaritas: {res['Similarity']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                st.write("---")