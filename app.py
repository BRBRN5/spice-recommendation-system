import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import os

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(
    page_title="Spice Recommendation System",
    layout="wide"
)


theme = st.toggle("☀️ / 🌙 Theme", value=False)

if theme:
    card_bg = "linear-gradient(135deg, #1e1e1e, #111111)"
    text_color = "white"
    page_bg = "#0b1120"
else:
    card_bg = "linear-gradient(135deg, #ffffff, #f3f4f6)"
    text_color = "#1f2937"
    page_bg = "#f7f3ee"
st.markdown(f"""
<style>
/* Animated background */

[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: fixed;
    top: -200px;
    left: -200px;
    width: 500px;
    height: 500px;

    background: radial-gradient(
        circle,
        rgba(53,94,59,0.25) 0%,
        rgba(53,94,59,0) 70%
    );

    animation: float1 12s ease-in-out infinite;

    z-index: 0;
}}

[data-testid="stAppViewContainer"]::after {{
    content: "";
    position: fixed;
    bottom: -200px;
    right: -200px;
    width: 500px;
    height: 500px;

    background: radial-gradient(
        circle,
        rgba(250,204,21,0.15) 0%,
        rgba(250,204,21,0) 70%
    );

    animation: float2 15s ease-in-out infinite;

    z-index: 0;
}}

@keyframes float1 {{
    0% {{
        transform: translate(0px, 0px);
    }}

    50% {{
        transform: translate(40px, 40px);
    }}

    100% {{
        transform: translate(0px, 0px);
    }}
}}

@keyframes float2 {{
    0% {{
        transform: translate(0px, 0px);
    }}

    50% {{
        transform: translate(-40px, -30px);
    }}

    100% {{
        transform: translate(0px, 0px);
    }}
}}
.recommendation-box {{
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);

    border: 1px solid rgba(255,255,255,0.15);

    box-shadow:
        0 8px 32px rgba(0,0,0,0.25);

    transition: all 0.3s ease;
}}

.recommendation-box:hover {{
    transform: translateY(-6px);

    box-shadow:
        0 12px 40px rgba(0,0,0,0.35);

    border: 1px solid rgba(250,204,21,0.4);
}}

/* Full app background */
[data-testid="stAppViewContainer"] {{
    background-color: {page_bg};
    position: relative;
    overflow: hidden;
}}

/* Main container */
[data-testid="stHeader"] {{
    background: transparent;
}}
/* Text colors */
h1, h2, h3, p, label {{
    color: {text_color};
}}

/* Input box */
.stTextInput input {{
    background-color: {"#1e1e1e" if theme else "#ffffff"};
    color: {text_color};
    border-radius: 10px;
    border: 1px solid #d1d5db;
}}

/* Placeholder text */
.stTextInput input::placeholder {{
    color: #9ca3af;
}}

/* Button */
.stButton>button {{
    background-color: #355e3b;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}}

.stButton>button:hover {{
    background-color: #2d4f32;
    color: white;
}}

</style>
""", unsafe_allow_html=True)




st.markdown("""
<div style="
    padding: 40px;
    border-radius: 20px;
    background: linear-gradient(135deg, #1f3b2d, #355e3b);
    color: white;
    margin-bottom: 40px;
">

<h1 style="font-size:60px; margin-bottom:10px;">
🌿 Spice Recommendation System
</h1>

<h3 style="font-size:28px; color:#dfeadf;">
Discover Traditional Herbs & Spices
</h3>

<p style="font-size:20px; margin-top:20px; line-height:1.8;">
Enter a disease or symptom to get spice recommendations
based on similar chemical compositions.
</p>

</div>
""", unsafe_allow_html=True)
st.markdown("""
<div style="
display:flex;
justify-content:space-between;
gap:25px;
margin-bottom:50px;
">

<div style="
flex:1;
background:rgba(255,255,255,0.08);
padding:25px;
border-radius:20px;
text-align:center;
backdrop-filter:blur(10px);
border:1px solid rgba(255,255,255,0.1);
">

<h2>🔍</h2>

<h3>Enter Symptoms</h3>

<p>
Type disease or symptoms like fever, cough, cold, etc.
</p>

</div>

<div style="
flex:1;
background:rgba(255,255,255,0.08);
padding:25px;
border-radius:20px;
text-align:center;
backdrop-filter:blur(10px);
border:1px solid rgba(255,255,255,0.1);
">

<h2>🧪</h2>

<h3>AI Chemical Analysis</h3>

<p>
The system analyzes medicinal and chemical similarities.
</p>

</div>

<div style="
flex:1;
background:rgba(255,255,255,0.08);
padding:25px;
border-radius:20px;
text-align:center;
backdrop-filter:blur(10px);
border:1px solid rgba(255,255,255,0.1);
">

<h2>🌿</h2>

<h3>Get Recommendations</h3>

<p>
Receive the most chemically similar medicinal spices.
</p>

</div>

</div>
""", unsafe_allow_html=True)

# =========================
# Load Dataset
# =========================
df = pd.read_csv("data.csv", encoding="cp1252")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Fill missing values
df = df.fillna("")

# =========================
# Check Required Columns
# =========================
required_columns = [
    "spice name",
    "traditional name",
    "uses",
    "origin",
    "chemical composition",
    "other uses"
]


for col in required_columns:
    if col not in df.columns:
        st.error(f"Missing column: {col}")
        st.stop()

# =========================
# Create TF-IDF Models
# =========================

# Model for medicinal uses
uses_vectorizer = TfidfVectorizer(stop_words="english")

uses_matrix = uses_vectorizer.fit_transform(
    df["uses"].astype(str)
)

# Model for chemical compositions
chemical_vectorizer = TfidfVectorizer(stop_words="english")

chemical_matrix = chemical_vectorizer.fit_transform(
    df["chemical composition"].astype(str)
)

# =========================
# Recommendation Function
# =========================
def recommend(query):

    # =========================
    # Step 1:
    # Find best spice using medicinal uses
    # =========================
    query_vector = uses_vectorizer.transform([query])

    use_scores = cosine_similarity(
        query_vector,
        uses_matrix
    )[0]

    # Best spice index
    best_match_index = np.argmax(use_scores)

    # =========================
    # Step 2:
    # Use ONLY chemical composition
    # similarity for recommendations
    # =========================
    best_chemical_vector = chemical_matrix[
        best_match_index
    ]

    chemical_scores = cosine_similarity(
        best_chemical_vector,
        chemical_matrix
    )[0]

    # =========================
    # Force top spice = 1.0
    # =========================
    chemical_scores[best_match_index] = 1.0

    # =========================
    # Get Top 5 Similar Spices
    # =========================
    top_indices = chemical_scores.argsort()[-5:][::-1]

    results = df.iloc[top_indices]

    scores = chemical_scores[top_indices]

    return results, scores

# =========================
# App Title
# =========================


# =========================
# User Input
# =========================
user_input = st.text_input(
    "Enter disease or symptom (e.g., fever, cough, cold)"
)

# =========================
# Recommendation Section
# =========================
search = st.button("Get Recommendations")

if user_input and (search or user_input):

    results, scores = recommend(user_input)

    st.subheader("Recommended Spices")

    for i, (idx, row) in enumerate(results.iterrows()):

        spice_name = row['spice name']

        # convert spice name to image filename
        image_name = "_".join(
    spice_name.strip().lower().split()
) + ".jpg"

        image_path = f"images/{image_name}"
        import base64

        with open(image_path, "rb") as image_file:
            img_base64 = base64.b64encode(image_file.read()).decode()

            match_percent = int(scores[i] * 100)

        chemicals = row['chemical composition'].split(",")

        chemical_tags = "".join(
            f"""<span style="
                background:#355e3b;
                color:white;
                padding:6px 12px;
                border-radius:20px;
                margin-right:8px;
                display:inline-block;
                margin-top:8px;
                font-size:14px;
            ">{chem.strip()}</span>"""
            for chem in chemicals
        )
        

        st.markdown(f"""
<div class="recommendation-box" style="
display:flex;
gap:40px;
padding:25px;
border-radius:20px;
background:{card_bg};
border:1px solid #d1d5db;
box-shadow:0 4px 12px rgba(0,0,0,0.08);
color:{text_color};
margin-bottom:30px;
align-items:flex-start;
transition:0.3s;
">

<div style="
width:180px;
height:180px;
overflow:hidden;
border-radius:15px;
flex-shrink:0;
">

<img src="data:image/jpeg;base64,{img_base64}"
style="
width:100%;
height:100%;
object-fit:cover;
border-radius:15px;
"/>

</div>

<div style="flex:1;">

<div style="
display:flex;
justify-content:space-between;
align-items:center;
">

<h2 style="margin:0;">🌿 {spice_name}</h2>

<div style="
background:#355e3b;
color:white;
padding:8px 16px;
border-radius:30px;
font-weight:bold;
font-size:15px;
">
{match_percent}% Match
</div>

</div>

<p style="margin-top:15px;">
<b>Traditional Name:</b> {row['traditional name']}
</p>

<p>
<b>Uses:</b> {row['uses']}
</p>

<p>
<b>Origin:</b> {row['origin']}
</p>

<p>
<b>Other Uses:</b> {row['other uses']}
</p>

<p>
<b>Chemical Composition:</b>
</p>

<div>
{chemical_tags}
</div>

</div>

</div>
""", unsafe_allow_html=True)

            

            

        # =========================
        # Bar Graph
        # =========================

    # =========================
    # Premium Similarity Graph
    # =========================

    st.subheader("Top Similar Spices")

    graph_df = pd.DataFrame({
        "Spice": results["spice name"].tolist(),
        "Similarity": [float(x) for x in scores]
    })

    fig = px.bar(
        graph_df,
        x="Similarity",
        y="Spice",
        orientation="h",
        text="Similarity",
        color="Similarity",
        color_continuous_scale=[
            "#355e3b",
            "#4ade80"
        ]
    )

    fig.update_layout(
        height=500,

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="white" if theme else "#1f2937",
            size=15
        ),

        title=dict(
            text=f"Top Similar Spices for '{user_input}'",
            x=0.5,
            font=dict(size=24)
        ),

        xaxis=dict(
            showgrid=False,
            zeroline=False,
            title="Chemical Similarity Score"
        ),

        yaxis=dict(
            showgrid=False,
            title=""
        ),

        coloraxis_showscale=False
    )

    fig.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside',

        marker=dict(
            line=dict(width=0)
        )
    )

    fig.update_yaxes(categoryorder="total ascending")

    st.plotly_chart(
        fig,
        use_container_width=True
        )

    # =========================
    # Heatmap Section
    # =========================

    st.subheader("Chemical Composition Similarity Heatmap")

    chemical_similarity_matrix = cosine_similarity(
            chemical_matrix
        )

    spice_names = df["spice name"].tolist()

    fig2, ax2 = plt.subplots(figsize=(14, 8))

    sns.heatmap(
            chemical_similarity_matrix,
            xticklabels=spice_names,
            yticklabels=spice_names,
            cmap="YlGnBu",
            ax=ax2
        )
    ax2.tick_params(axis='x', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)

    ax2.set_title(
            "Chemical Composition Similarity Heatmap"
        )

    plt.xticks(rotation=90)

    plt.yticks(rotation=0)

    plt.tight_layout()

    st.pyplot(fig2)
    # =========================
# Floating AI Assistant
# =========================

    st.markdown("---")


st.caption(
    "AI assistant responses are for educational purposes only."
)