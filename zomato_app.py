import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")

st.set_page_config(
    page_title="Zomato Analytics",
    page_icon="🍽️",
    layout="wide"
)

# -------------------------
# TEXT CLEANING
# -------------------------

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):
    text = str(text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)

# -------------------------
# LOAD DATA
# -------------------------

@st.cache_data
def load_data(rest_file, review_file):

    rest_df = pd.read_csv(rest_file)
    reviews_df = pd.read_csv(review_file)

    rest_df["Cost"] = rest_df["Cost"].astype(str).str.replace(",", "")
    rest_df["Cost"] = pd.to_numeric(rest_df["Cost"], errors="coerce")

    reviews_df["Rating"] = pd.to_numeric(reviews_df["Rating"], errors="coerce")

    avg_rating = reviews_df.groupby("Restaurant")["Rating"].mean().reset_index()
    avg_rating.columns = ["Name", "AvgRating"]

    rest_df = rest_df.merge(avg_rating, on="Name", how="left")

    return rest_df, reviews_df

# -------------------------
# CLUSTERING
# -------------------------

@st.cache_resource
def run_clustering(df):

    features = df[["Cost","AvgRating"]].dropna()

    scaler = MinMaxScaler()
    X = scaler.fit_transform(features)

    wcss=[]
    sil=[]

    for k in range(2,10):

        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)

        wcss.append(kmeans.inertia_)
        sil.append(silhouette_score(X,labels))

    # FIXED PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=6, random_state=42)
    labels = kmeans.fit_predict(X_pca)

    return labels,X_pca,wcss,sil

# -------------------------
# SENTIMENT MODEL
# -------------------------

@st.cache_resource
def train_sentiment(df):

    df = df.dropna(subset=["Review","Rating"])

    df["clean"] = df["Review"].apply(clean_text)

    df["label"] = df["Rating"].apply(lambda x: 1 if x>3 else 0)

    vectorizer = TfidfVectorizer(max_features=3000)

    X = vectorizer.fit_transform(df["clean"])
    y = df["label"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    lr = LogisticRegression()
    lr.fit(X_train,y_train)

    rf = RandomForestClassifier()
    rf.fit(X_train,y_train)

    return lr,rf,vectorizer

# -------------------------
# FILE UPLOAD
# -------------------------

if "rest_df" not in st.session_state:
    st.session_state.rest_df=None

if st.session_state.rest_df is None:

    st.title("🍽️ Zomato Analytics")

    rest_file = st.file_uploader("Upload Restaurant Metadata CSV")
    review_file = st.file_uploader("Upload Reviews CSV")

    if rest_file and review_file:

        rest_df,reviews_df = load_data(rest_file,review_file)

        st.session_state.rest_df = rest_df
        st.session_state.reviews_df = reviews_df

        st.success("Data Loaded Successfully!")

        st.rerun()

    st.stop()

rest_df = st.session_state.rest_df
reviews_df = st.session_state.reviews_df

# -------------------------
# SIDEBAR
# -------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "EDA",
        "Cuisine Analysis",
        "Cost Analysis",
        "Clustering",
        "Sentiment Analysis",
        "Review Analyzer",
        "Data Explorer"
    ]
)

# -------------------------
# OVERVIEW
# -------------------------

if page=="Overview":

    st.title("Zomato Restaurant Dashboard")

    col1,col2,col3,col4,col5 = st.columns(5)

    col1.metric("Restaurants",len(rest_df))
    col2.metric("Reviews",len(reviews_df))
    col3.metric("Average Rating",round(rest_df["AvgRating"].mean(),2))
    col4.metric("Average Cost",f"₹{rest_df['Cost'].mean():.0f}")
    col5.metric("Unique Reviewers",reviews_df["Reviewer"].nunique())

    st.divider()

    top10 = rest_df.sort_values("AvgRating",ascending=False).head(10)

    fig = px.bar(
        top10,
        x="AvgRating",
        y="Name",
        orientation="h",
        title="Top Restaurants by Rating"
    )

    st.plotly_chart(fig,use_container_width=True)

# -------------------------
# EDA
# -------------------------

if page=="EDA":

    st.header("Rating Distribution")

    fig = px.histogram(
        rest_df,
        x="AvgRating",
        nbins=20
    )

    st.plotly_chart(fig,use_container_width=True)

# -------------------------
# CUISINE ANALYSIS
# -------------------------

if page=="Cuisine Analysis":

    st.header("Popular Cuisines")

    cuisines = rest_df["Cuisines"].dropna().str.split(",").explode()

    top = cuisines.value_counts().head(20)

    fig = px.bar(
        top,
        title="Top Cuisines"
    )

    st.plotly_chart(fig,use_container_width=True)

# -------------------------
# COST ANALYSIS
# -------------------------

if page=="Cost Analysis":

    st.header("Cost Distribution")

    fig = px.histogram(rest_df,x="Cost")

    st.plotly_chart(fig,use_container_width=True)

    fig2 = px.scatter(
        rest_df,
        x="Cost",
        y="AvgRating",
        hover_name="Name"
    )

    st.plotly_chart(fig2,use_container_width=True)

# -------------------------
# CLUSTERING
# -------------------------

if page=="Clustering":

    st.header("Restaurant Clustering")

    if st.button("Run Clustering"):

        labels,X_pca,wcss,sil = run_clustering(rest_df)

        fig = px.line(
            y=wcss,
            title="Elbow Curve"
        )

        st.plotly_chart(fig,use_container_width=True)

        fig2 = px.scatter(
            x=X_pca[:,0],
            y=X_pca[:,1],
            color=labels.astype(str),
            title="PCA Cluster Visualization"
        )

        st.plotly_chart(fig2,use_container_width=True)

# -------------------------
# SENTIMENT ANALYSIS
# -------------------------

if page=="Sentiment Analysis":

    st.header("Review Sentiment Analysis")

    if st.button("Train Sentiment Models"):

        lr,rf,vectorizer = train_sentiment(reviews_df)

        st.session_state.lr_model = lr
        st.session_state.rf_model = rf
        st.session_state.vectorizer = vectorizer

        st.success("Models Trained Successfully!")

# -------------------------
# REVIEW ANALYZER
# -------------------------

if page=="Review Analyzer":

    st.header("Analyze Review Sentiment")

    review = st.text_area("Enter a Review")

    if st.button("Analyze"):

        model = st.session_state.lr_model
        vectorizer = st.session_state.vectorizer

        clean = clean_text(review)

        X = vectorizer.transform([clean])

        pred = model.predict(X)[0]

        if pred==1:
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😞")

# -------------------------
# DATA EXPLORER
# -------------------------

if page=="Data Explorer":

    tab1,tab2 = st.tabs(["Restaurants","Reviews"])

    with tab1:
        st.dataframe(rest_df,use_container_width=True)

    with tab2:
        st.dataframe(reviews_df,use_container_width=True)