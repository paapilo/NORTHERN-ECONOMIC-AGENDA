# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.ticker as ticker
import ast  # For parsing sector lists

# --- Streamlit App Title ---
st.title("Idea Submissions Dashboard")

# --- File Upload Section ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully!")
else:
    try:
        df = pd.read_csv("idea_submissions.csv")
        st.info("Loaded default CSV file from local disk.")
    except FileNotFoundError:
        st.error("No CSV file found. Please upload a CSV file.")
        st.stop()

# --- Raw Data Display ---
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.dataframe(df)

# --- Data Cleaning ---
columns_to_delete = [
    'id', 'name', 'email', 'phone', 'document_url',
    'supporting_url', 'created_at', 'objective',
    'stakeholders', 'implementation', 'financial_structure',
    'economic_outcomes'
]
df.drop(columns=[col for col in columns_to_delete if col in df.columns], inplace=True)

# --- Cleaned Data Display ---
if st.checkbox("Show cleaned data"):
    st.subheader("Cleaned Data")
    st.dataframe(df)

# --- Helper Functions ---

# Parse sectors safely
def parse_sectors(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

# Bar chart plotting function
def plot_bar_chart(data, title, xlabel, ylabel, color='skyblue'):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    bars = ax.bar(data.index, data.values, color=color)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticklabels(data.index, rotation=45, ha='right', fontsize=10)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Force integer y-axis

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10)

    st.pyplot(fig)

# --- Sector Statistics ---

# Parse and explode sectors
df['sectors_parsed'] = df['sectors'].apply(parse_sectors)
df_exploded = df.explode('sectors_parsed')

# Proper sector counting
sector_counts = df_exploded['sectors_parsed'].value_counts().reset_index()
sector_counts.columns = ['Sector', 'Number of Submissions']

# Add total row
total_submissions = sector_counts['Number of Submissions'].sum()
total_row = pd.DataFrame({'Sector': ['Total'], 'Number of Submissions': [total_submissions]})
sector_counts_display = pd.concat([sector_counts, total_row], ignore_index=True)

# Display sectors table
st.subheader("Submissions per Sector (Cleaned)")
st.dataframe(sector_counts_display)

# --- Submissions per Sector Bar Chart ---
st.subheader("Submissions per Sector")

# Plot without "Total" row
sector_counts_no_total = sector_counts[sector_counts['Sector'] != 'Total']
plot_bar_chart(
    data=sector_counts_no_total.set_index('Sector')['Number of Submissions'],
    title="Submissions per Sector",
    xlabel="Sector",
    ylabel="Number of Submissions",
    color='lightgreen'
)

# --- Other Distributions ---

# Set Seaborn style
sns.set_style('whitegrid')

# Zone Distribution
st.subheader("Submissions per Zone")
zone_counts = df['zone'].value_counts()
plot_bar_chart(zone_counts, "Submissions per Zone", "Zone", "Number of Submissions", color='cornflowerblue')

# State Distribution
st.subheader("Submissions per State")
state_counts = df['author_state'].value_counts()
plot_bar_chart(state_counts, "Submissions per State", "State", "Number of Submissions", color='salmon')

#####################
# --- WordCloud + Sentiment Analysis Section ---

import nltk
from wordcloud import WordCloud
from textblob import TextBlob

# Download NLTK stopwords (first time only)
nltk.download('stopwords')
from nltk.corpus import stopwords

st.header("☁️ WordCloud and Sentiment Analysis for Project Descriptions")

# --- Preprocess Descriptions ---
if 'description' not in df.columns:
    st.warning("'Description' column is missing in your data.")
else:
    descriptions = df['description'].dropna()

    if descriptions.empty:
        st.info("No descriptions available to analyze.")
    else:
        # --- WordCloud ---
        st.subheader("WordCloud of Project Descriptions")

        # Combine all text
        full_text = " ".join(descriptions)

        # Clean text (remove weird characters)
        full_text = full_text.encode('ascii', 'ignore').decode()  # Remove strange characters

        # Generate WordCloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=set(stopwords.words('english'))
        ).generate(full_text)

        # Plot WordCloud
        fig_wc, ax_wc = plt.subplots(figsize=(12, 6), dpi=150)
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)

        # --- Sentiment Analysis ---
        st.subheader("Sentiment Analysis of Project Descriptions")

        # Analyze each description
        df['sentiment_polarity'] = descriptions.apply(lambda x: TextBlob(x).sentiment.polarity)

        # Classify sentiment
        def classify_sentiment(score):
            if score > 0.1:
                return 'Positive'
            elif score < -0.1:
                return 'Negative'
            else:
                return 'Neutral'

        df['sentiment_label'] = df['sentiment_polarity'].apply(classify_sentiment)

        # Show sentiment counts
        sentiment_counts = df['sentiment_label'].value_counts()

        st.dataframe(sentiment_counts.rename_axis('Sentiment').reset_index(name='Number of Projects'))

        # Plot sentiment distribution
        fig_sentiment, ax_sentiment = plt.subplots(figsize=(8, 5), dpi=150)
        bars = ax_sentiment.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'gray'])
        ax_sentiment.set_title('Sentiment Distribution of Project Descriptions')
        ax_sentiment.set_xlabel('Sentiment')
        ax_sentiment.set_ylabel('Number of Projects')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax_sentiment.annotate(f'{int(height)}',
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3),
                                  textcoords="offset points",
                                  ha='center', va='bottom')

        st.pyplot(fig_sentiment)