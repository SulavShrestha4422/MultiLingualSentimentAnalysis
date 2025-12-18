import streamlit as st
import pandas as pd
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import re
import time
import random
import pathlib 
import csv 
# Ensure 'pyarrow' is installed in your environment (pip install pyarrow)

# --- CORE PIPELINE LIBRARIES ---
import praw
import numpy as np
import torch
from langdetect import detect, DetectorFactory, LangDetectException
from transformers import AutoModelForSequenceClassification, XLMRobertaTokenizer
import nltk
from nltk.corpus import stopwords

# Set seed for reproducibility in langdetect
DetectorFactory.seed = 42

# --- 0. CONFIGURATION AND CONSTANTS ---

# --- PRAW CONFIGURATION (Use Streamlit secrets or environment variables) ---
# Option 1: Use Streamlit secrets (recommended for deployment)
# Option 2: Use environment variables (recommended for local development)
try:
    # Try to load from Streamlit secrets first
    CLIENT_ID = st.secrets["REDDIT_CLIENT_ID"]
    CLIENT_SECRET = st.secrets["REDDIT_CLIENT_SECRET"]
    USER_AGENT = st.secrets.get("REDDIT_USER_AGENT", "multilingual-sentiment-dashboard")
except (FileNotFoundError, KeyError):
    # Fall back to environment variables
    CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
    CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
    USER_AGENT = os.getenv("REDDIT_USER_AGENT", "multilingual-sentiment-dashboard")

# Validate credentials are loaded
if not CLIENT_ID or not CLIENT_SECRET:
    st.error("⚠️ Reddit API credentials not found! Please configure them in .streamlit/secrets.toml or as environment variables.")
    st.info("""
    **Setup Instructions:**
    
    **Option 1: Streamlit Secrets (Recommended)**
    1. Create `.streamlit/secrets.toml` in your project directory
    2. Add the following:
    ```toml
    REDDIT_CLIENT_ID = "your_client_id_here"
    REDDIT_CLIENT_SECRET = "your_client_secret_here"
    REDDIT_USER_AGENT = "multilingual-sentiment-dashboard"
    ```
    
    **Option 2: Environment Variables**
    ```bash
    export REDDIT_CLIENT_ID="your_client_id_here"
    export REDDIT_CLIENT_SECRET="your_client_secret_here"
    export REDDIT_USER_AGENT="multilingual-sentiment-dashboard"
    ```
    """)

# --- PATH & DATA CONFIGURATION ---
SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
TARGET_LANGS = ['en', 'de', 'hi', 'ar']

# Use pathlib to define absolute paths based on the script's location
BASE_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR.parent / 'data'
RAW_FILE = DATA_DIR / 'raw_multilingual_data.parquet'
CLEANED_FILE = DATA_DIR / 'cleaned_multilingual_data.parquet'
LABELED_FILE = DATA_DIR / 'labeled_multilingual_data.parquet'

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True) 

# --- NLTK & DEVICE SETUP ---
try:
    nltk.data.find('corpora/stopwords')
except Exception:
    nltk.download('stopwords', quiet=True)

# Define stop words for cleaning (combines English, German, and Arabic)
try:
    ALL_STOPWORDS = set(stopwords.words('english') + stopwords.words('german') + stopwords.words('arabic'))
except Exception:
    ALL_STOPWORDS = set(['the', 'and', 'a', 'in', 'is', 'it'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. PIPELINE FUNCTIONS (STEPS 2, 3, 4) ---

@st.cache_resource
def load_model_resources():
    """Loads transformer model and tokenizer once."""
    with st.spinner("Loading multilingual models (XLMRoberta)... This may take a moment."):
        tokenizer = XLMRobertaTokenizer.from_pretrained(SENTIMENT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL).to(device)
    return tokenizer, model

def run_data_collection(topics_dict, max_total_posts, status_text):
    """Executes Step 2: Data Collection (Scraping Reddit)."""
    status_text.text("STEP 2/4: Connecting to Reddit and scraping data...")
    
    limit_per_lang = max(5, int(max_total_posts / len(TARGET_LANGS)))
    
    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT
        )
        reddit.read_only = True
    except Exception as e:
        st.error(f"PRAW Initialization Error: {e}. Check your credentials.")
        return False

    all_data = []
    
    REDDIT_SOURCES = {
        'en': {'subreddits': ['artificial', 'tech', 'science'], 'queries': [topics_dict['en']]},
        'de': {'subreddits': ['de', 'wissenschaft', 'technik'], 'queries': [topics_dict['de']]},
        'hi': {'subreddits': ['india', 'tech', 'scienceindia'], 'queries': [topics_dict['hi']]},
        'ar': {'subreddits': ['arabs', 'egypt', 'saudiarabia'], 'queries': [topics_dict['ar']]}
    }
    
    limit_per_source_type = int(limit_per_lang / 2)

    for lang_code, data_sources in REDDIT_SOURCES.items():
        # Scrape Subreddits
        sub_limit = int(limit_per_source_type / len(data_sources['subreddits']))
        for sub_name in data_sources['subreddits']:
            try:
                subreddit = reddit.subreddit(sub_name)
                for submission in subreddit.top(time_filter="year", limit=sub_limit):
                    full_text = f"{submission.title} {submission.selftext}"
                    if full_text.strip():
                        all_data.append({'text': full_text.strip(), 'language_guess': lang_code,
                                        'source_type': 'Reddit_Subreddit', 'source_name': sub_name, 
                                        'raw_timestamp': submission.created_utc})
            except Exception: pass
        
        # Scrape Search
        for query in data_sources['queries']:
            try:
                for submission in reddit.subreddit('all').search(query, sort='relevance', limit=limit_per_source_type):
                    full_text = f"{submission.title} {submission.selftext}"
                    if full_text.strip():
                        all_data.append({'text': full_text.strip(), 'language_guess': lang_code,
                                        'source_type': 'Reddit_Search', 'source_name': f"Search: {query}", 
                                        'raw_timestamp': submission.created_utc})
            except Exception: pass
        time.sleep(random.randint(2, 5)) 
    
    st.write(f"DEBUG: STEP 2 (Collection) scraped {len(all_data)} potential posts.") 

    raw_df = pd.DataFrame(all_data)
    raw_df.to_parquet(str(RAW_FILE), index=False)
    status_text.text(f"STEP 2/4: Data collection complete. Collected {len(raw_df)} posts.")
    return len(raw_df) > 0

# --- STEP 3 LOGIC (Data Cleaning and Filtering) ---
def clean_text(text):
    if not isinstance(text, str): return ""
    # Remove URLs, user mentions, and moderator actions
    text = re.sub(r'http\S+|www.\S+|@\w+|\[removed\]|\[deleted\]', '', text, flags=re.IGNORECASE)
    # Remove non-alphanumeric characters (keep periods, question marks, hyphens)
    text = re.sub(r'[^\w\s.?!-]', ' ', text)
    # Collapse multiple spaces and strip whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Filter out very short posts that lack meaningful sentiment
    return text if len(text) > 20 else ""

def safe_lang_detect(text):
    if not text: return np.nan
    try:
        lang = detect(text)
        return lang if lang in TARGET_LANGS else 'other'
    except LangDetectException:
        return 'undetermined'

def run_preprocessing(status_text):
    """Executes Step 3: Cleaning and Language Verification."""
    status_text.text("STEP 3/4: Preprocessing and language filtering...")
    try:
        df = pd.read_parquet(str(RAW_FILE))
        
        # 1. Create a new column for cleaned text
        df['cleaned_text'] = df['text'].apply(clean_text)
        df = df[df['cleaned_text'] != ""].reset_index(drop=True)
        df['detected_language'] = df['cleaned_text'].apply(safe_lang_detect)
        
        # Filter to only retain posts in target languages
        filtered_df = df[df['detected_language'].isin(TARGET_LANGS)].copy()
        
        # Drop the original raw 'text' column to avoid duplicates
        if 'text' in filtered_df.columns:
            filtered_df.drop(columns=['text'], inplace=True)
        
        # Rename the cleaned text to the standard 'text' column name
        filtered_df.rename(columns={'cleaned_text': 'text', 'detected_language': 'language'}, inplace=True)
        
        # Explicitly select the final desired columns only
        filtered_df = filtered_df[['text', 'language', 'source_type', 'source_name']] 
        
        st.write(f"DEBUG: STEP 3 (Preprocessing) retained {len(filtered_df)} posts. Final columns: {list(filtered_df.columns)}")

        filtered_df.to_parquet(str(CLEANED_FILE), index=False)
        status_text.text(f"STEP 3/4: Preprocessing complete. {len(filtered_df)} posts retained.")
        return len(filtered_df) > 0
    except Exception as e:
        st.error(f"Preprocessing Error: {e}")
        return False

# --- STEP 4 LOGIC (Sentiment Modeling) ---
def analyze_sentiment(text, tokenizer, model):
    """Performs sentiment analysis."""
    if not text or not isinstance(text, str): return np.nan
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs.logits.softmax(dim=-1).cpu().numpy()[0]
        sentiment_labels = model.config.id2label
        predicted_id = np.argmax(scores)
        return sentiment_labels[predicted_id]
    except Exception:
        return "PROCESSING_ERROR"

def run_modeling(status_text):
    """Executes Step 4: Sentiment Analysis with atomic write protection."""
    status_text.text(f"STEP 4/4: Running sentiment analysis on {device}...")
    temp_file = None
    
    try:
        df = pd.read_parquet(str(CLEANED_FILE))
        if df.empty: 
            st.error("Modeling Error: Input DataFrame from Step 3 is empty.")
            return False

        tokenizer, model = load_model_resources()

        # Apply Sentiment Analysis
        df['sentiment'] = df['text'].apply(lambda x: analyze_sentiment(x, tokenizer, model))
        df = df[df['sentiment'] != "PROCESSING_ERROR"]
        
        st.write(f"DEBUG: STEP 4 (Modeling) labeled {len(df)} posts.")

        # Verify schema before writing
        required_cols = ['text', 'language', 'source_type', 'source_name', 'sentiment']
        if not all(col in df.columns for col in required_cols):
            st.error(f"FATAL SCHEMA ERROR: DataFrame missing required columns before final write. Found: {list(df.columns)}")
            return False

        # Atomic Write: Write to temp file, then rename
        temp_file = LABELED_FILE.parent / f"{LABELED_FILE.stem}_temp{LABELED_FILE.suffix}"
        
        # Write to temporary file
        df.to_parquet(str(temp_file), index=False)
        
        # Validate the temporary file
        if not os.path.exists(temp_file):
            st.error("Atomic Write Failed: Temporary file was not created.")
            return False
            
        temp_size = os.path.getsize(temp_file)
        if temp_size < 500:
            st.warning(f"Atomic Write Failed: Temporary file is only {temp_size} bytes.")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False
        
        # Atomic commit: Rename temp file to final destination
        os.replace(temp_file, LABELED_FILE)
        
        # Final validation
        final_size = os.path.getsize(LABELED_FILE)
        st.write(f"DEBUG: Final labeled file written successfully: {final_size} bytes")
        
        status_text.text(f"STEP 4/4: Modeling complete. {len(df)} posts labeled.")
        return len(df) > 0
        
    except Exception as e:
        st.error(f"Modeling Error: {e}")
        # Clean up temp file on error
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        return False

# --- 2. STREAMLIT UI & DASHBOARD LOGIC ---

def load_labeled_data(path):
    """Loads the final labeled data - NO CACHING to ensure fresh reads."""
    if not os.path.exists(path):
        st.info("No labeled data file found. Please run the pipeline first.")
        return pd.DataFrame()
        
    file_size = os.path.getsize(path)
    if file_size < 100:
        st.warning("File is under 100 bytes - likely corrupted.")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(str(path), engine='pyarrow')
        st.success(f"✅ Loaded {len(df)} posts from file ({file_size:,} bytes)")
    except Exception as e:
        st.error(f"❌ Read Error: {e}")
        return pd.DataFrame() 

    if 'sentiment' not in df.columns:
        st.error(f"❌ Missing 'sentiment' column. Found: {list(df.columns)}")
        return pd.DataFrame()

    # Capitalize sentiment labels to standardize (model returns lowercase)
    df['sentiment'] = df['sentiment'].str.capitalize()
    
    # Filter to only valid sentiment labels
    valid_sentiments = ['Negative', 'Neutral', 'Positive']
    df = df[df['sentiment'].isin(valid_sentiments)].copy()
    
    if df.empty:
        st.error("❌ No valid sentiment data after filtering")
        return pd.DataFrame()
    
    # Set categorical order for proper sorting
    df['sentiment'] = pd.Categorical(df['sentiment'], categories=valid_sentiments, ordered=True)
    
    st.write(f"📊 Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
    
    return df

# Visualization Functions
def sentiment_by_language_chart(data):
    if data.empty: return alt.Chart(pd.DataFrame()).mark_bar().encode()
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('language:N', title="Language"),
        y=alt.Y('count():Q', stack="normalize", axis=alt.Axis(format='%', title="Percentage of Posts")),
        color=alt.Color('sentiment:N', sort=['Negative', 'Neutral', 'Positive'], scale=alt.Scale(range=['#e41a1c', '#ffffb3', '#4daf4a'])),
        column=alt.Column('language:N', header=alt.Header(titleOrient="bottom", labelOrient="bottom")),
    ).properties(title='Sentiment Distribution by Language').interactive() 
    return chart

def overall_sentiment_distribution(data):
    if data.empty: return alt.Chart(pd.DataFrame()).mark_bar().encode()
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('sentiment:N', sort=['Negative', 'Neutral', 'Positive'], title="Sentiment"),
        y=alt.Y('count():Q', title="Total Post Count"),
        color=alt.Color('sentiment:N', sort=['Negative', 'Neutral', 'Positive'], scale=alt.Scale(range=['#e41a1c', '#ffffb3', '#4daf4a'])),
        tooltip=['sentiment', alt.Tooltip('count():Q', title='Post Count')]
    ).properties(title='Overall Sentiment Distribution').interactive()
    return chart

def create_wordcloud(text_data):
    if text_data.empty: return None
    text = " ".join(text_data['text'].astype(str).tolist())
    text = re.sub(r'[^\w\s]', '', text.lower())
    wordcloud = WordCloud(
        width=800, height=400, background_color='white', stopwords=ALL_STOPWORDS, 
        min_font_size=5, collocations=False
    ).generate(text)
    return wordcloud

# --- Summary Table Function ---
def create_sentiment_summary_table(data):
    """Creates a table summarizing sentiment counts and percentages by language."""
    if data.empty: return pd.DataFrame()
        
    count_df = data.groupby(['language', 'sentiment']).size().reset_index(name='Count')
    total_count = count_df.groupby('language')['Count'].transform('sum')
    count_df['Percentage'] = (count_df['Count'] / total_count * 100).round(1)

    pivot_count = count_df.pivot_table(index='language', columns='sentiment', values='Count', fill_value=0)
    pivot_percent = count_df.pivot_table(index='language', columns='sentiment', values='Percentage', fill_value=0.0)
    
    sentiment_cols = ['Negative', 'Neutral', 'Positive']
    for col_name in sentiment_cols:
        if col_name not in pivot_count.columns: pivot_count[col_name] = 0
        if col_name not in pivot_percent.columns: pivot_percent[col_name] = 0.0

    combined_df = pd.DataFrame(index=pivot_count.index)
    for col_name in sentiment_cols:
        combined_df[f'{col_name} Count'] = pivot_count[col_name]
        combined_df[f'{col_name} %'] = pivot_percent[col_name].astype(str) + '%'
        
    return combined_df.reset_index().rename(columns={'language': 'Language'})


# --- MAIN STREAMLIT APP ---

st.title("🚀 Multilingual Sentiment Pipeline Dashboard")
st.markdown("Run the full data analysis pipeline by defining a topic and post limit below.")

# --- SIDEBAR: Clear Data Button ---
st.sidebar.header("🛠️ Maintenance")

# Manual file test button
if st.sidebar.button("🔬 Deep File Diagnostic"):
    st.sidebar.subheader("Deep Diagnostic Results")
    
    test_path = LABELED_FILE
    st.sidebar.write(f"Testing file: `{test_path}`")
    st.sidebar.write(f"File exists: {os.path.exists(test_path)}")
    
    if os.path.exists(test_path):
        st.sidebar.write(f"File size: {os.path.getsize(test_path):,} bytes")
        
        # Try multiple read methods
        st.sidebar.write("---")
        st.sidebar.write("**Method 1: Direct PyArrow**")
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(test_path)
            st.sidebar.success(f"✅ PyArrow read: {table.num_rows} rows")
            st.sidebar.write(f"Schema: {table.schema}")
        except Exception as e:
            st.sidebar.error(f"❌ PyArrow failed: {e}")
        
        st.sidebar.write("---")
        st.sidebar.write("**Method 2: Pandas Default**")
        try:
            df1 = pd.read_parquet(test_path)
            st.sidebar.success(f"✅ Pandas default: {len(df1)} rows, {list(df1.columns)}")
        except Exception as e:
            st.sidebar.error(f"❌ Pandas default failed: {e}")
        
        st.sidebar.write("---")
        st.sidebar.write("**Method 3: Pandas with PyArrow Engine**")
        try:
            df2 = pd.read_parquet(test_path, engine='pyarrow')
            st.sidebar.success(f"✅ Pandas+PyArrow: {len(df2)} rows")
            st.sidebar.write("First 3 rows:")
            st.sidebar.dataframe(df2.head(3))
        except Exception as e:
            st.sidebar.error(f"❌ Pandas+PyArrow failed: {e}")
        
        st.sidebar.write("---")
        st.sidebar.write("**Method 4: Pandas with FastParquet**")
        try:
            df3 = pd.read_parquet(test_path, engine='fastparquet')
            st.sidebar.success(f"✅ Pandas+FastParquet: {len(df3)} rows")
        except Exception as e:
            st.sidebar.error(f"❌ Pandas+FastParquet failed: {e}")

if st.sidebar.button("🗑️ Clear All Data Files"):
    deleted_count = 0
    for file in [RAW_FILE, CLEANED_FILE, LABELED_FILE]:
        if os.path.exists(file):
            try:
                os.remove(file)
                st.sidebar.success(f"✓ Deleted {file.name}")
                deleted_count += 1
            except Exception as e:
                st.sidebar.error(f"✗ Failed to delete {file.name}: {e}")
    
    if deleted_count > 0:
        st.sidebar.info("All data files cleared. Run the pipeline again.")
    else:
        st.sidebar.info("No data files found to delete.")

# --- TOPIC INPUT FORM (Step 1 of the Dashboard) ---

with st.form("topic_form"):
    st.header("1. Define Analysis Parameters")
    
    # Input: Maximum Total Posts
    max_total_posts = st.number_input(
        "Maximum Total Posts to Collect (Across All 4 Languages)", 
        min_value=100, 
        max_value=4000, 
        value=1000, 
        step=100,
        help="The total number of posts the scraper will attempt to retrieve. This limit is split evenly among English, German, Hindi, and Arabic."
    )
    
    st.subheader("Topic Definitions")
    
    col_en, col_de = st.columns(2)
    topic_en = col_en.text_input("English Topic (EN)", value="Space Exploration") 
    topic_de = col_de.text_input("German Topic (DE)", value="Weltraumforschung") 
    
    col_hi, col_ar = st.columns(2)
    topic_hi = col_hi.text_input("Hindi Topic (HI)", value="अंतरिक्ष अन्वेषण")
    topic_ar = col_ar.text_input("Arabic Topic (AR)", value="استكشاف الفضاء")
    
    run_button = st.form_submit_button("Start Full Analysis Pipeline (Steps 2-4)")

# --- PIPELINE EXECUTION ---

if run_button:
    if not all([topic_en, topic_de, topic_hi, topic_ar]):
        st.error("Please provide a topic for all four languages.")
    else:
        topics_dict = {
            'en': topic_en, 'de': topic_de, 'hi': topic_hi, 'ar': topic_ar
        }
        
        pipeline_status = st.empty()
        
        with st.spinner("Pipeline Running... Do not close the browser or terminal."):
            try:
                # Steps 2, 3, 4 execution
                if not run_data_collection(topics_dict, max_total_posts, pipeline_status):
                    st.error("Pipeline Failed at Step 2: Data collection returned no posts. Please check your **PRAW credentials** and **Administrator access**.")
                    st.stop()
                
                if not run_preprocessing(pipeline_status):
                    st.error("Pipeline Failed at Step 3: No posts remained after cleaning/language filtering. Try a different topic or increase the post limit.")
                    st.stop()

                if not run_modeling(pipeline_status):
                    st.error("Pipeline Failed at Step 4: No posts remained after modeling. **Model execution error or corrupted file detected.**")
                    st.stop()
                
                pipeline_status.success("✅ Full Pipeline Completed Successfully!")
                st.balloons()
                st.rerun()  # Rerun immediately to show results 

            except Exception as e:
                st.error(f"An unexpected error occurred during the pipeline execution: {e}")
                st.stop()

# --- DISPLAY RESULTS (Step 5) ---

st.markdown("---")
st.header("2. Analysis Results")

# --- Diagnostic Check ---
file_path = LABELED_FILE
file_exists = os.path.exists(file_path)
file_size = os.path.getsize(file_path) if file_exists else 0

if not file_exists:
    st.info("📭 No data file found yet. Please run the pipeline above to analyze sentiment data.")
elif file_size == 0:
    st.warning("⚠️ Data file exists but is empty. Please clear data files and run the pipeline again.")
else:
    st.success(f"📁 Data file ready: {file_size:,} bytes")
    
    # Load data - no caching, fresh read every time
    df = load_labeled_data(file_path)

if df.empty:
    st.info("👆 Run the pipeline above to generate sentiment analysis data.")
else:
    # --- Sidebar Filters (for results) ---
    st.sidebar.header("Filter Results")
    available_langs = sorted(df['language'].unique().tolist())
    selected_languages = st.sidebar.multiselect("Select Language(s)", options=available_langs, default=available_langs)
    
    available_sources = sorted(df['source_name'].unique().tolist())
    selected_sources = st.sidebar.multiselect("Select Source/Topic", options=available_sources, default=available_sources)
    
    filtered_df = df[(df['language'].isin(selected_languages)) & (df['source_name'].isin(selected_sources))].reset_index(drop=True)
    st.sidebar.info(f"Showing {len(filtered_df):,} records out of {len(df):,}")

    if filtered_df.empty:
        st.warning("Filters resulted in no data. Please adjust your sidebar selections.")
    else:
        # Row 1: Key Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Posts Labeled", f"{len(df):,}")
        col2.metric("Posts in Current View", f"{len(filtered_df):,}")
        
        searched_topic = filtered_df[filtered_df['source_name'].str.contains('Search:')]['source_name'].iloc[-1].split(':')[-1].strip() if not filtered_df[filtered_df['source_name'].str.contains('Search:')].empty else "N/A"
        col3.metric("Topic Analyzed", searched_topic)

        # --- Sentiment Summary Table ---
        st.subheader("Sentiment Summary (Counts & Percentages by Language)")
        sentiment_table_df = create_sentiment_summary_table(filtered_df)
        st.dataframe(sentiment_table_df, use_container_width=True)

        # Row 2: Sentiment Charts
        st.subheader("Sentiment Distribution Charts")
        st.altair_chart(sentiment_by_language_chart(filtered_df), use_container_width=True)
        st.altair_chart(overall_sentiment_distribution(filtered_df), use_container_width=True)
        
        # Row 3: Word Cloud
        st.subheader("Language Insights: Word Clouds")
        lang_for_wc_options = filtered_df['language'].unique().tolist()
        if lang_for_wc_options:
            lang_for_wc = st.selectbox("Select Language for Word Cloud Generation", options=lang_for_wc_options)
            wc_data = filtered_df[filtered_df['language'] == lang_for_wc]
            
            if not wc_data.empty:
                wordcloud = create_wordcloud(wc_data)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                ax.set_title(f"Most Frequent Words in {lang_for_wc.upper()} Posts")
                st.pyplot(fig)
            else:
                st.info(f"No posts found for {lang_for_wc} based on current filters.")