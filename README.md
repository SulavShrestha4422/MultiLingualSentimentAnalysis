# Multilingual Sentiment Analysis Dashboard

A Streamlit dashboard for analyzing sentiment across multiple languages (English, German, Hindi, Arabic) using Reddit data.

## Setup Instructions

### Prerequisites
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Reddit API credentials

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Create Conda environment

**Option A: From environment.yml (Recommended)**
```bash
conda env create -f environment.yml
conda activate multilingual-sentiment
```

**Option B: Manual setup**
```bash
conda create -n multilingual-sentiment python=3.10
conda activate multilingual-sentiment
pip install -r requirements.txt
```

### 3. Configure Reddit API credentials

Create a file `.streamlit/secrets.toml` in the project directory:
```toml
REDDIT_CLIENT_ID = "your_client_id_here"
REDDIT_CLIENT_SECRET = "your_client_secret_here"
REDDIT_USER_AGENT = "your_app_name"
```

**Get Reddit API credentials:**
1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Select "script" as the app type
4. Copy your client ID and secret

### 4. Run the app
```bash
# Make sure your conda environment is activated
conda activate multilingual-sentiment

# Run the Streamlit app
streamlit run your_app_name.py
```





## Technologies Used
- **Streamlit** - Web app framework
- **Transformers** - Hugging Face models
- **PRAW** - Reddit API wrapper
- **Pandas** - Data manipulation
- **PyTorch** - Deep learning backend
- **Altair** - Declarative visualizations

## Environment Details
- **Python Version:** 3.10+
- **Environment Manager:** Anaconda
- **Key Dependencies:** See `environment.yml` or `requirements.txt`

## Troubleshooting

### "Reddit API credentials not found"
- Make sure `.streamlit/secrets.toml` exists and contains valid credentials
- Check that the file is in the correct location (project root)

### Conda environment issues
```bash
# Remove and recreate environment
conda deactivate
conda env remove -n multilingual-sentiment
conda env create -f environment.yml
```

### Model download issues
The app downloads transformer models on first run (~500MB). Ensure you have:
- Stable internet connection
- Sufficient disk space
- No firewall blocking Hugging Face CDN

## Contributing
Pull requests are welcome! For major changes, please open an issue first.

## License
[Your chosen license - e.g., MIT]