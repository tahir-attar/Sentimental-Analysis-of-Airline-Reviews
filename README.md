# BERT Airline Sentiment Analysis Dashboard

Aspect-Based Sentiment Analysis (ABSA) for airline reviews using BERT transformer model.

## Overview

This project analyzes airline reviews to extract sentiment for different aspects like food, seat comfort, crew service, baggage handling, and more. It processes incoming flight data and provides real-time sentiment tracking via a Streamlit dashboard.

## Features

- **BERT-based Sentiment Analysis** - Uses pre-trained BERT model for accurate sentiment classification
- **Aspect-Based Analysis** - Extracts sentiment for specific flight aspects (food, seat, crew, etc.)
- **Real-time Dashboard** - Live monitoring of sentiment metrics using Streamlit
- **Flight Data Pipeline** - Processes incoming flight CSVs automatically
- **Historical Tracking** - Maintains history of sentiment trends per flight

## Project Structure

```
bert_airline_sentiment_model/
├── FINAL_PRODUCT.py                          # Main Streamlit application
├── ABSA_UI/
│   ├── Model/                                # BERT model files (tokenizer, config, weights)
│   ├── incoming_flights/                     # Flight CSV files for processing
│   ├── airline_reviews_with_aspect_sentiments.csv  # Output data
│   └── state.json                            # Application state tracking
├── requirements.txt                          # Python dependencies
└── README.md                                 # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Tahir-2802/Sentimental-Analysis-of-Airline-Reviews.git
cd Sentimental-Analysis-of-Airline-Reviews
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```bash
python -c "import nltk; nltk.download('punkt')"
```

4. **Add BERT Model Files**:
   - The trained BERT model files are too large for GitHub
   - Place your model files in `ABSA_UI/Model/`:
     - `config.json`
     - `vocab.txt`
     - `model.safetensors` (or `pytorch_model.bin`)
     - `training_args.bin`
   - Or train your own model using your airline reviews dataset

## Usage

Run the Streamlit dashboard:

```bash
streamlit run FINAL_PRODUCT.py
```

The application will start on `http://localhost:8501`

### Dashboard Features
- **Auto-refresh** - Continuously monitors incoming flight data
- **Aspect Gauges** - Visual indicators for each sentiment aspect
- **Flight Tracking** - Real-time processing of new flights
- **History Charts** - Sentiment trends over time

## Configuration

Edit `FINAL_PRODUCT.py` to modify:
- `ASPECTS` - Sentiment aspects to track
- `INCOMING_DIR` - Directory with flight CSV files
- `HISTORY_CAP` - Maximum history entries per aspect
- `MAX_SEQ_LEN` - BERT tokenizer max sequence length

## Model Details

- **Model Type**: BERT (Bidirectional Encoder Representations from Transformers)
- **Input**: Review text
- **Output**: Sentiment classification (Positive/Negative/Neutral)
- **Location**: `ABSA_UI/Model/`

## Data Format

### Input (incoming_flights/*.csv)
```
Cleaned_Reviews
"Great flight experience with excellent crew service"
"Food quality was poor but seat was comfortable"
```

### Output (airline_reviews_with_aspect_sentiments.csv)
```
Review_Text,Aspect_Sentiments,Flight_No,Route,Date,Time
"Great service","{\"crew\": \"Positive\", \"service\": \"Positive\"}",AI301,BOM-DEL,2025-10-01,06:00
```

## Dependencies

- **pandas** - Data manipulation
- **transformers** - BERT model and tokenizer
- **torch** - PyTorch deep learning framework
- **nltk** - Natural language processing (sentence tokenization)
- **plotly** - Interactive visualizations
- **streamlit** - Web dashboard framework
- **scikit-learn** - Machine learning utilities

## License

MIT License

## Author

Created for airline sentiment analysis project

## Support

For issues or questions, please open an issue on GitHub.
