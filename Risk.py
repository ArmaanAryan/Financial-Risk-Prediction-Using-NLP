import os
import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime, timedelta
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from ta.momentum import RSIIndicator
import nltk
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import random
from typing import List, Dict, Tuple, Any, Optional

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)

# --- Download NLTK data ---
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logging.error(f"Failed to download NLTK punkt: {e}")
    raise

# --- Set random seeds for reproducibility ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# --- PARAMETERS ---
LAMBDA_DECAY = 0.15
SENTIMENT_WINDOW = 7
NEGATIVE_THRESHOLD = -0.02
BATCH_SIZE = 64
LOOKBACK_DAYS = 180

# --- DEVICE SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# --- MODEL LOADING ---
def load_models() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification]:
    """Load FinBERT and Financial RoBERTa models with tokenizers."""
    finbert_model_name = "ProsusAI/finbert"
    roberta_model_name = "soleimanian/financial-roberta-large-sentiment"
    try:
        finbert_tok = AutoTokenizer.from_pretrained(finbert_model_name)
        finbert = AutoModelForSequenceClassification.from_pretrained(finbert_model_name).to(device)
        roberta_tok = AutoTokenizer.from_pretrained(roberta_model_name)
        roberta = AutoModelForSequenceClassification.from_pretrained(roberta_model_name).to(device)
        logging.info("Models loaded successfully.")
        return finbert_tok, finbert, roberta_tok, roberta
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        raise

finbert_tok, finbert, roberta_tok, roberta = load_models()

# --- PREPROCESSING FUNCTIONS ---
def preprocess_text(text: str) -> str:
    """Preprocess text for sentiment analysis."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r"tesla inc|tesla motors|tsla", "<COMPANY>", text)
    text = re.sub(r"elon musk", "<PERSON>", text)
    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens[:512])

def get_sentiment_probs(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    batch_size: int = BATCH_SIZE
) -> np.ndarray:
    """Compute sentiment probabilities in batches."""
    all_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
            all_probs.extend(probs.tolist())
    return np.array(all_probs)

def base_sentiment_score(probs: np.ndarray) -> float:
    """Convert [neg, neu, pos] to score in [-1, 1]."""
    return float(probs[2] - probs[0])

def temporal_decay_score(score: float, delta_days: int, lambd: float = LAMBDA_DECAY) -> float:
    """Apply temporal decay to sentiment score."""
    return float(score) * np.exp(-lambd * delta_days)

# --- DATA FETCHING ---
def fetch_stock_data(ticker: str = "TSLA", days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Fetch stock data using yfinance or generate synthetic data."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            logging.warning("No stock data available. Using synthetic data.")
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            np.random.seed(RANDOM_SEED)
            sentiment_scores = np.random.choice([-1, 0, 1], size=len(dates), p=[0.35, 0.3, 0.35])
            price_changes = np.random.normal(0, 0.50, len(dates)) + 0.12 * sentiment_scores
            df = pd.DataFrame({
                'Date': dates,
                'Close': 100 * (1 + price_changes).cumprod(),
                'Volume': np.random.normal(1_000_000, 200_000, len(dates))
            })
        df = df.reset_index()
        df['date'] = pd.to_datetime(df['Date']).dt.date
        df = df[['date', 'Close', 'Volume']]
        df.columns = ['date', 'stock_price', 'volume']
        df['volume'] = df['volume'].clip(lower=100_000)
        logging.info(f"Stock data shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error fetching stock data: {e}")
        print(f"Stock data fetch failed: {e}")
        raise

def fetch_x_posts(
    query: str = "Tesla OR TSLA -from:elonmusk",
    days: int = LOOKBACK_DAYS,
    max_posts: int = 1000
) -> pd.DataFrame:
    """Generate synthetic posts data aligned with stock data dates."""
    try:
        stock_df = fetch_stock_data(ticker="TSLA", days=days)
        dates = stock_df['date'].values
        np.random.seed(RANDOM_SEED)
        sentiment_scores = np.random.choice([-1, 0, 1], size=len(dates), p=[0.35, 0.3, 0.35])
        posts = []
        for date, sentiment in zip(dates, sentiment_scores):
            num_posts = np.random.randint(12, 20)
            for _ in range(num_posts):
                if sentiment == 1:
                    text = random.choice([
                        f"<COMPANY> stock skyrockets on breakthrough innovation!",
                        f"<COMPANY> beats earnings expectations, shares soar.",
                        f"Strong <COMPANY> outlook drives bullish sentiment."
                    ])
                elif sentiment == -1:
                    text = random.choice([
                        f"<COMPANY> stock tanks after earnings miss.",
                        f"<COMPANY> faces lawsuits, shares plummet.",
                        f"Negative <COMPANY> news triggers sell-off."
                    ])
                else:
                    text = random.choice([
                        f"<COMPANY> stock remains stable as market awaits new developments.",
                        f"<COMPANY> trading flat amid mixed economic signals.",
                        f"Investors hold steady on <COMPANY> pending quarterly report."
                    ])
                posts.append({'date': date, 'text': text})
        df = pd.DataFrame(posts)
        df['date'] = pd.to_datetime(df['date']).dt.date
        logging.info(f"Generated {len(df)} synthetic posts for {len(dates)} days.")
        return df
    except Exception as e:
        logging.error(f"Error generating synthetic posts: {e}")
        print(f"Synthetic posts generation failed: {e}")
        raise

# --- DATA PROCESSING ---
def process_data(stock_df: pd.DataFrame, posts_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess, compute sentiment, engineer features, and align with market data."""
    logging.info(f"Stock data shape: {stock_df.shape}, Posts data shape: {posts_df.shape}")
    posts_df['processed_text'] = posts_df['text'].apply(preprocess_text)
    logging.info(f"Processed texts: {posts_df['processed_text'].notnull().sum()}")
    texts = posts_df['processed_text'].tolist()
    finbert_probs = get_sentiment_probs(texts, finbert_tok, finbert)
    roberta_probs = get_sentiment_probs(texts, roberta_tok, roberta)
    posts_df[['fin_neg', 'fin_neu', 'fin_pos']] = finbert_probs
    posts_df[['roberta_neg', 'roberta_neu', 'roberta_pos']] = roberta_probs
    posts_df['finbert_score'] = [base_sentiment_score(p) for p in finbert_probs]
    posts_df['roberta_score'] = [base_sentiment_score(p) for p in roberta_probs]
    posts_df['finbert_conf'] = np.max(finbert_probs, axis=1)
    posts_df['roberta_conf'] = np.max(roberta_probs, axis=1)
    posts_df['finbert_DCS'] = posts_df['finbert_score'] * posts_df['finbert_conf']
    posts_df['roberta_DCS'] = posts_df['roberta_score'] * posts_df['roberta_conf']
    max_date = pd.to_datetime(posts_df['date']).max()
    posts_df['days_ago'] = (max_date - pd.to_datetime(posts_df['date'])).dt.days
    posts_df['finbert_TDF'] = posts_df.apply(lambda x: temporal_decay_score(x['finbert_score'], x['days_ago']), axis=1)
    posts_df['roberta_TDF'] = posts_df.apply(lambda x: temporal_decay_score(x['roberta_score'], x['days_ago']), axis=1)
    sentiment_df = posts_df.groupby('date').agg({
        'finbert_TDF': 'mean',
        'roberta_TDF': 'mean',
        'finbert_DCS': ['mean', 'var'],
        'roberta_DCS': ['mean', 'var']
    }).reset_index()
    sentiment_df.columns = ['date', 'finbert_TDF', 'roberta_TDF', 'finbert_DCS', 'finbert_DCS_var', 'roberta_DCS', 'roberta_DCS_var']
    sentiment_df['AMSI'] = (sentiment_df['finbert_TDF'] + sentiment_df['roberta_TDF']) / 2
    sentiment_df['AMSI_rolling'] = sentiment_df['AMSI'].rolling(window=SENTIMENT_WINDOW, min_periods=1).mean()
    stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date
    df = stock_df.merge(sentiment_df, on='date', how='left')
    df[['AMSI_rolling', 'finbert_DCS', 'roberta_DCS', 'finbert_DCS_var', 'roberta_DCS_var']] = df[['AMSI_rolling', 'finbert_DCS', 'roberta_DCS', 'finbert_DCS_var', 'roberta_DCS_var']].fillna(0)
    df['rsi_7'] = RSIIndicator(close=df['stock_price'], window=7).rsi().bfill()
    df['return'] = df['stock_price'].pct_change()
    df['risk_label'] = (df['return'].shift(-1) < NEGATIVE_THRESHOLD).astype(int)
    df['volume_change'] = df['volume'].pct_change().fillna(0)
    df['AMSI_rsi_interaction'] = df['AMSI_rolling'] * df['rsi_7']
    df['volume_sentiment_interaction'] = df['volume_change'] * df['AMSI_rolling']
    df['sentiment_trend'] = df['AMSI_rolling'].diff().fillna(0)
    feature_stats = df[['stock_price', 'AMSI_rolling', 'rsi_7', 'finbert_DCS', 'roberta_DCS', 'roberta_DCS_var', 'volume_change', 'AMSI_rsi_interaction', 'volume_sentiment_interaction', 'sentiment_trend']].describe()
    logging.info(f"Feature statistics:\n{feature_stats}")
    logging.info(f"Merged data shape: {df.shape}")
    return df

# --- MODEL TRAINING ---
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> XGBClassifier:
    """Train XGBoost model with manual early stopping."""
    xgb_model = XGBClassifier(
        learning_rate=0.05,
        max_depth=4,
        n_estimators=200,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=RANDOM_SEED,
        n_jobs=-1,
        base_score=0.5,
        scale_pos_weight=sum(y_train == 0) / sum(y_train == 1) * 1.5 if sum(y_train == 1) > 0 else 1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0
    )
    best_mcc = -1
    best_model = None
    patience = 20
    patience_counter = 0
    for i in range(1, 201):
        xgb_model.n_estimators = i
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_pred = xgb_model.predict(X_val)
        val_mcc = matthews_corrcoef(y_val, val_pred)
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            best_model = xgb_model
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            logging.info(f"Early stopping at iteration {i}, best validation MCC: {best_mcc:.4f}")
            break
    if best_model is None:
        best_model = xgb_model
    logging.info(f"Model training completed. Best validation MCC: {best_mcc:.4f}")
    return best_model

# --- EVALUATION ---
def evaluate_model(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_file: str = "evaluation_results.txt"
) -> Tuple[str, float, float, float]:
    """Evaluate model and save results."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    if len(np.unique(y_test)) > 1:
        report = classification_report(y_test, y_pred, digits=3, zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        mcc = matthews_corrcoef(y_test, y_pred)
    else:
        logging.warning("Only one class in test set. Metrics undefined.")
        report = "Classification report undefined: single class in test set."
        f1 = 0.0
        auc = float('nan')
        mcc = 0.0
    with open(output_file, 'w') as f:
        f.write("Classification Report:\n")
        f.write(str(report) + "\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"AUC-ROC: {auc:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n")
    logging.info(f"Evaluation results saved to {output_file}")
    return report, f1, auc, mcc

# --- FEATURE IMPORTANCE ---
def plot_feature_importance(
    model: XGBClassifier,
    feature_names: List[str],
    output_file: str = "feature_importance.png"
) -> pd.DataFrame:
    """Plot and save feature importance."""
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    fi_df = fi_df.sort_values(by='importance', ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(y='feature', x='importance', hue='feature', data=fi_df, palette='viridis', legend=False)
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    logging.info(f"Feature importance plot saved to {output_file}")
    return fi_df

# --- PREDICTION FUNCTION ---
def predict_risk(
    model: XGBClassifier,
    scaler: StandardScaler,
    input_data: Dict[str, List[Any]]
) -> np.ndarray:
    """Predict downside risk probability for new data."""
    expected_features = ['AMSI_rolling', 'rsi_7', 'finbert_DCS', 'roberta_DCS', 'roberta_DCS_var', 'volume_change', 'AMSI_rsi_interaction', 'volume_sentiment_interaction', 'sentiment_trend']
    input_df = pd.DataFrame(input_data)
    missing_cols = [col for col in expected_features if col not in input_df.columns]
    if missing_cols:
        raise ValueError(f"Input data missing required features: {missing_cols}")
    input_scaled = scaler.transform(input_df[expected_features])
    pred_prob = model.predict_proba(input_scaled)[:, 1]
    return pred_prob

# --- MAIN EXECUTION ---
def main():
    """Main function to run the pipeline."""
    try:
        stock_df = fetch_stock_data(ticker="TSLA")
        posts_df = fetch_x_posts(query="Tesla OR TSLA -from:elonmusk", max_posts=1000)
        df = process_data(stock_df, posts_df)
        feature_cols = ['AMSI_rolling', 'rsi_7', 'finbert_DCS', 'roberta_DCS', 'roberta_DCS_var', 'volume_change', 'AMSI_rsi_interaction', 'volume_sentiment_interaction', 'sentiment_trend']
        features_df = df[feature_cols].dropna()
        labels = df.loc[features_df.index, 'risk_label'].values
        logging.info(f"Features DataFrame shape: {features_df.shape}, Label distribution: {np.bincount(labels)}")
        if len(features_df) < 3:
            raise ValueError("Insufficient data for training.")
        # Plot label distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(x=pd.Series(labels, dtype='category'), hue=pd.Series(labels, dtype='category'), palette='viridis', legend=False)
        plt.xlabel('Risk Label')
        plt.ylabel('Count')
        plt.title('Label Distribution')
        plt.savefig('label_distribution.png')
        plt.close()
        logging.info("Label distribution plot saved to label_distribution.png")
        # Feature scaling
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        scaled_stats = pd.DataFrame(features_scaled, columns=feature_cols).describe()
        logging.info(f"Scaled feature statistics:\n{scaled_stats}")
        # Stratified split
        X_temp, X_test, y_temp, y_test = train_test_split(
            features_scaled, labels, test_size=0.15, stratify=labels, random_state=RANDOM_SEED
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.3, stratify=y_temp, random_state=RANDOM_SEED
        )
        # Log training labels before SMOTE
        logging.info(f"Train labels before SMOTE: {np.bincount(y_train)}")
        # Apply SMOTE if sufficient positive samples
        if sum(y_train == 1) >= 4:
            logging.info("Applying SMOTE...")
            smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=3, sampling_strategy=1.0)
            try:
                X_train, y_train = smote.fit_resample(X_train, y_train)
                logging.info(f"Train labels after SMOTE: {np.bincount(y_train)}")
            except ValueError as e:
                logging.warning(f"SMOTE failed: {e}. Proceeding without SMOTE.")
        else:
            logging.warning(f"SMOTE skipped: insufficient positive samples ({sum(y_train == 1)})")
        logging.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
        logging.info(f"Train labels: {np.bincount(y_train)}, Val labels: {np.bincount(y_val)}, Test labels: {np.bincount(y_test)}")
        model = train_model(X_train, y_train, X_val, y_val)
        # Log validation performance
        val_pred = model.predict(X_val)
        val_report = classification_report(y_val, val_pred, digits=3, zero_division=0)
        logging.info(f"Validation Classification Report:\n{val_report}")
        report, f1, auc, mcc = evaluate_model(model, X_test, y_test)
        print("Classification Report:\n", report)
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"MCC: {mcc:.4f}")
        fi_df = plot_feature_importance(model, feature_cols)
        print("\nFeature Importance:")
        print(fi_df)
        example = {
            'AMSI_rolling': [0.1],
            'rsi_7': [df['rsi_7'].iloc[-1] if not df['rsi_7'].empty else 55.0],
            'finbert_DCS': [0.2],
            'roberta_DCS': [0.16],
            'roberta_DCS_var': [0.01],
            'volume_change': [0.0],
            'AMSI_rsi_interaction': [0.1 * (df['rsi_7'].iloc[-1] if not df['rsi_7'].empty else 55.0)],
            'volume_sentiment_interaction': [0.0],
            'sentiment_trend': [0.0]
        }
        pred_prob = predict_risk(model, scaler, example)
        print(f"\nPredicted Downside Risk Probability for Example: {pred_prob[0]:.4f}")
        plt.figure(figsize=(10, 6))
        plt.plot(df['date'], df['stock_price'], label='Stock Price', color='blue')
        plt.plot(df['date'], df['AMSI_rolling'] * df['stock_price'].std() + df['stock_price'].mean(), label='AMSI Rolling (Scaled)', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('TSLA Stock Price vs. AMSI Rolling')
        plt.legend()
        plt.savefig('stock_vs_amsi.png')
        plt.close()
        logging.info("Stock vs. AMSI plot saved to stock_vs_amsi.png")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()