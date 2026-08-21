import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="Virality Predictor Pro", layout="wide")

FEATURES = [
    'follower_count', 'following_count', 'account_age_days', 
    'verified_status', 'past_viral_tweets', 'avg_daily_tweets',
    'character_count', 'hashtag_count', 'mention_count', 'url_count', 
    'media_attached', 'sentiment_score', 'subjectivity_score', 'reading_level',
    'hour_of_day', 'day_of_week', 'is_weekend', 'mins_since_last_post',
    'trending_topic_match', 'author_reply_rate', 'account_engagement_rate', 'bot_probability'
]

@st.cache_data
def load_or_generate_data():
    try:
        df = pd.read_csv('tweets.csv', thousands=',')
        return df[FEATURES + ['is_viral']]
    except FileNotFoundError:
        np.random.seed(42)
        n_samples = 2000
        data = {feat: np.abs(np.random.randn(n_samples)) * np.random.randint(1, 100) for feat in FEATURES}
        
        data['verified_status'] = np.random.randint(0, 2, n_samples)
        data['media_attached'] = np.random.randint(0, 2, n_samples)
        data['is_weekend'] = np.random.randint(0, 2, n_samples)
        data['trending_topic_match'] = np.random.randint(0, 2, n_samples)
        
        viral_logic = (data['follower_count'] * 0.4 + data['sentiment_score'] * 0.3 + data['past_viral_tweets'] * 0.5)
        threshold = np.percentile(viral_logic, 80)
        data['is_viral'] = (viral_logic > threshold).astype(int)
        
        return pd.DataFrame(data)

df = load_or_generate_data()

@st.cache_resource
def train_hybrid_model(data):
    X = data[FEATURES]
    y = data['is_viral']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]
    
    meta_model = LogisticRegression()
    
    hybrid_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
    hybrid_model.fit(X_train_scaled, y_train)
    
    rf_model = hybrid_model.named_estimators_['rf']
    importances = rf_model.feature_importances_
    
    accuracy = hybrid_model.score(X_test_scaled, y_test)
    
    return hybrid_model, scaler, importances, accuracy

model, scaler, feature_importances, accuracy = train_hybrid_model(df)

st.title("🚀 Advanced Virality Prediction Engine")
st.markdown(f"**Model Architecture:** Stacked Ensemble (Random Forest + Gradient Boosting) → Logistic Regression Meta-Learner | **Test Accuracy:** `{accuracy*100:.2f}%`")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Live Prediction Panel")
    
    with st.form("prediction_form"):
        st.subheader("👤 User Profile")
        f_followers = st.number_input("Follower Count", value=10000, step=1000)
        f_following = st.number_input("Following Count", value=500, step=100)
        f_age = st.number_input("Account Age (Days)", value=365, step=30)
        f_verified = st.selectbox("Verified Status", ["No", "Yes"])
        f_past_viral = st.number_input("Past Viral Tweets", value=0, step=1)
        f_avg_daily = st.slider("Avg Daily Tweets", 0.0, 50.0, 2.0)
        
        st.subheader("📝 Content Metrics")
        f_chars = st.slider("Character Count", 10, 280, 150)
        f_hashtags = st.slider("Hashtag Count", 0, 15, 2)
        f_mentions = st.slider("Mention Count", 0, 10, 1)
        f_urls = st.slider("URL Count", 0, 5, 0)
        f_media = st.selectbox("Media Attached", ["No", "Yes"])
        f_sentiment = st.slider("Sentiment Score", 0.0, 100.0, 50.0)
        f_subjectivity = st.slider("Subjectivity Score", 0.0, 100.0, 50.0)
        f_reading = st.slider("Reading Level (Grade)", 1.0, 20.0, 8.0)
        
        st.subheader("⏳ Temporal Dynamics")
        f_hour = st.slider("Hour of Day (0-23)", 0, 23, 12)
        f_day = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
        f_weekend = st.selectbox("Is Weekend", ["No", "Yes"])
        f_mins_since = st.number_input("Mins Since Last Post", value=120, step=15)
        
        st.subheader("🌐 Network Context")
        f_trending = st.selectbox("Matches Trending Topic", ["No", "Yes"])
        f_reply_rate = st.slider("Author Reply Rate (%)", 0.0, 100.0, 15.0)
        f_engagement = st.slider("Account Engagement Rate (%)", 0.0, 100.0, 5.0)
        f_bot = st.slider("Bot Probability (%)", 0.0, 100.0, 1.0)
        
        submitted = st.form_submit_button("Predict Virality")

with col2:
    if submitted:
        input_data = {
            'follower_count': f_followers,
            'following_count': f_following,
            'account_age_days': f_age,
            'verified_status': 1 if f_verified == "Yes" else 0,
            'past_viral_tweets': f_past_viral,
            'avg_daily_tweets': f_avg_daily,
            'character_count': f_chars,
            'hashtag_count': f_hashtags,
            'mention_count': f_mentions,
            'url_count': f_urls,
            'media_attached': 1 if f_media == "Yes" else 0,
            'sentiment_score': f_sentiment,
            'subjectivity_score': f_subjectivity,
            'reading_level': f_reading,
            'hour_of_day': f_hour,
            'day_of_week': f_day,
            'is_weekend': 1 if f_weekend == "Yes" else 0,
            'mins_since_last_post': f_mins_since,
            'trending_topic_match': 1 if f_trending == "Yes" else 0,
            'author_reply_rate': f_reply_rate,
            'account_engagement_rate': f_engagement,
            'bot_probability': f_bot
        }
        
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        
        st.header("Prediction Results")
        if prediction == 1:
            st.success(f"🔥 **VIRAL** (Probability: {prob*100:.1f}%)")
        else:
            st.error(f"📉 **NOT VIRAL** (Probability: {prob*100:.1f}%)")
        
        st.divider()

    st.header("The Mechanics: Feature Weights")
    
    importance_df = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=True)
    
    fig = px.bar(
        importance_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        color='Importance',
        color_continuous_scale='reds',
        template='plotly_dark'
    )
    fig.update_layout(title="Full Model Analysis", margin=dict(l=0, r=0, t=40, b=0), height=600)
    st.plotly_chart(fig, use_container_width=True)
