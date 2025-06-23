import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from wordcloud import WordCloud, STOPWORDS
import ast
from datetime import datetime
import re
from collections import Counter
import string
from prophet import Prophet

# Session state to control sidebar
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Sentiment vs Price"

# Load sentiment data
@st.cache_data
def load_data():
    df = pd.read_csv("merged_sentiment_cleaned_202005_202504.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['nlp_features'] = df['nlp_features'].apply(ast.literal_eval)
    df['tokens'] = df['nlp_features'].apply(lambda x: x.get('tokens', []))
    df['month'] = df['date'].dt.to_period('M')
    return df

df = load_data()

# Sidebar (hidden for Prediction)
if st.session_state.active_tab != "Prediction":
    st.sidebar.title("Filters")
    filter_mode = st.sidebar.radio("Filter by", ["Date Range", "Single Day"])

    if filter_mode == "Date Range":
        start_date = st.sidebar.date_input("Start Date", df['date'].min())
        end_date = st.sidebar.date_input("End Date", df['date'].max())
        mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
    else:
        selected_date = st.sidebar.date_input("Select Date", value=pd.to_datetime("2024-12-01"))
        mask = df['date'] == pd.to_datetime(selected_date)

    filtered_df = df[mask]
else:
    filtered_df = df.copy()

# Main Tabs
st.title("SP500 News Sentiment Dashboard")
tabs = st.tabs(["Sentiment vs Price", "Mention & Alert", "Word Cloud", "Prediction"])

# Tab 1: Sentiment vs Price
with tabs[0]:
    st.header("Sentiment and S&P500 Price Trend")

    # Company options
    company_options = filtered_df['related'].unique().tolist()
    selected_companies = st.multiselect("Select Company/Companies", company_options, default=["S&P 500"])

    # Filter sentiment data
    sentiment_df = filtered_df[filtered_df['related'].isin(selected_companies)]
    daily_sentiment = sentiment_df.groupby('date')['sentiment'].mean().reset_index()

    # Load and filter S&P500 price data
    price_df = pd.read_csv("sp500_price_202005_202504.csv")
    price_df['date'] = pd.to_datetime(price_df['date'])
    price_df = price_df[price_df['date'].between(filtered_df['date'].min(), filtered_df['date'].max())]

    # Conbine dara
    df_plot = pd.merge(daily_sentiment, price_df[['date', 'close']], on='date', how='inner')
    df_plot.rename(columns={'close': 'Close Price', 'sentiment': 'Sentiment'}, inplace=True)

    if df_plot.empty:
        st.warning("No data available for the selected filters.")
    else:
        # Draw
        base = alt.Chart(df_plot).encode(x='date:T')

        line_price = base.mark_line(color='blue').encode(
            y=alt.Y('Close Price:Q',
                    axis=alt.Axis(title='S&P500 Price', titleColor='blue'),
                    scale=alt.Scale(domain=[df_plot['Close Price'].min()*0.98, df_plot['Close Price'].max()*1.02])),
            tooltip=['date:T', alt.Tooltip('Close Price:Q', format=',.2f')]
        )

        line_sentiment = base.mark_line(color='orange').encode(
            y=alt.Y('Sentiment:Q',
                    axis=alt.Axis(title='Sentiment', titleColor='orange'),
                    scale=alt.Scale(domain=[-1.1, 1.1])),
            tooltip=['date:T', alt.Tooltip('Sentiment:Q', format='.3f')]
        )

        chart = alt.layer(line_price, line_sentiment).resolve_scale(y='independent').interactive()
        st.altair_chart(chart, use_container_width=True)

        # Stats
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Price Stats (S&P 500)")
            st.metric("Min Price", round(df_plot['Close Price'].min(), 2))
            st.metric("Max Price", round(df_plot['Close Price'].max(), 2))
            st.metric("Mean Price", round(df_plot['Close Price'].mean(), 2))
            st.metric("Std Dev Price", round(df_plot['Close Price'].std(), 2))
        with col2:
            st.subheader("Sentiment Stats (Selected Company)")
            st.metric("Min Sentiment", round(df_plot['Sentiment'].min(), 3))
            st.metric("Max Sentiment", round(df_plot['Sentiment'].max(), 3))
            st.metric("Mean Sentiment", round(df_plot['Sentiment'].mean(), 3))
            st.metric("Std Dev Sentiment", round(df_plot['Sentiment'].std(), 3))

# Tab 2: Mentions
with tabs[1]:
    st.session_state.active_tab = "Mention & Alert"
    st.header("Company Mentions and Alerts")
    mention_df = filtered_df[filtered_df['related'] != 'S&P 500']
    summary = mention_df.groupby("related").agg(
        mention_count=('title', 'count'),
        avg_sentiment=('sentiment', 'mean')
    ).reset_index()
    summary['alert'] = summary['avg_sentiment'].apply(lambda x: '❗️' if x < -0.5 else '')
    st.dataframe(summary.sort_values("mention_count", ascending=False))

# Tab 3: Word Cloud
with tabs[2]:
    st.session_state.active_tab = "Word Cloud"
    st.header("Sentiment Word Cloud")

    all_tokens_raw = [token.lower() for tokens in filtered_df['tokens'] for token in tokens if isinstance(token, str)]
    cleaned_tokens = [re.sub(r'[^\w\s]', '', token) for token in all_tokens_raw if token.isalpha()]
    stopwords = set(STOPWORDS).union({
        'the', 'in', 'it', 'of', 'to', 'and', 'as', 'for', 'on', 'is', 'its', 'with', 'are', 'a', 'an', 'this', 'that'
    })
    filtered_tokens = [word for word in cleaned_tokens if word not in stopwords and len(word) > 1]

    wordcloud = WordCloud(width=1000, height=500, background_color='white', stopwords=stopwords).generate(" ".join(filtered_tokens))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    st.subheader("Top 5 Keywords and Related Headlines")
    top_tokens = Counter(filtered_tokens).most_common(5)
    for word, _ in top_tokens:
        st.markdown(f"**{word}**")
        try:
            pattern = re.compile(rf'\b{re.escape(word)}\b', flags=re.IGNORECASE)
            headlines = filtered_df[filtered_df['title'].str.contains(pattern, na=False)]['title'].drop_duplicates().head(5).tolist()
            for h in headlines:
                st.markdown(f"- {h}")
        except re.error:
            st.markdown("_Error parsing keyword pattern_")

# Tab 4: Prediction
with tabs[3]:
    st.session_state.active_tab = "Prediction"
    st.header("S&P 500 Price Prediction")

    st.info("This prediction is based on historical prices only and does not consider news sentiment.")

    df_price = pd.read_csv("sp500_price_202005_202504.csv")
    df_price = df_price.rename(columns={"date": "ds", "close": "y"})
    df_price["ds"] = pd.to_datetime(df_price["ds"])

    m = Prophet()
    m.fit(df_price)

    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    st.caption(f"Forecasting starts from the next day after the last available price date: {df_price['ds'].max().strftime('%Y-%m-%d')}")

    fig1 = m.plot(forecast)
    st.pyplot(fig1)

    st.subheader("Forecast Table (Next 30 Days)")
    forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_display = forecast_display[forecast_display['ds'] > df_price['ds'].max()]
    forecast_display = forecast_display.rename(columns={
        'ds': 'Date',
        'yhat': 'Predicted Price',
        'yhat_lower': 'Lower Bound',
        'yhat_upper': 'Upper Bound'
    })
    st.dataframe(forecast_display.reset_index(drop=True))
