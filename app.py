import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from wordcloud import WordCloud, STOPWORDS
import ast
from datetime import datetime
import re
from collections import Counter
from prophet import Prophet
import joblib
import plotly.graph_objects as go

# --- Session state to track active tab ---
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Sentiment vs Price"

tab_labels = ["Sentiment vs Price", "Mention & Alert", "Word Cloud", "Prediction"]
tabs = st.tabs(tab_labels)

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("merged_sentiment_cleaned_202005_202504.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['nlp_features'] = df['nlp_features'].apply(ast.literal_eval)
    df['tokens'] = df['nlp_features'].apply(lambda x: x.get('tokens', []))
    df['month'] = df['date'].dt.to_period('M')
    return df

df = load_data()

# --- Sidebar ---
st.sidebar.title("Filters")
filter_mode = st.sidebar.radio("Filter by", ["Date Range", "Single Day"], key="filter_mode")

if filter_mode == "Date Range":
    start_date_raw = st.sidebar.date_input("Start Date", df['date'].min(), key="start_date")
    end_date_raw = st.sidebar.date_input("End Date", df['date'].max(), key="end_date")
    start_date = pd.to_datetime(start_date_raw)
    end_date = pd.to_datetime(end_date_raw)
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
else:
    selected_date_raw = st.sidebar.date_input("Select Date", value=pd.to_datetime("2024-12-01"), key="single_day")
    selected_date = pd.to_datetime(selected_date_raw)
    start_date = selected_date
    end_date = selected_date
    mask = df['date'] == selected_date

filtered_df = df[mask]

# --- Load price data (used in multiple tabs) ---
price_df = pd.read_csv("sp500_price_202005_202504.csv")
price_df['date'] = pd.to_datetime(price_df['date'])
price_filtered = price_df[price_df['date'].between(start_date, end_date)]

# --- Sentiment vs Price Tab ---
with tabs[0]:
    st.session_state.active_tab = tab_labels[0]
    st.header("Sentiment and S&P500 Price Trend")

    # --- Company dropdown ---
    company_options = sorted(filtered_df['related'].dropna().unique().tolist())
    selected_companies = st.multiselect(
        "Select Company/Companies",
        options=company_options,
        default=company_options  # 預設全選
    )

    sentiment_df = filtered_df[filtered_df['related'].isin(selected_companies)]
    daily_sentiment = sentiment_df.groupby('date', as_index=False)['sentiment'].mean()

    # --- Merge with price data (always show price data) ---
    df_plot = pd.merge(
        price_filtered[['date', 'close']],
        daily_sentiment,
        on='date',
        how='left'
    )
    df_plot.rename(columns={'close': 'Close Price', 'sentiment': 'Sentiment'}, inplace=True)

    if df_plot.empty or df_plot['Close Price'].isna().all():
        st.warning("No price data available for the selected filters.")
    else:
        base = alt.Chart(df_plot).encode(x='date:T')
        line_price = base.mark_line(color='blue').encode(y=alt.Y('Close Price:Q', title="S&P500 Price"))
        chart_layers = [line_price]

        if df_plot['Sentiment'].notna().any():
            line_sentiment = base.mark_line(color='orange').encode(y=alt.Y('Sentiment:Q', title="Sentiment Score"))
            chart_layers.append(line_sentiment)

        chart = alt.layer(*chart_layers).resolve_scale(y='independent').interactive()
        st.altair_chart(chart, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Min Price", round(df_plot['Close Price'].min(), 2))
            st.metric("Max Price", round(df_plot['Close Price'].max(), 2))
            st.metric("Mean Price", round(df_plot['Close Price'].mean(), 2))
            st.metric("Std Dev Price", round(df_plot['Close Price'].std(), 2))
        with col2:
            if df_plot['Sentiment'].notna().any():
                st.metric("Min Sentiment", round(df_plot['Sentiment'].min(), 3))
                st.metric("Max Sentiment", round(df_plot['Sentiment'].max(), 3))
                st.metric("Mean Sentiment", round(df_plot['Sentiment'].mean(), 3))
                st.metric("Std Dev Sentiment", round(df_plot['Sentiment'].std(), 3))
            else:
                st.markdown("No sentiment data available.")

# --- Mention & Alert Tab ---
with tabs[1]:
    st.session_state.active_tab = tab_labels[1]
    st.header("Company Mentions and Alerts")
    mention_df = filtered_df[filtered_df['related'] != 'S&P 500']
    summary = mention_df.groupby("related").agg(
        mention_count=('title', 'count'),
        avg_sentiment=('sentiment', 'mean')
    ).reset_index()
    summary['alert'] = summary['avg_sentiment'].apply(lambda x: '❗️' if x < -0.5 else '')
    st.dataframe(summary.sort_values("mention_count", ascending=False))

# --- Word Cloud Tab ---
with tabs[2]:
    st.session_state.active_tab = tab_labels[2]
    st.header("Sentiment Word Cloud")

    tokens = [t.lower() for tokens in filtered_df['tokens'] for t in tokens if isinstance(t, str)]
    tokens = [re.sub(r'[^\w\s]', '', t) for t in tokens if t.isalpha()]
    stopwords = set(STOPWORDS).union({'the', 'in', 'it', 'of', 'to', 'and', 'as', 'for', 'on', 'is', 'its'})
    tokens = [word for word in tokens if word not in stopwords and len(word) > 1]

    if tokens:
        wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(" ".join(tokens))
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        st.subheader("Top 5 Keywords and Related Headlines")
        top_words = Counter(tokens).most_common(5)
        for word, _ in top_words:
            st.markdown(f"**{word}**")
            try:
                pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
                headlines = filtered_df[filtered_df['title'].str.contains(pattern, na=False)]['title'].drop_duplicates().head(5)
                for h in headlines:
                    st.markdown(f"- {h}")
            except re.error:
                st.markdown("_Regex error occurred_")
    else:
        st.warning("No tokens available to generate word cloud.")

# --- Prediction Tab ---
with tabs[3]:
    st.session_state.active_tab = tab_labels[3]
    st.header("S&P 500 Price Prediction")
    st.caption("⚠️ This page is not applicable to filters.")

    df_price = price_df.copy()
    df_price['ds'] = df_price['date']
    df_price['y'] = df_price['close']

    m = Prophet()
    m.fit(df_price[['ds', 'y']])
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_price['ds'], y=df_price['y'], mode='markers', name='Actual Price', marker=dict(color='black', size=4)))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', fill='tonexty',
                             fillcolor='rgba(0,0,255,0.2)', line=dict(width=0), showlegend=False))

    fig.update_layout(
        title='S&P 500 Forecast with Confidence Interval',
        xaxis_title='Date', yaxis_title='Price',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Table (Next 30 Days)")
    forecast_display = forecast[forecast['ds'] > df_price['ds'].max()].iloc[:30]
    forecast_display = forecast_display[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_display.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
    st.dataframe(forecast_display.reset_index(drop=True))
