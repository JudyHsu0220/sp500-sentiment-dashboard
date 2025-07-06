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
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# --- Session state ---
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Sentiment vs Price"

tab_labels = ["Sentiment vs Price", "Mention & Alert", "Word Cloud", "Prediction"]
tabs = st.tabs(tab_labels)

@st.cache_data
def load_data():
    df = pd.read_csv("merged_sentiment_cleaned_202005_202504.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['nlp_features'] = df['nlp_features'].apply(ast.literal_eval)
    df['tokens'] = df['nlp_features'].apply(lambda x: x.get('tokens', []))
    df['month'] = df['date'].dt.to_period('M')
    return df

df = load_data()

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

company_price_df = pd.read_csv("company_price_202005_202504.csv")
company_price_df['date'] = pd.to_datetime(company_price_df['date'])

# --- Sentiment vs Price Tab ---
with tabs[0]:
    st.session_state.active_tab = tab_labels[0]
    st.header("Sentiment and S&P500 Price Trend")

    company_options = sorted(filtered_df['related'].dropna().unique().tolist())
    selected_company = st.selectbox("Select Company", options=["S&P 500"] + company_options, index=0)

    if selected_company == "S&P 500":
        sentiment_df = filtered_df[filtered_df['related'] == "S&P 500"]
        price_filtered = pd.read_csv("sp500_price_202005_202504.csv")
        price_filtered['date'] = pd.to_datetime(price_filtered['date'])
        price_filtered = price_filtered[price_filtered['date'].between(start_date, end_date)]
        price_col = "close"
        price_label = "S&P500 Price"
    else:
        sentiment_df = filtered_df[filtered_df['related'] == selected_company]
        price_filtered = company_price_df[(company_price_df['company'] == selected_company) &
                                          (company_price_df['date'].between(start_date, end_date))]
        price_col = "price"
        price_label = f"{selected_company} Price"

    daily_sentiment = sentiment_df.groupby('date', as_index=False)['sentiment'].mean()
    df_plot = pd.merge(price_filtered[['date', price_col]], daily_sentiment, on='date', how='left')
    df_plot.rename(columns={price_col: 'Close Price', 'sentiment': 'Sentiment'}, inplace=True)

    if df_plot.empty or df_plot['Close Price'].isna().all():
        st.warning("No price data available for the selected filters.")
    else:
        df_plot['Sentiment_display'] = df_plot['Sentiment'].round(2).astype(str).replace("nan", "N/A")
        df_plot['Close Price'] = df_plot['Close Price'].round(2)
        df_plot['date_str'] = df_plot['date'].dt.strftime('%Y-%m-%d')

        base = alt.Chart(df_plot).encode(x='date:T')

        price_line = base.mark_line(color='blue').encode(
            y=alt.Y('Close Price:Q', title="Price")
        )

        sentiment_points = base.mark_point(color='orange', size=40).encode(
            y=alt.Y('Sentiment:Q', title="Sentiment Score"),
            tooltip=[
                alt.Tooltip('date_str:N', title='Date'),
                alt.Tooltip('Close Price:Q', title='Price'),
                alt.Tooltip('Sentiment_display:N', title='Sentiment')
            ]
        )

        chart = alt.layer(price_line, sentiment_points).resolve_scale(y='independent').interactive()
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
    st.header("Mention Volume and Sentiment Alert")

    mention_volume = filtered_df.groupby('related', as_index=False).size().sort_values('size', ascending=False)
    top_mentions = mention_volume.head(20)

    st.subheader("Top 20 Mentioned Companies")
    st.bar_chart(data=top_mentions.set_index('related'), use_container_width=True)

    sentiment_summary = filtered_df.groupby('related')['sentiment'].mean().reset_index()
    negative_alerts = sentiment_summary[sentiment_summary['sentiment'] < -0.3].sort_values('sentiment')

    st.subheader("Companies with Most Negative Sentiment")
    st.dataframe(negative_alerts.head(10).rename(columns={'sentiment': 'Avg Sentiment'}))

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

    # Prepare data
    df_price = price_df.copy()
    df_price['ds'] = df_price['date']
    df_price['y'] = df_price['close']

    # Prophet forecast
    m = Prophet()
    m.fit(df_price[['ds', 'y']])
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    # Align forecast with actuals using index
    forecast['actual'] = np.interp(
        forecast['ds'].astype(np.int64),
        df_price['ds'].astype(np.int64),
        df_price['y']
    )

    # Merge date to use as customdata
    forecast['date_str'] = forecast['ds'].dt.strftime('%Y-%m-%d')

    # Build plotly chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='blue'),
        customdata=forecast[['date_str', 'actual', 'yhat_upper', 'yhat_lower']],
        hovertemplate=(
            'Date: %{customdata[0]}<br>'
            'Predicted Price: %{y:.2f}<br>'
            'Actual Price: %{customdata[1]:.2f}<br>'
            'Upper Bound: %{customdata[2]:.2f}<br>'
            'Lower Bound: %{customdata[3]:.2f}<extra></extra>'
        )
    ))

    # Add actual price as dots (no hover to avoid redundancy)
    fig.add_trace(go.Scatter(
        x=df_price['ds'],
        y=df_price['y'],
        mode='markers',
        name='Actual Price',
        marker=dict(color='black', size=4),
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=True,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        name='Lower Bound',
        fill='tonexty',
        fillcolor='rgba(0,0,255,0.2)',
        line=dict(width=0),
        showlegend=True,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title='S&P 500 Forecast with Confidence Interval',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Forecast Table
    st.subheader("Forecast Table (Next 30 Days)")
    forecast_display = forecast[forecast['ds'] > df_price['ds'].max()].iloc[:30]
    forecast_display = forecast_display[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_display.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
    st.dataframe(forecast_display.reset_index(drop=True))
