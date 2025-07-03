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

# --- Load price data ---
price_df = pd.read_csv("sp500_price_202005_202504.csv")
price_df['date'] = pd.to_datetime(price_df['date'])
price_filtered = price_df[price_df['date'].between(start_date, end_date)]

# --- Load company-level price data for task 2 ---
company_price_df = pd.read_csv("https://raw.githubusercontent.com/JudyHsu0220/sp500-sentiment-dashboard/refs/heads/main/company_price_202005_202504.csv")
company_price_df['date'] = pd.to_datetime(company_price_df['date'])

# --- Sentiment vs Price Tab ---
with tabs[0]:
    st.session_state.active_tab = tab_labels[0]
    st.header("Sentiment and Company Price Trend")

    # --- Company dropdown (single-select) ---
    company_options = sorted(filtered_df['related'].dropna().unique().tolist())
    selected_company = st.selectbox("Select Company", options=company_options, index=None, placeholder="")

    if selected_company:
        sentiment_df = filtered_df[filtered_df['related'] == selected_company]
        sentiment_daily = sentiment_df.groupby('date', as_index=False)['sentiment'].mean()

        company_price_filtered = company_price_df[(company_price_df['company'] == selected_company) &
                                                  (company_price_df['date'].between(start_date, end_date))]

        df_plot = pd.merge(
            company_price_filtered[['date', 'close']],
            sentiment_daily,
            on='date',
            how='left'
        )
        df_plot.rename(columns={'close': 'Price', 'sentiment': 'Sentiment'}, inplace=True)

        if df_plot.empty or df_plot['Price'].isna().all():
            st.warning("No company price data available for the selected filters.")
        else:
            base = alt.Chart(df_plot).encode(x='date:T')
            line_price = base.mark_line(color='blue').encode(y=alt.Y('Price:Q', title="Price"))
            chart_layers = [line_price]

            if df_plot['Sentiment'].notna().any():
                line_sentiment = base.mark_line(color='orange').encode(y=alt.Y('Sentiment:Q', title="Sentiment"))
                chart_layers.append(line_sentiment)

            chart = alt.layer(*chart_layers).resolve_scale(y='independent').interactive()
            st.altair_chart(chart, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Min Price", round(df_plot['Price'].min(), 2))
                st.metric("Max Price", round(df_plot['Price'].max(), 2))
                st.metric("Mean Price", round(df_plot['Price'].mean(), 2))
                st.metric("Std Dev Price", round(df_plot['Price'].std(), 2))
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
    st.header("Topic Modeling with BERTopic")

    headlines = filtered_df.dropna(subset=['title'])['title'].drop_duplicates().tolist()

    if len(headlines) > 5:
        try:
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            topic_model = BERTopic(embedding_model=embedding_model)
            topics, probs = topic_model.fit_transform(headlines)

            topic_info = topic_model.get_topic_info()
            top_topics = topic_info[topic_info.Topic != -1].nlargest(5, 'Count')['Topic'].tolist()

            shown_headlines = set()
            for topic_id in top_topics:
                words = ", ".join([word for word, _ in topic_model.get_topic(topic_id)])
                st.markdown(f"**Topic {topic_id}: {words}**")
                topic_headlines = topic_model.get_documents_per_topic()[topic_id]
                unique_headlines = [h for h in topic_headlines if h not in shown_headlines][:5]
                shown_headlines.update(unique_headlines)
                for h in unique_headlines:
                    st.markdown(f"- {h}")
        except Exception as e:
            st.error(f"Topic modeling failed: {e}")
    else:
        st.warning("Not enough headlines for topic modeling.")

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

    forecast['actual'] = np.interp(
        forecast['ds'].astype(np.int64),
        df_price['ds'].astype(np.int64),
        df_price['y']
    )
    forecast['date_str'] = forecast['ds'].dt.strftime('%Y-%m-%d')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted Price',
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
    fig.add_trace(go.Scatter(x=df_price['ds'], y=df_price['y'], mode='markers', name='Actual Price',
                             marker=dict(color='black', size=4), hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound',
                             line=dict(width=0), showlegend=True, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound',
                             fill='tonexty', fillcolor='rgba(0,0,255,0.2)', line=dict(width=0), showlegend=True,
                             hoverinfo='skip'))

    fig.update_layout(
        title='S&P 500 Forecast with Confidence Interval',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Table (Next 30 Days)")
    forecast_display = forecast[forecast['ds'] > df_price['ds'].max()].iloc[:30]
    forecast_display = forecast_display[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_display.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
    st.dataframe(forecast_display.reset_index(drop=True))
