# app.py - Only the changed parts

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from analytics import BatchAnalytics
from ai_agent import BatchWatchAgent
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="BatchWatch AI", layout="wide", page_icon="🤖")

@st.cache_data
def load_data(path="autosys_jobs.csv"):
    return pd.read_csv(path, parse_dates=['run_date', 'start_time', 'end_time'])

@st.cache_resource
def init_analytics(df):
    return BatchAnalytics(df, lookback_days=14)

@st.cache_resource
def init_agent(_analytics, max_tokens=4000):
    """Initialize AI agent - uses your llm_chat function"""
    return BatchWatchAgent(_analytics, max_tokens=max_tokens)

# Load data
df = load_data()
if df.empty:
    st.error("No data available")
    st.stop()

analytics = init_analytics(df)

# Sidebar
st.sidebar.title("🤖 BatchWatch AI")
role = st.sidebar.radio("Role", ["Batch Operations Team", "Application Owner"])

# Token limit control
max_tokens = st.sidebar.slider("Max Response Tokens", 1000, 8000, 4000, 500)

# Initialize agent with YOUR llm_chat
agent = init_agent(analytics, max_tokens=max_tokens)

app_names = sorted(df['application'].unique())
if role == "Batch Operations Team":
    selected_apps = st.sidebar.multiselect("Applications", app_names, default=app_names[:2])
else:
    selected_apps = [st.sidebar.selectbox("Your Application", app_names)]

if not selected_apps:
    st.warning("Please select at least one application")
    st.stop()

# Filter data
filtered_df = df[df['application'].isin(selected_apps)]

# Main dashboard
st.title("🤖 BatchWatch AI - Intelligent Batch Monitoring")
st.caption(f"Monitoring {len(selected_apps)} application(s) | Role: {role}")

# Key Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

total_jobs = filtered_df['job_id'].nunique()
recent_failures = filtered_df[
    (filtered_df['status'] == 'Failure') & 
    (filtered_df['run_date'] >= datetime.now() - timedelta(days=1))
].shape[0]
long_runners = analytics.identify_long_running_jobs()
long_runners = long_runners[long_runners['application'].isin(selected_apps)]
alerts = [a for a in analytics.generate_alerts() if a.application in selected_apps]
critical_alerts = sum(1 for a in alerts if a.severity == "Critical")

col1.metric("Total Jobs", total_jobs)
col2.metric("Recent Failures (24h)", recent_failures, delta=None, delta_color="inverse")
col3.metric("Long Runners", len(long_runners), delta=None, delta_color="inverse")
col4.metric("Active Alerts", len(alerts))
col5.metric("Critical Alerts", critical_alerts, delta=None, delta_color="inverse")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", 
    "⚠️ Long Running Jobs", 
    "❌ Failure Analysis", 
    "🔮 Predictions", 
    "🚨 Alerts & SLA", 
    "🤖 AI Assistant"
])

# === TAB 1: Overview ===
with tab1:
    st.subheader("📊 Application Overview")
    
    for app in selected_apps:
        with st.expander(f"**{app}**", expanded=True):
            app_df = filtered_df[filtered_df['application'] == app]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Jobs", app_df['job_id'].nunique())
            col2.metric("Avg Runtime", f"{app_df['runtime_min'].mean():.1f} min")
            col3.metric("Success Rate", f"{(app_df['status'] == 'Success').sum() / len(app_df) * 100:.1f}%")
            col4.metric("Executions", len(app_df))
            
            # Runtime trend
            st.write("**Runtime Trend (Last 30 Days)**")
            recent = app_df[app_df['run_date'] >= datetime.now() - timedelta(days=30)]
            daily_avg = recent.groupby(recent['run_date'].dt.date)['runtime_min'].mean().reset_index()
            
            fig = px.line(daily_avg, x='run_date', y='runtime_min', 
                         title=f"{app} - Daily Average Runtime")
            fig.add_hline(y=daily_avg['runtime_min'].mean(), 
                         line_dash="dash", line_color="red",
                         annotation_text="Average")
            st.plotly_chart(fig, use_container_width=True)

# === TAB 2: Long Running Jobs ===
with tab2:
    st.subheader("⚠️ Long Running Jobs Analysis")
    st.caption("Jobs running >1.5x their average runtime (last 14 days)")
    
    threshold = st.slider("Threshold Multiplier", 1.0, 3.0, 1.5, 0.1)
    
    long_runners_full = analytics.identify_long_running_jobs(threshold_multiplier=threshold)
    long_runners_filtered = long_runners_full[long_runners_full['application'].isin(selected_apps)]
    
    if long_runners_filtered.empty:
        st.success("✅ No long-running jobs detected with current threshold")
    else:
        st.warning(f"⚠️ {len(long_runners_filtered)} long-running jobs detected")
        
        # Display table
        st.dataframe(long_runners_filtered, use_container_width=True)
        
        # Visualization
        st.write("**Runtime Comparison**")
        fig = go.Figure()
        for _, row in long_runners_filtered.head(10).iterrows():
            fig.add_trace(go.Bar(
                name=row['job_id'],
                x=['Average', 'Current'],
                y=[row['avg_runtime'], row['runtime_min']],
                text=[f"{row['avg_runtime']:.1f}", f"{row['runtime_min']:.1f}"],
                textposition='auto'
            ))
        fig.update_layout(title="Long Runners - Average vs Current Runtime",
                         barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # AI Analysis
    st.write("---")
    st.write("**🤖 AI Analysis**")
    if st.button("Analyze Long Running Jobs", key="analyze_long"):
        with st.spinner("🤖 BatchWatch is analyzing..."):
            response = agent.query(
                "Analyze the long-running jobs. What are the root causes and what actions should we take?",
                application_filter=selected_apps
            )
            st.write("**Answer:**")
            st.write(response['answer'])
            st.caption(f"Confidence: {response['confidence'].upper()} | Context docs: {response['total_context_docs']}")
            
            with st.expander("View Context Used"):
                for i, ctx in enumerate(response['relevant_context'], 1):
                    st.text(f"Context {i}:")
                    st.code(ctx)
                    st.write("---")

# === TAB 3: Failure Analysis ===
with tab3:
    st.subheader("❌ Failure Analysis")
    
    failure_analysis = analytics.analyze_failures()
    
    # Job-level failure rate
    st.write("**Jobs with Highest Failure Rates**")
    job_failures = failure_analysis['by_job']
    job_failures_filtered = job_failures[job_failures['application'].isin(selected_apps)]
    
    if not job_failures_filtered.empty:
        top_failures = job_failures_filtered.head(10)
        st.dataframe(top_failures, use_container_width=True)
        
        fig = px.bar(top_failures, x='job_id', y='failure_rate',
                    color='application', text='failure_rate',
                    title="Top 10 Jobs by Failure Rate (%)")
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No job failures to analyze")
    
    # AI Analysis
    st.write("---")
    st.write("**🤖 AI Analysis**")
    if st.button("Analyze Failure Patterns", key="analyze_failures"):
        with st.spinner("🤖 Analyzing failure patterns..."):
            response = agent.query(
                "Analyze the failure patterns. Are there any concerning trends? Which jobs need immediate attention?",
                application_filter=selected_apps
            )
            st.write("**Answer:**")
            st.write(response['answer'])
            st.caption(f"Confidence: {response['confidence'].upper()}")

# === TAB 4: Predictions ===
with tab4:
    st.subheader("🔮 Runtime Predictions")
    st.caption("AI-powered predictions based on historical trends")
    
    prediction_days = st.slider("Prediction Horizon (days ahead)", 1, 30, 14)
    
    for app in selected_apps:
        st.write(f"### {app}")
        
        app_df = filtered_df[filtered_df['application'] == app]
        jobs = app_df['job_id'].unique()
        
        predictions = []
        for job_id in jobs[:20]:  # Limit to first 20
            pred = analytics.predict_future_runtime(job_id, days_ahead=prediction_days)
            if pred:
                predictions.append({
                    'Job ID': pred.job_id,
                    'Predicted Runtime (min)': pred.predicted_runtime_min,
                    'Confidence Range': f"{pred.confidence_interval[0]:.1f} - {pred.confidence_interval[1]:.1f}",
                    'Risk Level': pred.risk_level,
                    'Based on Runs': pred.based_on_runs
                })
        
        if predictions:
            pred_df = pd.DataFrame(predictions)
            st.dataframe(pred_df, use_container_width=True)
        else:
            st.info(f"Not enough historical data for predictions in {app}")

# === TAB 5: Alerts & SLA ===
with tab5:
    st.subheader("🚨 Alerts & SLA Monitoring")
    
    if not alerts:
        st.success("✅ No active alerts")
    else:
        st.warning(f"⚠️ {len(alerts)} active alerts")
        
        for alert in alerts:
            severity_icon = {"Critical": "🔴", "Warning": "🟡", "Info": "🔵"}
            
            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown(f"### {severity_icon[alert.severity]}")
                with col2:
                    st.markdown(f"**{alert.application}** - {alert.alert_type}")
                    st.write(alert.message)
                    st.caption(f"💡 {alert.recommended_action}")
                st.divider()

# === TAB 6: AI Assistant ===
with tab6:
    st.subheader("🤖 AI Assistant")
    st.caption("Ask BatchWatch AI anything about your batch jobs")
    
    # Executive Summary
    st.write("**📋 Executive Summary**")
    if st.button("Generate Dashboard Summary"):
        with st.spinner("🤖 Generating comprehensive summary..."):
            summary = agent.get_dashboard_summary(selected_apps)
            st.markdown(summary)
    
    st.write("---")
    
    # Quick questions
    st.write("**Quick Questions**")
    quick_questions = [
        "Which jobs are most likely to cause SLA breaches?",
        "What are the top 3 failure patterns I should address?",
        "Which application needs the most attention?",
        "What's causing the long-running jobs?",
        "Are there any concerning trends in the last week?",
    ]
    
    selected_question = st.selectbox("Select a question", ["Custom..."] + quick_questions)
    
    if selected_question == "Custom...":
        user_question = st.text_area(
            "Ask your question:",
            placeholder="E.g., Which jobs are predicted to breach SLA next week?",
            height=100
        )
    else:
        user_question = selected_question
        st.info(f"Selected: {user_question}")
    
    if st.button("Ask AI", type="primary"):
        if user_question.strip():
            with st.spinner("🤖 BatchWatch AI is analyzing..."):
                response = agent.query(user_question, application_filter=selected_apps)
                
                st.write("### Answer")
                st.markdown(response['answer'])
                
                st.write(f"**Confidence:** {response['confidence'].upper()}")
                st.write(f"**Context Documents Used:** {response['total_context_docs']}")
                
                with st.expander("📚 View Context Sources"):
                    for i, ctx in enumerate(response['relevant_context'], 1):
                        st.text(f"Source {i}:")
                        st.code(ctx)
                        st.write("---")
        else:
            st.warning("Please enter a question")

# === FOOTER ===
st.write("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"📊 Data loaded: {len(df)} execution records")
with col2:
    st.caption(f"🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col3:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# Auto-refresh option
auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=False)
if auto_refresh:
    import time
    time.sleep(60)
    st.rerun()

