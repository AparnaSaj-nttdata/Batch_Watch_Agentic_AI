import streamlit as st
import pandas as pd
from datetime import timedelta
import os
import time

if "history_store" not in st.session_state:
    st.session_state.history_store = {}

# LangChain & Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory

# --- Analytics Helpers ---

@st.cache_data
def load_autosys_csv(path="../data/autosys_jobs.csv"):
    try:
        df = pd.read_csv(path, parse_dates=["run_date"])
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

@st.cache_data
def detect_long_running(df, lookback_days=14, threshold=1.5):
    if df.empty:
        return pd.DataFrame()
    latest_date = df['run_date'].max()
    cutoff = latest_date - timedelta(days=lookback_days)
    recent_df = df[df['run_date'] >= cutoff]
    avg_runtime = recent_df.groupby('job_id')['runtime_min'].mean()
    latest_runs = recent_df.sort_values('run_date').groupby('job_id').tail(1)
    merged = latest_runs.merge(avg_runtime.rename('avg_runtime'), left_on='job_id', right_index=True)
    merged['ratio'] = merged['runtime_min'] / merged['avg_runtime']
    flagged = merged[merged['ratio'] > threshold][['job_id', 'box_name', 'run_date', 'runtime_min', 'avg_runtime', 'ratio']]
    return flagged.reset_index(drop=True)

@st.cache_data
def failure_aggregates(df):
    failures = df[df['status'] == 'Failure'].copy()
    if failures.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    failures['day'] = failures['run_date'].dt.date
    failures['week'] = failures['run_date'].dt.to_period('W').astype(str)
    failures['month'] = failures['run_date'].dt.to_period('M').astype(str)
    daily = failures.groupby('day').size().reset_index(name='failures')
    weekly = failures.groupby('week').size().reset_index(name='failures')
    monthly = failures.groupby('month').size().reset_index(name='failures')
    return daily, weekly, monthly

@st.cache_data
def app_level_aggregates(df):
    agg = df.groupby('application').agg(
        job_count=('job_id', 'nunique'),
        avg_start_time=('run_date', 'min'),
        last_completion_time=('run_date', 'max'),
        avg_runtime=('runtime_min', 'mean'),
        fail_count=('status', lambda x: (x == 'Failure').sum())
    ).reset_index()
    return agg

def predict_completion(df, application, current_time=None):
    app_df = df[df['application'] == application]
    if app_df.empty:
        return None
    avg_runtime = app_df['runtime_min'].mean()
    last_job = app_df.sort_values('run_date').iloc[-1]
    if current_time is None:
        current_time = pd.Timestamp.now()
    expected_completion = last_job['run_date'] + timedelta(minutes=avg_runtime)
    return {
        'application': application,
        'last_job_start': last_job['run_date'],
        'expected_completion': expected_completion,
        'avg_runtime': avg_runtime,
    }

def predict_future_runtime(df, job_id, future_days=14):
    job_df = df[df['job_id'] == job_id].sort_values('run_date')
    if job_df.empty:
        return None
    lookback_df = job_df[job_df['run_date'] >= (job_df['run_date'].max() - timedelta(days=30))]
    avg_runtime = lookback_df['runtime_min'].mean()
    future_date = job_df['run_date'].max() + timedelta(days=future_days)
    return {
        'job_id': job_id,
        'avg_runtime': avg_runtime,
        'future_date': future_date
    }

def jobs_to_documents(df):
    docs = []
    for _, row in df.iterrows():
        doc = (
            f"Application: {row['application']}\n"
            f"Job ID: {row['job_id']}\n"
            f"Box Name: {row['box_name']}\n"
            f"Run Date: {row['run_date']}\n"
            f"Runtime (min): {row['runtime_min']}\n"
            f"Status: {row['status']}"
        )
        docs.append(doc)
    failures = df[df['status'] == 'Failure'].copy()
    if not failures.empty:
        failures['day'] = failures['run_date'].dt.date
        failures['week'] = failures['run_date'].dt.to_period('W').astype(str)
        failures['month'] = failures['run_date'].dt.to_period('M').astype(str)
        daily = failures.groupby(['application','day']).size().reset_index(name='failures')
        for _, row in daily.iterrows():
            docs.append(f"Failure aggregate - Application: {row['application']} - Date: {row['day']}, Failure count: {row['failures']}")
        weekly = failures.groupby(['application','week']).size().reset_index(name='failures')
        for _, row in weekly.iterrows():
            docs.append(f"Failure aggregate - Application: {row['application']} - Week: {row['week']}, Failure count: {row['failures']}")
        monthly = failures.groupby(['application','month']).size().reset_index(name='failures')
        for _, row in monthly.iterrows():
            docs.append(f"Failure aggregate - Application: {row['application']} - Month: {row['month']}, Failure count: {row['failures']}")
    agg = app_level_aggregates(df)
    for _, row in agg.iterrows():
        docs.append(
            f"Application summary: {row['application']}\n"
            f"Total jobs: {row['job_count']}\n"
            f"Avg Start Time: {row['avg_start_time']}\n"
            f"Last Completion Time: {row['last_completion_time']}\n"
            f"Avg Runtime: {row['avg_runtime']}\n"
            f"Failures: {row['fail_count']}"
        )
    # Add future runtime predictions for each job
    for job_id in df['job_id'].unique():
        pred = predict_future_runtime(df, job_id, future_days=14)
        if pred:
            docs.append(
                f"Future runtime prediction for Job {job_id}: "
                f"On {pred['future_date']}, expected average runtime is {pred['avg_runtime']:.2f} minutes."
            )
    return docs

@st.cache_resource
def init_agent(df):
    embeddings = OllamaEmbeddings(model="all-minilm")
    job_docs = jobs_to_documents(df)
    index_dir = "faiss_index_dir"
    if os.path.exists(index_dir):
        import shutil
        shutil.rmtree(index_dir)
    vs = FAISS.from_texts(job_docs, embedding=embeddings)
    vs.save_local(index_dir)
    retriever = vs.as_retriever(search_kwargs=dict(k=10))
    llm = ChatOllama(model="phi4-mini:3.8b")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are BatchWatch, an AI agent monitoring Autosys batch jobs. Focus on job runtimes, failures, SLA patterns, and application-level analytics. Use historical data and provided predictions to estimate future average runtimes (e.g., 2 weeks from now) for any job when asked."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    def get_session_history(session_id: str):
        if session_id not in st.session_state.history_store:
            st.session_state.history_store[session_id] = InMemoryChatMessageHistory()
        return st.session_state.history_store[session_id]
    chain = prompt | llm
    runnable = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    return runnable, retriever, llm, vs

st.set_page_config(page_title="BatchWatch Agent", layout="wide")
st.title("🤖 BatchWatch Agent (POC - Modern LangChain)")

df = load_autosys_csv()
if df.empty:
    st.warning("No data loaded. Please check your CSV file.")
    st.stop()

# --- Role-based view ---
role = st.selectbox("Select your role", ["Batch Team", "Application Owner"])
app_names = sorted(df['application'].unique())

if role == "Batch Team":
    selected_apps = st.multiselect("Choose Application(s) to Monitor", app_names)
else:
    selected_apps = [st.selectbox("Choose Application", app_names)]

filtered_df = df[df['application'].isin(selected_apps)]

runnable, retriever, llm, vs = init_agent(df)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Job Overview", "⚠️ Long Runners", "❌ Failures", "🧠 AI Agent", "📈 App Analytics", "🚨 Alerts"
])

# --- Tab 1: Job Overview ---
with tab1:
    st.subheader("Autosys Job Data")
    st.dataframe(filtered_df.head(20))

# --- Tab 2: Long Running Jobs ---
with tab2:
    st.subheader("Long Running Jobs vs Historical Averages")
    for app in selected_apps:
        app_df = filtered_df[filtered_df['application'] == app]
        st.markdown(f"#### Application: {app}")
        long_running_jobs = detect_long_running(app_df, lookback_days=14, threshold=1.5)
        st.dataframe(long_running_jobs)
        if not long_running_jobs.empty:
            st.warning(f"{len(long_running_jobs)} long-running jobs detected in {app} (last 2 weeks).")

    user_input = st.text_area("Ask about long-running jobs:", key="long_running_ai_input")
    if st.button("Ask AI (Long Runners)", key="ask_ai_long_running"):
        if user_input.strip():
            with st.spinner("🤖 BatchWatch is analyzing long runners..."):
                ai_query = f"{user_input} [Applications: {', '.join(selected_apps)}]"
                docs = vs.similarity_search(ai_query, k=10)
                context_docs = "\n---\n".join([d.page_content for d in docs])

                long_runners_context = []
                for app in selected_apps:
                    app_df = filtered_df[filtered_df['application'] == app]
                    long_jobs = detect_long_running(app_df, lookback_days=14, threshold=1.5)
                    if not long_jobs.empty:
                        long_runners_context.append(
                            f"Long running jobs for {app} (last 2 weeks):\n" +
                            long_jobs.to_string(index=False)
                        )
                long_runners_str = "\n\n".join(long_runners_context) if long_runners_context else "No long running jobs detected in the last 2 weeks."

                full_context = f"{context_docs}\n\n{long_runners_str}"

                prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "You are BatchWatch, an AI agent monitoring Autosys batch jobs. "
                     "Focus on job runtimes, failures, SLA patterns, and application-level analytics. "
                     "When asked about long-running jobs, use the provided list of jobs running longer than average over the last 2 weeks."),
                    ("system", "{context}"),
                    ("human", "{input}")
                ])
                agent_chain = prompt | llm

                response = agent_chain.invoke({"context": full_context, "input": ai_query})
                st.write("🤖 **BatchWatch:**", response.content)

# --- Tab 3: Failure Aggregates ---
with tab3:
    st.subheader("Failure Aggregates")
    for app in selected_apps:
        st.markdown(f"#### Application: {app}")
        app_df = filtered_df[filtered_df['application'] == app]
        daily, weekly, monthly = failure_aggregates(app_df)
        st.write("**Daily Failures:**")
        st.dataframe(daily)
        st.write("**Weekly Failures:**")
        st.dataframe(weekly)
        st.write("**Monthly Failures:**")
        st.dataframe(monthly)

    user_input = st.text_area("Ask about failures:", key="failures_ai_input")
    if st.button("Ask AI (Failures)", key="ask_ai_failures"):
        if user_input.strip():
            with st.spinner("🤖 BatchWatch is analyzing failures..."):
                ai_query = f"{user_input} [Applications: {', '.join(selected_apps)}]"
                docs = vs.similarity_search(ai_query, k=10)
                context_docs = "\n---\n".join([d.page_content for d in docs])

                agg_context = []
                for app in selected_apps:
                    app_df = filtered_df[filtered_df['application'] == app]
                    daily, weekly, monthly = failure_aggregates(app_df)
                    if not daily.empty:
                        agg_context.append(f"Daily Failures for {app}:\n" + daily.to_string(index=False))
                    if not weekly.empty:
                        agg_context.append(f"Weekly Failures for {app}:\n" + weekly.to_string(index=False))
                    if not monthly.empty:
                        agg_context.append(f"Monthly Failures for {app}:\n" + monthly.to_string(index=False))
                aggregates_context = "\n\n".join(agg_context)

                full_context = f"{context_docs}\n\n{aggregates_context}" if aggregates_context else context_docs

                prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "You are BatchWatch, an AI agent monitoring Autosys batch jobs. "
                     "Focus on job runtimes, failures, SLA patterns, and application-level analytics. "
                     "When asked about failures, use the daily/weekly/monthly aggregates provided to answer precisely."),
                    ("system", "{context}"),
                    ("human", "{input}")
                ])
                agent_chain = prompt | llm

                response = agent_chain.invoke({"context": full_context, "input": ai_query})
                st.write("🤖 **BatchWatch:**", response.content)

# --- Tab 4: AI Agent (Freeform) ---
with tab4:
    st.subheader("AI Remediation & Focus Visualization")
    st.caption("💬 The agent uses FAISS memory + Ollama embeddings to focus on relevant job segments.")
    st.markdown(f"**Current Applications:** `{', '.join(selected_apps)}`")

    user_input = st.text_area(
        "Ask BatchWatch something about jobs, failures, SLAs, or application-level stats across selected applications:",
        key="main_ai_input"
    )

    if st.button("Ask AI (General)", key="ask_ai_main"):
        if user_input.strip():
            with st.spinner("🤖 BatchWatch is analyzing job patterns..."):
                ai_query = f"{user_input} [Applications: {', '.join(selected_apps)}]"
                docs = vs.similarity_search(ai_query, k=10)
                if docs:
                    st.info("📌 **Data segments AI is focusing on:**")
                    for d in docs:
                        st.write("-", d.page_content)
                context = "\n---\n".join([d.page_content for d in docs])

                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are BatchWatch, an AI agent monitoring Autosys batch jobs. Focus on job runtimes, failures, SLA patterns, and application-level analytics. Use historical data and provided predictions to estimate future average runtimes (e.g., 2 weeks from now) for any job when asked."),
                    ("system", "{context}"),
                    ("human", "{input}")
                ])
                agent_chain = prompt | llm

                response = agent_chain.invoke({"context": context, "input": ai_query})
                st.write("🤖 **BatchWatch:**", response.content)

# --- Tab 5: Application-Level Analytics & Prediction ---
with tab5:
    st.subheader("📈 Application Analytics")
    agg = app_level_aggregates(filtered_df)
    st.write("**Application Summaries:**")
    st.dataframe(agg)

    for app in selected_apps:
        st.markdown(f"### Application: {app}")
        app_df = filtered_df[filtered_df['application'] == app]
        st.write(f"**Jobs for {app}:**")
        st.dataframe(app_df)

        # --- Future Runtime Prediction Table ---
        st.write("#### 🔮 Future Average Runtime Predictions (2 Weeks Ahead)")

        if not app_df.empty:
            future_preds = []
            for job_id in app_df['job_id'].unique():
                pred = predict_future_runtime(df, job_id, future_days=14)
                if pred:
                    future_preds.append({
                        "Job ID": pred['job_id'],
                        "Predicted Avg Runtime (min) in 2 Weeks": f"{pred['avg_runtime']:.2f}",
                        "Prediction Date": pred['future_date']
                    })
            if future_preds:
                st.dataframe(pd.DataFrame(future_preds))
            else:
                st.info("No jobs with enough data for future runtime prediction.")

            # Optionally: let user pick one to highlight
            selected_job = st.selectbox(
                f"Highlight prediction for a job in {app}", 
                app_df['job_id'].unique(), 
                key=f"future_job_{app}"
            )
            prediction = predict_future_runtime(df, selected_job, future_days=14)
            if prediction:
                st.info(
                    f"**Job ID:** `{prediction['job_id']}`\n\n"
                    f"**Predicted average runtime in 2 weeks:** `{prediction['avg_runtime']:.2f}` min\n"
                    f"**Date predicted for:** `{prediction['future_date']}`"
                )
            else:
                st.warning("Not enough data to predict runtime for this job.")
        else:
            st.info("No jobs found for prediction.")

        # Existing prediction for app-level completion
        st.write(f"**Expected Completion Time Prediction for {app}:**")
        app_prediction = predict_completion(df, app)
        if app_prediction:
            st.write(
                f"Last job started at: `{app_prediction['last_job_start']}`  \n"
                f"Average job runtime: `{app_prediction['avg_runtime']:.2f}` min  \n"
                f"Predicted completion: `{app_prediction['expected_completion']}`"
            )
        else:
            st.write("No jobs found for prediction.")

        st.write(f"**Runtime Trend for {app}**")
        if not app_df.empty:
            app_df_sorted = app_df.sort_values('run_date')
            st.line_chart(app_df_sorted.set_index('run_date')['runtime_min'])

# --- Tab 6: Alerts & Anomalies ---
with tab6:
    st.subheader("🚨 Alerts & Anomalies")

    alerts = []
    for app in selected_apps:
        app_df = filtered_df[filtered_df['application'] == app]
        long_runners = detect_long_running(app_df, threshold=2.0)
        if not long_runners.empty:
            alerts.append(f"Application {app}: {len(long_runners)} jobs running >2x average.")
        recent_failures = app_df[app_df['status'] == 'Failure']
        recent_failures = recent_failures[recent_failures['run_date'] > pd.Timestamp.now() - pd.Timedelta(days=1)]
        if not recent_failures.empty:
            alerts.append(f"Application {app}: {len(recent_failures)} failures in last 24h.")

    if alerts:
        for alert in alerts:
            st.error(alert)
    else:
        st.success("No critical alerts for selected applications.")

    # Show anomaly jobs/failures
    for app in selected_apps:
        app_df = filtered_df[filtered_df['application'] == app]
        st.write(f"**Anomalous Jobs for {app}:**")
        st.dataframe(detect_long_running(app_df, threshold=2.0))
        st.write(f"**Recent Failures (24h) for {app}:**")
        st.dataframe(app_df[(app_df['status'] == 'Failure') & (app_df['run_date'] > pd.Timestamp.now() - pd.Timedelta(days=1))])

    # --- AI-Powered Alert Summary ---
    if st.button("Generate AI Dashboard Summary", key="dashboard_ai_summary"):
        with st.spinner("🤖 BatchWatch is summarizing dashboard..."):
            summary_context = ""
            for app in selected_apps:
                app_df = filtered_df[filtered_df['application'] == app]
                long_runners = detect_long_running(app_df, threshold=2.0)
                recent_failures = app_df[(app_df['status'] == 'Failure') & (app_df['run_date'] > pd.Timestamp.now() - pd.Timedelta(days=1))]
                agg = app_level_aggregates(app_df)
                prediction = predict_completion(df, app)
                summary_context += (
                    f"\n---\nApplication: {app}\n"
                    f"Summary:\n{agg.to_string(index=False)}\n"
                    f"Long Runners (>2x average):\n{long_runners.to_string(index=False)}\n"
                    f"Recent Failures (24h):\n{recent_failures.to_string(index=False)}\n"
                    f"Prediction:\n{prediction}\n"
                )

            ai_prompt = (
                "Provide a concise summary for batch operations, highlighting recent failures, anomalous job runs (>2x average), and expected completion times for each application. "
                "If there are critical issues, mention them at the top."
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are BatchWatch, an AI agent monitoring Autosys batch jobs. Use the summaries, anomalies, and predictions to give a useful dashboard summary for operations."),
                ("system", "{context}"),
                ("human", "{input}")
            ])
            agent_chain = prompt | llm
            response = agent_chain.invoke({"context": summary_context, "input": ai_prompt})
            st.write("🤖 **BatchWatch Dashboard Summary:**", response.content)

# --- Real-Time Refresh ---
st.markdown("---")
REFRESH_INTERVAL_SEC = 60
auto_refresh = st.checkbox(f"Auto-refresh dashboard every {REFRESH_INTERVAL_SEC} seconds", value=False)
if st.button("Refresh Now"):
    st.experimental_rerun()
if auto_refresh:
    st.caption("Auto-refresh is ON. Dashboard will reload in the background.")
    time.sleep(REFRESH_INTERVAL_SEC)
    st.experimental_rerun()
else:
    st.caption("Auto-refresh is OFF. Click 'Refresh Now' to reload.")

# --- End of App ---
