# BatchWatch AI - Intelligent Batch Job Monitoring System

An AI-powered monitoring and prediction system for Autosys batch jobs using a custom LLM API. It provides real-time analytics, long-running job detection, failure trends, future runtime prediction, SLA risk alerts, and an AI assistant for natural-language insights.

---

## 🌟 Features

- **Real-time Monitoring**: Track job execution status, runtime, and failures across multiple applications.
- **Long Running Job Detection**: Automatically identify jobs running significantly longer than historical averages.
- **Failure Pattern Analysis**: Aggregate and analyze failures on a daily, weekly, and monthly basis.
- **Predictive Analytics**: Forecast future job runtimes and identify potential SLA breach risks.
- **AI-Powered Insights**: Use natural language to ask questions about job performance and get context-aware answers.
- **Role-Based Dashboards**: Separate views for Batch Operations Teams (cross-application) and Application Owners (single-application).
- **Executive Summaries**: Auto-generate dashboard summaries for leadership with a single click.

---

## 📋 Requirements

- Python 3.9+
- Access to your custom NTT AI API with valid credentials.
- A network environment that can successfully reach your API endpoint.

---

## 📦 Project Structure

```
batch-watch-ai/
├── app.py                      # Main Streamlit application (UI and dashboard)
├── analytics.py                # Analytics engine (long runners, failures, predictions, alerts)
├── ai_agent.py                 # AI agent (RAG via TF-IDF + your custom LLM)
├── data_generator.py           # Synthetic data generator for creating a sample dataset
├── requirements.txt            # Project dependencies
├── services/
│   ├── __init__.py
│   ├── LLM.py                  # Your MyCustomLLM wrapper class
│   ├── auth.py                 # Your authenticate() implementation
│   └── llm_service.py          # Your llm_chat(prompt, max_tokens) function
├── .env                        # NTT_ID and any other secrets (do not commit to git)
├── data/
│   └── autosys_jobs.csv        # Generated data file
└── vector_store.pkl            # Cached knowledge base for RAG (auto-generated)
```

---

## 🚀 Installation

1.  **Create and activate a virtual environment:**

    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

---

## 🔐 Configuration

1.  **Create `.env` in the project root directory:**

    ```python
    # .env
    NTT_ID = "your-actual-ntt-id-here"
    NTT_SECRET = "your-actual-secret-key"
    ```


2.  **Verify your `services/llm_service.py` is correct (as you provided):**

    ```python
    from services.LLM import MyCustomLLM
    from services.auth import authenticate
    from dotenv import load_dotenv

    def llm_chat(prompt, max_tokens=4000):
        token = authenticate()
        llm = MyCustomLLM(
            api_url='https://api.ntth.ai/v1/chat',
            token=token,
            # model_id="cc5bab32-9ccf-472b-9d76-91fc2ec5b047", # GPT-4o-mini
            model_id="6c26a584-a988-4fed-92ea-f6501429fab9", # GPT-4o
            # model_id="a4022a51-2f02-4ba7-8a31-d33c7456b58e", # Gemini 2.5 Flash
            ID=NTT_ID,
            max_tokens=max_tokens
        )
        return llm._call(prompt)
    ```

4.  Ensure your `services/LLM.py` file contains your working `MyCustomLLM` class that correctly calls the API.

---

## 🧪 Generate Sample Data

Run the data generator script from your terminal. This will create the `data/autosys_jobs.csv` file.

```bash
python data_generator.py
```

---

## ▶️ Run the Application

Launch the Streamlit app from your terminal.

```bash
streamlit run app.py
```

The application will open in your default web browser, typically at `http://localhost:8501`.

---

## 🧭 Using the App

-   **Role Selection (Sidebar)**:
    -   `Batch Operations Team`: Select multiple applications for a cross-application monitoring view.
    -   `Application Owner`: Select a single application for a deep-dive, focused view.
-   **Max Response Tokens (Sidebar)**: Adjust the slider to control the maximum length of answers from the AI Assistant.

### Tabs Overview

-   **📊 Overview**: High-level application KPIs, runtime trends, and job distribution charts.
-   **⚠️ Long Running Jobs**: Detects jobs running >1.5x their historical average (threshold is adjustable). Provides root cause indicators and an AI analysis button.
-   **❌ Failure Analysis**: Aggregates failures by day/week/month, shows job-level failure rates, and offers AI-powered pattern analysis.
-   **🔮 Predictions**: Forecasts future job runtimes with confidence intervals and risk levels. Estimates application completion times.
-   **🚨 Alerts & SLA**: Displays active alerts by severity and tracks SLA compliance rates over time.
-   **🤖 AI Assistant**: Ask questions in natural language. Features executive summary generation, quick questions, and shows the context used for answers.

**Tip**: If data seems stale, use the **Refresh Data** button in the app's footer to clear caches and reload everything.

---

## 🛠️ Troubleshooting

-   **Error: API request failed - 400 Bad Request**:
    -   Verify your `authenticate()` function returns the correct token format (e.g., does it need a `Bearer ` prefix?).
    -   Confirm the `modelId` and other payload fields in `services/LLM.py` match your API's specification exactly.

-   **Error querying LLM: 'token' or other auth issues**:
    -   This often happens due to caching. Click the **Refresh Data** button in the app to clear the cache.
    -   Run the authentication test script below to ensure your `auth.py` and `ntt_secrets.py` are correct.

-   **SSL errors with model downloads**:
    -   This should not occur. The app now uses TF-IDF for embeddings, which requires no external model downloads from HuggingFace.

-   **Cache-related or stale data**:
    -   Use the **Refresh Data** button in the app.
    -   As a last resort, manually delete the `vector_store.pkl` file to force a full knowledge base rebuild on the next app run.

---

## 🧪 Quick Tests
-   **Validate Analytics Engine (in a Python shell)**:

    ```python
    from analytics import BatchAnalytics
    import pandas as pd
    df = pd.read_csv('data/autosys_jobs.csv', parse_dates=['run_date','start_time','end_time'])
    analytics = BatchAnalytics(df)
    print('Long runners found:', len(analytics.identify_long_running_jobs()))
    print('Failures by job:', len(analytics.analyze_failures()['by_job']))
    print('Sample prediction:', analytics.predict_future_runtime(df['job_id'].iloc[0], days_ahead=14))
    ```

---

## 🔒 Security

-   **Never commit secrets to version control.** Add the following files to your `.gitignore`:
    -   `.env`
    -   `services/auth.py`
    -   `vector_store.pkl`
-   Tokens are retrieved on each call, not stored long-term in the application state.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                   │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐  │
│  │ Overview │  Long    │ Failures │Predictions│   AI     │  │
│  │          │ Runners  │          │           │Assistant │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
           ┌─────────────┴─────────────┐
           │                           │
┌──────────▼─────────┐        ┌────────▼────────┐
│  BatchAnalytics     │        │ BatchWatchAgent │
│  (analytics.py)     │        │  (ai_agent.py)  │
│ - long runners      │        │ - TF-IDF RAG    │
│ - failures          │◄───────┤ - prompt build  │
│ - predictions       │        │ - llm_chat call │
│ - alerts            │        └────────┬────────┘
└─────────────────────┘                 │
                                ┌───────▼────────┐
                                │  llm_chat()    │
                                │ (llm_service)  │
                                └───────┬────────┘
                                ┌───────▼────────┐
                                │  MyCustomLLM   │
                                │   (LLM.py)     │
                                └───────┬────────┘
                                ┌───────▼────────┐
                                │ authenticate() │
                                │   (auth.py)    │
                                └───────┬────────┘
                                ┌───────▼────────┐
                                │  NTT AI API    │
                                └────────────────┘
```

---

## 🔮 Future Enhancements

-   Real-time streaming updates and notifications (e.g., to Slack/Teams).
-   Deeper anomaly detection using machine learning models.
-   Dependency-aware critical path analysis for more accurate SLA predictions.
-   Cost and resource optimization insights based on job performance.
-   User authentication and role-based access control within the app.

---

## ✅ Quick Start Summary

1.  Configure `.env` and `services/auth.py`.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Generate sample data: `python data_generator.py`
4.  Launch the app: `streamlit run app.py`

Enjoy using BatchWatch AI!
