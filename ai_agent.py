"""
AI Agent for BatchWatch using custom LLM API and TF-IDF embeddings
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from analytics import BatchAnalytics
from services.llm_service import llm_chat


class SimpleVectorStore:
    """
    Simple vector store using TF-IDF
    No external downloads required - works in any environment
    """
    
    def __init__(self):
        """Initialize vector store with TF-IDF"""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=1,
            max_df=0.95
        )
        self.documents = []
        self.embeddings = None
    
    def add_documents(self, documents: List[str]):
        """
        Add documents to the vector store
        
        Args:
            documents: List of document strings
        """
        self.documents.extend(documents)
        
        # Generate TF-IDF embeddings
        if len(self.documents) > 0:
            try:
                self.embeddings = self.vectorizer.fit_transform(self.documents)
                print(f"✓ Vector store built with {len(self.documents)} documents")
            except Exception as e:
                print(f"Error building vector store: {e}")
                self.embeddings = None
    
    def similarity_search(self, query: str, k: int = 10) -> List[str]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if not self.documents or self.embeddings is None:
            return []
        
        try:
            # Transform query
            query_embedding = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top k indices
            top_k = min(k, len(self.documents))
            top_k_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Filter out very low similarity scores
            results = []
            for idx in top_k_indices:
                if similarities[idx] > 0.01:  # Minimum similarity threshold
                    results.append(self.documents[idx])
            
            return results
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return self.documents[:k]  # Fallback to first k documents
    
    def save(self, path: str):
        """Save vector store to disk"""
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'vectorizer': self.vectorizer,
                    'embeddings': self.embeddings
                }, f)
            print(f"✓ Vector store saved to {path}")
        except Exception as e:
            print(f"Error saving vector store: {e}")
    
    def load(self, path: str):
        """Load vector store from disk"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.vectorizer = data['vectorizer']
                self.embeddings = data['embeddings']
            print(f"✓ Vector store loaded from {path}")
        except Exception as e:
            print(f"Error loading vector store: {e}")


class BatchWatchAgent:
    """
    AI Agent for batch job monitoring using your custom LLM API
    """
    
    def __init__(self, analytics: 'BatchAnalytics', max_tokens: int = 4000):
        """
        Initialize BatchWatch AI Agent
        
        Args:
            analytics: BatchAnalytics instance with job data
            max_tokens: Maximum tokens for LLM responses
        """
        self.analytics = analytics
        self.max_tokens = max_tokens
        self.vector_store = SimpleVectorStore()
        
        # Try to load existing knowledge base
        if os.path.exists('vector_store.pkl'):
            try:
                self.vector_store.load('vector_store.pkl')
                print("✓ Loaded existing knowledge base")
            except:
                print("Building new knowledge base...")
                self._build_knowledge_base()
        else:
            print("Building knowledge base for the first time...")
            self._build_knowledge_base()
    
    def _build_knowledge_base(self):
        """Build comprehensive knowledge base from analytics"""
        print("Building knowledge base...")
        documents = []
        
        # 1. Job execution history (sample recent ones)
        recent_jobs = self.analytics.df.tail(300)
        for _, job in recent_jobs.iterrows():
            doc = f"""Job Execution: {job['job_id']} ({job['application']})
Date: {job['run_date']} | Runtime: {job['runtime_min']:.1f}min | Status: {job['status']}
Schedule: {job['schedule_type']} | Delay: {job.get('delay_min', 0):.1f}min
SLA Breach: {'Yes' if job.get('sla_breach', False) else 'No'}"""
            documents.append(doc)
        
        # 2. Long running job analysis
        try:
            long_runners = self.analytics.identify_long_running_jobs()
            if not long_runners.empty:
                for _, job in long_runners.head(20).iterrows():
                    doc = f"""⚠️ Long Running Job: {job['job_id']} ({job['application']})
Current: {job['runtime_min']:.1f}min | Average: {job['avg_runtime']:.1f}min | Ratio: {job['runtime_ratio']:.2f}x
Z-Score: {job['z_score']:.2f} | Cause: {job['potential_cause']}
Risk Level: HIGH - May impact SLA"""
                    documents.append(doc)
        except Exception as e:
            print(f"Warning: Could not analyze long runners: {e}")
        
        # 3. Failure patterns
        try:
            failure_analysis = self.analytics.analyze_failures()
            
            for _, failure in failure_analysis['by_job'].head(30).iterrows():
                risk = 'HIGH' if failure['failure_rate'] > 10 else 'MEDIUM' if failure['failure_rate'] > 5 else 'LOW'
                doc = f"""❌ Failure Pattern: {failure['job_id']} ({failure['application']})
Failure Rate: {failure['failure_rate']:.1f}% | Failures: {failure['failure_count']}/{failure['total_runs']}
Risk: {risk} | Action: {'Immediate investigation' if risk == 'HIGH' else 'Monitor'}"""
                documents.append(doc)
            
            # Application failures
            for _, app_fail in failure_analysis['by_application'].iterrows():
                doc = f"""Application Failures: {app_fail['application']}
Total: {app_fail['total_failures']} | Unique Jobs: {app_fail['unique_failed_jobs']}
Period: {app_fail['first_failure']} to {app_fail['last_failure']}"""
                documents.append(doc)
        except Exception as e:
            print(f"Warning: Could not analyze failures: {e}")
        
        # 4. Predictions
        try:
            for job_id in list(self.analytics.df['job_id'].unique())[:50]:
                pred = self.analytics.predict_future_runtime(job_id, days_ahead=14)
                if pred:
                    doc = f"""🔮 Prediction: {pred.job_id} ({pred.application})
2-Week Forecast: {pred.predicted_runtime_min:.1f}min
Range: {pred.confidence_interval[0]:.1f}-{pred.confidence_interval[1]:.1f}min
Risk: {pred.risk_level} | Based on: {pred.based_on_runs} runs"""
                    documents.append(doc)
        except Exception as e:
            print(f"Warning: Could not generate predictions: {e}")
        
        # 5. Application summaries
        for app in self.analytics.df['application'].unique():
            try:
                app_df = self.analytics.df[self.analytics.df['application'] == app]
                success_rate = (app_df['status'] == 'Success').sum() / len(app_df) * 100
                
                doc = f"""📊 Application Summary: {app}
Jobs: {app_df['job_id'].nunique()} | Executions: {len(app_df)}
Avg Runtime: {app_df['runtime_min'].mean():.1f}min ± {app_df['runtime_min'].std():.1f}
Success Rate: {success_rate:.1f}% | Failures: {100-success_rate:.1f}%
Avg Delay: {app_df.get('delay_min', pd.Series([0])).mean():.1f}min"""
                documents.append(doc)
            except Exception as e:
                print(f"Warning: Could not summarize {app}: {e}")
        
        # 6. Active alerts
        try:
            alerts = self.analytics.generate_alerts()
            for alert in alerts[:20]:
                doc = f"""🚨 Alert: {alert.application} - {alert.job_id}
Severity: {alert.severity} | Type: {alert.alert_type}
Message: {alert.message}
Action: {alert.recommended_action}"""
                documents.append(doc)
        except Exception as e:
            print(f"Warning: Could not generate alerts: {e}")
        
        print(f"Adding {len(documents)} documents to vector store...")
        self.vector_store.add_documents(documents)
        
        # Save for future use
        try:
            self.vector_store.save('vector_store.pkl')
        except Exception as e:
            print(f"Warning: Could not save vector store: {e}")
        
        print("✓ Knowledge base built successfully!")
    
    def _format_prompt(self, question: str, context: str, system_message: str) -> str:
        """
        Format prompt for your LLM API
        
        Args:
            question: User question
            context: Retrieved context
            system_message: System instructions
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""{system_message}

==================== CONTEXT INFORMATION ====================

{context}

==================== END CONTEXT ====================

QUESTION: {question}

Please answer the question based ONLY on the context information provided above. Be specific with job IDs, applications, and metrics.

ANSWER:"""
        return prompt
    
    def query(self, question: str, application_filter: Optional[List[str]] = None) -> Dict:
        """
        Query the agent with context-aware retrieval
        
        Args:
            question: User question
            application_filter: Optional list of applications to focus on
            
        Returns:
            Dict with answer, relevant context, and confidence
        """
        # Enhance query with application filter
        enhanced_query = question
        if application_filter:
            enhanced_query += f" Applications: {', '.join(application_filter)}"
        
        # Retrieve relevant context
        relevant_docs = self.vector_store.similarity_search(enhanced_query, k=15)
        
        # Filter by application if specified
        if application_filter and relevant_docs:
            filtered_docs = [doc for doc in relevant_docs if any(app in doc for app in application_filter)]
            if len(filtered_docs) >= 5:
                relevant_docs = filtered_docs
        
        context = "\n\n---\n\n".join(relevant_docs[:10])
        
        system_message = """You are BatchWatch AI, an expert in Autosys batch job monitoring and SLA management.

Your capabilities:
- Analyze job runtime patterns and predict future execution times
- Identify jobs at risk of SLA breach with specific metrics
- Diagnose root causes of failures and delays
- Provide actionable recommendations for batch operations teams

When answering:
1. Be specific with job IDs, applications, and metrics (numbers, percentages)
2. Explain WHY a job might be at risk (not just that it is)
3. Provide concrete next steps and recommendations
4. Use the historical data and predictions provided in the context
5. If you're uncertain or lack data, say so clearly
6. Prioritize critical issues (SLA breaches, failures) over minor issues"""
        
        # Format the full prompt
        full_prompt = self._format_prompt(question, context, system_message)
        
        try:
            answer = llm_chat(full_prompt, max_tokens=self.max_tokens)
            
            confidence = "high" if len(relevant_docs) >= 8 else "medium" if len(relevant_docs) >= 4 else "low"
            
            return {
                "answer": answer,
                "relevant_context": relevant_docs[:5],
                "confidence": confidence,
                "total_context_docs": len(relevant_docs)
            }
        except Exception as e:
            error_msg = f"Error querying LLM: {str(e)}"
            print(error_msg)
            return {
                "answer": f"{error_msg}\n\nRelevant context found:\n{context[:500]}...",
                "relevant_context": relevant_docs[:5],
                "confidence": "error",
                "total_context_docs": len(relevant_docs)
            }
    
    def get_dashboard_summary(self, applications: List[str]) -> str:
        """
        Generate executive summary for dashboard
        
        Args:
            applications: List of applications to summarize
            
        Returns:
            str: Executive summary
        """
        try:
            # Gather key metrics
            alerts = [a for a in self.analytics.generate_alerts() if a.application in applications]
            long_runners = self.analytics.identify_long_running_jobs()
            long_runners = long_runners[long_runners['application'].isin(applications)]
            failure_analysis = self.analytics.analyze_failures()
            app_failures = failure_analysis['by_application']
            app_failures = app_failures[app_failures['application'].isin(applications)]
            
            critical_count = sum(1 for a in alerts if a.severity == "Critical")
            warning_count = sum(1 for a in alerts if a.severity == "Warning")
            
            # Build summary context
            summary_parts = [
                f"Applications Monitored: {', '.join(applications)}",
                f"Active Alerts: {len(alerts)} total ({critical_count} critical, {warning_count} warnings)",
                f"Long Running Jobs (>1.5x average): {len(long_runners)}",
                f"Recent Failures: {app_failures['total_failures'].sum() if not app_failures.empty else 0}",
                "\nTop Priority Alerts:"
            ]
            
            for alert in alerts[:5]:
                summary_parts.append(f"- [{alert.severity}] {alert.application}: {alert.message}")
            
            summary_context = "\n".join(summary_parts)
            
            # Create question for dashboard summary
            question = f"""Based on the following monitoring data:

{summary_context}

Provide a concise executive summary for batch operations leadership. Include:
1. Overall health status (Good/Concerning/Critical)
2. Top 3 critical issues requiring immediate attention
3. Key trends or patterns observed
4. Recommended priority actions for the next 24 hours

Keep it brief but actionable (max 250 words)."""
            
            # Query using your llm_chat
            response = self.query(question, application_filter=applications)
            return response['answer']
            
        except Exception as e:
            error_msg = f"Error generating summary: {str(e)}"
            print(error_msg)
            return error_msg
    
    def refresh_knowledge_base(self):
        """Force rebuild of knowledge base with latest data"""
        print("Refreshing knowledge base...")
        self._build_knowledge_base()
    
    def save_knowledge_base(self, path: str = "vector_store.pkl"):
        """Save knowledge base to disk"""
        self.vector_store.save(path)
    
    def load_knowledge_base(self, path: str = "vector_store.pkl"):
        """Load knowledge base from disk"""
        self.vector_store.load(path)
