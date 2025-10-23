import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class JobPrediction:
    job_id: str
    application: str
    predicted_runtime_min: float
    confidence_interval: Tuple[float, float]
    prediction_date: datetime
    based_on_runs: int
    risk_level: str  # Low, Medium, High
    
@dataclass
class SLAAlert:
    application: str
    job_id: str
    alert_type: str  # runtime_breach, failure_spike, delay_pattern
    severity: str  # Critical, Warning, Info
    message: str
    timestamp: datetime
    recommended_action: str

class BatchAnalytics:
    def __init__(self, df: pd.DataFrame, lookback_days: int = 14):
        self.df = df.copy()
        self.lookback_days = lookback_days
        self._preprocess()
    
    def _preprocess(self):
        """Prepare data for analysis"""
        self.df['run_date'] = pd.to_datetime(self.df['run_date'])
        self.df['start_time'] = pd.to_datetime(self.df['start_time'])
        self.df['end_time'] = pd.to_datetime(self.df['end_time'])
        self.df['day_of_week'] = self.df['run_date'].dt.dayofweek
        self.df['is_weekend'] = self.df['day_of_week'] >= 5
        self.df['is_month_end'] = self.df['run_date'].dt.day >= 28
        
    # Use Case 1: Long Running Jobs
    def identify_long_running_jobs(self, threshold_multiplier: float = 1.5) -> pd.DataFrame:
        """
        Identify jobs running longer than historical average
        Returns detailed analysis with root cause indicators
        """
        cutoff_date = self.df['run_date'].max() - timedelta(days=self.lookback_days)
        recent_df = self.df[self.df['run_date'] >= cutoff_date].copy()
        
        # Calculate statistics by job
        job_stats = recent_df.groupby('job_id').agg({
            'runtime_min': ['mean', 'std', 'min', 'max', 'count'],
            'application': 'first',
            'box_name': 'first',
            'run_date': 'max'
        }).reset_index()
        
        job_stats.columns = ['job_id', 'avg_runtime', 'std_runtime', 'min_runtime', 
                             'max_runtime', 'run_count', 'application', 'box_name', 'last_run']
        
        # Get latest run for each job
        latest_runs = recent_df.sort_values('run_date').groupby('job_id').tail(1)
        
        # Merge and calculate anomalies
        analysis = latest_runs.merge(job_stats[['job_id', 'avg_runtime', 'std_runtime']], 
                                     on='job_id', how='left')
        
        analysis['runtime_ratio'] = analysis['runtime_min'] / analysis['avg_runtime']
        analysis['z_score'] = (analysis['runtime_min'] - analysis['avg_runtime']) / (analysis['std_runtime'] + 1)
        
        # Flag long runners
        long_runners = analysis[
            (analysis['runtime_ratio'] > threshold_multiplier) | 
            (analysis['z_score'] > 2)
        ].copy()
        
        # Add context
        long_runners['potential_cause'] = long_runners.apply(self._diagnose_long_run, axis=1)
        
        return long_runners[['job_id', 'application', 'box_name', 'run_date', 
                            'runtime_min', 'avg_runtime', 'runtime_ratio', 'z_score',
                            'potential_cause']].sort_values('runtime_ratio', ascending=False)
    
    def _diagnose_long_run(self, row) -> str:
        """Diagnose potential causes of long runtime"""
        causes = []
        if row.get('is_weekend', False):
            causes.append("Weekend execution")
        if row.get('is_month_end', False):
            causes.append("Month-end load")
        if row.get('cpu_usage_pct', 0) > 85:
            causes.append("High CPU usage")
        if row.get('delay_min', 0) > 10:
            causes.append(f"Start delayed by {row['delay_min']:.0f}min")
        
        return "; ".join(causes) if causes else "Normal variance"
    
    # Use Case 2: Failure Analysis
    def analyze_failures(self) -> Dict[str, pd.DataFrame]:
        """
        Comprehensive failure analysis: daily, weekly, monthly
        Returns dictionary with multiple aggregation levels
        """
        failures = self.df[self.df['status'] == 'Failure'].copy()
        
        if failures.empty:
            return {
                'daily': pd.DataFrame(),
                'weekly': pd.DataFrame(),
                'monthly': pd.DataFrame(),
                'by_job': pd.DataFrame(),
                'by_application': pd.DataFrame()
            }
        
        # Temporal aggregations
        failures['date'] = failures['run_date'].dt.date
        failures['week'] = failures['run_date'].dt.to_period('W')
        failures['month'] = failures['run_date'].dt.to_period('M')
        
        daily = failures.groupby(['application', 'date']).agg({
            'job_id': 'count',
            'runtime_min': 'mean'
        }).rename(columns={'job_id': 'failure_count', 'runtime_min': 'avg_runtime'}).reset_index()
        
        weekly = failures.groupby(['application', 'week']).agg({
            'job_id': ['count', 'nunique'],
        }).reset_index()
        weekly.columns = ['application', 'week', 'failure_count', 'unique_jobs']
        
        monthly = failures.groupby(['application', 'month']).agg({
            'job_id': ['count', 'nunique'],
        }).reset_index()
        monthly.columns = ['application', 'month', 'failure_count', 'unique_jobs']
        
        # Job-level failure rate
        by_job = failures.groupby(['application', 'job_id']).size().reset_index(name='failure_count')
        total_runs = self.df.groupby(['application', 'job_id']).size().reset_index(name='total_runs')
        by_job = by_job.merge(total_runs, on=['application', 'job_id'])
        by_job['failure_rate'] = (by_job['failure_count'] / by_job['total_runs'] * 100).round(2)
        by_job = by_job.sort_values('failure_rate', ascending=False)
        
        # Application-level summary
        by_app = failures.groupby('application').agg({
            'job_id': ['count', 'nunique'],
            'run_date': ['min', 'max']
        }).reset_index()
        by_app.columns = ['application', 'total_failures', 'unique_failed_jobs', 
                          'first_failure', 'last_failure']
        
        return {
            'daily': daily,
            'weekly': weekly,
            'monthly': monthly,
            'by_job': by_job,
            'by_application': by_app
        }
    
    # Use Case 3: Runtime Prediction
    def predict_future_runtime(self, job_id: str, future_date: Optional[datetime] = None,
                               days_ahead: int = 14) -> Optional[JobPrediction]:
        """
        Predict future runtime using time series analysis
        Accounts for trends, seasonality, and variance
        """
        if future_date is None:
            future_date = datetime.now() + timedelta(days=days_ahead)
        
        job_history = self.df[self.df['job_id'] == job_id].sort_values('run_date')
        
        if len(job_history) < 5:
            return None
        
        # Use recent data with more weight
        recent = job_history.tail(30)
        
        # Calculate trend
        recent['days_ago'] = (recent['run_date'].max() - recent['run_date']).dt.days
        if len(recent) > 1:
            trend = np.polyfit(recent['days_ago'], recent['runtime_min'], 1)[0]
        else:
            trend = 0
        
        # Base prediction
        base_runtime = recent['runtime_min'].mean()
        std_runtime = recent['runtime_min'].std()
        
        # Adjust for future date characteristics
        adjustment = 1.0
        if future_date.weekday() >= 5:  # Weekend
            adjustment *= 1.15
        if future_date.day >= 28:  # Month-end
            adjustment *= 1.25
        
        # Project trend
        days_forward = (future_date - recent['run_date'].max()).days
        predicted_runtime = (base_runtime + trend * days_forward) * adjustment
        
        # Confidence interval (95%)
        confidence_margin = 1.96 * std_runtime
        ci_lower = max(0, predicted_runtime - confidence_margin)
        ci_upper = predicted_runtime + confidence_margin
        
        # Risk assessment
        risk_level = "Low"
        sla_threshold = job_history['sla_threshold_min'].iloc[0] if 'sla_threshold_min' in job_history.columns else base_runtime * 1.3
        
        if predicted_runtime > sla_threshold:
            risk_level = "High"
        elif predicted_runtime > sla_threshold * 0.9:
            risk_level = "Medium"
        
        return JobPrediction(
            job_id=job_id,
            application=job_history['application'].iloc[0],
            predicted_runtime_min=round(predicted_runtime, 2),
            confidence_interval=(round(ci_lower, 2), round(ci_upper, 2)),
            prediction_date=future_date,
            based_on_runs=len(recent),
            risk_level=risk_level
        )
    
    def predict_application_completion(self, application: str, 
                                       start_time: Optional[datetime] = None) -> Dict:
        """
        Predict when all jobs for an application will complete
        Critical for SLA monitoring
        """
        app_jobs = self.df[self.df['application'] == application]
        
        if app_jobs.empty:
            return None
        
        if start_time is None:
            start_time = datetime.now()
        
        # Get predictions for all jobs
        predictions = []
        for job_id in app_jobs['job_id'].unique():
            pred = self.predict_future_runtime(job_id, start_time)
            if pred:
                predictions.append({
                    'job_id': job_id,
                    'predicted_runtime': pred.predicted_runtime_min,
                    'risk_level': pred.risk_level
                })
        
        if not predictions:
            return None
        
        # Calculate critical path (assuming sequential execution for simplicity)
        total_predicted_runtime = sum(p['predicted_runtime'] for p in predictions)
        expected_completion = start_time + timedelta(minutes=total_predicted_runtime)
        
        high_risk_jobs = [p['job_id'] for p in predictions if p['risk_level'] == 'High']
        
        return {
            'application': application,
            'start_time': start_time,
            'expected_completion': expected_completion,
            'total_runtime_min': round(total_predicted_runtime, 2),
            'job_count': len(predictions),
            'high_risk_jobs': high_risk_jobs,
            'risk_summary': f"{len(high_risk_jobs)} of {len(predictions)} jobs at high risk"
        }
    
    # SLA Monitoring & Alerts
    def generate_alerts(self, severity_threshold: str = "Warning") -> List[SLAAlert]:
        """
        Generate actionable alerts for batch operations team
        """
        alerts = []
        current_time = datetime.now()
        
        # Check for recent long runners
        long_runners = self.identify_long_running_jobs(threshold_multiplier=1.8)
        for _, job in long_runners.iterrows():
            if (current_time - job['run_date']).days <= 1:
                alerts.append(SLAAlert(
                    application=job['application'],
                    job_id=job['job_id'],
                    alert_type="runtime_breach",
                    severity="Critical" if job['runtime_ratio'] > 2.5 else "Warning",
                    message=f"Job running {job['runtime_ratio']:.1f}x longer than average ({job['runtime_min']:.0f} min vs {job['avg_runtime']:.0f} min avg)",
                    timestamp=current_time,
                    recommended_action=f"Investigate: {job['potential_cause']}"
                ))
        
        # Check for failure spikes
        failure_analysis = self.analyze_failures()
        recent_failures = failure_analysis['daily']
        
        if not recent_failures.empty:
            recent_failures = recent_failures[
                pd.to_datetime(recent_failures['date']) >= current_time - timedelta(days=2)
            ]
            
            for _, row in recent_failures.iterrows():
                if row['failure_count'] >= 3:
                    alerts.append(SLAAlert(
                        application=row['application'],
                        job_id="MULTIPLE",
                        alert_type="failure_spike",
                        severity="Critical",
                        message=f"{row['failure_count']} failures detected on {row['date']}",
                        timestamp=current_time,
                        recommended_action="Review failed jobs and check system resources"
                    ))
        
        # Check for upcoming SLA risks
        for app in self.df['application'].unique():
            completion_pred = self.predict_application_completion(app)
            if completion_pred and completion_pred['high_risk_jobs']:
                alerts.append(SLAAlert(
                    application=app,
                    job_id="APPLICATION_LEVEL",
                    alert_type="sla_risk",
                    severity="Warning",
                    message=f"{len(completion_pred['high_risk_jobs'])} jobs predicted to breach SLA",
                    timestamp=current_time,
                    recommended_action=f"Monitor jobs: {', '.join(completion_pred['high_risk_jobs'][:3])}"
                ))
        
        # Filter by severity
        severity_order = {"Info": 0, "Warning": 1, "Critical": 2}
        threshold_level = severity_order.get(severity_threshold, 1)
        
        return [a for a in alerts if severity_order.get(a.severity, 0) >= threshold_level]
