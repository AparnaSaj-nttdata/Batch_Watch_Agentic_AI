import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

class AutosysDataGenerator:
    def __init__(self, num_jobs=30, num_days=90, num_apps=5, seed=42):
        self.num_jobs = num_jobs
        self.num_days = num_days
        self.num_apps = num_apps
        random.seed(seed)
        np.random.seed(seed)
        
        self.all_app_names = [
            "FinanceOps", "HRSuite", "SupplyChain", "CustomerPortal", "AnalyticsHub",
            "PaymentsCore", "InventoryPlus", "RiskEngine", "MobileBanking", "ERPCloud",
            "DataLake", "BillingPro", "CRM360", "ECommerce", "SupportDesk"
        ]
        
    def generate(self):
        """Generate realistic Autosys job execution data"""
        app_names = random.sample(self.all_app_names, self.num_apps)
        job_ids = [f"JOB_{i:03d}" for i in range(1, self.num_jobs + 1)]
        box_names = [f"BOX_{chr(65 + i % 5)}" for i in range(self.num_jobs)]
        schedule_types = ["Daily", "Weekly", "Monthly"]
        
        rows = []
        today = datetime.now()
        
        for job_idx, job_id in enumerate(job_ids):
            box_name = box_names[job_idx]
            schedule_type = random.choice(schedule_types)
            base_runtime = random.randint(20, 120)
            application = app_names[job_idx % self.num_apps]
            
            # Define dependency chain (for SLA impact analysis)
            dependency_level = job_idx % 3  # 0=root, 1=middle, 2=leaf
            
            for day_offset in range(self.num_days):
                # Skip if not scheduled (weekly/monthly)
                if schedule_type == "Weekly" and day_offset % 7 != 0:
                    continue
                if schedule_type == "Monthly" and day_offset % 30 != 0:
                    continue
                    
                run_date = today - timedelta(days=day_offset)
                scheduled_hour = random.randint(0, 23)
                scheduled_minute = random.choice([0, 15, 30, 45])
                
                # Simulate realistic delays
                delay = self._calculate_delay(schedule_type, day_offset)
                start_time_dt = run_date.replace(
                    hour=scheduled_hour, 
                    minute=scheduled_minute
                ) + timedelta(minutes=int(delay))
                
                # Calculate runtime with patterns
                runtime = self._calculate_runtime(
                    base_runtime, 
                    run_date, 
                    job_idx, 
                    day_offset
                )
                
                # Determine status with realistic failure patterns
                status = self._determine_status(job_idx, day_offset, runtime)
                
                # Calculate SLA buffer
                sla_time = base_runtime * 1.3  # 30% buffer
                sla_breach = runtime > sla_time
                
                end_time_dt = start_time_dt + timedelta(minutes=max(1, runtime))
                
                rows.append({
                    "job_id": job_id,
                    "box_name": box_name,
                    "application": application,
                    "run_date": run_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "schedule_type": schedule_type,
                    "scheduled_time": f"{scheduled_hour:02d}:{scheduled_minute:02d}",
                    "start_time": start_time_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": end_time_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": status,
                    "runtime_min": round(runtime, 2),
                    "delay_min": round(delay, 2),
                    "dependency_level": dependency_level,
                    "sla_threshold_min": round(sla_time, 2),
                    "sla_breach": sla_breach,
                    "cpu_usage_pct": round(random.uniform(20, 95), 2),
                    "memory_usage_mb": round(random.uniform(500, 4000), 2)
                })
        
        df = pd.DataFrame(rows)
        df.sort_values(["application", "job_id", "run_date"], 
                       ascending=[True, True, False], inplace=True)
        return df
    
    def _calculate_delay(self, schedule_type, day_offset):
        """Realistic delay patterns"""
        if day_offset < 7:  # Recent jobs
            return np.random.choice([0, 0, 0, 2, 5, 10, 30], p=[0.6, 0.15, 0.1, 0.05, 0.05, 0.03, 0.02])
        return np.random.choice([0, 0, 5], p=[0.8, 0.15, 0.05])
    
    def _calculate_runtime(self, base_runtime, run_date, job_idx, day_offset):
        """Runtime with realistic patterns"""
        runtime = base_runtime + np.random.normal(0, 10)
        
        # Weekend effect
        if run_date.weekday() >= 5:
            runtime *= 1.15
        
        # Month-end spike
        if run_date.day >= 28:
            runtime *= 1.25
        
        # Random long-running
        if random.random() < 0.05:
            runtime *= random.uniform(1.8, 3.0)
        
        # Growth trend over time (data volume increase)
        runtime *= (1 + (self.num_days - day_offset) * 0.001)
        
        return max(1, runtime)
    
    def _determine_status(self, job_idx, day_offset, runtime):
        """Status with realistic failure patterns"""
        failure_prob = 0.05
        
        # Higher failure for long runs
        if runtime > 200:
            failure_prob = 0.15
        
        # Specific jobs more prone to failure
        if job_idx % 7 == 0:
            failure_prob *= 1.5
        
        return "Failure" if random.random() < failure_prob else "Success"

if __name__ == "__main__":
    generator = AutosysDataGenerator(num_jobs=50, num_days=90, num_apps=5)
    df = generator.generate()
    df.to_csv("autosys_jobs.csv", index=False)
    print(f"Generated {len(df)} execution records")
    print(df.head(10))
