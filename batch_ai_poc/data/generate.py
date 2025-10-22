import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Parameters
num_jobs = 30
num_days = 90

# Application Names (choose as many as you want)
all_app_names = [
    "FinanceOps", "HRSuite", "SupplyChain", "CustomerPortal", "AnalyticsHub",
    "PaymentsCore", "InventoryPlus", "RiskEngine", "MobileBanking", "ERPCloud",
    "DataLake", "BillingPro", "CRM360", "ECommerce", "SupportDesk"
]
num_apps = 5  # Pick any number <= len(all_app_names)
app_names = random.sample(all_app_names, num_apps)

job_ids = [f"JOB_{i:02d}" for i in range(1, num_jobs+1)]
box_names = [f"BOX_{chr(65 + i%5)}" for i in range(num_jobs)]  # BOX_A to BOX_E
schedule_types = ["Daily", "Weekly", "Monthly"]

# For reproducibility
random.seed(42)
np.random.seed(42)

rows = []
today = datetime.now()

for job_idx, job_id in enumerate(job_ids):
    box_name = box_names[job_idx]
    schedule_type = random.choice(schedule_types)
    base_runtime = random.randint(20, 120)  # base runtime in minutes

    # Assign application (evenly distributed)
    application = app_names[job_idx % num_apps]

    for day_offset in range(num_days):
        run_date = today - timedelta(days=day_offset)
        scheduled_hour = random.randint(0, 23)
        scheduled_minute = random.choice([0, 15, 30, 45])
        scheduled_time = f"{scheduled_hour:02d}:{scheduled_minute:02d}"

        # Simulate start time (may be delayed)
        delay = np.random.choice([0, 0, 0, 5, 10])  # mostly on time, sometimes delayed
        start_time_dt = run_date.replace(hour=scheduled_hour, minute=scheduled_minute) + timedelta(minutes=int(delay))
        start_time = start_time_dt.strftime("%H:%M")

        # Simulate runtime
        runtime = base_runtime + np.random.normal(0, 10)
        # Occasionally, inject a long-running scenario
        if random.random() < 0.03:
            runtime *= random.uniform(1.5, 2.0)
        # Occasionally, inject failures
        status = "Success"
        if random.random() < 0.07:
            status = "Failure"
            runtime += np.random.choice([-20, 30])

        # End time
        end_time_dt = start_time_dt + timedelta(minutes=max(1, runtime))
        end_time = end_time_dt.strftime("%H:%M")

        rows.append({
            "job_id": job_id,
            "box_name": box_name,
            "application": application,
            "run_date": run_date.strftime("%Y-%m-%d"),
            "schedule_type": schedule_type,
            "scheduled_time": scheduled_time,
            "start_time": start_time,
            "end_time": end_time,
            "status": status,
            "runtime_min": round(runtime, 2)
        })

# Create DataFrame
df = pd.DataFrame(rows)
df.sort_values(["job_id", "run_date"], ascending=[True, False], inplace=True)
df.to_csv("autosys_jobs.csv", index=False)

print(f"Generated {len(df)} rows for {num_jobs} jobs over {num_days} days and {num_apps} applications: {app_names}")
print(df.head())
