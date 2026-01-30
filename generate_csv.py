import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create 48 hours of dummy data
start_date = datetime.now() - timedelta(days=2)
data = {
    'timestamp': [(start_date + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S') for i in range(48)],
    'aqi': np.random.randint(40, 150, size=48),
    'city': ['London'] * 48
}

df = pd.DataFrame(data)

# Ensure the 'data' directory exists
import os
os.makedirs('data', exist_ok=True)

df.to_csv('data/historical_aqi.csv', index=False)
print("✅ Created data/historical_aqi.csv with 48 rows of sample data.")