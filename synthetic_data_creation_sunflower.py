import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 150 sunlight hours between 1 and 15
sunlight_hours = np.random.uniform(1, 15, 150)

# Assume: height = 5 * sunlight_hours + 30 + random noise
heights = 5 * sunlight_hours + 30 + np.random.normal(0, 5, 150)

# Create a DataFrame
data = pd.DataFrame({
    'SunlightHours': sunlight_hours,
    'HeightCm': heights
})

# Save to CSV
data.to_csv('sunflower_data.csv', index=False)

print("Data created and saved to 'sunflower_data.csv'")

