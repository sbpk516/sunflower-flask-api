# app.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

def generate_data(filename, num_samples=150):
    """Generate synthetic sunflower data and save to a CSV file."""
    np.random.seed(42)  # For reproducibility

    # Generate random sunlight hours between 1 and 15
    sunlight_hours = np.random.uniform(1, 15, num_samples)

    # Generate heights: height = 5 * sunlight_hours + 30 + small random noise
    heights = 5 * sunlight_hours + 30 + np.random.normal(0, 5, num_samples)

    # Create and save DataFrame
    data = pd.DataFrame({
        'SunlightHours': sunlight_hours,
        'HeightCm': heights
    })
    data.to_csv(filename, index=False)
    print(f"Data generated and saved to {filename}")

def load_data(filename):
    """Load data from a CSV file."""
    df = pd.read_csv(filename)
    X = df[['SunlightHours']].values  # 2D feature array
    Y = df['HeightCm'].values         # 1D target array
    return X, Y

def train_linear_regression(X_train, Y_train):
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

def evaluate_model(model, X_test, Y_test):
    """Evaluate the model with R-squared score."""
    score = model.score(X_test, Y_test)
    print(f"Model R² score: {score:.2f}")
    return score

def predict(model, X_new):
    """Predict new values using the trained model."""
    return model.predict(X_new)

def save_model(model, filename):
    """Save the trained model to a file."""
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    """Load a trained model from a file."""
    return joblib.load(filename)

def main():
    # Step 1: Generate data
    data_file = 'sunflower_data.csv'
    # generate_data(data_file, num_samples=150)

    # Step 2: Load data
    X, Y = load_data(data_file)

    # Step 3: Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Step 4: Train model
    model = train_linear_regression(X_train, Y_train)

    # Step 5: Evaluate model
    evaluate_model(model, X_test, Y_test)

    # Step 6: Predict new values
    X_new = np.array([[10], [12], [14]])  # New sunlight values
    Y_pred = predict(model, X_new)

    print("\nPredictions for new sunlight hours:")
    for hours, height in zip(X_new.flatten(), Y_pred):
        print(f"For {hours} hours of sunlight, predicted height: {height:.2f} cm")

    # Step 7: Save model
    model_file = 'sunflower_model.pkl'
    save_model(model, model_file)

    # Step 8: Load model and predict again (simulating production use)
    loaded_model = load_model(model_file)
    Y_loaded_pred = predict(loaded_model, X_new)

    print("\nPredictions from loaded model:")
    for hours, height in zip(X_new.flatten(), Y_loaded_pred):
        print(f"For {hours} hours of sunlight, predicted height: {height:.2f} cm")

if __name__ == "__main__":
    main()
