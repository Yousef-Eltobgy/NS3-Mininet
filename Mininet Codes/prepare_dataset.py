import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def prepare_network_anomaly_dataset(file_path='network_dataset_labeled.csv'):
    """
    Loads and preprocesses the Network Anomaly Dataset.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found. Please ensure it's in the current directory.")
        print("You can download it from: https://www.kaggle.com/datasets/kaiser14/network-anomaly-dataset")
        return None, None, None, None, None

    # Drop irrelevant columns (e.g., IDs, timestamps if not used for time-series analysis)
    # Assuming 'timestamp' and 'Routers' might not be directly used as features for a simple ANN/SOANN
    # Adjust this based on actual column names and your specific needs
    if 'timestamp' in df.columns:
        df = df.drop('timestamp', axis=1)
    if 'Routers' in df.columns:
        df = df.drop('Routers', axis=1)
    if 'Planned route' in df.columns:
        df = df.drop('Planned route', axis=1)
    if 'Network measure' in df.columns:
        df = df.drop('Network measure', axis=1)

    # Identify target variable and features
    # Assuming 'Anomaly' is the target variable for classification
    if 'anomaly' not in df.columns:
        print("Error: 'Anomaly' column not found. Please check the dataset.")
        return None, None, None, None, None

    X = df.drop('anomaly', axis=1)
    y = df['anomaly']

    # Handle categorical features (if any) - using LabelEncoder for simplicity
    # You might need OneHotEncoder for more complex categorical features
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # Encode target variable (if not already numerical)
    if y.dtype == 'object':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print("Dataset prepared successfully.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test, X.shape[1]

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, input_dim = prepare_network_anomaly_dataset()
    if X_train is not None:
        print("Data preparation complete. Ready for model training.")

