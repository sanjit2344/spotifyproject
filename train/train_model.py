import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
file_path = 'dataset.csv'  # Update this to the correct path of your dataset
df = pd.read_csv(file_path)

# Drop missing values and duplicates
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Encode categorical variables
label_encoders = {}
for column in ['artists', 'album_name', 'track_name', 'track_genre']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Select relevant features and target variable
features = ['artists', 'album_name', 'track_name', 'duration_ms', 'explicit', 
            'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
            'time_signature', 'track_genre']
target = 'popularity'

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'popularity_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("Model and encoders saved successfully.")
