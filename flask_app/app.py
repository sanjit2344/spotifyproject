from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and encoders
model = joblib.load('popularity_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form

    # Extract feature values
    artists = label_encoders['artists'].transform([form_data['artists']])[0]
    album_name = label_encoders['album_name'].transform([form_data['album_name']])[0]
    track_name = label_encoders['track_name'].transform([form_data['track_name']])[0]
    duration_ms = int(form_data['duration_ms'])
    explicit = int(form_data['explicit'])
    danceability = float(form_data['danceability'])
    energy = float(form_data['energy'])
    key = int(form_data['key'])
    loudness = float(form_data['loudness'])
    mode = int(form_data['mode'])
    speechiness = float(form_data['speechiness'])
    acousticness = float(form_data['acousticness'])
    instrumentalness = float(form_data['instrumentalness'])
    liveness = float(form_data['liveness'])
    valence = float(form_data['valence'])
    tempo = float(form_data['tempo'])
    time_signature = int(form_data['time_signature'])
    track_genre = label_encoders['track_genre'].transform([form_data['track_genre']])[0]

    # Create input array
    input_features = np.array([[artists, album_name, track_name, duration_ms, explicit, 
                                danceability, energy, key, loudness, mode, speechiness, 
                                acousticness, instrumentalness, liveness, valence, tempo, 
                                time_signature, track_genre]])
    
    # Predict popularity
    predicted_popularity = model.predict(input_features)[0]

    return render_template('index.html', prediction_text=f'Predicted Popularity: {predicted_popularity:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
