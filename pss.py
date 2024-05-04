import time
import joblib
import requests
from fastapi import FastAPI

# Initialize FastAPI app
app = FastAPI()

# Load the model
model = joblib.load('model.pkl')

# Function to collect CO level data from the endpoint
def collect_co_level():
    response = requests.get("http://127.0.0.1:8000/mq7_data")
    co_levels = response.json()
    avg_co_level = sum(co_levels) / len(co_levels)
    return avg_co_level

# Function to collect PM2.5 level data from the endpoint
def collect_pm_level():
    response = requests.get("http://127.0.0.1:8000/pm_data")
    pm_levels = response.json()
    avg_pm_level = sum(pm_levels) / len(pm_levels)
    return avg_pm_level

# Function to collect temperature and humidity data from the endpoint
def collect_temperature_humidity():
    response = requests.get("http://127.0.0.1:8000/dht_data")
    dht_data = response.json()
    temperatures_c = [entry["temperature_c"] for entry in dht_data]
    humidity_values = [entry["humidity"] for entry in dht_data]
    avg_temperature = sum(temperatures_c) / len(temperatures_c)
    avg_humidity = sum(humidity_values) / len(humidity_values)
    return avg_temperature, avg_humidity

# Route to get prediction data
@app.get("/prediction")
async def get_prediction():
    avg_co_level = collect_co_level()
    avg_pm_level = collect_pm_level()
    avg_temperature, avg_humidity = collect_temperature_humidity()
    
    # Hardcoded latitude and longitude
    latitude = 12.3456
    longitude = 78.9123

    # Prepare the data
    sensor_data = [avg_co_level, avg_pm_level, avg_temperature, avg_humidity, latitude, longitude]

    # Preprocess the data
    new_data = [sensor_data]

    # Make predictions
    prediction = model.predict(new_data)

    # Convert numpy.int64 objects to Python integers
    prediction = int(prediction[0])

    # Create key-value pairs for response
    prediction_data = {
        'co': float(avg_co_level),
        'pm_lvl': float(avg_pm_level),
        'temperature': float(avg_temperature),
        'humidity': float(avg_humidity),
        'latitude': latitude,
        'longitude': longitude,
        'predicted_pollution': prediction
    }

    return prediction_data
