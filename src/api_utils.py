import requests
import pandas as pd
import time

def get_weather_data(lat, lon):
    """Fetch weather data from Open-Meteo."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    response = requests.get(url).json()
    return pd.DataFrame(response['hourly'])

def get_aqicn_data(city_token, city_name):
    """Fetch real-time AQI from AQICN."""
    url = f"https://api.waqi.info/feed/{city_name}/?token={city_token}"
    response = requests.get(url).json()
    return response['data']['aqi']

def get_openaq_data(city):
    """Fetch historical/recent air quality from OpenAQ."""
    url = f"https://api.openaq.org/v2/measurements?city={city}&limit=100"
    # Note: OpenAQ often requires headers for API keys in newer versions
    response = requests.get(url).json()
    return pd.DataFrame(response['results'])

def get_karachi_historical_aqi():
    """Fetch real historical AQI data for Karachi from OpenAQ."""
    # Karachi coordinates roughly: lat 24.86, lon 67.00
    url = "https://api.openaq.org/v2/measurements"
    params = {
        "city": "Karachi",
        "parameter": "pm25", # Fine particulate matter
        "limit": 1000,
        "date_from": "2024-01-01T00:00:00Z"
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json()['results']
        data = []
        for r in results:
            data.append({
                'timestamp': r['date']['utc'],
                'aqi': r['value'],
                'city': 'Karachi'
            })
        return pd.DataFrame(data)
    else:
        print(f"Error fetching from OpenAQ: {response.status_code}")
        return pd.DataFrame()