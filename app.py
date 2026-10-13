from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io

app = Flask(__name__)

# --- METADATA & HUMAN-READABLE LABELS ---
VAR_METADATA = {
    'temperature_c': {'name': 'Air Temperature', 'unit': '°C', 'icon': 'fa-temperature-high', 'color': 'text-amber-400'},
    'humidity_pct': {'name': 'Relative Humidity', 'unit': '%', 'icon': 'fa-droplet', 'color': 'text-cyan-400'},
    'pressure_hpa': {'name': 'Atmospheric Pressure', 'unit': 'hPa', 'icon': 'fa-gauge-high', 'color': 'text-indigo-400'},
    'wind_speed_kmh': {'name': 'Wind Speed', 'unit': 'km/h', 'icon': 'fa-wind', 'color': 'text-blue-400'},
    'rainfall_mm': {'name': 'Precipitation', 'unit': 'mm', 'icon': 'fa-cloud-rain', 'color': 'text-teal-400'}
}

def generate_capstone_dataset():
    """Generates 180-day synthetic weather dataset (Jan 1 - Jun 29, 2025)."""
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=180, freq="D")
    
    temp = 28 + 4 * np.sin(np.linspace(0, 3*np.pi, 180)) + np.random.normal(0, 2, 180)
    humidity = 89.08 - 0.74 * temp + np.random.normal(0, 5, 180)
    pressure = 1012 + np.random.normal(0, 3, 180)
    wind = np.abs(np.random.normal(7.7, 2.1, 180))
    rain = np.where(np.random.rand(180) > 0.75, np.random.exponential(5, 180), 0)

    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'temperature_c': np.round(temp, 1),
        'humidity_pct': np.round(humidity, 1),
        'pressure_hpa': np.round(pressure, 1),
        'wind_speed_kmh': np.round(wind, 1),
        'rainfall_mm': np.round(rain, 1)
    })

    # Add missing values for cleaning pipeline demonstration
    for col in VAR_METADATA.keys():
        idx = np.random.choice(180, 1)
        df.loc[idx, col] = np.nan

    return df

def run_python_pipeline(df):
    """Executes Python Data Pipeline (Cleaning, Statistics, Regression)."""
    logs = [f"Loaded dataset containing {len(df)} records."]
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    numeric_cols = list(VAR_METADATA.keys())
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Deduplication
    dups = df.duplicated(subset=['date']).sum()
    if dups > 0:
        df = df.drop_duplicates(subset=['date'])
        logs.append(f"Deduplicated {dups} timestamp entries.")

    # Median Imputation
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logs.append(f"Imputed missing {VAR_METADATA[col]['name']} using median ({round(median_val, 1)} {VAR_METADATA[col]['unit']}).")

    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%b')
    df['temp_7d_rolling'] = df['temperature_c'].rolling(7, min_periods=1).mean().round(2)
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')

    # Statistics Calculation
    stats = []
    for col in numeric_cols:
        meta = VAR_METADATA[col]
        stats.append({
            'key': col,
            'label': meta['name'],
            'unit': meta['unit'],
            'icon': meta['icon'],
            'color': meta['color'],
            'mean': float(np.round(df[col].mean(), 2)),
            'median': float(np.round(df[col].median(), 2)),
            'std': float(np.round(df[col].std(), 2)),
            'min': float(np.round(df[col].min(), 2)),
            'max': float(np.round(df[col].max(), 2))
        })

    # Linear Regression (Temp vs Humidity)
    X = df[['temperature_c']].values
    y = df['humidity_pct'].values
    model = LinearRegression().fit(X, y)
    
    slope = float(np.round(model.coef_[0], 3))
    intercept = float(np.round(model.intercept_, 3))
    r2 = float(np.round(model.score(X, y), 3))

    # Monthly Aggregation
    monthly = df.groupby(['month', 'month_name'], as_index=False).agg(
        mean_temp=('temperature_c', 'mean'),
        mean_humidity=('humidity_pct', 'mean'),
        total_rain=('rainfall_mm', 'sum')
    ).round(2).to_dict(orient='records')

    records = df[['date_str', 'temperature_c', 'humidity_pct', 'pressure_hpa', 'wind_speed_kmh', 'rainfall_mm', 'temp_7d_rolling']].to_dict(orient='records')

    return {
        'logs': logs,
        'stats': stats,
        'regression': {'slope': slope, 'intercept': intercept, 'r2': r2},
        'monthly': monthly,
        'records': records
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF-8")))
            return jsonify(run_python_pipeline(df))
    
    return jsonify(run_python_pipeline(generate_capstone_dataset()))

@app.route('/api/predict', methods=['GET'])
def predict():
    try:
        temp = float(request.args.get('temp', 28.0))
        # Equation: Humidity = 89.08 - 0.74 * Temp
        predicted_humidity = np.round(89.08 - 0.74 * temp, 1)
        
        if predicted_humidity > 70:
            comfort = "High Humidity / Sticky"
            badge = "bg-amber-500/20 text-amber-300"
        elif 45 <= predicted_humidity <= 70:
            comfort = "Optimal Comfort"
            badge = "bg-emerald-500/20 text-emerald-300"
        else:
            comfort = "Dry Conditions"
            badge = "bg-cyan-500/20 text-cyan-300"

        return jsonify({
            'temperature': temp,
            'predicted_humidity': float(predicted_humidity),
            'comfort_status': comfort,
            'comfort_badge': badge
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)