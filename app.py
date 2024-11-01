from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load and train the model
try:
    # Load the dataset
    data = pd.read_csv('heart.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: heart.csv file not found.")
    exit()

# Separate features and target
X = data.drop('target', axis=1)  # Ensure 'target' matches your dataset
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Route for the main page
@app.route('/')
def index():
    # Format fields for display
    formatted_fields = [field.replace('_', ' ').title() for field in X.columns]
    return render_template('index.html', fields=formatted_fields)

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        features = [float(request.form[field]) for field in X.columns]
        # Scale and predict
        features = scaler.transform([features])
        prediction = model.predict(features)[0]
        output = "Heart Disease Present" if prediction == 1 else "No Heart Disease"
    except ValueError:
        output = "Invalid input. Please enter numeric values for all fields."
    except Exception as e:
        output = f"Error: {str(e)}"
    
    # Prepare the formatted fields again for the prediction result page
    formatted_fields = [field.replace('_', ' ').title() for field in X.columns]
    return render_template('index.html', prediction=output, fields=formatted_fields)

if __name__ == "__main__":
    app.run(debug=True)
