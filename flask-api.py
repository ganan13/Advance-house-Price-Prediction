from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


model = joblib.load('random_forest_regressor_model.joblib')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = [data['quality'], data['livingArea'], data['isNeighborhood'],data['basementArea'],data['garageArea'],data['basementFinishedArea'],data['firstFloorArea'],data['yearBuilt'],data['lotSize'],data['yearReconstructed'],data['unfinishedArea'],data['overAllCondition'],data['porchArea'],data['isCentralAir'],data['countFireplaces']]  

        # Make a prediction using the loaded model
        predicted_price = model.predict([features])[0]

        # Create a JSON response with the predicted price
        response = {'predicted_price': predicted_price}

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
# import joblib

# # Load your trained RandomForestRegressor model
# model = joblib.load('random_forest_regressor_model.joblib')

# # Sample input data (customize this based on your model's input format)
# sample_input = [5, 12,1,10,12,10,1,1999,100,1996,100,3,100,1,3]  # Replace with your actual sample input data

# # Make a prediction using the loaded model
# predicted_price = model.predict([sample_input])[0]

# print(f'Predicted Price: {predicted_price}')
