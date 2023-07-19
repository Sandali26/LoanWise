from flask import Flask, jsonify, request
from API.services.loadModel import LoanDefaultPrediction

app = Flask(__name__)
model = LoanDefaultPrediction.load_model()

@app.route("/predict", methods=["POST"])
def predict_loan_default():
    data = request.get_json()
    # Process the data and get the prediction using the loaded model
    # Assuming data is in the format required for the model
    prediction = LoanDefaultPrediction.predict(data, model)
    return jsonify({"prediction": prediction.item()})

if __name__ == "_main_":
    app.run(debug=True)