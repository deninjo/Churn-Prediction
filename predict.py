import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = f'model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


app = Flask('churn')



@app.route('/predict', methods=['POST']) # post bc we are sending some info about the cust
def predict():  
    customer = request.get_json() # parses json customer to dict

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'Churn probability': float(y_pred),
        'Churn': bool(churn)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=9696)
