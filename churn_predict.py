import pickle


model_file = f'model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

customer = {'gender': 'female',
 'seniorcitizen': 0,
 'partner': 'yes',
 'dependents': 'yes',
 'phoneservice': 'yes',
 'multiplelines': 'yes',
 'internetservice': 'fiber_optic',
 'onlinesecurity': 'yes',
 'onlinebackup': 'no',
 'deviceprotection': 'yes',
 'techsupport': 'no',
 'streamingtv': 'yes',
 'streamingmovies': 'yes',
 'contract': 'month-to-month',
 'paperlessbilling': 'yes',
 'paymentmethod': 'electronic_check',
 'tenure': 7,
 'monthlycharges': 104.2,
 'totalcharges': 1743.5}

X = dv.transform([customer])
var = model.predict_proba(X)[0, 1]


print('input: ', customer)
print('churn probability', var)
