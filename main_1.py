from flask import Flask, request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np

# Initalise the Flask app
app = Flask(__name__)

cols = ['LengthVer', 'LengthDia', 'LengthCro', 'Height', 'Width']


@app.route('/')
def home():
	return render_template("webpage.html",pred="")


@app.route('/predict', methods=['POST'])
def predict():

	int_features = [x for x in request.form.values()]
	final = np.array(int_features)
	data_unseen = pd.DataFrame([final], columns=cols)
	# prediction = predict_model(model, data=data_unseen, round = 0)
	# load the model from disk
	try:
		loaded_model = pickle.load(open('pre.pkl', 'rb'))
		y_pred = loaded_model.predict(data_unseen)
		predictions = [value for value in y_pred]
		print(predictions)
	except Exception as e:
		print(e)
	
	# print(predictions)
	return render_template('webpage.html', pred="Expected value will be: {:.2f} gms".format(float(predictions[0])))


if __name__ == '__main__':
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)
