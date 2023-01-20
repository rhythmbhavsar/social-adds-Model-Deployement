from flask import Flask, request, jsonify
import pickle as pkl
import numpy as np

print("Running ML API")

model = pkl.load(open('social_add.pkl', 'rb'))
sc = pkl.load(open('sc.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return "API home page"

@app.route('/predict', methods=['POST'])
def predict():
    age = request.form.get('age')
    estimeted_sallery = request.form.get('estimeted_sallery')

    input_query = np.array([[age, estimeted_sallery]])
    
    result = model.predict(sc.transform(input_query))
    # result = {'age':age, 'estimeted_sallery':estimeted_sallery}
    return jsonify({'Outcome': str(result)}) 

if __name__ == "__main__":
    app.run()
