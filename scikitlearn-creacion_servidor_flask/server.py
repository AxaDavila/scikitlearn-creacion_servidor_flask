import joblib
import numpy as np

from flask import Flask
from flask import jsonify

app = Flask(__name__)

#POSTMAN PARA PRUEBAS, definicion de metodos y path desde API 
@app.route('/predict', methods=['GET'])
def predict():
    X_test = np.array([7.594444821,7.479555538,1.616463184,1.53352356,0.796666503,0.635422587,0.362012237,0.315963835,2.277026653])
    prediction = model.predict(X_test.reshape(10,-1)) #llamada del numero de listas del 1, al -1, para que sea un array de una sola o muchas filas
    return jsonify({'prediccion' : list(prediction)})

if __name__ == "__main__":
    model = joblib.load('./models/best_model.pkl')
    app.run(port=7879) #definicion de la ruta donde se expondra la api

