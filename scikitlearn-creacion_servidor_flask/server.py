import joblib
import numpy as np
import warnings

from flask import Flask, jsonify, request

app = Flask(__name__)

# Cargar modelo al importar el módulo
try:
    model = joblib.load('./models/best_model.pkl')
except FileNotFoundError:
    model = None
    app.logger.error("models/best_model.pkl no encontrado. Ejecuta el entrenamiento primero.")
except Exception as e:
    model = None
    app.logger.error(f"Error cargando modelo: {e}")


@app.route('/')
def index():
    return jsonify({
        'message': 'API de predicción',
        'model_loaded': model is not None,
        'endpoints': ['/predict']
    })


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no cargado'}), 500

    # permitir POST con JSON {"features": [...]}
    if request.method == 'POST':
        data = request.get_json(silent=True)
        if not data or 'features' not in data:
            return jsonify({'error': 'Se requiere JSON con la clave "features"'}), 400
        try:
            X = np.array(data['features'], dtype=float)
        except Exception as e:
            return jsonify({'error': f'Invalid features: {e}'}), 400
    else:
        # GET: ejemplo por defecto
        X = np.array([7.594444821,7.479555538,1.616463184,1.53352356,0.796666503,0.635422587,0.362012237,0.315963835,2.277026653])

    # Validar dimensión si el modelo expone n_features_in_
    if hasattr(model, 'n_features_in_'):
        expected = int(model.n_features_in_)
        if X.size != expected:
            return jsonify({'error': f'Número de features inválido: se esperaba {expected}, recibido {X.size}'}), 400

    try:
        # Asegurar forma (1, n_features)
        X_reshaped = X.reshape(1, -1)
        prediction = model.predict(X_reshaped)
        return jsonify({'prediccion': list(prediction)})
    except Exception as e:
        app.logger.exception('Error durante predict')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(port=7879)

