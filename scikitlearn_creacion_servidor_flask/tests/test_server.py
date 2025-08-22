import json
import os
import pytest

from server import app, model


def test_model_loaded():
    assert model is not None, "El modelo debe estar cargado en tests (ejecuta main.py para crearlo)"
    assert os.path.exists('./models/best_model.pkl')


def test_predict_get():
    client = app.test_client()
    r = client.get('/predict')
    assert r.status_code == 200
    data = r.get_json()
    assert 'prediccion' in data


def test_predict_post():
    client = app.test_client()
    payload = {'features': [7.594444821,7.479555538,1.616463184,1.53352356,0.796666503,0.635422587,0.362012237,0.315963835,2.277026653]}
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    data = r.get_json()
    assert 'prediccion' in data
