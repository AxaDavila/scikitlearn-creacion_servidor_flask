# profesional_scikitlearn_platzi
Repositorio de código usado durante el Curso Profesional de Scikit-Learn para Platzi.

En este repositorio podrás encontrar 16 ramas relacionadas con el curso, en orden:

1. preparacion_datos_pca
2. implementacion_algoritmo_pca
3. kernel_y_pca
4. implementacion_lasso_ridge 
5. preparacion_regresion_robusta
6. implementacion_regresion_robusta
7. preparacion_datos_bagging
8. implementacion_bagging
9. implementacion_boosting
10. implementacion_kmeans
11. implementacion_meanshift
12. implementacion_crossval
13. implementacion_randomizedSearchCV
14. revision_arquitectura
15. creacion_exportacion_modelo
16. creacion_servidor_flask

Resumen

¿Cómo publicar un modelo de Machine Learning utilizando Flask?
Al finalizar el desarrollo de un modelo de Machine Learning, el siguiente paso es hacerlo accesible para otros usuarios. Esto se logra a través de la creación de una API que permita interactuar con el modelo desde la web. En este artículo, aprenderemos cómo desplegar un modelo utilizando Flask, un servidor Python ligero, instalándolo y configurándolo en un entorno local.

¿Qué es Flask y cómo instalarlo?
Flask es un micro framework de Python que permite crear servidores web de manera rápida y sencilla. Para instalar Flask, es fundamental asegurarse de estar dentro del entorno de trabajo adecuado para evitar instalaciones globales. Utiliza el siguiente comando para instalarlo:

pip install Flask

¿Qué estructura debe tener el proyecto?
El proyecto debe tener una estructura organizada para facilitar el desarrollo y despliegue del modelo. Aquí un ejemplo de cómo podría estar configurado:

Entorno: Mantener un entorno virtual aislado para las dependencias del proyecto.
Carpetas:
Entrada: Datos de entrada al modelo.
Modelos: Contiene el mejor modelo encontrado.
Utilidades y ejecución: Scripts principales para la ejecución del proyecto.
Además, se necesita un archivo para la configuración del servidor, denominado server.py, que contendrá toda la lógica para ejecutar la API.

¿Cómo configurar el servidor Flask?
Primero, importa las librerías necesarias en el archivo server.py. Aquí un ejemplo de cómo empezar:

import joblib
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)
Después, carga el modelo utilizando la librería joblib:

model = joblib.load('models/best_model.pkl')
¿Cómo definir rutas y métodos en Flask?
Para que el servidor pueda responder a las solicitudes, define una ruta con el método que desees utilizar. Para un ejemplo sencillo con el método GET, la configuración sería:

@app.route('/predict', methods=['GET'])
def predict():
    sample_data = np.array([[/* datos de prueba sin Country, Rank y Score */]])
    prediction = model.predict(sample_data)
    return jsonify({'prediction': prediction.tolist()})
¿Cómo ejecutar el servidor y probar las predicciones?
Ejecuta el servidor especificando el puerto que prefieras. Es recomendable utilizar puertos altos:

if __name__ == '__main__':
    app.run(port=8080)
Luego de ejecutar el servidor, dirígete a tu navegador web e ingresa la URL local con el puerto especificado y la ruta definida (/predict) para obtener un archivo JSON con las predicciones.

¿Qué hacer con las predicciones obtenidas?
Las predicciones obtenidas en formato JSON pueden ser tratadas en diversas aplicaciones, ya sean basadas en JavaScript (front-end web) o Android (aplicaciones móviles). Así, puedes convertir tu modelo de inteligencia artificial en una solución aplicable a diferentes plataformas.

Con estos pasos, se consigue una arquitectura modular y extensible para llevar modelos de Machine Learning a producción. Continúa explorando el vasto mundo del desarrollo de APIs y cómo integrar modelos de inteligencia artificial en soluciones completas. ¡El éxito está a solo un paso de distancia!

## Uso rápido (reproducible)

1. Crear y activar el entorno virtual (desde la raíz del repo):

```bash
python3 -m venv scikitlearn-creacion_servidor_flask
source scikitlearn-creacion_servidor_flask/bin/activate
```

2. Instalar dependencias (usando `requirements.txt` con versiones pinneadas):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Entrenar y generar el modelo (genera `models/best_model.pkl`):

```bash
python main.py
```

4. Ejecutar el servidor Flask:

```bash
python server.py
# por defecto escucha en http://127.0.0.1:7879
```

5. Probar la API:

```bash
# GET de ejemplo
curl http://127.0.0.1:7879/predict

# POST con features (JSON)
curl -X POST -H "Content-Type: application/json" \
    -d '{"features":[7.59,7.48,1.61,1.53,0.79,0.63,0.36,0.31,2.27]}' \
    http://127.0.0.1:7879/predict
```

6. Ejecutar tests (pytest):

```bash
pip install pytest
pytest -q
```

Notas:
- Asegúrate de usar el intérprete/venv correcto (la reproducibilidad depende de la combinación Python+paquetes+SO).
- Si necesitas aislar por completo el entorno, considera usar `conda` o un `Dockerfile`.
