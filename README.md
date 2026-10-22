# TP MLP en 20 Newsgroups

Trabajo de Redes Neuronales basado en el notebook de Regresión Softmax de la cátedra, extendido a un MLP. El desarrollo, las explicaciones, los análisis y las figuras se concentran en [RESUELTO_MLP.ipynb](RESUELTO_MLP.ipynb).

**Resultados del 19 y 21/10/2026.** El notebook conserva salidas para leer sin entrenar de nuevo. Se conservan los 36 entrenamientos originales y se agregan seis: cuatro comparaciones MLP de LR/batch y dos entrenamientos del baseline Softmax. Total: **42 entrenamientos TNG y 618 épocas**, incluidos seis trials podados, trece comparaciones MLP, sobreajuste, selección y ajustes finales.

Resultado final: **accuracy test 0,678173 y F1 macro 0,668141** sobre 7532 documentos. La búsqueda seleccionó una oculta de 128, ReLU, Adam, TF-IDF de hasta 10000 características, stopwords y lematización. Validación fijó dos épocas para el ajuste final con todo el train oficial. No se reajustaron HP después de evaluar test. El bonus incluye Softmax histórico (accuracy 0,652947; F1 macro 0,643932) y un lineal nuevo con condiciones comunes (accuracy 0,663967; F1 macro 0,654344; 23 épocas elegidas con validación). El notebook distingue la comparación histórica de la comparación con condiciones comunes y explica sus límites.

## Instalación y ejecución

Entorno de referencia: Windows x64, Python 3.12.5, PyTorch 2.14.0 CPU. Se usa CPU con dos hilos. Reservar al menos 3 GiB libres de disco (recomendados 5), y al menos 2 GiB de RAM disponibles antes de iniciar. Las matrices densas y otras aplicaciones pueden requerir más RAM. El código comprueba recursos antes de descargar o densificar.

Desde una carpeta que contenga los cinco archivos de entrega, crear un entorno y abrir Jupyter allí:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name python3 --display-name "Python 3 (TP MLP)"
.\.venv\Scripts\python.exe -m jupyterlab .
```

Elegir el kernel de ese entorno, abrir `RESUELTO_MLP.ipynb` y ejecutar **Restart Kernel and Run All Cells**. La notebook no instala paquetes automáticamente. La primera ejecución descarga recursos NLTK y 20 Newsgroups, por lo que necesita Internet; los datos quedan en `datos/`. Cada ejecución crea otro identificador y entrena pesos nuevos: volver a ejecutar todo repite el presupuesto completo, no restaura resultados.

`requirements.txt` declara dependencias directas con las versiones de referencia. Incluye el índice oficial de PyTorch CPU además de PyPI. `requirements-lock.txt` conserva las 178 versiones del entorno preparado, incluidas las transitivas; puede usarse con `pip install -r requirements-lock.txt` para reproducir esa combinación en Windows/Python 3.12. No es un lock universal para otros sistemas (incluye paquetes específicos de Windows). La compatibilidad de una instalación nueva en otra computadora todavía no se ha comprobado.

## Experimentos

- Búsqueda Optuna: 24 trials de hasta 8 épocas; siete HP de Softmax más activación, 1–3 ocultas, anchos por capa, Adam/RMSprop/SGD y sus momentos/parámetros. Seis trials iniciales completos garantizan cobertura; el resto admite poda.
- Trece comparaciones controladas de 25 épocas, con representación y split fijos: ancho, profundidad, activación, optimizador, LR y batch.
- Caso de sobreajuste de 40 épocas; segundo caso previsto de 60 si no aparece el patrón requerido.
- MLP finalista de 30 épocas: elegir duración por mínima pérdida de validación, entrenar de nuevo con todo train oficial y evaluar test una vez.
- Bonus Softmax: una sola capa lineal con las mismas matrices/HP de entrenamiento del MLP; selección de duración en 30 épocas de validación, ajuste nuevo con todo train y una evaluación final. La representación se eligió por la búsqueda MLP, sin búsqueda propia del lineal. No se cambian decisiones a partir de test.

El split es 80/20 estratificado sobre train oficial (semilla 42); TF-IDF se ajusta solo con train de cada fase. El test no interviene en la selección. Las celdas explican inicialización, métricas, tiempos y límites. Una semilla y un presupuesto finito no permiten afirmar optimalidad o significancia estadística.

## Archivos generados y seguimiento

Se crean `datos/` y `salidas/`, excluidos de Git. Cada ejecución guarda JSON por entrenamiento, tabla CSV, estudio Optuna SQLite, MLflow SQLite, eventos TensorBoard y el modelo final en `salidas/mlp_20newsgroups/<id_ejecucion>/`. Los modelos y logs se generan al ejecutar; no forman parte de los cinco archivos de entrega.

Para abrir las interfaces después o durante una ejecución, desde esta misma carpeta y con el entorno activo, reemplazar el identificador por el impreso en el notebook:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri "sqlite:///salidas/mlp_20newsgroups/<id_ejecucion>/mlflow.db" --port 5001
.\.venv\Scripts\python.exe -m tensorboard.main --logdir "salidas/mlp_20newsgroups/<id_ejecucion>/tensorboard" --port 6007
```

Usar puertos disponibles. Las interfaces no tienen que estar abiertas para registrar. Las figuras y tablas están guardadas dentro del notebook para leerlas sin repetir entrenamientos.

## Fuente

Base: [01_Softmax_Regression_20Newsgroups.ipynb, commit original](https://github.com/cselmo/TP-Regresion-Softmax-TNG/blob/9267a2ff7839aad6fdceda2185baedabca9a4697/01_Softmax_Regression_20Newsgroups.ipynb). Se conservan los datos y las herramientas del notebook de la cátedra; la arquitectura MLP, el protocolo de búsqueda, las comparaciones y los criterios de evaluación se explican en el desarrollo.

Las salidas guardadas combinan la ejecución MLP del 19/10 y la ampliación del 21/10; no se repitió el test MLP para agregar el bonus. La versión base tuvo una ejecución completa desde una carpeta independiente; los seis entrenamientos añadidos y su presentación se ejecutaron en otro kernel nuevo. No se repitió un Run All completo de los 42 entrenamientos ni una instalación nueva en otra computadora para esta ampliación.
