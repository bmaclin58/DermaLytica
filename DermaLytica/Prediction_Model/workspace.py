import tensorflow as tf
modelpath = r"C:\Users\maclinbc\Django Python\DermaLytica\DermaLytica\Prediction_Model\AI_Models\KERAS_model.tflite"
tf.lite.experimental.Analyzer.analyze(model_path=modelpath)
