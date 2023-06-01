import onnx
from onnx_tf.backend import prepare

#Convert Tensorflow
ONNX_Model = onnx.load("./ONNXModels/Efficientnet-sim.onnx")
tf_rep = prepare(ONNX_Model)
tf_rep.export_graph("./TFModels/Efficientnet.onnx")

#Convert TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("./TFModels/Efficientnet.onnx")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [
tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
converter.allow_custom_ops=True
Tflite_quanit_model = converter.convert()

tflite_models_dir = pathlib.Path("./TFLiteModels/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_fp16_file = tflite_models_dir/"Efficientnet.tflite"
tflite_model_fp16_file.write_bytes(Tflite_quanit_model)