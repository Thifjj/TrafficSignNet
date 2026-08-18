import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize

# =========================
# CONFIGURAÇÕES
# =========================

CALIB_PATH = "data/gtsrb/split/val"
MODEL_PATH = "models/keras/TrafficSignNet_FP32_TF212.h5"
OUTPUT_PATH = "models/keras/TrafficSignNet_INT8.h5"

IMAGE_SIZE = (32, 32)
BATCH_SIZE = 32
CALIB_BATCHES = 32


# =========================
# DATASET DE CALIBRAÇÃO
# =========================

calib_dataset = tf.keras.utils.image_dataset_from_directory(
    CALIB_PATH,
    labels=None,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

# IMPORTANTE:
# deixe esta normalização somente se o modelo foi treinado
# recebendo imagens entre 0 e 1.
calib_dataset = calib_dataset.map(
    lambda x: tf.cast(x, tf.float32) / 255.0
)

# aproximadamente 1024 imagens
calib_dataset = calib_dataset.take(CALIB_BATCHES)


# =========================
# CARREGAR MODELO FP32
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

print("Modelo carregado:")
model.summary()


# =========================
# QUANTIZAÇÃO
# =========================

quantizer = vitis_quantize.VitisQuantizer(model)

quantized_model = quantizer.quantize_model(
    calib_dataset=calib_dataset
)


# =========================
# SALVAR
# =========================

quantized_model.save(OUTPUT_PATH)

print()
print("Quantização concluída!")
print("Modelo salvo em:", OUTPUT_PATH)