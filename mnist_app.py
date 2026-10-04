"""
mnist_app.py
============
MNIST projesinin Controller (Ana yonetim) katmani.

Bu dosya arayuz cizmez veya model egitmez. Sadece gerekli parcalari
(veri seti, model, isleyici) hafizaya alir ve visualization.py
icerisindeki arayuzu baslatir.
"""

import json
import os
from tensorflow.keras.datasets import mnist

from mnist_core import ImagePreprocessor, MNISTClassifier
from visualization import CustomerDemoApp  # Yeni View modulumuzu cagiriyoruz

BEST_METHOD_PARAMS_PATH = "best_method_params.json"
BEST_PARAMS_PATH = "best_params.json"
MODEL_PATH = "mnist_model.keras"


def load_preprocessing_params():
    """Preprocessing ayarlarini json'dan okur."""
    if os.path.exists(BEST_METHOD_PARAMS_PATH):
        path = BEST_METHOD_PARAMS_PATH
    elif os.path.exists(BEST_PARAMS_PATH):
        path = BEST_PARAMS_PATH
    else:
        raise FileNotFoundError(
            "Parametre dosyasi bulunamadi. Once 'python parameter_test.py' "
            "ve 'python method_test.py' calistirman gerekiyor."
        )
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    if "input_method" not in params:
        params["input_method"] = "edges"
    return params


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"'{MODEL_PATH}' bulunamadi. Once 'python train_model.py' calistirarak "
            "modeli egitip kaydetmen gerekiyor."
        )

    # 1. Model ve isleyici parametrelerini yukle (Model Katmani)
    params = load_preprocessing_params()
    preprocessor = ImagePreprocessor.from_params(params)

    print(f"Kayitli model yukleniyor: {MODEL_PATH} (yeniden eğitim YAPILMIYOR)")
    classifier = MNISTClassifier.load(MODEL_PATH)

    # 2. Veri setini yukle (Data Katmani)
    (_, _), (X_test, y_test) = mnist.load_data()

    # 3. Arayuzu calistir (View Katmani)
    print("Musteri arayuzu aciliyor...")
    app = CustomerDemoApp(X_test, y_test, preprocessor, classifier)
    app.mainloop()


if __name__ == "__main__":
    main()