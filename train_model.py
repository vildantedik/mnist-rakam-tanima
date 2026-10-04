"""
train_model.py
===============
Bu script, mnist_core.py'deki siniflari kullanarak modeli
BIR KEZ egitir, degerlendirir ve diske kaydeder.

'best_method_params.json' (veya yoksa 'best_params.json') dosyasindaki
en iyi ayarlari okur, egitir ve su dosyalari uretir:
  - mnist_model.keras              -> egitilmis model (mnist_app.py bunu yukler)
  - confusion_matrix_metrikler.png -> teknik performans raporu

NOT: Bu script SADECE modeli/parametreleri degistirdiginde tekrar
     calistirilmasi gereken bir dosyadir. Gunluk kullanimda
     'python mnist_app.py' calistirman yeterlidir - o dosya artik
     egitim YAPMAZ, sadece burada kaydedilen modeli yukler.

Calistirmak icin:
    python train_model.py
"""

import json
import os

from tensorflow.keras.datasets import mnist

from mnist_core import (
    ImagePreprocessor,
    MNISTClassifier,
    ModelEvaluator,
    augment_training_data,
)

BEST_METHOD_PARAMS_PATH = "best_method_params.json"
BEST_PARAMS_PATH = "best_params.json"  # fallback
MODEL_PATH = "mnist_model.keras"
EPOCHS = 15
BATCH_SIZE = 32


def load_params():
    if os.path.exists(BEST_METHOD_PARAMS_PATH):
        with open(BEST_METHOD_PARAMS_PATH, "r", encoding="utf-8") as f:
            params = json.load(f)
        print(f"'{BEST_METHOD_PARAMS_PATH}' yuklendi.")
        return params
    elif os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
            base = json.load(f)
        print(f"'{BEST_METHOD_PARAMS_PATH}' bulunamadi, '{BEST_PARAMS_PATH}' kullaniliyor "
              "(edges yontemi, BatchNorm/Augmentation kapali varsayilan).")
        return {
            "input_method": "edges",
            "canny_low": base["canny_low"],
            "canny_high": base["canny_high"],
            "blur_kernel": base["blur_kernel"],
            "dropout": base["dropout"],
            "layer_sizes": base["layer_sizes"],
            "use_batchnorm": False,
            "use_augmentation": False,
        }
    else:
        raise FileNotFoundError(
            "Ne 'best_method_params.json' ne de 'best_params.json' bulundu. "
            "Once 'python parameter_test.py' ve ardindan 'python method_test.py' calistirman gerekiyor."
        )


def main():
    params = load_params()
    print("Kullanilan parametreler:", json.dumps(params, indent=2, ensure_ascii=False))

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    preprocessor = ImagePreprocessor.from_params(params)

    if params.get("use_augmentation", False):
        print("\nVeri artirma uygulaniyor (dondurme/kaydirma/zoom ile ek ornekler)...")
        X_train_final_raw, y_train_final = augment_training_data(X_train, y_train)
    else:
        X_train_final_raw, y_train_final = X_train, y_train

    print(f"\nVeri on isleniyor (yontem: '{params['input_method']}')...")
    X_train_processed = preprocessor.process_batch(X_train_final_raw)
    X_test_processed = preprocessor.process_batch(X_test)

    print("\nModel egitiliyor...")
    classifier = MNISTClassifier(
        input_dim=preprocessor.input_dim,
        dropout_rate=params["dropout"],
        layer_sizes=params["layer_sizes"],
        use_batchnorm=params.get("use_batchnorm", False),
    )
    classifier.summary()
    classifier.train(X_train_processed, y_train_final, X_test_processed, y_test,
                      epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    print("\nModel degerlendiriliyor...")
    evaluator = ModelEvaluator(classifier, X_test_processed, y_test)
    evaluator.print_summary()
    evaluator.save_technical_report()

    print(f"\nModel diske kaydediliyor: {MODEL_PATH}")
    classifier.save(MODEL_PATH)

    print("\n" + "=" * 60)
    print("EGITIM TAMAMLANDI")
    print(f"Model kaydedildi: {MODEL_PATH}")
    print("=" * 60)
if __name__ == "__main__":
    main()