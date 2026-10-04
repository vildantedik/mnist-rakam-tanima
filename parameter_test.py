"""
parameter_test.py
==================
Bu script, MNIST ANN projesi icin en iyi hiperparametreleri bulmak amaciyla
kademeli (sequential) bir arama yapar:

  Asama 1 -> Preprocessing parametreleri (Canny threshold + Gaussian Blur kernel)
  Asama 2 -> Dropout orani
  Asama 3 -> Model mimarisi (katman/norron sayilari)

NOT: Bu dosya sadece DENEY amaclidir. Uretim kodunda (mnist_app.py) tekrar
     calistirilmaz. Sonuclar 'best_params.json' dosyasina yazilir, uretim
     kodu bu dosyayi okuyarak modeli tek seferde, dogrudan en iyi
     parametrelerle egitir.

Calistirmak icin:
    python parameter_test.py
"""

import json
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

EPOCHS = 15
BATCH_SIZE = 32


# ============================================================
# YARDIMCI FONKSIYONLAR
# ============================================================
def preprocess_image(img, canny_low, canny_high, blur_kernel):
    equalized = cv2.equalizeHist(img)
    blurred = cv2.GaussianBlur(equalized, blur_kernel, 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    return edges


def prepare_data(X_raw, canny_low, canny_high, blur_kernel):
    processed = np.array([preprocess_image(img, canny_low, canny_high, blur_kernel) for img in X_raw])
    processed = processed.astype('float32') / 255.0
    processed = processed.reshape(processed.shape[0], 28 * 28)
    return processed


def build_model(dropout_rate, layer_sizes):
    layers = []
    for i, size in enumerate(layer_sizes):
        if i == 0:
            layers.append(Dense(size, activation='relu', input_shape=(784,)))
        else:
            layers.append(Dense(size, activation='relu'))
        if i < len(layer_sizes) - 1:
            layers.append(Dropout(dropout_rate))
    layers.append(Dense(10, activation='softmax'))
    model = Sequential(layers)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("Veri yuklendi:", X_train.shape, X_test.shape)

    # --------------------------------------------------------
    # ASAMA 1: PREPROCESSING ARAMASI
    # --------------------------------------------------------
    preprocessing_configs = [
        {'canny_low': 30, 'canny_high': 100, 'blur_kernel': (3, 3), 'isim': 'Dusuk Threshold + Kucuk Blur'},
        {'canny_low': 30, 'canny_high': 100, 'blur_kernel': (5, 5), 'isim': 'Dusuk Threshold + Normal Blur'},
        {'canny_low': 50, 'canny_high': 150, 'blur_kernel': (3, 3), 'isim': 'Normal Threshold + Kucuk Blur'},
        {'canny_low': 50, 'canny_high': 150, 'blur_kernel': (5, 5), 'isim': 'Normal Threshold + Normal Blur (Baseline)'},
        {'canny_low': 70, 'canny_high': 200, 'blur_kernel': (3, 3), 'isim': 'Yuksek Threshold + Kucuk Blur'},
        {'canny_low': 70, 'canny_high': 200, 'blur_kernel': (5, 5), 'isim': 'Yuksek Threshold + Normal Blur'},
    ]

    print("\n" + "=" * 60)
    print("ASAMA 1: PREPROCESSING PARAMETRE ARAMASI")
    print("=" * 60)

    prep_results = []
    for cfg in preprocessing_configs:
        print(f"\nTest: {cfg['isim']}")
        X_tr = prepare_data(X_train, cfg['canny_low'], cfg['canny_high'], cfg['blur_kernel'])
        X_te = prepare_data(X_test, cfg['canny_low'], cfg['canny_high'], cfg['blur_kernel'])

        model = build_model(0.5, (256, 128, 64))
        history = model.fit(X_tr, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                             validation_data=(X_te, y_test), verbose=0)

        val_loss = min(history.history['val_loss'])
        val_acc = history.history['val_accuracy'][-1]
        prep_results.append({**cfg, 'val_loss': round(val_loss, 4), 'val_acc': round(val_acc, 4)})
        print(f"  -> val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    prep_df = pd.DataFrame(prep_results)
    print("\nSONUC TABLOSU:\n", prep_df.drop(columns=['blur_kernel']).to_string(index=False))

    best_prep = prep_results[prep_df['val_loss'].idxmin()]
    print(f"\n>>> EN IYI PREPROCESSING: {best_prep['isim']}")

    # --------------------------------------------------------
    # ASAMA 2: DROPOUT ARAMASI
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ASAMA 2: DROPOUT PARAMETRE ARAMASI")
    print("=" * 60)

    X_tr_best = prepare_data(X_train, best_prep['canny_low'], best_prep['canny_high'], best_prep['blur_kernel'])
    X_te_best = prepare_data(X_test, best_prep['canny_low'], best_prep['canny_high'], best_prep['blur_kernel'])

    dropout_results = []
    for dr in [0.2, 0.3, 0.4, 0.5, 0.6]:
        print(f"\nDropout = {dr}")
        model = build_model(dr, (256, 128, 64))
        history = model.fit(X_tr_best, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                             validation_data=(X_te_best, y_test), verbose=0)
        val_loss = min(history.history['val_loss'])
        val_acc = history.history['val_accuracy'][-1]
        dropout_results.append({'dropout': dr, 'val_loss': round(val_loss, 4), 'val_acc': round(val_acc, 4)})
        print(f"  -> val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    dropout_df = pd.DataFrame(dropout_results)
    print("\nSONUC TABLOSU:\n", dropout_df.to_string(index=False))

    best_dropout = dropout_results[dropout_df['val_loss'].idxmin()]['dropout']
    print(f"\n>>> EN IYI DROPOUT: {best_dropout}")

    # --------------------------------------------------------
    # ASAMA 3: MIMARI ARAMASI
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ASAMA 3: MODEL MIMARISI ARAMASI")
    print("=" * 60)

    arch_configs = [
        {'layer_sizes': (128, 64), 'isim': 'Kucuk (128-64)'},
        {'layer_sizes': (256, 128, 64), 'isim': 'Orta (256-128-64)'},
        {'layer_sizes': (512, 256, 128), 'isim': 'Buyuk (512-256-128)'},
    ]

    arch_results = []
    for cfg in arch_configs:
        print(f"\nTest: {cfg['isim']}")
        model = build_model(best_dropout, cfg['layer_sizes'])
        history = model.fit(X_tr_best, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                             validation_data=(X_te_best, y_test), verbose=0)
        val_loss = min(history.history['val_loss'])
        val_acc = history.history['val_accuracy'][-1]
        arch_results.append({**cfg, 'val_loss': round(val_loss, 4), 'val_acc': round(val_acc, 4)})
        print(f"  -> val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    arch_df = pd.DataFrame(arch_results)
    print("\nSONUC TABLOSU:\n", arch_df.drop(columns=['layer_sizes']).to_string(index=False))

    best_arch = arch_results[arch_df['val_loss'].idxmin()]['layer_sizes']
    print(f"\n>>> EN IYI MIMARI: {best_arch}")

    # --------------------------------------------------------
    # SONUCLARI best_params.json DOSYASINA YAZ
    # --------------------------------------------------------
    best_params = {
        'canny_low': best_prep['canny_low'],
        'canny_high': best_prep['canny_high'],
        'blur_kernel': list(best_prep['blur_kernel']),
        'dropout': best_dropout,
        'layer_sizes': list(best_arch),
    }

    with open('best_params.json', 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("ARAMA TAMAMLANDI - En iyi parametreler 'best_params.json' dosyasina yazildi:")
    print(json.dumps(best_params, indent=2, ensure_ascii=False))
    print("=" * 60)



if __name__ == "__main__":
    main()