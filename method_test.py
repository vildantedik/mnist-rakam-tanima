"""
method_test.py
===============
Bu script, MNIST projesinde ozellikle karisan rakamlarin (3, 5, 8, 9)
taninma basarisini artirmak icin UC farkli iyilestirme yontemini
kademeli olarak karsilastirir:

  ASAMA A -> Girdi temsili yontemi
             (sadece kenar / sadece ham piksel / hibrit [piksel+kenar])
  ASAMA B -> Model mimarisi
             (buyuk mimari / derin-dar mimari+BatchNorm / buyuk+BatchNorm)
  ASAMA C -> Veri artirma (Data Augmentation) - var / yok

SECIM OLCUSU: Her asamada, hedef rakamlarin (3, 5, 8, 9) ORTALAMA
F1-SKORU esas alinir (genel accuracy degil) - cunku projenin amaci
tam olarak bu rakamlardaki performansi iyilestirmek.

Sonuclar 'best_method_params.json' dosyasina yazilir.
Bu dosya, mnist_app.py tarafinda kullanilmak uzere hazirlanmistir
(ayri bir entegrasyon adiminda mnist_app.py bu dosyayi okuyacak
sekilde guncellenebilir).

Calistirmak icin:
    python method_test.py

NOT: Toplamda ~8 model egitilir (3 + 3 + 2), her biri 15 epoch.
     Suresi CPU'da yaklasik 20-35 dakika arasi olabilir.
"""

import json
import os

import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

EPOCHS = 15
BATCH_SIZE = 32
TARGET_DIGITS = [3, 5, 8, 9]   # projenin odaklandigi "zor" rakamlar
BEST_PARAMS_PATH = "best_params.json"          # parameter_test.py ciktisi (varsa)
OUTPUT_PATH = "best_method_params.json"


# ============================================================
# ON ISLEME / VERI HAZIRLAMA FONKSIYONLARI
# ============================================================
def edge_image(img, canny_low, canny_high, blur_kernel):
    equalized = cv2.equalizeHist(img)
    blurred = cv2.GaussianBlur(equalized, blur_kernel, 0)
    return cv2.Canny(blurred, canny_low, canny_high)


def build_edges_dataset(X_raw, canny_low, canny_high, blur_kernel):
    processed = np.array([edge_image(img, canny_low, canny_high, blur_kernel) for img in X_raw])
    processed = processed.astype("float32") / 255.0
    return processed.reshape(processed.shape[0], 28 * 28)


def build_raw_dataset(X_raw):
    flat = X_raw.astype("float32") / 255.0
    return flat.reshape(flat.shape[0], 28 * 28)


def build_hybrid_dataset(X_raw, canny_low, canny_high, blur_kernel):
    edges_data = build_edges_dataset(X_raw, canny_low, canny_high, blur_kernel)
    raw_data = build_raw_dataset(X_raw)
    return np.concatenate([raw_data, edges_data], axis=1)  # (N, 1568)


def build_dataset(method, X_raw, canny_low, canny_high, blur_kernel):
    if method == "edges":
        return build_edges_dataset(X_raw, canny_low, canny_high, blur_kernel)
    elif method == "raw":
        return build_raw_dataset(X_raw)
    elif method == "hybrid":
        return build_hybrid_dataset(X_raw, canny_low, canny_high, blur_kernel)
    raise ValueError(f"Bilinmeyen method: {method}")


# ============================================================
# MODEL KURMA FONKSIYONU
# ============================================================
def build_model(input_dim, layer_sizes, dropout_rate, use_batchnorm=False):
    layers = []
    for i, size in enumerate(layer_sizes):
        if i == 0:
            layers.append(Dense(size, activation="relu", input_shape=(input_dim,)))
        else:
            layers.append(Dense(size, activation="relu"))
        if use_batchnorm:
            layers.append(BatchNormalization())
        if i < len(layer_sizes) - 1:
            layers.append(Dropout(dropout_rate))
    layers.append(Dense(10, activation="softmax"))
    model = Sequential(layers)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ============================================================
# DEGERLENDIRME YARDIMCISI
# ============================================================
def evaluate(model, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    target_f1_avg = float(np.mean([f1_per_class[d] for d in TARGET_DIGITS]))
    return {
        "accuracy": accuracy,
        "target_f1_avg": target_f1_avg,
        "f1_per_class": f1_per_class,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
    }


def print_target_digits(label, metrics):
    print(f"  [{label}] Genel Accuracy: {metrics['accuracy']:.4f}  |  "
          f"Hedef Rakamlar (3,5,8,9) Ort. F1: {metrics['target_f1_avg']:.4f}")
    for d in TARGET_DIGITS:
        print(f"      Rakam {d}: Precision={metrics['precision_per_class'][d]:.3f}  "
              f"Recall={metrics['recall_per_class'][d]:.3f}  F1={metrics['f1_per_class'][d]:.3f}")


# ============================================================
# ANA AKIS
# ============================================================
def main():
    # Onceki asamadan (parameter_test.py) bulunan preprocessing/dropout degerlerini
    # yukle; yoksa makul varsayilanlar kullan.
    if os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
            prev = json.load(f)
        canny_low = prev["canny_low"]
        canny_high = prev["canny_high"]
        blur_kernel = tuple(prev["blur_kernel"])
        base_dropout = prev["dropout"]
        print(f"'{BEST_PARAMS_PATH}' bulundu, degerler kullaniliyor: "
              f"Canny({canny_low},{canny_high}), Blur{blur_kernel}, Dropout={base_dropout}")
    else:
        canny_low, canny_high, blur_kernel, base_dropout = 50, 150, (5, 5), 0.4
        print("'best_params.json' bulunamadi, varsayilan degerler kullaniliyor.")

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("Veri yuklendi:", X_train.shape, X_test.shape)

    # ------------------------------------------------------------------
    # ASAMA A: GIRDI TEMSILI YONTEMI (edges / raw / hybrid)
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("ASAMA A: GIRDI TEMSILI YONTEMI KARSILASTIRMASI")
    print("(Mimari sabit: 256-128-64, Dropout sabit: {:.1f}, BatchNorm: Yok)".format(base_dropout))
    print("=" * 65)

    methods = ["edges", "raw", "hybrid"]
    stage_a_results = []

    for method in methods:
        print(f"\nTest ediliyor: girdi yontemi = '{method}'")
        X_tr = build_dataset(method, X_train, canny_low, canny_high, blur_kernel)
        X_te = build_dataset(method, X_test, canny_low, canny_high, blur_kernel)

        model = build_model(input_dim=X_tr.shape[1], layer_sizes=(256, 128, 64),
                             dropout_rate=base_dropout, use_batchnorm=False)
        model.fit(X_tr, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_data=(X_te, y_test), verbose=0)

        metrics = evaluate(model, X_te, y_test)
        print_target_digits(method, metrics)
        stage_a_results.append({"method": method, **metrics})

    best_a = max(stage_a_results, key=lambda r: r["target_f1_avg"])
    BEST_METHOD = best_a["method"]
    print(f"\n>>> EN IYI GIRDI YONTEMI: '{BEST_METHOD}' "
          f"(Hedef Rakamlar Ort. F1: {best_a['target_f1_avg']:.4f})")

    # En iyi yontemle veri setlerini hazirla (sonraki asamalarda tekrar kullanilacak)
    X_train_best_method = build_dataset(BEST_METHOD, X_train, canny_low, canny_high, blur_kernel)
    X_test_best_method = build_dataset(BEST_METHOD, X_test, canny_low, canny_high, blur_kernel)

    # ------------------------------------------------------------------
    # ASAMA B: MODEL MIMARISI (BatchNorm dahil)
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"ASAMA B: MIMARI KARSILASTIRMASI (Girdi yontemi: '{BEST_METHOD}')")
    print("=" * 65)

    arch_configs = [
        {"isim": "Buyuk (512-256-128), BatchNorm Yok", "layer_sizes": (512, 256, 128), "use_bn": False},
        {"isim": "Derin-Dar (256-128-64-32) + BatchNorm", "layer_sizes": (256, 128, 64, 32), "use_bn": True},
        {"isim": "Buyuk (512-256-128) + BatchNorm", "layer_sizes": (512, 256, 128), "use_bn": True},
    ]

    stage_b_results = []
    for cfg in arch_configs:
        print(f"\nTest ediliyor: {cfg['isim']}")
        model = build_model(input_dim=X_train_best_method.shape[1], layer_sizes=cfg["layer_sizes"],
                             dropout_rate=base_dropout, use_batchnorm=cfg["use_bn"])
        model.fit(X_train_best_method, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_data=(X_test_best_method, y_test), verbose=0)

        metrics = evaluate(model, X_test_best_method, y_test)
        print_target_digits(cfg["isim"], metrics)
        stage_b_results.append({**cfg, **metrics})

    best_b = max(stage_b_results, key=lambda r: r["target_f1_avg"])
    BEST_ARCHITECTURE = best_b["layer_sizes"]
    BEST_USE_BN = best_b["use_bn"]
    print(f"\n>>> EN IYI MIMARI: {best_b['isim']} "
          f"(Hedef Rakamlar Ort. F1: {best_b['target_f1_avg']:.4f})")

    # ------------------------------------------------------------------
    # ASAMA C: VERI ARTIRMA (Data Augmentation) - var / yok
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("ASAMA C: VERI ARTIRMA (DATA AUGMENTATION) KARSILASTIRMASI")
    print("=" * 65)

    # Augmentasyon islemi orijinal (28x28) goruntuler uzerinde yapilir,
    # ardindan secilen girdi yontemiyle (edges/raw/hybrid) islenir.
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )
    X_train_4d = X_train.reshape(-1, 28, 28, 1)
    aug_iterator = datagen.flow(X_train_4d, y_train, batch_size=len(X_train), shuffle=False)
    X_train_aug_4d, y_train_aug = next(aug_iterator)
    X_train_aug = np.clip(X_train_aug_4d.reshape(-1, 28, 28), 0, 255).astype("uint8")

    X_train_combined_raw = np.concatenate([X_train, X_train_aug], axis=0)
    y_train_combined = np.concatenate([y_train, y_train_aug], axis=0)

    X_train_combined = build_dataset(BEST_METHOD, X_train_combined_raw, canny_low, canny_high, blur_kernel)

    stage_c_results = []

    print("\nTest ediliyor: Veri artirma YOK (baseline)")
    model_no_aug = build_model(input_dim=X_train_best_method.shape[1], layer_sizes=BEST_ARCHITECTURE,
                                dropout_rate=base_dropout, use_batchnorm=BEST_USE_BN)
    model_no_aug.fit(X_train_best_method, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      validation_data=(X_test_best_method, y_test), verbose=0)
    metrics_no_aug = evaluate(model_no_aug, X_test_best_method, y_test)
    print_target_digits("Augmentation YOK", metrics_no_aug)
    stage_c_results.append({"augmentation": False, **metrics_no_aug})

    print("\nTest ediliyor: Veri artirma VAR (orijinal + dondurulmus/kaydirilmis kopyalar)")
    model_aug = build_model(input_dim=X_train_combined.shape[1], layer_sizes=BEST_ARCHITECTURE,
                             dropout_rate=base_dropout, use_batchnorm=BEST_USE_BN)
    model_aug.fit(X_train_combined, y_train_combined, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_data=(X_test_best_method, y_test), verbose=0)
    metrics_aug = evaluate(model_aug, X_test_best_method, y_test)
    print_target_digits("Augmentation VAR", metrics_aug)
    stage_c_results.append({"augmentation": True, **metrics_aug})

    best_c = max(stage_c_results, key=lambda r: r["target_f1_avg"])
    BEST_USE_AUGMENTATION = best_c["augmentation"]
    print(f"\n>>> EN IYI SECIM: Augmentation = {BEST_USE_AUGMENTATION} "
          f"(Hedef Rakamlar Ort. F1: {best_c['target_f1_avg']:.4f})")

    # ------------------------------------------------------------------
    # OZET TABLO (dokumantasyon icin)
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("TUM ASAMALARIN OZETI")
    print("=" * 65)

    summary_a = pd.DataFrame([{"Yontem": r["method"], "Accuracy": round(r["accuracy"], 4),
                                "Hedef Rakamlar Ort. F1": round(r["target_f1_avg"], 4)} for r in stage_a_results])
    print("\nAsama A - Girdi Temsili:\n", summary_a.to_string(index=False))

    summary_b = pd.DataFrame([{"Mimari": r["isim"], "Accuracy": round(r["accuracy"], 4),
                                "Hedef Rakamlar Ort. F1": round(r["target_f1_avg"], 4)} for r in stage_b_results])
    print("\nAsama B - Mimari:\n", summary_b.to_string(index=False))

    summary_c = pd.DataFrame([{"Augmentation": r["augmentation"], "Accuracy": round(r["accuracy"], 4),
                                "Hedef Rakamlar Ort. F1": round(r["target_f1_avg"], 4)} for r in stage_c_results])
    print("\nAsama C - Veri Artirma:\n", summary_c.to_string(index=False))

    # ------------------------------------------------------------------
    # SONUCLARI KAYDET
    # ------------------------------------------------------------------
    best_method_params = {
        "input_method": BEST_METHOD,
        "canny_low": canny_low,
        "canny_high": canny_high,
        "blur_kernel": list(blur_kernel),
        "dropout": base_dropout,
        "layer_sizes": list(BEST_ARCHITECTURE),
        "use_batchnorm": BEST_USE_BN,
        "use_augmentation": BEST_USE_AUGMENTATION,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(best_method_params, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(f"DENEY TAMAMLANDI - En iyi kombinasyon '{OUTPUT_PATH}' dosyasina yazildi:")
    print(json.dumps(best_method_params, indent=2, ensure_ascii=False))
    print("=" * 65)


if __name__ == "__main__":
    main()