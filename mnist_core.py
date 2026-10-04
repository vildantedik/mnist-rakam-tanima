"""
mnist_core.py
=============
MNIST projesinin paylasilan CEKIRDEK sinif ve fonksiyonlari.

Bu modul tek basina calistirilmaz - hem 'train_model.py' (egitim)
hem de 'mnist_app.py' (musteri arayuzu) tarafindan import edilir.
Boylece egitim mantigi ile arayuz mantigi birbirinden ayrilmis olur.

Icerik:
  - ImagePreprocessor : goruntu on isleme + girdi temsili (edges/raw/hybrid)
  - MNISTClassifier   : ANN modelini sarmalayan sinif (kurma/egitme/kaydetme/yukleme)
  - ModelEvaluator    : confusion matrix + precision/recall/f1 hesaplama ve raporlama
  - augment_training_data : veri artirma yardimcisi
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
)


# ============================================================
# GORUNTU ON ISLEME SINIFI (edges / raw / hybrid destekli)
# ============================================================
class ImagePreprocessor:
    """Histogram esitleme -> Gaussian Blur -> Canny Edge zincirini uygular
    ve secilen 'method'e gore model girdisi uretir:
      - 'edges'  : sadece kenar haritasi        (784 boyut)
      - 'raw'    : sadece ham piksel             (784 boyut)
      - 'hybrid' : ham piksel + kenar birlikte   (1568 boyut)
    """

    def __init__(self, method, canny_low, canny_high, blur_kernel):
        self.method = method
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.blur_kernel = tuple(blur_kernel)

    def steps(self, img):
        """Ara adimlari doner: (esitlenmis, bulanik, kenarlar) - gorsellestirme icin."""
        equalized = cv2.equalizeHist(img)
        blurred = cv2.GaussianBlur(equalized, self.blur_kernel, 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        return equalized, blurred, edges

    def feature_vector(self, img):
        """Secilen yontem uzerinden tek bir goruntuyu duz (flatten) vektore cevirir."""
        _, _, edges = self.steps(img)
        raw_flat = (img.astype("float32") / 255.0).flatten()
        edges_flat = (edges.astype("float32") / 255.0).flatten()

        if self.method == "edges":
            return edges_flat
        elif self.method == "raw":
            return raw_flat
        elif self.method == "hybrid":
            return np.concatenate([raw_flat, edges_flat])
        raise ValueError(f"Bilinmeyen method: {self.method}")

    def process_batch(self, X_raw):
        """Bir goruntu dizisini isleyip (N, boyut) formatinda dondurur."""
        return np.array([self.feature_vector(img) for img in X_raw], dtype="float32")

    def prepare_single_for_model(self, img):
        """Tek bir goruntuyu model girdisi formatina (1, boyut) cevirir."""
        return self.feature_vector(img).reshape(1, -1)

    @property
    def input_dim(self):
        return 1568 if self.method == "hybrid" else 784

    @classmethod
    def from_params(cls, params):
        """best_method_params.json / best_params.json sozlugunden ornek olusturur."""
        return cls(
            method=params.get("input_method", "edges"),
            canny_low=params["canny_low"],
            canny_high=params["canny_high"],
            blur_kernel=params["blur_kernel"],
        )


# ============================================================
# ANN MODEL SINIFI (BatchNorm destekli, kaydet/yukle ozellikli)
# ============================================================
class MNISTClassifier:
    """Dense/Dropout(/BatchNorm) katmanlarindan olusan Sequential ANN modelini sarmalar.
    Egitilmis modeli diske kaydedip, tekrar egitmeden geri yukleyebilir."""

    def __init__(self, input_dim=None, dropout_rate=None, layer_sizes=None, use_batchnorm=False):
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.layer_sizes = tuple(layer_sizes) if layer_sizes else None
        self.use_batchnorm = use_batchnorm
        self.model = self._build() if input_dim is not None else None

    def _build(self):
        layers = []
        for i, size in enumerate(self.layer_sizes):
            if i == 0:
                layers.append(Dense(size, activation="relu", input_shape=(self.input_dim,)))
            else:
                layers.append(Dense(size, activation="relu"))
            if self.use_batchnorm:
                layers.append(BatchNormalization())
            if i < len(self.layer_sizes) - 1:
                layers.append(Dropout(self.dropout_rate))
        layers.append(Dense(10, activation="softmax"))

        model = Sequential(layers)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def summary(self):
        self.model.summary()

    def train(self, X_train, y_train, X_val, y_val, epochs=15, batch_size=32, verbose=1):
        return self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=verbose,
        )

    def predict_probs(self, X):
        return self.model.predict(X, verbose=0)

    def predict_classes(self, X):
        return np.argmax(self.predict_probs(X), axis=1)

    def predict_single(self, x_flat):
        """Tek bir ornek icin (1, boyut) girdi alir, (tahmin, guven_yuzdesi) doner."""
        probs = self.model.predict(x_flat, verbose=0)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class] * 100)
        return pred_class, confidence

    # --------------------------------------------------------
    # KAYDETME / YUKLEME
    # --------------------------------------------------------
    def save(self, path):
        """Egitilmis modeli diske kaydeder (Keras native format, .keras uzantili)."""
        self.model.save(path)

    @classmethod
    def load(cls, path):
        """Diskten egitilmis bir modeli yukler - YENIDEN EGITIM YAPMAZ."""
        instance = cls()  # bos instance, mimari bilgisi gerekmiyor
        instance.model = load_model(path)
        return instance


# ============================================================
# VERI ARTIRMA YARDIMCISI
# ============================================================
def augment_training_data(X_train, y_train, rotation_range=10, width_shift_range=0.1,
                           height_shift_range=0.1, zoom_range=0.1):
    """Egitim setine dondurme/kaydirma/zoom ile uretilen ek ornekler ekler.
    Donen veri, ORIJINAL goruntu formatinda (28x28, uint8) olur - preprocessing
    sonrasinda uygulanmasi gerekir."""
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        zoom_range=zoom_range,
    )
    X_train_4d = X_train.reshape(-1, 28, 28, 1)
    iterator = datagen.flow(X_train_4d, y_train, batch_size=len(X_train), shuffle=False)
    X_aug_4d, y_aug = next(iterator)
    X_aug = np.clip(X_aug_4d.reshape(-1, 28, 28), 0, 255).astype("uint8")

    X_combined = np.concatenate([X_train, X_aug], axis=0)
    y_combined = np.concatenate([y_train, y_aug], axis=0)
    return X_combined, y_combined


# ============================================================
# PERFORMANS DEGERLENDIRME SINIFI
# ============================================================
class ModelEvaluator:
    """Confusion matrix ve precision/recall/f1 metriklerini hesaplar,
    teknik raporu dosyaya kaydeder (musteriye gosterilmez)."""

    def __init__(self, classifier: MNISTClassifier, X_test, y_test):
        self.classifier = classifier
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = classifier.predict_classes(X_test)

        self.conf_matrix = confusion_matrix(y_test, self.y_pred)
        self.accuracy = accuracy_score(y_test, self.y_pred) * 100
        self.precision = precision_score(y_test, self.y_pred, average=None, zero_division=0)
        self.recall = recall_score(y_test, self.y_pred, average=None, zero_division=0)
        self.f1 = f1_score(y_test, self.y_pred, average=None, zero_division=0)
        self.macro_precision = precision_score(y_test, self.y_pred, average="macro", zero_division=0)
        self.macro_recall = recall_score(y_test, self.y_pred, average="macro", zero_division=0)
        self.macro_f1 = f1_score(y_test, self.y_pred, average="macro", zero_division=0)

    def print_summary(self):
        print(f"\nGenel Dogruluk (Accuracy): %{self.accuracy:.2f}")
        print(f"Genel Precision (macro avg): {self.macro_precision:.4f}")
        print(f"Genel Recall (macro avg): {self.macro_recall:.4f}")
        print(f"Genel F1-Score (macro avg): {self.macro_f1:.4f}")

    def save_technical_report(self, path="confusion_matrix_metrikler.png"):
        """2x2 teknik grafik: Confusion Matrix + Precision + Recall + F1.
        SADECE dosyaya kaydedilir, ekrana acilmaz."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 13))
        fig.suptitle(
            f"Model Performans Analizi  —  Genel Dogruluk: %{self.accuracy:.2f}",
            fontsize=14, fontweight="bold", y=0.98
        )

        im = axes[0, 0].imshow(self.conf_matrix, cmap="Blues")
        axes[0, 0].set_title("Confusion Matrix", fontsize=13, fontweight="bold")
        axes[0, 0].set_xlabel("Tahmin Edilen Rakam")
        axes[0, 0].set_ylabel("Gercek Rakam")
        axes[0, 0].set_xticks(range(10))
        axes[0, 0].set_yticks(range(10))
        for i in range(10):
            for j in range(10):
                renk = "white" if self.conf_matrix[i, j] > self.conf_matrix.max() / 2 else "black"
                axes[0, 0].text(j, i, self.conf_matrix[i, j], ha="center", va="center", color=renk, fontsize=8)
        plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

        self._bar_panel(axes[0, 1], self.precision, self.macro_precision, "Precision (Kesinlik)", "Precision")
        self._bar_panel(axes[1, 0], self.recall, self.macro_recall, "Recall (Duyarlilik)", "Recall")
        self._bar_panel(axes[1, 1], self.f1, self.macro_f1, "F1-Score", "F1-Score")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Teknik rapor kaydedildi: {path}")

    @staticmethod
    def _bar_panel(ax, values, macro_avg, title, ylabel):
        colors = ["seagreen" if v >= 0.95 else "orange" if v >= 0.90 else "crimson" for v in values]
        ax.bar(range(10), values, color=colors)
        ax.axhline(y=macro_avg, color="black", linestyle="--", linewidth=1, label=f"Ortalama: {macro_avg:.3f}")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Rakam")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(10))
        ax.set_ylim(0.85, 1.0)
        ax.legend(loc="lower right", fontsize=9)
        for i, v in enumerate(values):
            ax.text(i, v + 0.003, f"{v:.2f}", ha="center", fontsize=8)