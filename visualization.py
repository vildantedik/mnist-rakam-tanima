"""
visualization.py
================
MNIST projesinin View (Arayuz) katmani.
Sadece gorsellestirme ve kullanici etkilesiminden sorumludur.
"""

import sys
import random
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Core bilesenleri sadece tip belirlemek (type hinting) icin import ediyoruz
from mnist_core import ImagePreprocessor, MNISTClassifier

BG_MAIN = "#F4F6F8"
BG_HEADER = "#1F2937"
BG_CARD = "#FFFFFF"
COLOR_CORRECT = "#1E8E4E"
COLOR_WRONG = "#D93025"
COLOR_ACCENT = "#2563EB"


class CustomerDemoApp(tk.Tk):
    def __init__(self, X_raw, y_raw, preprocessor: ImagePreprocessor, classifier: MNISTClassifier):
        super().__init__()
        self.X_raw = X_raw
        self.y_raw = y_raw
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.index = 0

        self.title("MNIST Rakam Tanıma Sistemi")
        self.geometry("1280x680")
        self.minsize(1100, 600)
        self.configure(bg=BG_MAIN)

        self._setup_style()
        self._build_header()
        self._build_body()
        self._build_footer()

        self.render(self.index)

    def _setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        
        style.configure("Nav.TButton", font=("Segoe UI", 11), padding=(14, 8))
        style.map("Nav.TButton", background=[("active", "#DCE3EA")])
        
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"),
                         padding=(14, 8), foreground="white", background=COLOR_ACCENT)
        style.map("Accent.TButton", background=[("active", "#1D4ED8")])

        style.configure("Green.TButton", font=("Segoe UI", 11, "bold"),
                         padding=(14, 8), foreground="white", background=COLOR_CORRECT)
        style.map("Green.TButton", background=[("active", "#17703D")])

        style.configure("Warning.TButton", font=("Segoe UI", 11, "bold"),
                         padding=(14, 8), foreground="white", background="#D97706")
        style.map("Warning.TButton", background=[("active", "#B45309")])

        style.configure("Red.TButton", font=("Segoe UI", 11, "bold"),
                         padding=(14, 8), foreground="white", background=COLOR_WRONG)
        style.map("Red.TButton", background=[("active", "#B3271E")])

    def _build_header(self):
        header = tk.Frame(self, bg=BG_HEADER, height=86)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text="MNIST Rakam Tanıma Sistemi", bg=BG_HEADER, fg="white",
                 font=("Segoe UI", 18, "bold")).pack(anchor="w", padx=28, pady=(14, 0))
        tk.Label(header, text="Yapay zeka, el yazısı bir rakamı adım adım tanır",
                 bg=BG_HEADER, fg="#B9C2CC", font=("Segoe UI", 11)).pack(anchor="w", padx=28)

    def _build_body(self):
        body = tk.Frame(self, bg=BG_MAIN)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=24, pady=20)

        image_card = tk.Frame(body, bg=BG_CARD, highlightbackground="#E2E5E9",
                               highlightthickness=1)
        image_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig, self.axes = plt.subplots(1, 4, figsize=(9.2, 4.0))
        self.fig.patch.set_facecolor(BG_CARD)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.05, wspace=0.35)

        self.canvas = FigureCanvasTkAgg(self.fig, master=image_card)
        self.canvas.get_tk_widget().configure(bg=BG_CARD, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        result_card = tk.Frame(body, bg=BG_CARD, highlightbackground="#E2E5E9",
                                highlightthickness=1, width=300)
        result_card.pack(side=tk.LEFT, fill=tk.Y, padx=(20, 0))
        result_card.pack_propagate(False)

        tk.Label(result_card, text="SONUÇ", bg=BG_CARD, fg="#6B7280",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=24, pady=(28, 4))

        center_frame = tk.Frame(result_card, bg=BG_CARD)
        center_frame.pack(expand=True) 

        self.digit_label = tk.Label(center_frame, text="?", bg=BG_CARD,
                                     font=("Segoe UI", 96, "bold"))
        self.digit_label.pack(pady=(0, 6))

        self.confidence_label = tk.Label(center_frame, text="", bg=BG_CARD, fg="#374151",
                                          font=("Segoe UI", 13))
        self.confidence_label.pack()

        self.status_badge = tk.Label(center_frame, text="", font=("Segoe UI", 11, "bold"),
                                      fg="white", padx=14, pady=6)
        self.status_badge.pack(pady=(18, 6))

        self.truth_label = tk.Label(center_frame, text="", bg=BG_CARD, fg="#6B7280",
                                     font=("Segoe UI", 10))
        self.truth_label.pack()

    def _build_footer(self):
        footer = tk.Frame(self, bg=BG_MAIN)
        footer.pack(side=tk.BOTTOM, fill=tk.X, padx=24, pady=(0, 20))

        nav_frame = tk.Frame(footer, bg=BG_MAIN)
        nav_frame.pack(side=tk.LEFT)

        ttk.Button(nav_frame, text="Önceki", style="Green.TButton",
                   command=self.show_previous).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(nav_frame, text="Rastgele Örnek", style="Accent.TButton",
                   command=self.show_random).pack(side=tk.LEFT, padx=8)
        ttk.Button(nav_frame, text="Hatalı Örnek Bul", style="Warning.TButton",
                   command=self.show_next_wrong).pack(side=tk.LEFT, padx=8)
        ttk.Button(nav_frame, text="Sonraki", style="Green.TButton",
                   command=self.show_next).pack(side=tk.LEFT, padx=8)

        self.status_label = tk.Label(footer, text="", bg=BG_MAIN, fg="#6B7280",
                                      font=("Segoe UI", 10))
        self.status_label.pack(side=tk.LEFT, padx=20)

        ttk.Button(footer, text="Kapat", style="Red.TButton",
                   command=self.quit_app).pack(side=tk.RIGHT)

    def show_next(self):
        self.index = (self.index + 1) % len(self.X_raw)
        self.render(self.index)

    def show_previous(self):
        self.index = (self.index - 1) % len(self.X_raw)
        self.render(self.index)

    def show_random(self):
        self.index = random.randint(0, len(self.X_raw) - 1)
        self.render(self.index)

    def show_next_wrong(self):
        self.status_label.config(text="Hatalı örnek aranıyor, veri seti taranıyor...")
        self.update() 
        
        start_index = self.index
        while True:
            self.index = (self.index + 1) % len(self.X_raw)
            original = self.X_raw[self.index]
            true_label = int(self.y_raw[self.index])
            
            model_input = self.preprocessor.prepare_single_for_model(original)
            pred_class, _ = self.classifier.predict_single(model_input)
            
            if pred_class != true_label:
                break
                
            if self.index == start_index: 
                break
                
        self.render(self.index)

    def quit_app(self):
        self.quit()
        self.destroy()
        sys.exit(0)

    def render(self, index):
        original = self.X_raw[index]
        true_label = int(self.y_raw[index])

        equalized, blurred, edges = self.preprocessor.steps(original)
        model_input = self.preprocessor.prepare_single_for_model(original)
        pred_class, confidence = self.classifier.predict_single(model_input)

        for ax in self.axes:
            ax.clear()
            ax.axis("off")

        steps_data = [
            (original, "1. Gösterilen Rakam"),
            (equalized, "2. Netleştirme"),
            (blurred, "3. Gürültü Giderme"),
            (edges, "4. Kenar Çıkarma"),
        ]
        for ax, (img, title) in zip(self.axes, steps_data):
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=10.5, fontweight="bold", color="#374151", pad=8)

        self.canvas.draw()

        is_correct = pred_class == true_label
        renk = COLOR_CORRECT if is_correct else COLOR_WRONG
        durum = "DOĞRU TAHMİN" if is_correct else "YANLIŞ TAHMİN"

        self.digit_label.config(text=str(pred_class), fg=renk)
        self.confidence_label.config(text=f"%{confidence:.1f} eminlik")
        self.status_badge.config(text=durum, bg=renk)
        self.truth_label.config(text=f"Gercek deger: {true_label}")

        self.status_label.config(text=f"Ornek {index + 1} / {len(self.X_raw)}")