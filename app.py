import streamlit as st
import os
import gdown
import joblib
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Tohum Analiz AI", page_icon="🌱", layout="centered")

# --- MODEL İNDİRME VE HAZIRLIK ---
MODELS_DIR = "ml_models"


@st.cache_resource
def download_models_folder():
    """Google Drive'dan modelleri indirir (Sadece 1 kere çalışır)"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # Klasör boşsa veya yoksa indir
    if not os.listdir(MODELS_DIR):
        try:
            # Gizli linki Streamlit Secrets'tan alıyoruz
            folder_url = st.secrets["drive_folder_url"]

            with st.spinner(
                "Modeller Google Drive'dan indiriliyor... (İlk açılışta 1-2 dk sürebilir)"
            ):
                gdown.download_folder(
                    url=folder_url, output=MODELS_DIR, quiet=False, use_cookies=False
                )
                st.success("Modeller başarıyla indi!")
        except Exception as e:
            st.error(f"Model indirme hatası: {e}")
            st.stop()
    return MODELS_DIR


# --- RESNET50 (ML MODELLERİ İÇİN) ---
@st.cache_resource
def load_feature_extractor():
    return ResNet50(
        weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3)
    )


# --- MODEL YÜKLEME ---
@st.cache_resource
def load_ai_model(model_path):
    filename = os.path.basename(model_path)
    ext = os.path.splitext(filename)[1].lower()

    model_data = {}

    if ext in [".keras", ".h5"]:
        # Deep Learning
        model = tf.keras.models.load_model(model_path)
        # Sınıf isimlerini bulmaya çalış
        json_path = model_path.replace(ext, ".json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                model_data["classes"] = json.load(f)
        else:
            model_data["classes"] = [
                f"Sınıf {i}" for i in range(model.output_shape[-1])
            ]

        model_data["type"] = "DL"
        model_data["model"] = model

    elif ext == ".joblib":
        # Machine Learning
        # Önce label encoder var mı diye bak (Klasörde olmalı)
        le_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
        if not os.path.exists(le_path):
            st.error(
                "HATA: label_encoder.joblib bulunamadı! Drive klasörüne yükledin mi?"
            )
            return None

        model = joblib.load(model_path)
        le = joblib.load(le_path)

        model_data["type"] = "ML"
        model_data["model"] = model
        model_data["classes"] = list(le.classes_)
        model_data["extractor"] = load_feature_extractor()

    return model_data


# --- TAHMİN ---
def predict(image, model_path):
    model_data = load_ai_model(model_path)
    if not model_data:
        return None

    img = image.resize((224, 224))
    img_array = np.array(img)

    probs = []

    if model_data["type"] == "DL":
        # DL modelleri genelde 0-1 arası normalize ister
        img_input = np.expand_dims(img_array / 255.0, axis=0)
        probs = model_data["model"].predict(img_input, verbose=0)[0]

    elif model_data["type"] == "ML":
        # ML modelleri ResNet özellikleri ister
        extractor = model_data["extractor"]
        x = preprocess_input(np.expand_dims(img_array, axis=0))
        features = extractor.predict(x, verbose=0)
        probs = model_data["model"].predict_proba(features)[0]

    # Sonuçları Sırala
    top_indices = probs.argsort()[-3:][::-1]
    results = []
    for i in top_indices:
        results.append(
            {"label": model_data["classes"][i], "score": float(probs[i]) * 100}
        )

    return results


# --- ARAYÜZ ---
st.title("🌱 Tohum Analiz Projesi")
st.caption("Bartın Üniversitesi - Yapay Zeka Destekli Tohum Tespiti")

# 1. Modelleri İndir (Otomatik)
if "drive_folder_url" not in st.secrets:
    st.error("Lütfen Streamlit Secrets ayarlarını yapın!")
    st.stop()

download_models_folder()

# 2. Modelleri Listele (Label Encoder hariç)
available_models = [
    f
    for f in os.listdir(MODELS_DIR)
    if f.endswith((".keras", ".joblib")) and "label_encoder" not in f
]

if not available_models:
    st.warning("Klasörde model bulunamadı.")
else:
    selected_model_name = st.selectbox("Model Seçiniz", available_models)
    selected_model_path = os.path.join(MODELS_DIR, selected_model_name)

    # 3. Resim Yükleme
    uploaded_file = st.file_uploader(
        "Tohum resmi yükleyin", type=["jpg", "png", "jpeg"]
    )

    if uploaded_file and st.button("Analiz Et"):
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Yüklenen Resim", width=300)

        with st.spinner("Analiz yapılıyor..."):
            sonuc = predict(img, selected_model_path)

        if sonuc:
            en_iyi = sonuc[0]
            st.success(f"Sonuç: **{en_iyi['label']}** (Güven: %{en_iyi['score']:.1f})")
            st.progress(int(en_iyi["score"]))

            # Diğer tahminler
            with st.expander("Diğer Olasılıklar"):
                for s in sonuc[1:]:
                    st.write(f"{s['label']}: %{s['score']:.1f}")
