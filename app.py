import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
from datetime import datetime
from utils import processing

def apply_blur(img_np):
    return cv2.GaussianBlur(img_np, (15, 15), 0)

def apply_morphology(img_bin, operation="dilation", se_shape="rect"):
    if se_shape == "rect":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    elif se_shape == "ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    else:
        kernel = np.ones((5, 5), np.uint8)

    if operation == "dilation":
        return cv2.dilate(img_bin, kernel, iterations=1)
    else:
        return cv2.erode(img_bin, kernel, iterations=1)

if not os.path.exists('assets'):
    os.makedirs('assets')

st.set_page_config(page_title="IMG APP", layout="wide")
st.title("📸 IMG APP - Pencitik")
st.markdown("Upload gambar dan gunakan fitur.")

st.subheader("Operasi Logika Antara 2 Gambar")
file1 = st.file_uploader("Gambar 1 (Biner)", type=["jpg", "jpeg", "png"], key="log1")
file2 = st.file_uploader("Gambar 2 (Biner) - Hanya untuk NOT, AND, OR, XOR", type=["jpg", "jpeg", "png"], key="log2")

if file1:
    img1 = Image.open(file1).convert("L")
    np1 = np.array(img1)
    _, bin1 = cv2.threshold(np1, 128, 255, cv2.THRESH_BINARY)

    log_ops = ["NOT", "AND", "OR", "XOR"]
    op = st.radio("Pilih Operasi Logika", log_ops, horizontal=True)

    if op == "NOT":
        result_np = cv2.bitwise_not(bin1)
        col1, col2 = st.columns(2)
        with col1: st.image(img1, caption="Gambar 1", width=300)
        with col2: st.image(result_np, caption=f"Hasil Logika: {op}", width=300)
    else:
        if file2:
            img2 = Image.open(file2).convert("L")
            np2 = np.array(img2)
            h, w = min(bin1.shape[0], np2.shape[0]), min(bin1.shape[1], np2.shape[1])
            bin1 = cv2.resize(bin1, (w, h))
            np2 = cv2.resize(np2, (w, h))
            _, bin2 = cv2.threshold(np2, 128, 255, cv2.THRESH_BINARY)

            if op == "AND":
                result_np = processing.logic_and(bin1, bin2)
            elif op == "OR":
                result_np = processing.logic_or(bin1, bin2)
            else:
                result_np = processing.logic_xor(bin1, bin2)

            col1, col2, col3 = st.columns(3)
            with col1: st.image(bin1, caption="Gambar 1", width=300)
            with col2: st.image(bin2, caption="Gambar 2", width=300)
            with col3: st.image(result_np, caption=f"Hasil Logika: {op}", width=300)
        else:
            st.warning("Silakan upload gambar ke-2 untuk operasi AND, OR, atau XOR")

    if st.button("💾 Simpan Hasil Logika"):
        Image.fromarray(result_np).save(f"assets/logika_{op.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        st.success("✅ Disimpan!")

else:
    st.info("⬆️ Upload gambar pertama terlebih dahulu.")

uploaded_file = st.file_uploader("Upload gambar (png/jpg/jpeg)", type=["png", "jpg", "jpeg"], key="main")

if uploaded_file:
    img = Image.open(uploaded_file)
    img_np = np.array(img.convert("RGB"))[:, :, ::-1]

    fitur_dipilih = st.selectbox("🔧 Pilih Fitur", [
        "Grayscale", "Biner", "Aritmatika (1 Gambar)", 
        "Blurring", "Histogram", "Morfologi"
    ])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Gambar Asli")
        st.image(img, width=300)

    with col2:
        st.markdown("### Hasil")

        if fitur_dipilih == "Grayscale":
            result = img.convert("L")
            st.image(result, caption="Grayscale", width=300)
            if st.button("💾 Simpan Grayscale"):
                result.save(f"assets/grayscale_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                st.success("✅ Disimpan!")

        elif fitur_dipilih == "Biner":
            gray = img.convert("L")
            thresh = 128
            fn = lambda x: 255 if x > thresh else 0
            result = gray.point(fn, mode='1')
            st.image(result, caption="Biner", width=300)
            if st.button("💾 Simpan Biner"):
                result.save(f"assets/biner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                st.success("✅ Disimpan!")

        elif fitur_dipilih == "Aritmatika (1 Gambar)":
            op = st.radio("Operasi", ["Penjumlahan", "Pengurangan", "Perkalian", "Pembagian"], horizontal=True)
            img2_np = img_np.copy()

            if op == "Penjumlahan":
                result_np = processing.arithmetic_add(img_np, img2_np)
            elif op == "Pengurangan":
                result_np = processing.arithmetic_subtract(img_np, img2_np)
            elif op == "Perkalian":
                result_np = cv2.convertScaleAbs(img_np * 1.2)
            else:  # Pembagian
                result_np = cv2.convertScaleAbs(img_np / 2)

            result = Image.fromarray(cv2.cvtColor(result_np, cv2.COLOR_BGR2RGB))
            st.image(result, caption=f"Hasil {op}", width=300)

            if st.button("💾 Simpan Aritmatika"):
                result.save(f"assets/arit1_{op.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                st.success("✅ Disimpan!")


        elif fitur_dipilih == "Blurring":
            result_np = apply_blur(img_np)
            result = Image.fromarray(cv2.cvtColor(result_np, cv2.COLOR_BGR2RGB))
            st.image(result, caption="Gaussian Blur", width=300)
            if st.button("💾 Simpan Blur"):
                result.save(f"assets/blur_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                st.success("✅ Disimpan!")

        elif fitur_dipilih == "Histogram":
            mode = st.radio("Mode", ["RGB", "Grayscale"])
            img_hist = img.convert("RGB") if mode == "RGB" else img.convert("L")
            img_np_hist = np.array(img_hist)
            fig, ax = plt.subplots()
            if mode == "RGB":
                for i, col in enumerate(("r", "g", "b")):
                    hist = cv2.calcHist([img_np_hist], [i], None, [256], [0, 256])
                    ax.plot(hist, color=col)
            else:
                hist = cv2.calcHist([img_np_hist], [0], None, [256], [0, 256])
                ax.plot(hist, color="black")
            ax.set_title(f"Histogram {mode}")
            st.pyplot(fig)
            st.info("📊 Histogram tidak dapat disimpan langsung.")

        elif fitur_dipilih == "Morfologi":
            gray = img.convert("L")
            img_np_bin = np.array(gray)
            _, bin_img = cv2.threshold(img_np_bin, 128, 255, cv2.THRESH_BINARY)
            se_shape = st.radio("Bentuk SE", ["rect", "ellipse"])
            result_np = apply_morphology(bin_img, "dilation", se_shape)
            result = Image.fromarray(result_np)
            st.image(result, caption=f"Dilasi: {se_shape}", width=300)
            if st.button("💾 Simpan Morfologi"):
                result.save(f"assets/morf_{se_shape}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                st.success("✅ Disimpan!")
else:
    st.info("⬆️ Upload gambar terlebih dahulu untuk mengkonversi ke Grayscale, Biner, dan Fitur lainnya.")
