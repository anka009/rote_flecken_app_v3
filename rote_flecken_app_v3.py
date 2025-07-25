# 📥 Imports
import streamlit as st
import random
import numpy as np
import cv2
from PIL import Image, ImageSequence

# 🧠 Initialisiere Session-State
def init_session_state():
    defaults = {
        "analyse_ergebnisse": [],
        "total_flecken": 0,
        "total_pixel_area": 0,
        "upload_key": "upload_key"
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

init_session_state()

# 🎛️ Sidebar-Steuerung
def sidebar_controls():
    st.sidebar.markdown("## ⚙️ Einstellungen")

    st.sidebar.slider("Hue min", 0, 180, 0, key="h_min")
    st.sidebar.slider("Hue max", 0, 180, 30, key="h_max")
    st.sidebar.slider("Sättigung min", 0, 255, 70, key="s_min")
    st.sidebar.slider("Helligkeit min", 0, 255, 50, key="v_min")
    st.sidebar.slider("🟢 Minimale Fleckfläche (Pixel)", 10, 1000, 50, step=10, key="min_area")

    if st.sidebar.button("🔁 Reset"):
        reset_keys = ["analyse_ergebnisse", "total_flecken", "total_pixel_area", "upload_key"]
        for key in reset_keys:
            st.session_state.pop(key, None)
        st.session_state["upload_key"] = f"upload_key_{random.randint(0, 100000)}"
        st.rerun()

sidebar_controls()

# 📤 Datei-Upload
def image_upload():
    return st.file_uploader(
        "📁 Bilder hochladen",
        type=["gif", "png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        key=st.session_state["upload_key"]
    )

# 🔬 Analyse-Funktion
def analyse_image(file, page_number, frame_rgb, settings):
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    lower = np.array([settings["h_min"], settings["s_min"], settings["v_min"]])
    upper = np.array([settings["h_max"], 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > settings["min_area"]]

    fleckenzahl = len(filtered)
    fläche_pixel = sum(cv2.contourArea(cnt) for cnt in filtered)
    fläche_mm2 = fläche_pixel / (10 ** 2)  # 10 Pixel pro mm

    st.session_state["analyse_ergebnisse"].append({
        "Datei": file.name,
        "Seite": page_number,
        "Fleckenzahl": fleckenzahl,
        "Fläche (mm²)": round(fläche_mm2, 2)
    })

    output = frame_rgb.copy()
    cv2.drawContours(output, filtered, -1, (0, 255, 0), 2)
    return fleckenzahl, fläche_pixel, fläche_mm2, output

# 🖼️ Hauptprozess
def process_uploaded_images(uploaded_files):
    if not uploaded_files:
        return

    settings = {
        "h_min": st.session_state["h_min"],
        "h_max": st.session_state["h_max"],
        "s_min": st.session_state["s_min"],
        "v_min": st.session_state["v_min"],
        "min_area": st.session_state["min_area"]
    }

    for file in uploaded_files:
        st.markdown(f"### 🖼️ Datei: `{file.name}`")
        try:
            image_pil = Image.open(file)
            frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(image_pil)]
        except Exception as e:
            st.error(f"❌ Fehler beim Laden: {e}")
            continue

        for idx, frame in enumerate(frames):
            if len(frames) > 1:
                st.subheader(f"📄 Seite {idx + 1}")
            image_np = np.array(frame)
            flecken, pixel_area, mm2_area, output = analyse_image(file, idx + 1, image_np, settings)

            st.success(f"🔴 Flecken: {flecken}")
            st.info(f"📐 Fläche: {pixel_area:.2f} px² • {mm2_area:.2f} mm²")
            st.image(output, caption="✅ Markierte Flecken", channels="RGB")

# 🚀 Main App
st.markdown("<h1 style='text-align:center;color:#C00000;'>🔬 Fleckenanalyse</h1><hr>", unsafe_allow_html=True)
uploaded_files = image_upload()
process_uploaded_images(uploaded_files)
