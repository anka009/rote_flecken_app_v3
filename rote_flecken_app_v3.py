import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageSequence
from streamlit_drawable_canvas import st_canvas

# 🧠 Session-State Initialisierung
if "analyse_ergebnisse" not in st.session_state:
    st.session_state["analyse_ergebnisse"] = []
if "total_flecken" not in st.session_state:
    st.session_state["total_flecken"] = 0
if "total_pixel_area" not in st.session_state:
    st.session_state["total_pixel_area"] = 0

# 🔄 Reset-Button
if st.sidebar.button("🧪 Neues Experiment starten"):
    st.session_state["analyse_ergebnisse"] = []
    st.session_state["total_flecken"] = 0
    st.session_state["total_pixel_area"] = 0
    st.rerun()

# 🎛️ Parametersteuerung
h_min = st.sidebar.slider("Hue min", 0, 180, 0)
h_max = st.sidebar.slider("Hue max", 0, 180, 30)
s_min = st.sidebar.slider("Sättigung min", 0, 255, 70)
v_min = st.sidebar.slider("Helligkeit min", 0, 255, 50)
min_area = st.sidebar.slider("🟢 Minimale Fleckfläche (Pixel)", 10, 1000, 50, 10)
min_canvas_area = st.sidebar.slider("🖍️ Mindestfläche für manuelle Flecken (px²)", 10, 1000, 50, 10)
pixels_per_mm = 10  # Skalierung anpassen

# 📁 Datei-Upload
uploaded_files = st.file_uploader("📥 Bilder hochladen", type=["gif", "png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)

if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        frames = [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(image)]

        for j, frame in enumerate(frames):
            st.subheader(f"📄 Seite {j + 1} von Datei: {uploaded_file.name}")
            image_np = np.array(frame)
            image_pil = Image.fromarray(image_np)

image_pil = Image.open(uploaded_file).convert("RGB")
st.image(image_pil, caption="📷 Vorschau", use_column_width=True)

# Zeichenfläche direkt über dem Originalbild
canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.3)",       # Transparente rote Füllung
    stroke_width=2,
    stroke_color="#ff0000",
    background_image=Image.open(uploaded_file),  # ✅ Verwende das Original-Upload-Objekt!
    height=image_pil.height if image_pil.height < 600 else 600,
    width=image_pil.width if image_pil.width < 800 else 800,
    drawing_mode="polygon",
    key=f"canvas_{uploaded_file.name}"
)

# Manuell gezeichnete Formen auswerten
gezeichnete_flecken = []
if canvas_result and canvas_result.json_data:
    for obj in canvas_result.json_data["objects"]:
        if obj["type"] == "polygon":
            points = obj["path"]
            if points and len(points) >= 3:
                x = [p[0] for p in points]
                y = [p[1] for p in points]
                area_px = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))
                if area_px >= min_canvas_area:
                    area_mm2 = round(area_px / (pixels_per_mm ** 2), 2)
                    gezeichnete_flecken.append({
                        "Datei": uploaded_file.name,
                        "Seite": 0,
                        "Fleckenzahl": 1,
                        "Fläche (mm²)": area_mm2,
                        "Quelle": "Manuell"
                    })

# Session-State aktualisieren
if gezeichnete_flecken:
    st.session_state["analyse_ergebnisse"].extend(gezeichnete_flecken)
    st.session_state["total_flecken"] += len(gezeichnete_flecken)
    st.session_state["total_pixel_area"] += sum(
        round(f["Fläche (mm²)"] * (pixels_per_mm ** 2), 2) for f in gezeichnete_flecken
    )
    st.success(f"🖌️ {len(gezeichnete_flecken)} Flecken direkt im Bild gezeichnet und übernommen!")

# 📊 Ergebnis-Tabelle
st.subheader("📊 Gesamt-Ergebnisse")
if st.session_state["analyse_ergebnisse"]:
    df = pd.DataFrame(st.session_state["analyse_ergebnisse"])
    st.dataframe(df)
    st.info(f"🔢 Gesamtflecken: {st.session_state['total_flecken']}, Gesamtfläche: {round(st.session_state['total_pixel_area'] / (pixels_per_mm ** 2), 2)} mm²")
else:
    st.warning("Keine Ergebnisse vorhanden – bitte Bild hochladen und analysieren!")
