import streamlit as st
import cv2
import numpy as np
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

# 🎛️ Farbfilter & Schwellenwerte
h_min = st.sidebar.slider("Hue min", 0, 180, 0)
h_max = st.sidebar.slider("Hue max", 0, 180, 30)
s_min = st.sidebar.slider("Sättigung min", 0, 255, 70)
v_min = st.sidebar.slider("Helligkeit min", 0, 255, 50)
min_area = st.sidebar.slider("🟢 Minimale Fleckfläche (Pixel)", 10, 1000, 50, 10)
min_canvas_area = st.sidebar.slider("🖍️ Mindestfläche für manuelle Flecken (px²)", 10, 1000, 50, 10)
pixels_per_mm = 10  # Skalierung anpassen

# 📁 Datei-Upload
uploaded_files = st.file_uploader("📥 Bilder hochladen", type=["gif", "png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)

# 💡 Canvas-Abschnitt nur für erste Datei (optional erweiterbar auf alle)
if uploaded_files:
    uploaded_file = uploaded_files[0]
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="📷 Vorschau", use_column_width=True)

    st.subheader("🖌️ Manuelle Fleckenzeichnung")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color="#ff0000",
        background_color="#ffffff",
        height=300,
        width=500,
        drawing_mode="polygon",
        key=f"canvas_{uploaded_file.name}"
    )

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

    if gezeichnete_flecken:
        st.session_state["analyse_ergebnisse"].extend(gezeichnete_flecken)
        st.session_state["total_flecken"] += len(gezeichnete_flecken)
        st.session_state["total_pixel_area"] += sum(
            round(fleck["Fläche (mm²)"] * (pixels_per_mm ** 2), 2) for fleck in gezeichnete_flecken
        )
        st.success(f"🤝 {len(gezeichnete_flecken)} manuelle Flecken hinzugefügt!")

# 🔍 Automatische Analyse aller Dateien + Frames
if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        frames = [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(image)]

        for j, frame in enumerate(frames):
            if len(frames) > 1:
                st.subheader(f"📄 Seite {j + 1} von Datei: {uploaded_file.name}")

            image_np = np.array(frame)
            hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, 255, 255))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            fleckenzahl = 0
            fläche_pixel = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= min_area:
                    fleckenzahl += 1
                    fläche_pixel += area

            fläche_mm2 = fläche_pixel / (pixels_per_mm ** 2)
            output = image_np.copy()
            cv2.drawContours(output, contours, -1, (255, 0, 0), 2)
            st.image(output, caption=f"🔬 Analyse Seite {j + 1}", use_column_width=True)

            st.session_state["analyse_ergebnisse"].append({
                "Datei": uploaded_file.name,
                "Seite": j + 1,
                "Fleckenzahl": fleckenzahl,
                "Fläche (mm²)": round(fläche_mm2, 2),
                "Quelle": "Automatisch"
            })

            st.session_state["total_flecken"] += fleckenzahl
            st.session_state["total_pixel_area"] += fläche_pixel

# 📊 Ergebnis-Tabelle
st.subheader("📊 Gesamt-Ergebnisse")
if st.session_state["analyse_ergebnisse"]:
    df = pd.DataFrame(st.session_state["analyse_ergebnisse"])
    st.dataframe(df)
    st.info(f"🔢 Gesamtflecken: {st.session_state['total_flecken']}, Gesamtfläche: {round(st.session_state['total_pixel_area'] / (pixels_per_mm ** 2), 2)} mm²")
else:
    st.warning("Keine Ergebnisse vorhanden – bitte Bild hochladen und analysieren!")
