import streamlit as st
from PIL import Image, ImageSequence
import numpy as np
import cv2
import pandas as pd
import io

# 🧠 Session-State Initialisierung
if "analyse_ergebnisse" not in st.session_state:
    st.session_state["analyse_ergebnisse"] = []

if "total_flecken" not in st.session_state:
    st.session_state["total_flecken"] = 0

if "total_pixel_area" not in st.session_state:
    st.session_state["total_pixel_area"] = 0

# 🧹 Reset-Button
if st.sidebar.button("🧹 Neues Experiment starten"):
    st.session_state["analyse_ergebnisse"] = []
    st.session_state["total_flecken"] = 0
    st.session_state["total_pixel_area"] = 0
    st.rerun()

# 🎨 Farbempfindlichkeit via Slider
h_min = st.sidebar.slider("Hue min", 0, 180, 0)
h_max = st.sidebar.slider("Hue max", 0, 180, 30)
s_min = st.sidebar.slider("Sättigung min", 0, 255, 70)
v_min = st.sidebar.slider("Helligkeit min", 0, 255, 50)
min_area = st.sidebar.slider("🟢 Minimale Fleckfläche (Pixel)", min_value=10, max_value=1000, value=50, step=10)

pixels_per_mm = 10  # oder passend zu deiner Skalierung

# 📤 Datei-Upload
uploaded_files = st.file_uploader("Bilder hochladen", type=["gif", "png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)

if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):
        st.header(f"🖼️ Datei: `{uploaded_file.name}`")

        try:
            image_pil = Image.open(uploaded_file)
            frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(image_pil)]
        except Exception as e:
            st.error(f"❌ Fehler beim Laden: {e}")
            continue

        for j, frame in enumerate(frames):
            if len(frames) > 1:
                st.subheader(f"📄 Seite {j + 1}")

            image_np = np.array(frame)
            hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            fleckenzahl = len(filtered)
            fläche_pixel = sum(cv2.contourArea(cnt) for cnt in filtered)
            fläche_mm2 = fläche_pixel / (pixels_per_mm ** 2)

            # 📈 Speicherung ins Session-State
            st.session_state["analyse_ergebnisse"].append({
                "Datei": uploaded_file.name,
                "Seite": j + 1,
                "Fleckenzahl": fleckenzahl,
                "Fläche (mm²)": round(fläche_mm2, 2)
            })

            st.success(f"🔴 Flecken: {fleckenzahl}")
            st.info(f"📐 Fläche: {fläche_pixel:.2f} Pixel² ({fläche_mm2:.2f} mm²)")
            output = image_np.copy()
            cv2.drawContours(output, filtered, -1, (0, 255, 0), 2)
            st.image(output, caption="Markierte Flecken", channels="RGB")

            # 🔢 Summierung
            st.session_state["total_flecken"] += fleckenzahl
            st.session_state["total_pixel_area"] += fläche_pixel
import pandas as pd

# In Tabelle umwandeln
df = pd.DataFrame(st.session_state["analyse_ergebnisse"])

# Schöne Anzeige im Streamlit-Dashboard
st.markdown("## 📊 Gesamttabelle aller Ergebnisse")
st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)

import io

excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
    df.to_excel(writer, index=False, sheet_name='Analyse')

st.download_button(
    label="📥 Tabelle als Excel herunterladen",
    data=excel_buffer.getvalue(),
    file_name="flecken_gesamttabelle.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
      

# 🌟 Gesamtergebnisse anzeigen
st.success(f"✅ Gesamte Fleckenanzahl: {st.session_state['total_flecken']}")
st.info(f"📐 Gesamtfläche (Pixel): {st.session_state['total_pixel_area']}")

# 📦 CSV-Export
df = pd.DataFrame(st.session_state["analyse_ergebnisse"])
csv_data = df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📄 Ergebnisse als CSV herunterladen",
    data=csv_data,
    file_name="flecken_analyse.csv",
    mime="text/csv"
)
