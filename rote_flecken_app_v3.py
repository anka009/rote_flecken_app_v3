import streamlit as st
from PIL import Image, ImageSequence
import numpy as np
import cv2
import pandas as pd
import io
import random

# 🧠 Session-State Initialisierung
for key in ["analyse_ergebnisse"]:
    if key not in st.session_state:
        st.session_state[key] = []

# 📘 HSV-Farbgruppen definieren
farbebereiche = {
    "Rot": [(0, 70, 50), (10, 255, 255)],
    "Grün": [(35, 70, 50), (85, 255, 255)],
    "Blau": [(100, 70, 50), (130, 255, 255)],
    "Gelb": [(20, 70, 50), (40, 255, 255)]
}
farb_codes = {
    "Rot": (0, 0, 255),
    "Grün": (0, 255, 0),
    "Blau": (255, 0, 0),
    "Gelb": (0, 255, 255)
}

# 🔧 Parameter
min_area = 50
max_area = 5000
pixels_per_mm = 10

# 📂 Upload
uploaded_files = st.file_uploader("📁 Bild-Upload", type=["gif", "png", "jpg", "jpeg", "tif", "tiff"],
                                   accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
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

            alle_flecken = []
            statistik = {}

            # 🔍 Durchlaufe Farbgruppen
            for farbe, (lower_hsv, upper_hsv) in farbebereiche.items():
                mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                gefiltert = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

                for cnt in gefiltert:
                    alle_flecken.append((cnt, farbe))
                    fläche = cv2.contourArea(cnt)
                    statistik.setdefault(farbe, {"Fleckenzahl": 0, "Gesamtfläche (px²)": 0})
                    statistik[farbe]["Fleckenzahl"] += 1
                    statistik[farbe]["Gesamtfläche (px²)"] += fläche

            # 🎨 Darstellung mit Füllung + Label
            overlay = image_np.copy()
            for cnt, farbe in alle_flecken:
                cv2.drawContours(overlay, [cnt], -1, farb_codes[farbe], -1)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(overlay, farbe, (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            result_img = cv2.addWeighted(overlay, 0.4, image_np, 0.6, 0)
            st.image(result_img, caption="🎨 Flecken nach Farben erkannt", channels="RGB")

            # 📊 Tabelle pro Farbe
            df_farben = pd.DataFrame.from_dict(statistik, orient="index").reset_index()
            df_farben.columns = ["Farbe", "Fleckenzahl", "Gesamtfläche (px²)"]
            df_farben["Gesamtfläche (mm²)"] = df_farben["Gesamtfläche (px²)"] / (pixels_per_mm ** 2)

            st.markdown("### 📊 Statistik nach Farbgruppen")
            st.dataframe(df_farben)

            st.session_state["analyse_ergebnisse"].append({
                "Datei": uploaded_file.name,
                "Seite": j + 1,
                "Flecken gesamt": len(alle_flecken)
            })

    # 📦 Gesamttabelle + Download
    df_summary = pd.DataFrame(st.session_state["analyse_ergebnisse"])
    st.markdown("## 🧾 Zusammenfassung")
    st.dataframe(df_summary)

    # CSV-Download
    csv_data = df_summary.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV herunterladen",
                       data=csv_data,
                       file_name="flecken_uebersicht.csv",
                       mime="text/csv")
