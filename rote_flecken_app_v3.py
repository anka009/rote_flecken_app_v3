import streamlit as st
from PIL import Image, ImageSequence
import numpy as np
import cv2
import pandas as pd

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

# 🎯 Parameter
min_area = 50
max_area = 2000
pixels_per_mm = 10

# 📂 Upload
uploaded_file = st.file_uploader("Bild hochladen", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)

    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    alle_flecken = []
    statistik = {}

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

    # 🖼️ Bild ausgeben
    overlay = image_np.copy()
    for cnt, farbe in alle_flecken:
        cv2.drawContours(overlay, [cnt], -1, farb_codes[farbe], -1)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(overlay, farbe, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    result = cv2.addWeighted(overlay, 0.4, image_np, 0.6, 0)
    st.image(result, caption="🎨 Flecken nach Farben")

    # 📊 Tabelle
    df = pd.DataFrame.from_dict(statistik, orient='index').reset_index()
    df.columns = ["Farbe", "Fleckenzahl", "Gesamtfläche (px²)"]
    st.dataframe(df)

    # 📥 Download (CSV)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Statistik herunterladen", data=csv, file_name="flecken_statistik.csv", mime="text/csv")
