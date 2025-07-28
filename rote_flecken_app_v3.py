import streamlit as st
from PIL import Image, ImageSequence
import numpy as np
import cv2
import pandas as pd
import io
import random

# 🧠 Session-State Initialisierung
if "analyse_ergebnisse" not in st.session_state:
    st.session_state["analyse_ergebnisse"] = []
if "total_flecken" not in st.session_state:
    st.session_state["total_flecken"] = 0
if "total_pixel_area" not in st.session_state:
    st.session_state["total_pixel_area"] = 0
if "upload_key" not in st.session_state:
    st.session_state["upload_key"] = f"upload_key_{random.randint(0, 100_000)}"

# 🔁 Reset-Button
if st.sidebar.button("🔁 Alles zurücksetzen"):
    for key in list(st.session_state.keys()):
        st.session_state.pop(key)
    st.session_state["upload_key"] = f"upload_key_{random.randint(0, 100_000)}"
    st.rerun()

# 🎛️ Farbempfindlichkeit
st.sidebar.markdown("## 🎨 Flecken-Farbparameter")
h_min = st.sidebar.slider("Hue min", 0, 180, 0)
h_max = st.sidebar.slider("Hue max", 0, 180, 30)
s_min = st.sidebar.slider("Sättigung min", 0, 255, 70)
v_min = st.sidebar.slider("Helligkeit min", 0, 255, 50)
min_area = st.sidebar.slider("🟩 Minimale Fleckfläche (Pixel)", 10, 5000, 50, 10)
max_area = st.sidebar.slider("🟥 Maximale Fleckfläche (Pixel)", 100, 10000, 2000, 50)

pixels_per_mm = 10  # Anpassbar je nach Skalierung

# ⚡ Kontrast-Einstellungen
st.sidebar.markdown("## ⚡ Kontrastverstärkung")
enable_clahe = st.sidebar.checkbox("Automatischer CLAHE-Boost", value=True)
contrast = st.sidebar.slider("Manueller Kontrast", 0.5, 2.5, 1.0, 0.1)
brightness = st.sidebar.slider("Manuelle Helligkeit", 0.5, 2.0, 1.0, 0.1)
st.sidebar.markdown("## 🧭 Gruppierungsparameter")
merge_radius = st.sidebar.slider("🔗 Gruppierungs-Radius (Pixel)", 0, 1000, 200, 10)

# 📁 Datei-Upload
uploaded_files = st.file_uploader(
    "📁 Bilder hochladen",
    type=["gif", "png", "jpg", "jpeg", "tif", "tiff"],
    accept_multiple_files=True,
    key=st.session_state["upload_key"]
)

# 🔬 Analyse
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

            # ✨ Manueller Kontrast & Helligkeit
            image_np = np.clip(contrast * image_np + (brightness - 1) * 255, 0, 255).astype(np.uint8)

            # 🔥 Optional CLAHE-Boost
            if enable_clahe:
                lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_boosted = clahe.apply(l)
                lab_boosted = cv2.merge((l_boosted, a, b))
                image_np = cv2.cvtColor(lab_boosted, cv2.COLOR_LAB2RGB)

            hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            filtered = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
            centers = []
filtered = []  # Initialisiere außerhalb, damit sie immer existiert

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
            # 🧪 Bildverarbeitung wie bisher
            # ...
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            filtered = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
            if filtered:
                centers = []
                for cnt in filtered:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        centers.append([cx, cy])
                # Hier weiter mit Clustering etc...
            else:
                st.warning("⚠️ Keine Konturen zum Verarbeiten gefunden.")


for cnt in filtered:
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append([cx, cy])
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=merge_radius, min_samples=1).fit(centers)
labels = db.labels_
clustered = {}
for label, cnt in zip(labels, filtered):
    clustered.setdefault(label, []).append(cnt)

merged_contours = [cv2.convexHull(np.vstack(group)) for group in clustered.values()]
output = image_np.copy()
cv2.drawContours(output, merged_contours, -1, (0, 255, 255), 2)

st.image(output, caption="🟠 Gruppierte Flecken", channels="RGB")
st.success(f"🧮 Gruppierte Fleckenanzahl: {len(merged_contours)}")

fleckenzahl = len(filtered)
fläche_pixel = sum(cv2.contourArea(cnt) for cnt in filtered)
fläche_mm2 = fläche_pixel / (pixels_per_mm ** 2)

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

st.session_state["total_flecken"] += fleckenzahl
st.session_state["total_pixel_area"] += fläche_pixel

# 📊 Gesamttabelle
df = pd.DataFrame(st.session_state["analyse_ergebnisse"])
st.markdown("## 📈 Gesamttabelle aller Ergebnisse")
st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)

# 📥 Excel-Download
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    df.to_excel(writer, index=False, sheet_name="Analyse")
st.download_button(
    label="📥 Excel herunterladen",
    data=excel_buffer.getvalue(),
    file_name="flecken_gesamttabelle.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# 📄 CSV-Export
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📄 CSV herunterladen",
    data=csv_data,
    file_name="flecken_analyse.csv",
    mime="text/csv"
)

# 🌟 Gesamtergebnisse
st.success(f"✅ Gesamte Fleckenanzahl: {st.session_state['total_flecken']}")
st.info(f"📐 Gesamtfläche (Pixel): {st.session_state['total_pixel_area']}")
