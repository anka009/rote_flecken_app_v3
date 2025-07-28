import streamlit as st
import cv2
import numpy as np
import pandas as pd
import io
from sklearn.cluster import DBSCAN

# 🔧 Parameter
merge_radius = 50
pixels_per_mm = 10

st.title("🩸 Fleckenerkennung & Analyse")

uploaded_file = st.file_uploader("📂 Bild hochladen", type=["jpg", "png", "jpeg", "tif", "bmp"])
if uploaded_file:
    image_bytes = uploaded_file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    st.image(image_np, caption="📷 Originalbild", channels="RGB")

    # 🔍 Vorverarbeitung
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]

    # 🧠 Zentren berechnen für Clustering
    centers = []
    for cnt in filtered:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append([cx, cy])

    if centers:
        centers_np = np.array(centers)
        db = DBSCAN(eps=merge_radius, min_samples=1).fit(centers_np)
        labels = db.labels_

        clustered = {}
        for label, cnt in zip(labels, filtered):
            clustered.setdefault(label, []).append(cnt)

        merged_contours = [cv2.convexHull(np.vstack(group)) for group in clustered.values()]

        # 🟦 Farbig eingerahmte Gruppendarstellung
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
        ]
        grouped_image = image_np.copy()
        for idx, (label, group) in enumerate(clustered.items()):
            color = colors[idx % len(colors)]
            contour = cv2.convexHull(np.vstack(group))
            cv2.drawContours(grouped_image, [contour], -1, color, 3)

            # 🏷️ Gruppennummer
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(grouped_image, f"Gruppe {idx+1}", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        st.image(grouped_image, caption="🖼️ Gruppierte Flecken farblich eingerahmt", channels="RGB")

        output_clustered = image_np.copy()
        cv2.drawContours(output_clustered, merged_contours, -1, (0, 255, 255), 2)
        st.image(output_clustered, caption="🟡 Gruppierte Flecken (Konvex)", channels="RGB")

        st.success(f"🧬 Gruppen: {len(merged_contours)}")
    else:
        st.warning("⚠️ Keine Zentren zum Clustern gefunden.")

    # 🟢 Einzel-Fleckdarstellung
    output_marked = image_np.copy()
    cv2.drawContours(output_marked, filtered, -1, (0, 255, 0), 2)
    st.image(output_marked, caption="🟢 Markierte Flecken", channels="RGB")

    # 📊 Statistik
    fleckenzahl = len(filtered)
    fläche_pixel = sum(cv2.contourArea(cnt) for cnt in filtered)
    fläche_mm2 = fläche_pixel / (pixels_per_mm ** 2)

    st.success(f"🔴 Flecken: {fleckenzahl}")
    st.info(f"📏 Fläche: {fläche_pixel:.2f} px² ({fläche_mm2:.2f} mm²)")

    # 🧾 Ergebnis speichern
    st.session_state.setdefault("analyse_ergebnisse", [])
    st.session_state.setdefault("total_flecken", 0)
    st.session_state.setdefault("total_pixel_area", 0)

    st.session_state["analyse_ergebnisse"].append({
        "Datei": uploaded_file.name,
        "Fleckenzahl": fleckenzahl,
        "Fläche (mm²)": round(fläche_mm2, 2)
    })

    st.session_state["total_flecken"] += fleckenzahl
    st.session_state["total_pixel_area"] += fläche_pixel

    # 📥 Gesamttabelle & Downloads
    df = pd.DataFrame(st.session_state["analyse_ergebnisse"])
    st.markdown("## 📊 Gesamttabelle")
    st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)

    # Excel-Download
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Analyse")
    st.download_button("📥 Excel herunterladen",
                       data=excel_buffer.getvalue(),
                       file_name="flecken_gesamttabelle.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # CSV-Download
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("📄 CSV herunterladen",
                       data=csv_data,
                       file_name="flecken_analyse.csv",
                       mime="text/csv")

    st.success(f"✅ Gesamtanzahl Flecken: {st.session_state['total_flecken']}")
    st.info(f"📐 Gesamtfläche (px): {st.session_state['total_pixel_area']:.2f}")
