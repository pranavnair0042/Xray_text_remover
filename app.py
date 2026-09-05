import streamlit as st
import numpy as np
import cv2
from PIL import Image
import easyocr
import pandas as pd
from datetime import datetime
import time

st.set_page_config(
    page_title="RadClean | Medical Image Anonymizer",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Clinical Dashboard Feel
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1rem;
    }
    /* Restrict image heights so all 3 fit within standard viewports without scrolling */
    div[data-testid="stImage"] img {
        max-height: 52vh !important;
        width: auto !important;
        object-fit: contain !important;
        margin: 0 auto;
        display: block;
        border-radius: 6px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)


def detect_text_regions(image_np, confidence_threshold=0.15):
    # Initializes safely inside the UI lifecycle with Streamlit caching
    reader = load_ocr_reader()
    
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    orig_h, orig_w = gray.shape[:2]
    all_boxes, all_texts, all_confs = [], [], []

    rotations = [
        (0, None),
        (90, cv2.ROTATE_90_CLOCKWISE),
        (180, cv2.ROTATE_180),
        (270, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]

    for angle, rot_code in rotations:
        rot_img = cv2.rotate(enhanced, rot_code) if rot_code is not None else enhanced
        
        results = reader.readtext(
            rot_img,
            text_threshold=0.35,
            low_text=0.25,
            link_threshold=0.2,
            mag_ratio=1.5,
            slope_ths=0.2
        )

        for (bbox, text, prob) in results:
            if prob >= confidence_threshold:
                pts = np.array(bbox, dtype=np.float32)

                if angle == 90:
                    orig_pts = np.zeros_like(pts)
                    orig_pts[:, 0] = pts[:, 1]
                    orig_pts[:, 1] = orig_h - 1 - pts[:, 0]
                elif angle == 180:
                    orig_pts = np.zeros_like(pts)
                    orig_pts[:, 0] = orig_w - 1 - pts[:, 0]
                    orig_pts[:, 1] = orig_h - 1 - pts[:, 1]
                elif angle == 270:
                    orig_pts = np.zeros_like(pts)
                    orig_pts[:, 0] = orig_w - 1 - pts[:, 1]
                    orig_pts[:, 1] = pts[:, 0]
                else:
                    orig_pts = pts

                x_min = int(max(0, np.min(orig_pts[:, 0])))
                x_max = int(min(orig_w, np.max(orig_pts[:, 0])))
                y_min = int(max(0, np.min(orig_pts[:, 1])))
                y_max = int(min(orig_h, np.max(orig_pts[:, 1])))

                w = x_max - x_min
                h = y_max - y_min

                is_duplicate = False
                for bx, by, bw, bh in all_boxes:
                    if abs(bx - x_min) < 20 and abs(by - y_min) < 20:
                        is_duplicate = True
                        break

                if not is_duplicate and w > 0 and h > 0:
                    all_boxes.append((x_min, y_min, w, h))
                    all_texts.append(text)
                    all_confs.append(prob)

    return all_boxes, all_texts, all_confs

def generate_seamless_mask(image_gray, boxes, pad=6):
    h_img, w_img = image_gray.shape[:2]
    stroke_mask = np.zeros((h_img, w_img), dtype=np.uint8)

    for (x, y, w, h) in boxes:
        x1 = max(0, x - pad - 4)
        y1 = max(0, y - pad - 6)
        x2 = min(w_img, x + w + pad + 4)
        y2 = min(h_img, y + h + pad + 6)

        roi = image_gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if np.mean(roi[thresh == 255]) < np.mean(roi[thresh == 0]):
            thresh = cv2.bitwise_not(thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        stroke_mask[y1:y2, x1:x2] = np.maximum(stroke_mask[y1:y2, x1:x2], dilated)

    return stroke_mask

# --- Sidebar Controls ---
st.sidebar.image("https://img.icons8.com/fluency/96/x-ray.png", width=60)
st.sidebar.title("Control Panel")

uploaded_file = st.sidebar.file_uploader("Upload X-Ray File", type=["png", "jpg", "jpeg"])

with st.sidebar.expander("⚙️ Pipeline Tuning", expanded=False):
    ocr_thresh = st.slider("Detection Confidence", 0.05, 0.70, 0.15, 0.05)
    inpaint_method = st.selectbox("Algorithm", ["Navier-Stokes", "Telea"])
    radius = st.slider("Reconstruction Radius", 3, 15, 7)
    pad_val = st.slider("Dilation Padding", 2, 10, 5)

# --- Header Section ---
st.markdown('<div class="main-header">🩻 Clinical X-Ray De-Identification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated multi-angle character extraction and contour-preserving inpainting.</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    t0 = time.time()
    pil_img = Image.open(uploaded_file).convert("L")
    img_gray = np.array(pil_img)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    # Process Pipelines
    with st.spinner("Executing multi-angle character detection..."):
        boxes, texts, confs = detect_text_regions(img_rgb, ocr_thresh)

    mask = generate_seamless_mask(img_gray, boxes, pad=pad_val)
    
    flag = cv2.INPAINT_NS if inpaint_method == "Navier-Stokes" else cv2.INPAINT_TELEA
    cleaned_img = cv2.inpaint(img_gray, mask, radius, flag)
    elapsed_time = time.time() - t0

    # High-level Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Labels Detected", len(texts))
    m2.metric("Mean Confidence", f"{np.mean(confs):.1%}" if confs else "N/A")
    m3.metric("Mask Area (px)", f"{np.count_nonzero(mask):,}")
    m4.metric("Latency", f"{elapsed_time:.2f}s")

    st.write("")

    # Visual Inspection Tabs
    tab1, tab2, tab3 = st.tabs(["🔬 Side-by-Side Comparison", "🎯 Mask Verification", "📋 Metadata Audit"])

    with tab1:
        col_orig, col_det, col_clean = st.columns(3)
        with col_orig:
            st.caption("**Original Source**")
            st.image(img_gray, use_container_width=True, clamp=True)
        
        with col_det:
            st.caption("**Detection Overlay**")
            annotated = img_rgb.copy()
            for (x, y, w, h), t in zip(boxes, texts):
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (239, 68, 68), 2)
                cv2.putText(annotated, t, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (239, 68, 68), 1)
            st.image(annotated, use_container_width=True)

        with col_clean:
            st.caption("**Anonymized Result**")
            st.image(cleaned_img, use_container_width=True, clamp=True)

    with tab2:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.caption("**Generated Stroke-Level Mask**")
            st.image(mask, use_container_width=True, clamp=True)
        with col_m2:
            st.caption("**Mask Heatmap Blend**")
            heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_rgb, 0.7, heatmap, 0.3, 0)
            st.image(overlay, use_container_width=True)

    with tab3:
        if texts:
            df = pd.DataFrame({
                "Annotation String": texts,
                "Confidence": [f"{c:.2%}" for c in confs],
                "Bounding Box": [f"x:{b[0]}, y:{b[1]}, w:{b[2]}, h:{b[3]}" for b in boxes],
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Audit Log (CSV)", csv_bytes, "audit_log.csv", "text/csv")
        else:
            st.info("No clinical text or identifiers found.")

    # Bottom Download Tray
    st.divider()
    success, buffer = cv2.imencode(".png", cleaned_img)
    if success:
        st.download_button(
            label="⬇️ Export Fully Anonymized X-Ray (PNG)",
            data=buffer.tobytes(),
            file_name=f"clean_{uploaded_file.name}",
            mime="image/png",
            type="primary"
        )
else:
    st.info("Upload an image in the left sidebar to start.")
