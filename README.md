# 🩻 RadClean: Medical X-Ray Text De-Identification & Anonymization

An end-to-end computer vision pipeline that automatically detects and removes burned-in patient identifiers, orientation tags, and clinical annotations from medical X-ray images while preserving underlying diagnostic tissue textures.

## 🚀 Key Features
- **Multi-Angle Detection:** Rotational OCR handling (0°, 90°, 180°, 270°) to reliably catch vertical and flipped medical markers.
- **Stroke-Level Thresholding:** Uses adaptive Otsu thresholding and morphological dilation to inpaint only the character strokes rather than destructive rectangular boxes.
- **Contour-Preserving Inpainting:** Solves background partial differential equations via Navier-Stokes boundary propagation for seamless restoration.
- **Interactive Dashboard:** Built with Streamlit, featuring confidence adjustments, mask inspection heatmaps, and audit CSV exports.

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **OCR Engine:** EasyOCR (CRAFT detector)
- **Computer Vision:** OpenCV (`cv2`)
- **Dashboard / UI:** Streamlit

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
   cd <your-repo-name>