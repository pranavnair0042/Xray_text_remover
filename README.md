# RadClean: Clinical X-Ray De-Identification & Text Anonymization

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://xraytextremover-vcvlxx3d28rfsss27ykrm3.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Headless-green.svg)](https://opencv.org/)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-PyTorch-orange.svg)](https://github.com/JaidedAI/EasyOCR)

An end-to-end computer vision web application designed to detect, extract, and completely inpaint burned-in clinical text annotations, patient identifiers, and orientation markers on medical X-rays without destroying background diagnostic anatomy.

🔗 **Live Deployment:** [xraytextremover-vcvlxx3d28rfsss27ykrm3.streamlit.app](https://xraytextremover-vcvlxx3d28rfsss27ykrm3.streamlit.app/)

---

## Key Highlights
* **Multi-Angle Detection:** Uses rotational coordinate mapping across 0°, 90°, 180°, and 270° orientations to capture tricky vertical and inverted hospital markers.
* **Stroke-Level Thresholding:** Employs Otsu segmentation with adaptive morphological dilation to target character ink exclusively, avoiding destructive rectangular cutout boxes.
* **Curvature-Preserving Inpainting:** Uses Navier-Stokes boundary propagation to seamlessly fill background grayscale gradients without leaving blurry silhouettes.
* **Interactive Web Interface:** Built with Streamlit, including detection confidence controls, mask visualization heatmaps, and downloadable CSV audit trails.

---

## Technical Pipeline
1. **Contrast Equalization:** Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied to enhance faint, low-contrast text strokes.
2. **Text Localization:** Multi-angle scanning using CRAFT (via EasyOCR) produces text bounding coordinates regardless of scan orientation.
3. **Refined Mask Generation:** Otsu thresholding extracts stroke contours within detected bounding boxes, followed by elliptical kernel dilation.
4. **Boundary Reconstruction:** Fluid-dynamics-based partial differential equations (Navier-Stokes) compute smooth color and texture propagation from surrounding pixels into the masked voids.

---

## Local Setup & Installation

If you want to run this application locally:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/pranavnair0042/Xray_text_remover.git](https://github.com/pranavnair0042/Xray_text_remover.git)
   cd Xray_text_remover