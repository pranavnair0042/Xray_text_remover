import streamlit as st
import mysql.connector
from mysql.connector import Error
import pandas as pd
from PIL import Image, ImageFilter
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis
from skimage.segmentation import felzenszwalb
from ultralytics import YOLO
import io
import cv2
import numpy as np
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = np.array(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_eq = cv2.equalizeHist(img_gray)
    img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    alpha = 1.5
    beta = 0
    img_contrast = cv2.convertScaleAbs(img_clahe, alpha=alpha, beta=beta)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img_sharp = cv2.filter2D(img_contrast, -1, kernel)
    img_denoised = cv2.fastNlMeansDenoising(img_sharp, h=10, templateWindowSize=7, searchWindowSize=21)
    blurred_img = cv2.GaussianBlur(img_denoised, (5, 5), 0)
    laplacian = cv2.Laplacian(blurred_img, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    processed_images = [img_gray, img_eq, img_blur, img_clahe, img_contrast, img_sharp, img_denoised, laplacian]
    titles = ['Grayscale', 'Equalized', 'Blurred', 'CLAHE', 'Contrast', 'Sharpened', 'Denoised', 'Laplacian']
    return processed_images, titles

def convert_to_bytes(img):
    is_success, buffer = cv2.imencode(".png", img)
    if is_success:
        return buffer.tobytes()
    return None


# Database connection function
def create_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='Triumph@9203',  # Replace with your MySQL password
            database='mydb'  # Replace with your database name
        )
        st.success("Connection to MySQL DB successful")
    except Error as e:
        st.error(f"The error '{e}' occurred")
    return connection

# Table creation function
def create_table():
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS defect_data2 (
        id INT AUTO_INCREMENT PRIMARY KEY,
        defect_no VARCHAR(255) NOT NULL,
        satellite VARCHAR(255) NOT NULL,
        component_name VARCHAR(255) NOT NULL,
        component_id VARCHAR(255) NOT NULL,
        defects_detected VARCHAR(255) NOT NULL,
        date DATE NOT NULL
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS defect_info (
        id INT AUTO_INCREMENT PRIMARY KEY,
        defect_no VARCHAR(255) NOT NULL,
        defect_types VARCHAR(255) NOT NULL,
        features VARCHAR(255) NOT NULL,
        user_remarks VARCHAR(255) NOT NULL,
        accept_reject VARCHAR(255) NOT NULL
    );
    """)
    connection.commit()
    cursor.close()
    connection.close()

# Data insertion functions
def insert_data(defect_no, satellite, component_name, component_id, defects_detected, date):
    connection = create_connection()
    cursor = connection.cursor()
    query = """
    INSERT INTO defect_data2 (defect_no, satellite, component_name, component_id, defects_detected, date)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    try:
        cursor.execute(query, (defect_no, satellite, component_name, component_id, defects_detected, date))
        connection.commit()
        st.success("Data inserted successfully")
    except Error as e:
        st.error(f"The error '{e}' occurred")
    cursor.close()
    connection.close() 

def insert_defect_info(defect_types, features, user_remarks, accept_reject):
    connection = create_connection()
    cursor = connection.cursor()

    latest_entry = fetch_latest_entry()
    defect_no = latest_entry['defect_no'] if latest_entry else None
    query = """
    INSERT INTO defect_info (defect_no, defect_types, features, user_remarks, accept_reject)
    VALUES (%s, %s, %s, %s, %s)
    """
    try:
        cursor.execute(query, (defect_no, defect_types, features, user_remarks, accept_reject))
        connection.commit()
        st.success("Defect information inserted successfully")
    except Error as e:
        st.error(f"The error '{e}' occurred")
    cursor.close()
    connection.close()

# Fetch latest entry function
def fetch_latest_entry():
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    query = "SELECT * FROM defect_data2 ORDER BY id DESC LIMIT 1"
    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    connection.close()
    return result

# Fetch all entries functions
def fetch_all_entries():
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    query = "SELECT * FROM defect_data2"
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    return result

def fetch_all_defect_info():
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    query = "SELECT * FROM defect_info"
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    return result

# Function to load model
@st.cache_resource
def load_model(path):
    model = YOLO(path)
    return model

# Function to predict
def predict(image, model):
    results = model(image)
    return results

# Function to draw bounding boxes and segment regions
def draw_boxes_and_segment(image, results, model):
    segmented_images = []
    defects_detected = []
    for result in results:
        for obj in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = obj[:6]
            label = model.names[int(cls)]
            defects_detected.append(label)  # Store the label
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Extract the region of interest (ROI)
            roi = image[int(y1):int(y2), int(x1):int(x2)]
            segmented_images.append(roi)
            
    return image, segmented_images, defects_detected

# Noise functions
def add_salt_and_pepper(image_array, amount):
    row, col, _ = image_array.shape
    s_vs_p = 0.5
    out = np.copy(image_array)
    num_salt = np.ceil(amount * image_array.size * s_vs_p)
    num_pepper = np.ceil(amount * image_array.size * (1.0 - s_vs_p))

    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_array.shape]
    out[coords[0], coords[1], :] = 1

    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_array.shape]
    out[coords[0], coords[1], :] = 0
    return out

def add_gaussian_noise(image_array, mean=0, var=0.1):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image_array.shape)
    noisy_image = image_array + gauss
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

# Filter functions
def apply_filter(image, filter_type):
    return image.filter(filter_type)

def apply_canny(image):
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def apply_sobel(image):
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    return np.uint8(sobel)

def apply_felzenszwalb(image, scale=255):
    try:
        # Convert image to grayscale
        img = np.array(image.convert('L'))  # Convert to grayscale
        
        # Ensure the image data is in the correct range [0, 255] and type uint8
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Apply Felzenszwalb segmentation
        segments = felzenszwalb(img, scale=scale)
        
        # Convert segment labels to a color image
        segmented_img = np.zeros_like(img)
        unique_segments = np.unique(segments)
        for i, seg in enumerate(unique_segments):
            segmented_img[segments == seg] = i * (255 // len(unique_segments))
        
        # Convert to PIL Image
        segmented_img = Image.fromarray(segmented_img)
        
        return segmented_img
    except Exception as e:
        st.error(f"Error applying Felzenszwalb segmentation: {e}")
        return None

def apply_gaussian_adaptive_thresholding(image):
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

# Feature extraction functions
def extract_features(roi):
    if len(np.array(roi).shape) == 3:
        gray_roi = cv2.cvtColor(np.array(roi), cv2.COLOR_BGR2GRAY)
    else:
        gray_roi = np.array(roi)
    glcm = graycomatrix(gray_roi, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    entropy = shannon_entropy(gray_roi)
    skewness = skew(gray_roi.flatten())
    kurt = kurtosis(gray_roi.flatten())
    return {'contrast': contrast, 'correlation': correlation, 'energy': energy, 'homogeneity': homogeneity, 'entropy': entropy, 'skewness': skewness, 'kurtosis': kurt}

# Streamlit UI section for segmentation
st.title("Satellite X-ray Defect Detection")


# Navigation options using radio buttons
options = ["Image Processing", "X-ray Evaluation Results", "X-ray Image Defect Detection – Screen 1", "Defect Information"]
choice = st.radio("Choose an option", options)


# Image upload widget
if choice == "Image Processing":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        image_cv = np.array(image.convert('RGB'))

        processed_images, titles = preprocess_image(image_cv)

        # Display images in rows of 3
        num_images = len(processed_images)
        cols = 3  # Number of images per row

        st.write("### Processed Images")
        num_rows = (num_images + cols - 1) // cols  # Calculate number of rows needed

        for i in range(num_rows):
            cols_in_row = st.columns(cols)
            for j in range(cols):
                index = i * cols + j
                if index < num_images:
                    with cols_in_row[j]:
                        st.image(processed_images[index], use_container_width=True, clamp=True, channels="GRAY")
                        img_bytes = convert_to_bytes(processed_images[index])
                        if img_bytes is not None:
                            st.download_button(
                                label=f"Download {titles[index]} Image",
                                data=img_bytes,
                                file_name=f"{titles[index]}.png",
                                mime="image/png"
                            )

elif choice == "X-ray Evaluation Results":
    st.header("X-ray Evaluation Results")
    st.subheader("Latest Entry:")
    latest_entry = fetch_latest_entry()
    st.write(latest_entry)
    st.subheader("All Entries:")
    all_entries = fetch_all_entries()
    df = pd.DataFrame(all_entries)
    st.dataframe(df)

elif choice == "X-ray Image Defect Detection – Screen 1":
    st.title("X-ray Image Defect Detection – Screen 1")

    if 'satellite' not in st.session_state:
        st.session_state.satellite = ''
    if 'component_name' not in st.session_state:
        st.session_state.component_name = ''

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.satellite = st.text_input("Satellite", st.session_state.satellite)
   
    with col2:
        st.session_state.component_name = st.text_input("Component Name", st.session_state.component_name)

    col3, col4 = st.columns([3, 1])
    with col3:
        uploaded_file = st.file_uploader("Upload X-ray Image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            st.image(image, caption='Uploaded X-ray Image', use_container_width=True)
            aspect_ratio = st.radio("Choose aspect ratio for cropping", ["Free", "1:1", "16:9", "4:3", "2:3"])
            if aspect_ratio == "Free":
                image = st_cropper(image, realtime_update=True)
            else:
                aspect_ratio_dict = {
                    "1:1": (1, 1),
                    "16:9": (16, 9),
                    "4:3": (4, 3),
                    "2:3": (2, 3)
                }
                image = st_cropper(image, aspect_ratio=aspect_ratio_dict[aspect_ratio], realtime_update=True)
                image_array = np.array(image)
        
    with col4:
        st.subheader("Noise Options")
        noise_option = st.radio("Select Noise Option", ["None", "Salt & Pepper", "Gaussian"])
        if noise_option == "Salt & Pepper" and uploaded_file is not None:
            amount = st.slider("Noise Amount", 0.01, 0.1, 0.05)
            noisy_image = add_salt_and_pepper(image_array, amount)
            st.image(noisy_image, caption='Image with Salt & Pepper Noise', use_container_width=True)
        elif noise_option == "Gaussian" and uploaded_file is not None:
            mean = st.slider("Gaussian Mean", -1.0, 1.0, 0.0)
            var = st.slider("Gaussian Variance", 0.01, 0.1, 0.05)
            noisy_image = add_gaussian_noise(image_array, mean, var)
            st.image(noisy_image, caption='Image with Gaussian Noise', use_container_width=True)

        st.subheader("Filter Options")
        filter_option = st.radio("Select Filter Option", ["None", "Average", "Median", "Min", "Max"])
        if filter_option != "None" and uploaded_file is not None:
            if filter_option == "Average":
                filtered_image = apply_filter(image, ImageFilter.BLUR)
            elif filter_option == "Median":
                filtered_image = apply_filter(image, ImageFilter.MedianFilter(size=3))
            elif filter_option == "Min":
                filtered_image = apply_filter(image, ImageFilter.MinFilter(size=3))
            elif filter_option == "Max":
                filtered_image = apply_filter(image, ImageFilter.MaxFilter(size=3))
            st.image(filtered_image, caption=f'Image with {filter_option} Filter', use_containern_width=True)

        st.subheader("Segmentation Techniques")
        segmentation_technique = st.radio("Select Segmentation Technique", ["None", "Canny", "Sobel", "Felzenwalb", "Gaussian Adaptive"])
        if segmentation_technique != "None" and uploaded_file is not None:
            if segmentation_technique == "Canny":
                segmented_image = apply_canny(image)
            elif segmentation_technique == "Sobel":
                segmented_image = apply_sobel(image)
            elif segmentation_technique == "Felzenwalb":
                segmented_image = apply_felzenszwalb(image)
            elif segmentation_technique == "Gaussian Adaptive":
                segmented_image = apply_gaussian_adaptive_thresholding(image)
            else:
                segmented_image = filtered_image
            st.image(segmented_image, caption=f'Image with {segmentation_technique} Segmentation', use_container_width=True, clamp=True)
    
    # Initialize session state for inputs and outputs if they don't exist
    if 'defect_no' not in st.session_state:
        st.session_state.defect_no = ''
   
    if 'features' not in st.session_state:
        st.session_state.features = ''
    if 'user_remarks' not in st.session_state:
        st.session_state.user_remarks = ''
    if 'accept_reject' not in st.session_state:
        st.session_state.accept_reject = 'Accept'

    # Initialize session state for detected data if they don't exist
    if 'detected_image' not in st.session_state:
        st.session_state.detected_image = None
    if 'rois' not in st.session_state:
        st.session_state.rois = []
    if 'defect_labels' not in st.session_state:
        st.session_state.defect_labels = []

    # Layout for buttons
    col1, col2, col3 = st.columns([1, 1, 1])

    # Button to detect defects
    with col1:
        detect_defects = st.button("Detect Defects")

    # Button to accept defects
    with col2:
        accept_defects = st.button("Accept")

    # Button to reject defects
    with col3:
        reject_defects = st.button("Reject")

    # Ensure uploaded_file is not None before proceeding
    if detect_defects and uploaded_file is not None:
        # Load the model and perform prediction
        model = load_model('bestlopandporosity.pt')
        results = predict(np.array(image), model)
        
        # Store the results in session state
        st.session_state.detected_image, st.session_state.rois, st.session_state.defect_labels = draw_boxes_and_segment(np.array(image), results, model)
        
        # Display the detected image
        st.image(st.session_state.detected_image, caption="Detected Defects", use_container_width=True)
        
        # Display detected objects
        st.write("Detected Objects:")
        for result in results:
            for obj in result.boxes.data.tolist():
                st.write(f'Object: {model.names[int(obj[5])]}, F1-Score: {obj[4]:.2f}')

    # Handle Accept button functionality
    if accept_defects and st.session_state.detected_image is not None:
        st.write("Defects accepted.")
        # You can add logic here to save the accepted data to a database or file if needed.

    # Handle Reject button functionality
    if reject_defects:
        st.write("Defects rejected. Clearing detected data.")
        st.session_state.detected_image = None
        st.session_state.rois = []
        st.session_state.defect_labels = []
        # Clear displayed images and other defect-related outputs
        st.experimental_rerun()

    # Display the stored detected image and ROIs if available
    if st.session_state.detected_image is not None:
        st.image(st.session_state.detected_image, caption="Detected Defects", use_container_width=True)
        for i, roi in enumerate(st.session_state.rois):
            st.subheader(f"Region {i+1}")
            st.image(roi, caption=f"Region {i+1}", use_container_width=True)

            features = extract_features(roi)
            st.write("Extracted Features:", features)

    # Data insertion section
    st.session_state.defect_no = st.text_input("Defect No", st.session_state.defect_no)
    
    component_id = uploaded_file.name if uploaded_file is not None else "No File"
    defects_detected = st.session_state.defect_labels[i] if st.session_state.defect_labels else "None"
    date = st.date_input("Date")

    # Insert data button
    if st.button("Insert Data"):
        insert_data(st.session_state.defect_no, st.session_state.satellite, st.session_state.component_name, component_id, defects_detected, date)

    # Additional defect information
    st.session_state.features = st.text_area("Features", st.session_state.features)
    st.session_state.user_remarks = st.text_area("User Remarks", st.session_state.user_remarks)
    st.session_state.accept_reject = st.radio("Accept/Reject", ["Accept", "Reject"], index=["Accept", "Reject"].index(st.session_state.accept_reject))

    # Insert defect information button
    if st.button("Insert Defect Information"):
        insert_defect_info(defects_detected, st.session_state.features, st.session_state.user_remarks, st.session_state.accept_reject)

elif choice == "Defect Information":
    st.header("Defect Information")
    st.subheader("All Defect Information:")
    all_defect_info = fetch_all_defect_info()
    df = pd.DataFrame(all_defect_info)
    st.dataframe(df)

create_table()
