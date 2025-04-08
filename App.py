import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFont
from PIL.ExifTags import TAGS  # For EXIF metadata
import io
import os
from rembg import remove  # type: ignore # For background removal
from streamlit_image_zoom import image_zoom  # type: ignore

# Page Config
st.set_page_config(
    page_title="SnapSaathi Studio.AI",
    page_icon="🖼️",
    layout="centered"
)

# Colorful & Styled Header using Markdown and HTML
st.markdown("""
    <div style='
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center; 
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    '>
        <h1 style='color: white; font-size: 3em;'>🪄SnapSaathi Studio.AI🤖</h1>
        <p style='color: white; font-size: 1.3em;'>
            🤖 Enhance your low-quality photos with AI magic 🪄📸
        </p>
    </div>
""", unsafe_allow_html=True)

# Optional Divider
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# Check OpenCV version and dnn_superres availability
try:
    if not hasattr(cv2, 'dnn_superres'):
        raise ImportError("Your OpenCV installation does not support dnn_superres.")
except ImportError as e:
    st.error(f"{e} Install with: pip install opencv-contrib-python")
    st.stop()

# Load OpenCV Super-Resolution Model
def load_superres_model(enhancement_level):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    model_path = "FSRCNN_x4.pb"  # Pre-trained model
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    try:
        sr.readModel(model_path)
        sr.setModel("fsrcnn", enhancement_level)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    return sr

def enhance_image(image, sr_model):
    image = np.array(image)

    # Ensure proper channel format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif image.shape[2] != 3:
        raise ValueError("Input image must have 3 channels.")

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Image has invalid dimensions.")

    try:
        enhanced_image = sr_model.upsample(image)
    except Exception as e:
        raise RuntimeError(f"Error during enhancement: {e}")

    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(enhanced_image)

# New Feature: AI Background Remover
def remove_background(image):
    try:
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        # Use rembg to remove the background
        result = remove(img_byte_arr)

        # Convert the result back to a PIL Image
        return Image.open(io.BytesIO(result))
    except Exception as e:
        st.error(f"Error removing background: {e}")
        return None

# New Feature: Cartoonify Filter
def cartoonify_image(image):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img_array, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return Image.fromarray(cartoon)

# New Feature: Vintage Filter
def apply_vintage_filter(img):
    img_array = np.array(img).astype(np.float32)
    vintage = img_array.copy()
    vintage[:, :, 0] *= 1.1  # Red channel
    vintage[:, :, 1] *= 0.9  # Green channel
    vintage[:, :, 2] *= 0.8  # Blue channel
    vintage = np.clip(vintage, 0, 255).astype(np.uint8)
    return Image.fromarray(vintage)

# Function to add text to an image
def add_text(img, text):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 30)  # Ensure arial.ttf exists on the system
    except IOError:
        st.error("Font file not found. Please ensure 'arial.ttf' is available.")
        return img
    draw.text((20, 20), text, fill="white", font=font)
    return img

# Function to blur the image
def blur_image(img, k):
    img = np.array(img)
    blurred = cv2.GaussianBlur(img, (k, k), 0)
    return Image.fromarray(blurred)

# Function to convert an image to a pencil sketch
def pencil_sketch(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return Image.fromarray(sketch)

# Sidebar Images and Navigation
AI_path = "AI.png"  # Ensure this file is in the same directory as your script
try:
    st.sidebar.image(AI_path)
except FileNotFoundError:
    st.sidebar.warning("AI.png file not found. Please check the file path.")

image_path = "image.png"  # Ensure this file is in the same directory as your script
try:
    st.sidebar.image(image_path)
except FileNotFoundError:
    st.sidebar.warning("image.png file not found. Please check the file path.")

# Sidebar Navigation
with st.sidebar:
    st.header("🖼️ Photo Enhancer Features")

    tab_selection = st.radio("Select a Feature:", [
        "🏠 Home Dashboard",
        "📤 Upload Image",
        "🧠 AI Super Resolution",
        "🎨 Apply Artistic Filters",
        "🖌️ Pencil Sketch Converter",
        "🌈 Image Color Enhancer",
        "🔍 Zoom & Inspect",
        "🌀 Rotate / Flip Image",
        "📐 Crop Image",
        "🧹 Denoise / Clean Image",
        "🔎 Sharpness & Clarity Adjuster",
        "🖼️ Add Watermark / Text",
        "📦 Batch Image Enhancer",
        "📁 Export & Download Options",
        "🤖 Chat With AI Photo Assistant"
    ])

# Developer Information
st.sidebar.markdown("👨👨‍💻 Developer: Abhishek 💖 Yadav")

developer_path = "pic.jpg"  # Ensure this file is in the same directory as your script
try:
    st.sidebar.image(developer_path)
except FileNotFoundError:
    st.sidebar.warning("pic.jpg file not found. Please check the file path.")


# Option to upload an image or take a photo with the webcam
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    # Add Text to Image Feature
    user_text = st.text_input("Enter text to add:")
    if user_text:
        with st.spinner("Adding text to image..."):
            final_image = add_text(image.copy(), user_text)
            st.image(final_image, caption="Image with Text", use_column_width=True)

    # Add Pencil Sketch Feature
    if st.checkbox("Convert to Pencil Sketch"):
        with st.spinner("Converting to pencil sketch..."):
            sketch_image = pencil_sketch(image)
            st.image(sketch_image, caption="Pencil Sketch", use_column_width=True)
            # Add download option
            with io.BytesIO() as output:
                sketch_image.save(output, format="PNG")
                st.download_button(
                    label="Download Pencil Sketch Image",
                    data=output.getvalue(),
                    file_name="pencil_sketch.png",
                    mime="image/png"
                )

    # Other features (e.g., Vintage Filter, Metadata Viewer, etc.)
    if st.checkbox("Apply Vintage Filter"):
        with st.spinner("Applying vintage filter..."):
            vintage_image = apply_vintage_filter(image)
            st.image(vintage_image, caption="Vintage Filter", use_column_width=True)
            # Add download option
            with io.BytesIO() as output:
                vintage_image.save(output, format="PNG")
                st.download_button(
                    label="Download Vintage Filter Image",
                    data=output.getvalue(),
                    file_name="vintage_image.png",
                    mime="image/png"
                )

    if st.checkbox("Adjust Contrast"):
        contrast = st.slider("Adjust Contrast", 0.5, 3.0, 1.0)
        with st.spinner("Adjusting contrast..."):
            enhancer = ImageEnhance.Contrast(image)
            contrast_image = enhancer.enhance(contrast)
            st.image(contrast_image, caption=f"Contrast: {contrast}", use_column_width=True)

    if st.checkbox("Blur Image"):
        blur_level = st.slider("Blur Level (odd only)", 1, 21, 5, step=2)
        with st.spinner("Applying blur..."):
            blurred_image = blur_image(image, blur_level)
            st.image(blurred_image, caption=f"Blurred (k={blur_level})", use_column_width=True)
            # Add download option
            with io.BytesIO() as output:
                blurred_image.save(output, format="PNG")
                st.download_button(
                    label="Download Blurred Image",
                    data=output.getvalue(),
                    file_name="blurred_image.png",
                    mime="image/png"
                )

elif st.checkbox("Use Webcam"):
    webcam_image = st.camera_input("Take a photo")
    if webcam_image:
        image = Image.open(webcam_image)

if image:
    st.write("Zoom and inspect the uploaded image:")
    image_zoom(image)

    # Enhancement Level Selection
    enhancement_level = st.selectbox("Select enhancement level:", [2, 4, 8])

    sr_model = load_superres_model(enhancement_level)

    if sr_model is None:
        st.stop()

    # Background Removal
    if st.checkbox("Remove Background"):
        with st.spinner("Removing background..."):
            bg_removed_image = remove_background(image)
            if bg_removed_image:
                st.image(bg_removed_image, caption="Background Removed Image", use_column_width=True)
                # Add download option
                with io.BytesIO() as output:
                    bg_removed_image.save(output, format="PNG")
                    st.download_button(
                        label="Download Background Removed Image",
                        data=output.getvalue(),
                        file_name="background_removed_image.png",
                        mime="image/png"
                    )

    # Cartoonify Filter
    if st.checkbox("Cartoonify Image"):
        with st.spinner("Applying cartoon effect..."):
            cartoon_image = cartoonify_image(image)
            st.image(cartoon_image, caption="Cartoonified Image", use_column_width=True)
            # Add download option
            with io.BytesIO() as output:
                cartoon_image.save(output, format="PNG")
                st.download_button(
                    label="Download Cartoonified Image",
                    data=output.getvalue(),
                    file_name="cartoonified_image.png",
                    mime="image/png"
                )

    # Enhance Image Button
    if st.button("Enhance Image"):
        with st.spinner("Enhancing Image..."):
            try:
                enhanced_image = enhance_image(image, sr_model)
            except Exception as e:
                st.error(f"Enhancement failed: {e}")
                st.stop()

        # Store base enhanced image for applying filters separately
        st.session_state.base_enhanced_image = enhanced_image

    # Once base enhanced image exists in session
    if "base_enhanced_image" in st.session_state:
        base_image = st.session_state.base_enhanced_image

        # Set default filter
        if "selected_filter" not in st.session_state:
            st.session_state.selected_filter = "None"

        # Dropdown to choose filter
        filter = st.selectbox(
            "Apply a filter:",
            ["None", "Grayscale", "Sepia", "Cartoon"],
            index=["None", "Grayscale", "Sepia", "Cartoon"].index(st.session_state.selected_filter),
            key="selected_filter"
        )

        # Start with base image
        enhanced_image = base_image.copy()

        # Apply selected filter
        if st.session_state.selected_filter == "Grayscale":
            enhanced_image = enhanced_image.convert("L")
        elif st.session_state.selected_filter == "Sepia":
            sepia = np.array(enhanced_image)
            sepia = cv2.transform(sepia, np.array([[0.393, 0.769, 0.189],
                                                   [0.349, 0.686, 0.168],
                                                   [0.272, 0.534, 0.131]]))
            sepia = np.clip(sepia, 0, 255).astype(np.uint8)
            enhanced_image = Image.fromarray(sepia)
        elif st.session_state.selected_filter == "Cartoon":
            cartoon = np.array(enhanced_image)
            gray = cv2.cvtColor(cartoon, cv2.COLOR_RGB2GRAY)
            edges = cv2.adaptiveThreshold(cv2.medianBlur(gray, 5), 255,
                                          cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(cartoon, 9, 300, 300)
            cartoon = cv2.bitwise_and(color, color, mask=edges)
            enhanced_image = Image.fromarray(cartoon)

        # Add sliders for brightness, saturation, contrast, sharpness
        brightness = st.slider("Brightness", 0.0, 2.0, 1.0)
        enhanced_image = ImageEnhance.Brightness(enhanced_image).enhance(brightness)

        saturation = st.slider("Saturation", 0.0, 2.0, 1.0)
        enhanced_image = ImageEnhance.Color(enhanced_image).enhance(saturation)

        contrast = st.slider("Contrast", 0.0, 2.0, 1.0)
        enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(contrast)

        sharpness = st.slider("Sharpness", 0.0, 2.0, 1.0)
        enhanced_image = ImageEnhance.Sharpness(enhanced_image).enhance(sharpness)

        # Add blur slider
        blur_value = st.slider("Blur", 0, 10, 0)
        if blur_value > 0:
            img_cv = np.array(enhanced_image)
            img_cv = cv2.GaussianBlur(img_cv, (2 * blur_value + 1, 2 * blur_value + 1), 0)
            enhanced_image = Image.fromarray(img_cv)

        # Add invert colors checkbox
        invert = st.checkbox("Invert Colors")
        if invert:
            inverted_img = ImageOps.invert(enhanced_image.convert("RGB"))
            enhanced_image = inverted_img

        # Add grayscale checkbox
        grayscale = st.checkbox("Grayscale")
        if grayscale:
            enhanced_image = enhanced_image.convert("L")

        # Add rotate and flip buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("↺ Rotate Left"):
                enhanced_image = enhanced_image.rotate(90, expand=True)
            if st.button("⇋ Flip Horizontal"):
                enhanced_image = enhanced_image.transpose(Image.FLIP_LEFT_RIGHT)
        with col2:
            if st.button("↻ Rotate Right"):
                enhanced_image = enhanced_image.rotate(-90, expand=True)
            if st.button("⇵ Flip Vertical"):
                enhanced_image = enhanced_image.transpose(Image.FLIP_TOP_BOTTOM)

        # Show both images
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", use_column_width=True)
        with col2:
            st.image(enhanced_image, caption=f"Enhanced ({st.session_state.selected_filter})", use_column_width=True)

        # Add download option
        download_format = st.selectbox("Select download format:", ["PNG", "JPEG"])
        file_extension = "png" if download_format == "PNG" else "jpg"
        file_path = f"enhanced.{file_extension}"
        enhanced_image.save(file_path)

        with open(file_path, "rb") as file:
            st.download_button(
                label=f"Download Enhanced Image ({download_format})",
                data=file,
                file_name=f"enhanced_image.{file_extension}",
                mime=f"image/{file_extension}"
            )
