import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Cat & Dog Image Classification",
    page_icon="🐾",
    layout="wide"
)


# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "cat_dog_cnn.h5"
HEADER_IMAGE_PATH = BASE_DIR / "cat and dog.jpg"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       APP BACKGROUND
       ===================================================== */

    .stApp {
        background-color: #fff4f6;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }


    /* =====================================================
       STREAMLIT HEADER
       ===================================================== */

    [data-testid="stHeader"] {
        background-color: #fff4f6 !important;
    }


    /* =====================================================
       NORMAL TEXT
       ===================================================== */

    .stMarkdown p {
        color: #76565b;
    }


    /* =====================================================
       HERO TITLE
       ===================================================== */

    .hero-title {
        color: #4b2529 !important;
        font-size: 48px !important;
        font-weight: 700 !important;
        line-height: 1.15 !important;
        margin-bottom: 15px !important;
    }

    .cat-color {
        color: #e63d5b !important;
    }

    .dog-color {
        color: #4b2529 !important;
    }


    /* =====================================================
       BADGE
       ===================================================== */

    .badge-text {
        color: #d92f50 !important;
        background-color: #ffe1e7;
        padding: 7px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }


    /* =====================================================
       DESCRIPTION
       ===================================================== */

    .description-text {
        color: #76565b !important;
        font-size: 17px !important;
        line-height: 1.6 !important;
    }


    /* =====================================================
       SECTION HEADINGS
       ===================================================== */

    .section-heading {
        color: #4b2529 !important;
        font-size: 22px !important;
        font-weight: 700 !important;
    }

    .section-subheading {
        color: #96747a !important;
        font-size: 14px !important;
    }


    /* =====================================================
       HEADER IMAGE
       ===================================================== */

    [data-testid="stImage"] img {
        border-radius: 15px !important;
    }


    /* =====================================================
       FILE UPLOADER
       ===================================================== */

    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 1px solid #f4cbd3 !important;
        border-radius: 16px !important;
        padding: 8px !important;
        box-shadow: 0 4px 15px rgba(210, 60, 90, 0.06);
    }


    /* =====================================================
       SMALL DROPZONE
       ===================================================== */

    [data-testid="stFileUploaderDropzone"] {
        background-color: #fff9fa !important;
        border: 2px dashed #ef9eaf !important;
        border-radius: 12px !important;

        min-height: 85px !important;
        height: 85px !important;

        padding: 6px !important;
    }


    /* =====================================================
       FILE UPLOADER BUTTON
       ===================================================== */

    [data-testid="stFileUploaderDropzone"] button {
        background-color: #e83e5d !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #d92f50 !important;
        color: white !important;
    }


    /* =====================================================
       UPLOADED FILE
       ===================================================== */

    [data-testid="stFileUploaderFile"] {
        background-color: #fff8fa !important;
        border: 1px solid #f0b9c5 !important;
        border-radius: 9px !important;
    }


    /* =====================================================
       SUCCESS MESSAGE
       ===================================================== */

    [data-testid="stAlert"] {
        border-radius: 10px !important;
    }


    /* =====================================================
       RESULT CONTAINER
       ===================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #f4cbd3 !important;
        border-radius: 18px !important;
        box-shadow: 0 4px 18px rgba(210, 60, 90, 0.06);
    }


    /* =====================================================
       RESULT TITLE
       ===================================================== */

    .result-title {
        color: #df3455 !important;
        font-size: 32px !important;
        font-weight: 700 !important;
        text-align: center !important;
    }

    .result-confidence {
        color: #86666d !important;
        font-size: 15px !important;
        text-align: center !important;
    }


    /* =====================================================
       PROGRESS BAR
       ===================================================== */

    .stProgress > div > div > div > div {
        background-color: #e83e5d !important;
    }


    /* =====================================================
       FEATURE CARD
       ===================================================== */

    .feature-box {
        background-color: #ffffff;
        border: 1px solid #f4cbd3;
        border-radius: 16px;
        padding: 18px 12px;
        min-height: 135px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(210, 60, 90, 0.05);
    }

    .feature-heading {
        color: #542c32 !important;
        font-size: 16px !important;
        font-weight: 700 !important;
    }

    .feature-description {
        color: #96747a !important;
        font-size: 13px !important;
    }


    /* =====================================================
       STREAMLIT MARKDOWN HEADINGS
       ===================================================== */

    h1, h2, h3 {
        color: #4b2529 !important;
    }


    /* =====================================================
       CAPTION
       ===================================================== */

    [data-testid="stCaptionContainer"] {
        color: #96747a !important;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer-text {
        color: #96747a !important;
        font-size: 13px !important;
        text-align: center !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


model = None


if not MODEL_PATH.exists():

    st.error(
        "❌ cat_dog_cnn.h5 model file not found."
    )

else:

    try:

        model = load_model()

    except Exception as e:

        st.error(
            "❌ Unable to load the CNN model."
        )

        st.code(str(e))


# =========================================================
# TOP BADGE
# =========================================================

st.markdown(
    "🐾 AI Powered • CNN Image Classifier",
    help="Cat and Dog image classification using CNN"
)


# =========================================================
# HERO SECTION
# =========================================================

hero_left, hero_right = st.columns(
    [1.35, 1],
    gap="large"
)


# =========================================================
# HERO LEFT
# =========================================================

with hero_left:

    st.markdown(
        """
        <h1 class="hero-title">
            <span class="cat-color">Cat</span>
            <span class="dog-color"> &amp; Dog</span><br>
            <span class="dog-color">Image Classification</span>
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "Upload an image and let the CNN model predict whether it is a Cat or a Dog.",
        help="Upload a JPG, JPEG, or PNG image."
    )


# =========================================================
# HERO RIGHT
# =========================================================

with hero_right:

    if HEADER_IMAGE_PATH.exists():

        st.image(
            str(HEADER_IMAGE_PATH),
            width=400
        )

    else:

        st.warning(
            "⚠️ Header image not found."
        )

        st.caption(
            "Keep 'cat and dog.jpg' in the same folder as cd.py."
        )


st.write("")
st.write("")


# =========================================================
# UPLOAD + RESULT
# =========================================================

upload_column, result_column = st.columns(
    [1, 1],
    gap="large"
)


# =========================================================
# UPLOAD SECTION
# =========================================================

with upload_column:

    st.markdown(
        "###  Upload Image"
    )

    st.caption(
        "Choose a Cat or Dog image (JPG, JPEG, or PNG)"
    )


    # -----------------------------------------------------
    # FILE UPLOADER
    # -----------------------------------------------------

    uploaded_file = st.file_uploader(
        "Drag & drop an image or use Browse files",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        label_visibility="visible"
    )


    # -----------------------------------------------------
    # UPLOAD STATUS
    # -----------------------------------------------------

    if uploaded_file is not None:

        st.success(
            "Image uploaded successfully"
        )

        st.caption(
            f"File: {uploaded_file.name}"
        )

    else:

        st.caption(
            "Maximum file size: 200MB • JPG, JPEG, PNG"
        )

        st.caption(
            "Image will be resized automatically to 128 × 128"
        )


# =========================================================
# PREDICTION SECTION
# =========================================================

with result_column:

    st.markdown(
        "### ✨ Prediction Result"
    )


    # =====================================================
    # BEFORE UPLOAD
    # =====================================================

    if uploaded_file is None:

        with st.container(border=True):

            st.write("")

            st.markdown(
                "## 🐾"
            )

            st.markdown(
                "### Upload an image to begin"
            )

            st.caption(
                "Your CNN prediction will appear here."
            )

            st.write("")


    # =====================================================
    # AFTER UPLOAD
    # =====================================================

    elif model is not None:

        try:

            # -------------------------------------------------
            # READ IMAGE
            # -------------------------------------------------

            image = Image.open(
                uploaded_file
            ).convert("RGB")


            # -------------------------------------------------
            # RESIZE IMAGE
            # -------------------------------------------------

            resized_image = image.resize(
                (128, 128)
            )


            # -------------------------------------------------
            # NUMPY CONVERSION
            # -------------------------------------------------

            image_array = np.asarray(
                resized_image,
                dtype=np.float32
            )


            # -------------------------------------------------
            # NORMALIZATION
            # -------------------------------------------------

            image_array = image_array / 255.0


            # -------------------------------------------------
            # BATCH DIMENSION
            # -------------------------------------------------

            image_array = np.expand_dims(
                image_array,
                axis=0
            )


            # -------------------------------------------------
            # PREDICTION
            # -------------------------------------------------

            prediction = model.predict(
                image_array,
                verbose=0
            )


            probability = float(
                prediction[0][0]
            )


            # -------------------------------------------------
            # CLASSIFICATION
            # -------------------------------------------------

            if probability >= 0.5:

                label = "Dog"
                emoji = "🐶"
                confidence = probability

            else:

                label = "Cat"
                emoji = "🐱"
                confidence = 1 - probability


            confidence_percent = confidence * 100


            # =================================================
            # RESULT CARD
            # =================================================

            with st.container(border=True):

                st.write("")


                # -------------------------------------------------
                # UPLOADED IMAGE
                # -------------------------------------------------

                st.image(
                    image,
                    caption="Uploaded Image",
                    width=300
                )


                st.write("")


                # -------------------------------------------------
                # PREDICTION RESULT
                # -------------------------------------------------

                st.markdown(
                    f"## {emoji} {label}"
                )


                st.markdown(
                    f"Confidence: {confidence_percent:.2f}%"
                )


                # -------------------------------------------------
                # CONFIDENCE BAR
                # -------------------------------------------------

                st.progress(
                    confidence,
                    text=f"Confidence: {confidence_percent:.2f}%"
                )


                st.write("")


        except Exception as e:

            st.error(
                "❌ Error while predicting the image."
            )

            st.code(
                str(e)
            )


# =========================================================
# FEATURES
# =========================================================

st.write("")
st.write("")

st.markdown(
    "### 💡 What this model does"
)

st.write("")


feature1, feature2, feature3, feature4 = st.columns(
    4,
    gap="medium"
)


# =========================================================
# FEATURE 1
# =========================================================

with feature1:

    with st.container(border=True):

        st.markdown("### 🧠")

        st.markdown(
            "**CNN Model**"
        )

        st.caption(
            "Learns visual features from images"
        )


# =========================================================
# FEATURE 2
# =========================================================

with feature2:

    with st.container(border=True):

        st.markdown("### ⚡")

        st.markdown(
            "**Fast Prediction**"
        )

        st.caption(
            "Generates predictions quickly"
        )


# =========================================================
# FEATURE 3
# =========================================================

with feature3:

    with st.container(border=True):

        st.markdown("### 🎯")

        st.markdown(
            "**Binary Classification**"
        )

        st.caption(
            "Classifies Cat or Dog"
        )


# =========================================================
# FEATURE 4
# =========================================================

with feature4:

    with st.container(border=True):

        st.markdown("### 🐾")

        st.markdown(
            "**Easy to Use**"
        )

        st.caption(
            "Upload and get your prediction"
        )


# =========================================================
# FOOTER
# =========================================================

st.write("")

st.caption(
    "🐾 Built with  using Streamlit & TensorFlow"
)