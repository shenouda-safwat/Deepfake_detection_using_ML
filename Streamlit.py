import streamlit as st
import numpy as np
import joblib
from PIL import Image
import cv2
from skimage.feature import hog, local_binary_pattern
from scipy.fftpack import dct
import xgboost as xgb

st.set_page_config(page_title="Deepfake Detection", page_icon="🧠", layout="centered")

@st.cache_resource
def load_model_and_selector():
    model = joblib.load('xgboost_model.pkl')
    selector = joblib.load('feature_selector.pkl')
    return model, selector

try:
    model, selector = load_model_and_selector()
except Exception as e:
    st.error(f"❌ invalid during load selector or model {e}")
    st.stop()

def extract_features_single(img):
    img = cv2.resize(img, (96, 96))  

    if len(img.shape) == 2:  
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # --- HOG Features ---
    hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), block_norm='L2-Hys',
                    visualize=False, feature_vector=True, channel_axis=-1)

    # --- LBP Histogram Features ---
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_img, P=16, R=2, method='uniform')
    (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    # --- Color Histogram Features ---
    if len(img.shape) == 3:
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([img_color], [0], None, [8], [0, 180]).flatten()
        hist_s = cv2.calcHist([img_color], [1], None, [8], [0, 256]).flatten()
        hist_v = cv2.calcHist([img_color], [2], None, [8], [0, 256]).flatten()
        color_hist = np.hstack([hist_h, hist_s, hist_v])
        color_hist = color_hist / (color_hist.sum() + 1e-7)
    else:
        color_hist = np.zeros(24)

    # --- DCT Features ---
    dct_features = dct(dct(img.T, norm='ortho').T, norm='ortho')
    dct_features = dct_features.flatten()
    dct_features = dct_features[:500]

    # --- Combine All ---
    combined = np.hstack([hog_features, lbp_hist, color_hist, dct_features])

    return combined

st.markdown("<h1 style='text-align: center; color: #6C63FF;'>🔮 Deepfake Detection App</h1>", unsafe_allow_html=True)
st.markdown("---")

uploaded_image = st.file_uploader("📤 **load your image**", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="📸  the images that has been loaded ", use_container_width=True)

    with st.spinner('🧠 loaded feature extraction..'):
        image = Image.open(uploaded_image).convert('RGB')
        image = np.array(image)

        features = extract_features_single(image)

        if features.shape[0] != selector.n_features_in_:
            st.error(f"❌   numper of Feature extraction ({features.shape[0]}) not equal number of feature extraction ({selector.n_features_in_})")
            st.stop()

        features_selected = selector.transform(features.reshape(1, -1))

        dmat_features = xgb.DMatrix(features_selected)

        pred_prob = model.predict(dmat_features)
        prob_fake = pred_prob[0]
        prob_real = 1 - prob_fake

    st.success("✅ the feature has ectraction successful !")
    st.markdown("---")

    st.subheader("🔎  result of analysis:")

    if prob_fake >= 0.5:
        st.error(f"⚠️ the image is **Fake (Fake)**")
        confidence = prob_fake
    else:
        st.success(f"✅  the image is **Real (Real)**")
        confidence = prob_real

    st.progress(float(confidence))
    st.markdown(f"### 🧠 Confirmation Rate: `{confidence*100:.2f}%`")

else:
    st.info("📂 please, laod your images")
