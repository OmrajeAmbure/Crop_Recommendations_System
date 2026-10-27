import streamlit as st
import numpy as np
import pickle
import warnings

warnings.filterwarnings(
    "ignore",
    message="Trying to unpickle estimator"
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SmartCrop",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# LOAD MACHINE LEARNING MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

def main():

    # CSS DESIGN
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 850px;
        }

        #MainMenu {
            visibility: hidden;
        }

        .main-title {
            text-align: center;
            color: mediumseagreen;
            font-size: 38px;
            font-weight: bold;
        }

        .subtitle {
            text-align: center;
            font-size: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------
    # TITLE
    # --------------------------------------------------

    st.markdown(
        '<div class="main-title">🌱 SmartCrop</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Intelligent Crop Recommendation System'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.info(
        "🌾 Enter your soil and weather details to get a crop recommendation.\n\n"
        "मराठी: तुमच्या जमिनीची आणि हवामानाची माहिती भरा "
        "आणि कोणते पीक योग्य ठरू शकते ते जाणून घ्या."
    )

    # --------------------------------------------------
    # EXAMPLE FOR FARMERS
    # --------------------------------------------------

    with st.expander(
        "🌾 Example: How to fill the details? (उदाहरण पाहा)"
    ):

        st.subheader("👨‍🌾 Example of a farmer's field")

        st.write(
            "Suppose a farmer has soil test results and weather data. "
            "The farmer should enter the values from the relevant "
            "reports and data sources."
        )

        st.markdown("### मराठीत सोपे उदाहरण")

        st.write(
            "समजा, एका शेतकऱ्याने आपल्या शेतातील मातीची तपासणी केली आहे. "
            "खालील आकडे फक्त उदाहरण म्हणून दिले आहेत. "
            "प्रत्यक्ष वापरासाठी आपल्या माती परीक्षण अहवालातील "
            "आणि संबंधित हवामानातील मूल्ये वापरा."
        )

        example_data = {
            "घटक (Input)": [
                "Nitrogen (N)",
                "Phosphorus (P)",
                "Potassium (K)",
                "Temperature",
                "Humidity",
                "Soil pH",
                "Rainfall"
            ],
            "उदाहरण (Example)": [
                "90",
                "42",
                "43",
                "25 °C",
                "80%",
                "6.5",
                "200 mm"
            ],
            "सोप्या मराठीत अर्थ": [
                "जमिनीतील नायट्रोजनचे प्रमाण",
                "जमिनीतील फॉस्फरसचे प्रमाण",
                "जमिनीतील पोटॅशियमचे प्रमाण",
                "वातावरणाचे तापमान",
                "हवेतील ओलावा",
                "जमिनीचा सामू",
                "संबंधित कालावधीतील पाऊस"
            ]
        }

        st.table(example_data)

        st.warning(
            "महत्त्वाचे: हे आकडे फक्त समजण्यासाठी आहेत. "
            "हे प्रत्यक्ष माती परीक्षणाचे निष्कर्ष नाहीत. "
            "माती परीक्षणातील N, P, K ची एकके आणि हवामानाच्या "
            "माहितीचा कालावधी तुमच्या Dataset शी जुळला पाहिजे."
        )

    st.divider()

    # --------------------------------------------------
    # SOIL NUTRIENTS
    # --------------------------------------------------

    st.subheader(
        "🌿 Soil Nutrients (जमिनीतील अन्नद्रव्ये)"
    )

    # NITROGEN
    N = st.number_input(
        "Nitrogen (N)",
        min_value=1,
        max_value=10000,
        value=50,
        step=1,
        help=(
            "Nitrogen content in the soil. "
            "It supports the growth of green leaves and stems."
        )
    )

    st.caption(
        "🌱 मराठी: जमिनीतील नायट्रोजनचे प्रमाण. "
        "पानांची आणि खोडाची वाढ होण्यास मदत करते."
    )

    # PHOSPHORUS
    P = st.number_input(
        "Phosphorus (P)",
        min_value=1,
        max_value=10000,
        value=50,
        step=1,
        help=(
            "Phosphorus content in the soil. "
            "It supports root growth and energy transfer in plants."
        )
    )

    st.caption(
        "🌱 मराठी: जमिनीतील फॉस्फरसचे प्रमाण. "
        "मुळांच्या वाढीस आणि वनस्पतींच्या ऊर्जा प्रक्रियेस मदत करते."
    )

    # POTASSIUM
    K = st.number_input(
        "Potassium (K)",
        min_value=1,
        max_value=10000,
        value=50,
        step=1,
        help=(
            "Potassium content in the soil. "
            "It supports plant health and water regulation."
        )
    )

    st.caption(
        "🌿 मराठी: जमिनीतील पोटॅशियमचे प्रमाण. "
        "झाडाचे आरोग्य आणि पाण्याचे व्यवस्थापन यासाठी मदत करते."
    )

    st.divider()

    # --------------------------------------------------
    # WEATHER CONDITIONS
    # --------------------------------------------------

    st.subheader(
        "🌦️ Weather Conditions (हवामानाची माहिती)"
    )

    # TEMPERATURE
    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=0.5,
        help=(
            "Temperature in degrees Celsius. "
            "Different crops grow well at different temperatures."
        )
    )

    st.caption(
        "🌡️ मराठी: वातावरणाचे तापमान डिग्री सेल्सिअसमध्ये भरा. "
        "वेगवेगळ्या पिकांना वेगवेगळे तापमान योग्य असते."
    )

    # HUMIDITY
    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=1.0,
        help=(
            "Relative humidity of the air, expressed as a percentage."
        )
    )

    st.caption(
        "💧 मराठी: हवेमधील ओलाव्याचे प्रमाण टक्क्यांमध्ये भरा. "
        "ही जमिनीतील ओलावा दर्शवत नाही."
    )

    # SOIL PH
    ph = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=6.5,
        step=0.1,
        help=(
            "Soil pH indicates whether the soil is acidic, "
            "neutral, or alkaline."
        )
    )

    st.caption(
        "🧪 मराठी: जमिनीचा सामू. "
        "pH 7 पेक्षा कमी असल्यास जमीन आम्लधर्मी, "
        "7 असल्यास उदासीन आणि 7 पेक्षा जास्त असल्यास "
        "अल्कधर्मी असते."
    )

    # RAINFALL
    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=10000.0,
        value=100.0,
        step=1.0,
        help=(
            "Rainfall in millimeters for the period "
            "represented in your dataset."
        )
    )

    st.caption(
        "🌧️ मराठी: संबंधित कालावधीत पडलेल्या पावसाचे "
        "प्रमाण मिलिमीटरमध्ये भरा."
    )

    st.divider()

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    st.subheader("🔍 Crop Prediction (पीक शिफारस)")

    if st.button(
        "🌱 Predict Suitable Crop",
        use_container_width=True
    ):

        try:
            # PREPARE INPUT FEATURES
            feature_list = [
                N,
                P,
                K,
                temperature,
                humidity,
                ph,
                rainfall
            ]

            # CONVERT TO NUMPY ARRAY
            single_pred = np.array(
                feature_list
            ).reshape(1, -1)

            # LOAD MODEL
            model = load_model()

            # PREDICT CROP
            prediction = model.predict(single_pred)

            # EXTRACT PREDICTION
            crop_name = str(prediction.item()).title()

            # DISPLAY RESULT
            st.success(
                f"🌾 Recommended Crop: {crop_name}"
            )

            st.write(
                f"**मराठी:** दिलेल्या माहितीच्या आधारे "
                f"मॉडेलने **{crop_name}** या पिकाची शिफारस केली आहे."
            )

            st.warning(
                "टीप: ही शिफारस Machine Learning मॉडेलच्या "
                "प्रशिक्षणातील डेटावर आधारित आहे. "
                "प्रत्यक्ष लागवडीपूर्वी माती परीक्षण, "
                "स्थानिक हवामान आणि कृषी तज्ज्ञांचा सल्ला घ्या."
            )

        except FileNotFoundError:
            st.error(
                "❌ model.pkl file was not found. "
                "Please keep model.pkl in the project folder."
            )

        except Exception as e:
            st.error(
                f"Prediction failed: {e}"
            )

    # --------------------------------------------------
    # FOOTER
    # --------------------------------------------------

    st.divider()

    st.caption(
        "🌱 SmartCrop | Intelligent Crop Recommendation System"
    )


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    main()
