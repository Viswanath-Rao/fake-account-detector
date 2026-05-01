import streamlit as st
import pickle
import numpy as np
import cv2
import time

# Load model
model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title="Fake Account Detector", layout="centered")

# ---------- CSS ----------
st.markdown("""
<style>
.bubble {
    position: fixed;
    bottom: -100px;
    border-radius: 50%;
    animation: rise linear infinite;
    z-index: 0;
    background: radial-gradient(circle, rgba(0,255,170,0.9) 0%, rgba(0,255,170,0.2) 60%, transparent 80%);

    box-shadow:
        0 0 20px rgba(0,255,170,0.8),
        0 0 40px rgba(0,255,170,0.6),
        0 0 80px rgba(0,255,170,0.4);
}

@keyframes rise {
    0% {
        transform: translateY(0) scale(1);
        opacity: 0.4;
    }
    100% {
        transform: translateY(-120vh) scale(1.2);
        opacity: 0;
    }
}
/* Remove default padding that creates ghost boxes */
section.main > div {
    padding-top: 0px !important;
}

.stApp {
    background: #0f2027;
    color: white;
}

/* Main card */
.card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(15px);
    padding: 30px;
    border-radius: 20px;
    max-width: 850px;
    margin: 40px auto;
    box-shadow: 0px 0px 40px rgba(0,0,0,0.8);
}

/* Left & Right cards */
.left-card {
    background: rgba(255,255,255,0.03);
    padding: 20px;
    border-radius: 15px;
}

.right-card {
    background: rgba(0,0,0,0.4);
    padding: 20px;
    border-radius: 15px;
}
.card {
    transition: 0.4s;
}

.card:hover {
    transform: scale(1.01);
}

/* Button */
div.stButton > button {
    background: linear-gradient(90deg, #00FFAA, #00cc88);
    color: black;
    font-weight: bold;
    border-radius: 12px;
    height: 50px;
    width: 100%;
}
div.stButton > button:hover {
    transform: scale(1.08);
    box-shadow: 0px 0px 20px #00ffaa;
}


/* keep UI above dots */


</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("""
<h1 style='text-align:center; animation: fadeIn 1.5s ease-in-out;'>
Fake Account Detection
</h1>

<p style='text-align:center; opacity:0.8; animation: fadeIn 2s ease-in-out;'>
Check if a profile looks real or fake
</p>

<style>
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(-30px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="bubble" style="left:10%; width:40px; height:40px; animation-duration:12s;"></div>
<div class="bubble" style="left:25%; width:20px; height:20px; animation-duration:10s;"></div>
<div class="bubble" style="left:40%; width:60px; height:60px; animation-duration:18s;"></div>
<div class="bubble" style="left:55%; width:25px; height:25px; animation-duration:14s;"></div>
<div class="bubble" style="left:70%; width:50px; height:50px; animation-duration:16s;"></div>
<div class="bubble" style="left:85%; width:30px; height:30px; animation-duration:11s;"></div>
""", unsafe_allow_html=True)

# ---------- MAIN CARD ----------
st.markdown("<div class='card'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='left-card'>", unsafe_allow_html=True)

    st.subheader("📊 Account Details")

    followers = st.number_input("Followers")
    following = st.number_input("Following")
    posts = st.number_input("Posts")

    check = st.button("🔍 Analyze")
    if st.button("🔄 Reset"): st.rerun()
    prediction = None   # ✅ prevent NameError

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='right-card'>", unsafe_allow_html=True)

    st.subheader("🖼 Profile Preview")

    uploaded_file = st.file_uploader("Upload Profile Image", type=["jpg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------- PREDICTION ----------
# ---------- PREDICTION ----------
if check and followers is not None:

    # ✅ Step 1: ratio
    ratio = followers / (following + 1)

    # ✅ Step 2: image detection
    image_flag = 0
    image_message = "No image uploaded"

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 1:
            image_flag = 0
            image_message = "👤 Single face detected (Good profile image)"

        elif len(faces) > 1:
            image_flag = 1
            image_message = "👥 Multiple faces detected (Suspicious profile)"

        else:
            image_flag = 1
            image_message = "🖼 No face detected (Cartoon / Landscape / Fake)"


        status = st.empty()

        status.info("🔍 Analyzing profile image...")
        time.sleep(1)
        status.info("📊 Checking followers & following ratio...")
        time.sleep(1)

        status.info("📸 Evaluating account activity...")
        time.sleep(1)

        status.success("✅ Analysis complete!")
        time.sleep(0.5)

        status.empty()

    # ✅ Step 3: features
    features = np.array([[followers, following, posts, ratio]])

    # ✅ Step 4: prediction
    prediction = model.predict(features)
    colA, colB, colC = st.columns(3)

    colA.metric("Followers", followers)
    colB.metric("Following", following)
    colC.metric("Posts", posts)

    proba = model.predict_proba(features)[0]
    fake_prob = proba[1]
    real_confidence = (1 - fake_prob) * 100
    st.markdown("<hr style='border:1px solid #00ffaa;'>", unsafe_allow_html=True)
    st.markdown(f"""
                <div style='text-align:center; padding:20px; border-radius:15px;
                background: linear-gradient(90deg, #00FFAA, #00cc88); color:black;'>
                <h2>Confidence: {round(real_confidence,2)}% Real</h2>
                </div>
                """, unsafe_allow_html=True)
    with st.spinner("🤖 AI is analyzing..."):
        time.sleep(1)
    # ✅ UI output
    st.progress(1 - fake_prob)
    st.markdown("### 🖼 Image Analysis")
    st.info(image_message)
    st.markdown(f"<p style='text-align:center;'>Confidence: {round(real_confidence,2)}% Real</p>", unsafe_allow_html=True)

    if prediction[0] == 1:

        color = "#ff4d4d"
        text = "🚫 Fake Account"
    else:
        color = "#00FFAA"
        text = "✅ Real Account"


    st.markdown(f"""
    <div style='text-align:center; padding:30px; border-radius:20px;
    background-color:#111; border: 2px solid {color};
    box-shadow: 0px 0px 20px {color};'>
    <h1 style='color:{color};'>{text}</h1>
    </div>
    """, unsafe_allow_html=True)



    # ✅ Explanation
    # 🧠 AI Chat Style Explanation

    explanation = ""
    if prediction[0] == 1:
        explanation += "🚫 This account looks *FAKE*.\n\n"
    else:
        explanation += "✅ This account looks *REAL*.\n\n"
    explanation += "**Here's why:**\n\n"
    if following > followers:
        explanation += "• Following is much higher than followers\n"
    if posts < 5:
        explanation += "• Very low number of posts\n"
    if image_flag == 1:
        explanation += "• Profile image looks suspicious (no face or multiple faces)\n"
    if image_flag == 0:
        explanation += "• Profile image has a clear single face\n"
    if followers > 1000 and following < followers:
        explanation += "• Healthy follower-to-following ratio\n"

# 🤖 Chat UI
    with st.chat_message("assistant"):
        st.markdown("🤖 Analyzing account...")
        st.markdown(explanation)

# 📄 Download Report
    report = f"""
    Result: {"Fake" if prediction[0]==1 else "Real"}
    Confidence: {round(real_confidence,2)}% Real
    Followers: {followers}
    Following: {following}
    Posts: {posts}
    Image Analysis: {image_message}
    """

    st.download_button("📄 Download Report", report, file_name="account_report.txt")
    st.caption("Uses activity metrics + follower ratio + profile image signals.")