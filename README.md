# fake-account-detector
Machine learning-based system for detecting fake social media profiles using behavioral data and image analysis.
Detect fake social media accounts using AI + image intelligence.

🌐 Live Demo--
https://fake-account-detector-3gappbvre8tqh2hzzg5fu4d.streamlit.app/

Features--
Detects fake accounts using ML model
Analyzes profile images using OpenCV (face detection)
Displays confidence score for predictions
Provides AI-style explanation for results
Modern UI with animations and interactive design

Tech Stack--
Python
Streamlit
Scikit-learn
OpenCV

How it Works--
User inputs followers, following, and post count
Uploads profile image (optional)
Model extracts features (ratio + image signals)
Predicts whether account is real or fake
Displays result with confidence and explanation

Model Logic--
Follower-to-following ratio plays a key role
Low post count and abnormal ratios indicate fake profiles
Image analysis checks:
Single face → more likely real
Multiple/no face → suspicious

Run Locally*
pip install -r requirements.txt 
streamlit run app.py


## 📸 Screenshots

### ✅ Real Account Detection
![Real](real.png)

### 🚫 Fake Account Detection
![Fake](fake.png)

Future Improvements--
Larger and more realistic dataset
Deep learning-based image analysis
Integration with real social media APIs

Author
Viswanath Rao
