from flask import Flask, render_template, request
import pickle
import numpy as np
import requests
from urllib.parse import urlparse
import tldextract

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# -------------------------------
# FEATURE EXTRACTION FUNCTION
# -------------------------------
def extract_features(url):
    features = []

    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    # -------------------------------
    # 🚨 HARD RULES (VERY IMPORTANT)
    # -------------------------------
    # Force phishing if obvious
    if "@" in url:
        return [0]*19 + [1]   # guaranteed phishing signal

    suspicious_words = ["login", "secure", "account", "bank", "verify", "update"]
    if any(word in url.lower() for word in suspicious_words):
        strong_flag = 1
    else:
        strong_flag = 0

    # -------------------------------
    # NORMAL FEATURES
    # -------------------------------
    features.append(len(url))                          # 1 URL length
    features.append(1 if "-" in domain else 0)         # 2 dash in domain

    ext = tldextract.extract(url)
    sub_count = len(ext.subdomain.split(".")) if ext.subdomain else 0
    features.append(sub_count)                         # 3 subdomains

    features.append(1 if parsed.scheme == "https" else 0)  # 4 HTTPS

    features.append(url.count("."))                    # 5 dots
    features.append(url.count("/"))                    # 6 slashes
    features.append(sum(c.isdigit() for c in url))     # 7 digits

    features.append(1 if "-" in domain else 0)         # 8 prefix/suffix

    features.append(strong_flag)                       # 9 suspicious words

    # -------------------------------
    # FETCH PAGE (SAFE)
    # -------------------------------
    try:
        response = requests.get(url, timeout=3)
        html = response.text.lower()

        features.append(html.count("<a"))              # 10 links
        features.append(html.count("http"))            # 11 external refs
        features.append(1 if "<form" in html else 0)   # 12 forms

    except:
        features.extend([0, 0, 0])

    # -------------------------------
    # FILL TO MATCH MODEL INPUT
    # -------------------------------
    while len(features) < 20:
        features.append(0)

    return features[:20]

# -------------------------------
# ROUTES
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    url = request.form["url"]

    # Ensure URL format
    if not url.startswith("http"):
        url = "http://" + url

    features = extract_features(url)
    final_features = np.array(features).reshape(1, -1)

    prediction = model.predict(final_features)

    result = "Legitimate Website ✅" if prediction[0] == 1 else "Phishing Website ⚠️"

    return render_template("index.html", prediction_text=result)

# -------------------------------
# RUN APP
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)