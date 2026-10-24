# 🛡️ Phishing Website Detection Using Supervised ML

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)](#)

An intelligent, end-to-end Machine Learning web system designed to detect and flag phishing websites in real time. The system combines lexical, structural, and content-based feature extraction with an optimized Supervised Random Forest Classifier and a lightweight Flask web user interface.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
  - [Dataset](#dataset)
  - [Preprocessing & Encoding](#preprocessing--encoding)
  - [Feature Selection (Chi-2)](#feature-selection-chi-2)
  - [Model Training](#model-training)
- [Features Extracted](#-features-extracted)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Web Application](#running-the-web-application)
- [Live Inference Example](#-live-inference-example)
- [Security & Best Practices](#-security--best-practices)
- [License](#-license)

---

## 🔍 Overview

Phishing attacks remain one of the most prevalent social engineering threats on the web, luring users to deceptive domains to harvest sensitive credentials, financial details, or personal data. 

This project delivers:
1. **Automated Feature Extraction**: Analyzes incoming URLs across 20 distinct lexical, domain-level, and HTML-level heuristics.
2. **Supervised Classification**: Evaluates feature vectors using an ensemble Random Forest model trained on verified legitimate and phishing websites.
3. **Interactive Web Interface**: Allows users or security analysts to submit any URL and obtain an instant classification (Legitimate ✅ vs. Phishing ⚠️).

---

## 🏛️ System Architecture

```mermaid
graph TD
    A[User Enters URL] --> B[Flask Web Server]
    B --> C{Hard Rule Checks}
    C -- Obvious Phishing Pattern (e.g. '@' symbol) --> G[Flag as Phishing]
    C -- Passes Heuristic Checks --> D[Feature Extraction Engine]
    D --> E[Lexical & Structural Analysis: URL length, dots, slashes, subdomains, HTTPS]
    D --> F[Content Inspection: Forms, external references, anchor tags]
    E --> H[20-Dimensional Vector]
    F --> H
    H --> I[Random Forest Classifier (model.pkl)]
    I --> J{Prediction}
    J -- Class 1 --> K[Legitimate Website ✅]
    J -- Class 0 --> L[Phishing Website ⚠️]
    G --> L
    K --> M[Display on Web UI]
    L --> M
```

---

## ✨ Key Features

- **Multi-Vector Feature Extraction**: Captures URL syntax, character distributions, domain trust indicators, SSL state, and page content.
- **High-Accuracy Random Forest Ensemble**: 100 decision trees tuned with `max_depth=20` to maximize detection rates while minimizing false positives.
- **Fast Real-Time Inference**: Evaluates URLs in milliseconds, with fallback timeouts on page downloads to guarantee system responsiveness.
- **Heuristic Guardrails**: Instant pre-filtering for blatant credential-harvesting indicators (e.g., embedded `@` symbols, deceptive keywords).
- **Clean & Modular Codebase**: Separation of concerns across preprocessing, model artifacts, and web server routines.

---

## 🧪 Machine Learning Pipeline

### Dataset
- **Total Samples**: 11,055 URLs
- **Raw Features**: 30 URL & domain attributes + 1 index column + 1 target column (`Result`)
- **Class Balance**: 6,157 Legitimate (~55.7%) | 4,898 Phishing (~44.3%)

### Preprocessing & Encoding
1. **Index Dropping**: Removed arbitrary identifier column.
2. **Target Encoding**: Remapped `-1` (Phishing) $\to$ `0`, and `1` (Legitimate) $\to$ `1`.
3. **Feature Remapping**:
   - Binary attributes: $\{-1, 1\} \to \{0, 1\}$
   - Ternary attributes: $\{-1, 0, 1\} \to \{0, 0.5, 1\}$ (preserves neutral/suspicious states).
4. **Collinearity Pruning**: Computed full Pearson correlation matrix; dropped features exceeding a correlation threshold of $> 0.90$ (e.g., `Favicon` dropped due to 0.94 correlation with `popUpWidnow`).

### Feature Selection (Chi-2)
Using `SelectKBest` with the Chi-Squared ($\chi^2$) statistic, the top 20 most discriminatory features were retained:

| # | Feature Name | Category | Description |
|---|--------------|----------|-------------|
| 1 | `having_IPhaving_IP_Address` | Lexical | Identifies if an IP address is used directly instead of a domain name |
| 2 | `URLURL_Length` | Lexical | Classifies URL length (standard, suspicious, or excessively long) |
| 3 | `Shortining_Service` | Domain | Detects URL shortening services (e.g., bit.ly, tinyurl) |
| 4 | `having_At_Symbol` | Lexical | Flags presence of `@` which causes browsers to ignore preceding authority |
| 5 | `Prefix_Suffix` | Domain | Checks for dash `-` in domain name, often used in fake brand imitations |
| 6 | `having_Sub_Domain` | Domain | Counts subdomain depth to identify nested misleading domains |
| 7 | `SSLfinal_State` | Security | Verifies SSL/TLS certificate validity, issuer, and age |
| 8 | `Domain_registeration_length`| WHOIS | Analyzes domain expiry period (phishing domains rarely pay for $>1$ yr) |
| 9 | `HTTPS_token` | Lexical | Detects tricky `https` token prepended into the domain text |
| 10 | `Request_URL` | Content | Evaluates ratio of external media/objects loaded from other domains |
| 11 | `URL_of_Anchor` | Content | Checks percentage of anchor links targeting external or empty (`#`) URLs |
| 12 | `Links_in_tags` | Content | Analyzes links inside `<meta>`, `<script>`, and `<link>` tags |
| 13 | `SFH` | Content | Server Form Handler check (empty, about:blank, or external domain) |
| 14 | `Abnormal_URL` | WHOIS | Checks whether host name is matched in WHOIS response |
| 15 | `age_of_domain` | WHOIS | Age of the registered domain ($\ge 6$ months threshold) |
| 16 | `DNSRecord` | DNS | Validates if domain possesses a valid DNS record |
| 17 | `web_traffic` | Network | Traffic rank according to Alexa/SimilarWeb rankings |
| 18 | `Page_Rank` | Network | Google PageRank measurement |
| 19 | `Google_Index` | Search | Verification whether URL is indexed by Google search crawler |
| 20 | `Statistical_report` | Reputation | Known presence on public blacklists (PhishTank, StopBadware) |

### Model Training
- **Model**: `RandomForestClassifier` (`n_estimators=100`, `max_depth=20`, `random_state=42`)
- **Split Strategy**: Stratified 80% Train (8,844 samples) / 20% Test (2,211 samples)
- **Serialization**: Saved as `model.pkl` for instant production loading.

---

## 📁 Repository Structure

```text
Phishing-Detection-Using-Supervised-ML-Model/
│
├── .gitignore                   # Excludes environments, secrets, caches, and local files
├── requirements.txt             # Project dependencies for production & development
├── README.md                    # Comprehensive documentation and usage guide
│
├── app.py                       # Flask web server & real-time prediction endpoint
├── phishing_preprocessing.py    # Complete data cleaning & feature selection pipeline
├── phishing_detection.ipynb     # Jupyter Notebook with exploratory analysis & model training
│
├── dataset.csv                  # Benchmark phishing dataset (11,055 records)
├── model.pkl                    # Serialized trained Random Forest Classifier
│
├── Templates/
│   └── index.html               # Clean HTML template for URL input and prediction feedback
│
├── pyrefly.toml                 # Linter / type-checking configuration
└── pyrightconfig.json           # Python language server settings
```

---

## 🚀 Getting Started

### Prerequisites
- **Python**: Version 3.9 or higher installed on your system
- **Pip**: Latest version of Python package manager
- **Git**: For cloning the repository

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AchyuthCV/Phishing-Detection-Using-Supervised-ML-Model.git
   cd Phishing-Detection-Using-Supervised-ML-Model
   ```

2. **Create a virtual environment:**
   ```bash
   # On macOS/Linux:
   python3 -m venv .venv
   source .venv/bin/activate

   # On Windows:
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Web Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```
2. **Access the application:**
   Open your browser and navigate to:
   ```text
   http://127.0.0.1:5000
   ```
3. **Analyze a URL:**
   Input any web address into the search bar and click **Check** to view the classification result.

---

## 🖥️ Live Inference Example

| Input URL | Prediction Result | Primary Factors |
| :--- | :--- | :--- |
| `https://www.google.com` | **Legitimate Website ✅** | Valid SSL, established domain age, high traffic rank |
| `http://verify-bank-security-update.account-login.ru` | **Phishing Website ⚠️** | Excessive subdomains, keywords (`verify`, `bank`, `update`), dashes |
| `http://secure@paypal-verify.com/login.html` | **Phishing Website ⚠️** | Embedded `@` sign, deceptive prefix/suffix pattern |

---

## 🔒 Security & Best Practices

- **Zero Secret Leaks**: Configured `.gitignore` enforces exclusion of `.env` files, API credentials, private SSH keys, and virtual environment directories.
- **Safe Web Requests**: The scraping module utilizes safe requests with stringent timeouts (`timeout=3s`) and error guards to prevent server freezing on dead or malicious hostnames.
- **Reproducible Pipeline**: `phishing_preprocessing.py` can be executed independently to reproduce feature matrices from raw data without data leakage.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — feel free to use and adapt it for academic, research, or commercial applications.
