import streamlit as st
import pickle
import re
import validators
import ssl
import socket
import tldextract
import whois
import pandas as pd
from datetime import datetime
import os

st.set_page_config(
    page_title="CyberGuard",
    page_icon="🛡",
    layout="wide"
)

model = pickle.load(open("model.pkl", "rb"))
vec = pickle.load(open("vectorizer.pkl", "rb"))

bad_domains = open("bad_domains.txt").read().splitlines()

HISTORY_FILE = "scan_history.csv"

#WORD RULES 
phish_words = [
    "urgent","verify","suspend","click","login","reset",
    "alert","immediately","prize","reward","bank",
    "otp","password","confirm","limited","expire"
]
def rule_score(text):
    score = 0
    reasons = []

    t = text.lower()

    for w in phish_words:
        if w in t:
            score += 1
            reasons.append(f"suspicious word: {w}")

    if "http" in t:
        score += 1
        reasons.append("contains link")

    if re.search(r"\$\d+", t):
        score += 1
        reasons.append("money amount mentioned")
    if t.count("!") >= 3:
        score += 1
        reasons.append("too many exclamation marks")

    return score, reasons

def check_ssl(domain):
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
            s.settimeout(3)
            s.connect((domain, 443))
            return True
    except:
        return False

def domain_age(domain):
    try:
        w = whois.whois(domain)
        created = w.creation_date

        if isinstance(created, list):
            created = created[0]

        if created:
            age_days = (datetime.now() - created).days
            return age_days
    except:
        pass
    return None

def url_score(url):
    score = 0
    reasons = []

    if len(url) > 60:
        score += 1
        reasons.append("long url")

    # Check for HTTP vs HTTPS
    if url.startswith("http://"):
        score += 2
        reasons.append("insecure HTTP protocol (not HTTPS)")
    elif not url.startswith("https://") and "://" not in url:
        # URL without protocol assumed insecure
        score += 2
        reasons.append("no HTTPS protocol")

    if re.search(r"\d", url):
        score += 1
        reasons.append("numbers in url")

    ext = tldextract.extract(url)
    domain = ext.domain + "." + ext.suffix

    # blacklist
    if domain.lower() in bad_domains:
        score += 3
        reasons.append("blacklisted domain")

    # risky tld
    risky = [".xyz",".top",".click",".gq",".tk"]
    if any(domain.endswith(r) for r in risky):
        score += 2
        reasons.append("risky TLD")

    # SSL - give higher penalty for missing SSL
    ssl_check = check_ssl(domain)
    if not ssl_check:
        score += 3
        reasons.append("no SSL certificate")
    else:
        score -= 1  # Slight reduction for having SSL

    # domain age
    age = domain_age(domain)
    if age is not None and age < 180:
        score += 2
        reasons.append("very new domain")
    elif age is not None and age < 365:
        score += 1
        reasons.append("domain less than 1 year old")

    # If no SSL, always add at least 1 to ensure suspicious detection
    if not ssl_check:
        score = max(score, 3)  # Minimum score of 3 for no SSL

    return score, reasons, domain

def attachment_check(name):
    risky_ext = [".exe",".bat",".scr",".js",".zip",".html"]
    if any(name.lower().endswith(e) for e in risky_ext):
        return 2, ["risky attachment type"]
    return 0, []

def save_history(text, url, score, result):
    row = {
        "time": datetime.now(),
        "text": text,
        "url": url,
        "score": score,
        "result": result
    }

    df = pd.DataFrame([row])

    if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
        df.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(HISTORY_FILE, index=False)

st.title("🛡 CyberGuard — Phishing Detection")

text = st.text_area("Email / Message Content")
url = st.text_input("URL")
attachment = st.text_input("Attachment file name (optional)")

if st.button("🔍 Scan Now"):

    total = 0
    reasons = []

    # ---- ML MODEL ----
    if text:
        X = vec.transform([text])
        prob = model.predict_proba(X)[0][1]
        ml_score = int(prob * 5)
        total += ml_score

        if ml_score >= 3:
            reasons.append("ML phishing pattern")

        r_score, r_reason = rule_score(text)
        total += r_score
        reasons += r_reason

    domain = None
    if url:
        if validators.url(url):
            u_score, u_reason, domain = url_score(url)
            total += u_score
            reasons += u_reason
        else:
            total += 2
            reasons.append("invalid url format")
    if attachment:
        a_score, a_reason = attachment_check(attachment)
        total += a_score
        reasons += a_reason

    if text and domain:
        if "@" in text:
            found = re.findall(r'@([A-Za-z0-9.-]+\.[A-Za-z]{2,})', text)
            if found and domain not in found[0]:
                total += 2
                reasons.append("email domain mismatch")

    # ---- RESULT ----
    if total >= 8:
        result = "HIGH RISK"
        st.error("🚨 HIGH PHISHING RISK")
    elif total >= 4:
        result = "SUSPICIOUS"
        st.warning("⚠ Suspicious")
    else:
        result = "SAFE"
        st.success("✅ Likely Safe")

    st.write("Risk Score:", total)

    if reasons:
        st.write("Reasons:")
        for r in reasons:
            st.write("•", r)

    save_history(text, url, total, result)


#ADMIN PANEL
st.subheader("📊 Scan History")

if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
    try:
        df = pd.read_csv(HISTORY_FILE)
        if not df.empty:
            st.dataframe(df.tail(20))
        else:
            st.info("No scan history yet.")
    except pd.errors.EmptyDataError:
        st.info("No scan history yet.")
else:
    st.info("No scan history yet.")
