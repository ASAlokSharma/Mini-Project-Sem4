"""
PHASE 5: Deployment, Automation & Alerts
==========================================
Three parts:

  Part A — Scheduled data fetching + model retraining (APScheduler)
  Part B — Email/SMS weather alerts (smtplib + optional Twilio)
  Part C — Deployment config files (Dockerfile, requirements.txt,
            streamlit secrets template)

SETUP:
  pip install apscheduler requests smtplib twilio python-dotenv
"""

# ══════════════════════════════════════════════════════════════════════
# PART A: Automated Scheduler
# ══════════════════════════════════════════════════════════════════════
# scheduler.py — run this as a background process alongside the dashboard

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/scheduler.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("scheduler")


def job_fetch_weather():
    """
    Runs every hour.
    Fetches fresh current conditions for all cities and appends to the
    rolling CSV — so the ML training dataset grows over time.
    """
    log.info("Running hourly weather fetch...")
    try:
        # Import here to avoid circular imports
        from phase1_data_collection import collect_all_data
        collect_all_data(output_dir="data/live")
        log.info("Hourly fetch complete")
    except Exception as e:
        log.error(f"Hourly fetch failed: {e}")


def job_retrain_models():
    """
    Runs every Sunday at 02:00.
    Incorporates the last week's new data and retrains the XGBoost model.
    Keeps the old model as a backup before replacing it.
    """
    log.info("Starting weekly model retraining...")
    try:
        import shutil
        from phase2_feature_engineering import run_preprocessing
        from phase3_model_training import run_training

        # Back up current model
        if os.path.exists("models/xgboost_model.pkl"):
            shutil.copy(
                "models/xgboost_model.pkl",
                f"models/xgboost_model_backup_{datetime.now().strftime('%Y%m%d')}.pkl"
            )
            log.info("Current model backed up")

        # Reprocess + retrain
        run_preprocessing(raw_csv="data/historical_weather.csv")
        run_training()
        log.info("Weekly retraining complete")

    except Exception as e:
        log.error(f"Retraining failed: {e}")


def job_check_alerts():
    """
    Runs every 30 minutes.
    Fetches current conditions and fires alerts if thresholds are breached.
    """
    log.info("Checking alert thresholds...")
    try:
        from phase5_alerts import check_and_send_alerts
        check_and_send_alerts()
    except Exception as e:
        log.error(f"Alert check failed: {e}")


def start_scheduler():
    os.makedirs("logs", exist_ok=True)
    scheduler = BackgroundScheduler(timezone="UTC")

    # Every hour at :05 past the hour
    scheduler.add_job(job_fetch_weather, CronTrigger(minute=5), id="fetch_weather")

    # Every day at 06:00 UTC — daily alert check
    scheduler.add_job(job_check_alerts, CronTrigger(hour=6, minute=0), id="daily_alerts")

    # Every Sunday at 02:00 UTC
    scheduler.add_job(
        job_retrain_models,
        CronTrigger(day_of_week="sun", hour=2, minute=0),
        id="retrain_models",
    )

    scheduler.start()
    log.info("Scheduler started. Jobs: " + str([j.id for j in scheduler.get_jobs()]))

    try:
        import time
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        log.info("Scheduler stopped")


# ══════════════════════════════════════════════════════════════════════
# PART B: Weather Alerts
# ══════════════════════════════════════════════════════════════════════
# phase5_alerts.py

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from dotenv import load_dotenv

load_dotenv()

# Alert thresholds — customise for your region
ALERT_RULES = [
    {
        "name":      "Extreme heat",
        "condition": lambda d: d["main"]["temp"] > 42,
        "message":   lambda d: f"Temperature is {d['main']['temp']:.1f}°C — extreme heat warning.",
        "severity":  "high",
    },
    {
        "name":      "Heavy rain",
        "condition": lambda d: d.get("rain", {}).get("1h", 0) > 20,
        "message":   lambda d: f"Heavy rain: {d['rain']['1h']:.1f}mm in the last hour.",
        "severity":  "medium",
    },
    {
        "name":      "High wind",
        "condition": lambda d: d["wind"]["speed"] > 15,
        "message":   lambda d: f"Strong winds: {d['wind']['speed']:.1f} m/s.",
        "severity":  "medium",
    },
    {
        "name":      "Thunderstorm",
        "condition": lambda d: d["weather"][0]["main"] == "Thunderstorm",
        "message":   lambda d: "Thunderstorm detected in your area.",
        "severity":  "high",
    },
]

CITIES = {
    "New Delhi": (28.6139, 77.2090),
    "Mumbai":    (19.0760, 72.8777),
}

def send_email_alert(subject: str, body: str):
    """
    Sends an HTML email via Gmail SMTP.
    Set GMAIL_USER and GMAIL_APP_PASSWORD in your .env file.
    (Use a Gmail App Password, not your regular password.)
    """
    sender   = os.getenv("GMAIL_USER")
    password = os.getenv("GMAIL_APP_PASSWORD")
    recipient= os.getenv("ALERT_EMAIL", sender)

    if not sender or not password:
        print("Email credentials not set — skipping email alert")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = recipient

    html_body = f"""
    <html><body>
    <div style="font-family: sans-serif; max-width: 500px; padding: 20px;">
      <h2 style="color: #D85A30;">⚠️ Weather Alert</h2>
      <p style="font-size: 16px;">{body}</p>
      <p style="color: gray; font-size: 12px;">
        Sent at {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} by your Weather Dashboard
      </p>
    </div>
    </body></html>
    """

    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
        print(f"Email alert sent: {subject}")
    except Exception as e:
        print(f"Email failed: {e}")


def send_sms_alert(message: str):
    """
    Sends an SMS via Twilio (optional).
    Requires: pip install twilio
    Set TWILIO_SID, TWILIO_AUTH, TWILIO_FROM, TWILIO_TO in .env
    """
    try:
        from twilio.rest import Client
        client = Client(
            os.getenv("TWILIO_SID"),
            os.getenv("TWILIO_AUTH"),
        )
        client.messages.create(
            body=f"🌦️ Weather Alert: {message}",
            from_=os.getenv("TWILIO_FROM"),
            to=os.getenv("TWILIO_TO"),
        )
        print("SMS alert sent")
    except ImportError:
        print("Twilio not installed — skipping SMS")
    except Exception as e:
        print(f"SMS failed: {e}")


def check_and_send_alerts():
    """
    Checks all cities against all rules and fires alerts as needed.
    Uses a simple file-based deduplication so the same alert
    isn't sent more than once per 6 hours.
    """
    OWM_API_KEY = os.getenv("OWM_API_KEY")
    sent_file   = "data/sent_alerts.txt"

    # Load already-sent alert keys
    sent = set()
    if os.path.exists(sent_file):
        with open(sent_file) as f:
            sent = set(f.read().splitlines())

    def alert_key(city, rule_name, hour_bucket):
        return f"{city}:{rule_name}:{hour_bucket}"

    new_sent = []

    for city, (lat, lon) in CITIES.items():
        try:
            resp = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"lat": lat, "lon": lon, "appid": OWM_API_KEY, "units": "metric"},
                timeout=10,
            )
            data = resp.json()
        except Exception as e:
            print(f"Failed to fetch {city}: {e}")
            continue

        for rule in ALERT_RULES:
            try:
                if rule["condition"](data):
                    # Deduplicate: only send once per 6-hour window
                    hour_bucket = datetime.now().strftime("%Y%m%d_%H")[:-1]  # floor to 6h
                    key = alert_key(city, rule["name"], hour_bucket)

                    if key not in sent:
                        msg = f"[{city}] {rule['message'](data)}"
                        print(f"ALERT ({rule['severity']}): {msg}")

                        send_email_alert(
                            subject=f"⚠️ Weather Alert — {city}: {rule['name']}",
                            body=msg,
                        )

                        if rule["severity"] == "high":
                            send_sms_alert(msg)

                        new_sent.append(key)

            except Exception as e:
                print(f"Rule '{rule['name']}' failed for {city}: {e}")

    # Persist sent keys (keep last 500 to avoid unbounded growth)
    all_sent = list(sent) + new_sent
    with open(sent_file, "w") as f:
        f.write("\n".join(all_sent[-500:]))


# ══════════════════════════════════════════════════════════════════════
# PART C: Deployment config files (printed for reference)
# ══════════════════════════════════════════════════════════════════════

DOCKERFILE = """
# Dockerfile — containerise the Streamlit app for any cloud platform

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run dashboard
ENTRYPOINT ["streamlit", "run", "phase4_dashboard.py",
            "--server.port=8501", "--server.address=0.0.0.0"]
"""

REQUIREMENTS = """
# requirements.txt — pin versions for reproducibility
streamlit==1.35.0
plotly==5.22.0
pandas==2.2.2
numpy==1.26.4
requests==2.32.3
scikit-learn==1.5.0
xgboost==2.0.3
tensorflow==2.16.1
apscheduler==3.10.4
python-dotenv==1.0.1
twilio==8.5.0
"""

STREAMLIT_SECRETS = """
# .streamlit/secrets.toml — used for Streamlit Community Cloud deployment
# Do NOT commit this file to git.

OWM_API_KEY = "your_openweathermap_key_here"
GMAIL_USER  = "your_gmail@gmail.com"
GMAIL_APP_PASSWORD = "your_16_char_app_password"
ALERT_EMAIL = "alerts@youremail.com"
"""

ENV_TEMPLATE = """
# .env — local development secrets (never commit to git)
OWM_API_KEY=your_openweathermap_api_key
GMAIL_USER=your_email@gmail.com
GMAIL_APP_PASSWORD=xxxx_xxxx_xxxx_xxxx
ALERT_EMAIL=your_alert_recipient@email.com
TWILIO_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH=your_auth_token
TWILIO_FROM=+1234567890
TWILIO_TO=+0987654321
"""

if __name__ == "__main__":
    import sys

    if "--write-configs" in sys.argv:
        os.makedirs(".streamlit", exist_ok=True)
        with open("Dockerfile", "w") as f:
            f.write(DOCKERFILE.strip())
        with open("requirements.txt", "w") as f:
            f.write(REQUIREMENTS.strip())
        with open(".streamlit/secrets.toml.template", "w") as f:
            f.write(STREAMLIT_SECRETS.strip())
        with open(".env.template", "w") as f:
            f.write(ENV_TEMPLATE.strip())
        print("Config files written: Dockerfile, requirements.txt, .streamlit/secrets.toml.template, .env.template")

    elif "--start-scheduler" in sys.argv:
        start_scheduler()

    else:
        print("Usage:")
        print("  python phase5_deployment.py --write-configs     # write Dockerfile + requirements")
        print("  python phase5_deployment.py --start-scheduler   # start background scheduler")
