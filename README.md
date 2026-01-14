# Project Title: IIITD Receptionist Chatbot

## 1. Description / Objective

This project implements an intelligent multilingual receptionist chatbot capable of handling appointment scheduling, rescheduling, and cancellation. Key features include:
Multilingual support (English, Hindi, French, Spanish)
Both text and voice input modes
Natural language processing for request understanding
Database integration for appointment management
PDF report generation
Secure data handling with encryption

---

## 2. Necessary Libraries / Installation Requirements

Install the required libraries by running:

```bash
pip install -r requirements.txt
```

Alternatively, manually install the libraries:

```bash
!pip install spacy openai-whisper torchaudio pydub soundfile gTTS ipython langdetect deep-translator fpdf
!python -m spacy download en_core_web_sm
```

---

## 3. Commands to Run the Project

Copy and paste the following commands into your terminal:

```bash
# Step 1: Clone the repository
git clone https://github.com/Gerick1107/IIS-Project.git

# Step 2: Navigate into the project directory
cd IIS-Project

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the chatbot
python main.py
```

---

## 4. File Structure

```
IIS-Project/
│
├── README.md                # Project overview and setup instructions
├── requirements.txt         # List of dependencies
├── main.py                  # Main chatbot application
├── data.json                # Default data file 
├── encryption.key           # Auto-generated encryption key for database
├── receptionist_db.sqlite   # SQLite database file (auto-generated)
├── style1colorlarge.png     # Logo for PDF reports
├── WhatsAppAudio2025-04-17  # Sample Audio Recording saying "Appointment Management"
| at03.00.58_7ec72598.waptt
├── appointment_*.txt        # Appointment details
├── report_*.pdf             # Generated PDF reports
```

---
> **Note:** Username: admin | Password: admin123 for initial database access
Link to Github: https://github.com/Gerick1107/IIS-Project.git
