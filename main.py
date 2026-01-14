import json
import random
import re
import os
import spacy
from fpdf import FPDF
from spacy.matcher import Matcher
from deep_translator import GoogleTranslator
from datetime import datetime
import whisper
import soundfile as sf
import numpy as np
import tempfile
import torch
import torchaudio
from gtts import gTTS
from IPython.display import Audio, display
import sqlite3
from sqlite3 import Error
from cryptography.fernet import Fernet
import getpass

torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("small", device=DEVICE)
CONFIDENCE_THRESHOLD = 0.5
SUPPORTED_LANGUAGES = {
    'english': 'en',
    'hindi': 'hi',
    'french': 'fr',
    'spanish': 'es'
}
WHISPER_LANG_MAP = {
    'en': 'english',
    'hi': 'hindi',
    'fr': 'french',
    'es': 'spanish'
}
current_language = 'en'
current_input_method = 'text'
enable_tts = False

def translate_to_english(text):
    if current_language == 'en':
        return text
    try:
        return GoogleTranslator(source=current_language, target='en').translate(text)
    except Exception as e:
        print(f"Translation error (to English): {e}")
        return text

def translate_from_english(text):
    if current_language == 'en':
        return text
    try:
        return GoogleTranslator(source='en', target=current_language).translate(text)
    except Exception as e:
        print(f"Translation error (from English): {e}")
        return text

def detect_input_language(text):
    try:
        from langdetect import detect, LangDetectException
        try:
            lang_code = detect(text)
            return SUPPORTED_LANGUAGES.get(WHISPER_LANG_MAP.get(lang_code, '').lower())
        except LangDetectException:
            return None
    except ImportError:
        if any(word in text.lower() for word in ["नमस्ते", "धन्यवाद"]):
            return 'hi'
        elif any(word in text.lower() for word in ["bonjour", "merci"]):
            return 'fr'
        elif any(word in text.lower() for word in ["hola", "gracias"]):
            return 'es'
        return None

def handle_language_switch(detected_lang, input_method):
    global current_language
    if detected_lang and detected_lang != current_language:
        lang_name = [k for k,v in SUPPORTED_LANGUAGES.items() if v == detected_lang][0]
        confirm_msg = {
            'text': f"I noticed you're speaking in {lang_name.capitalize()}. Would you like to switch languages? (yes/no)",
            'audio': "I noticed a different language. Should I switch to match?"
        }[input_method]
        print_translated(confirm_msg)
        response = get_translated_input("").lower()
        if response in ['yes', 'y', 'हाँ', 'oui', 'sí']:
            current_language = detected_lang
            print_translated(f"Language switched to {lang_name.capitalize()}!")
            return True
    return False

def check_whisper_confidence(result):
    if not result or 'segments' not in result:
        return (False, "", False)
    segments = result['segments']
    if not segments:
        return (False, "", False)
    avg_confidence = sum(seg['confidence'] for seg in segments) / len(segments)
    combined_text = ' '.join(seg['text'] for seg in segments).strip()
    if avg_confidence < CONFIDENCE_THRESHOLD:
        return (False, combined_text, True)
    return (True, combined_text, False)

def get_translated_input(prompt):
    global current_input_method, current_language
    translated_prompt = translate_from_english(prompt)
    if current_input_method == 'text':
        user_input = input(f"{translated_prompt}\nYou: ").strip()
        if not user_input or user_input.lower() in ["umm...", "umm","um","uhh..","uhh","uh","hmm...","hmm","hm","..."]:
            print_translated("Could you please provide more details?")
            return get_translated_input(prompt)
        detected_lang = detect_input_language(user_input)
        handle_language_switch(detected_lang, 'text')
        return translate_to_english(user_input)
    else:
        print(f"{translated_prompt}")
        print(translate_from_english("Please upload your audio file (or type 'text' to switch to text input):"))
        audio_path = input().strip()
        if audio_path.lower() == 'text':
            current_input_method = 'text'
            return get_translated_input(prompt)
        if not os.path.exists(audio_path):
            print(translate_from_english("Error: File not found. Please check the path and try again."))
            return get_translated_input(prompt)
        try:
            audio_data = None
            sample_rate = 16000
            try:
                audio_data, sample_rate = sf.read(audio_path)
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
            except:
                try:
                    waveform, sample_rate = torchaudio.load(audio_path)
                    audio_data = waveform.numpy().squeeze()
                except Exception as e:
                    raise ValueError(f"Failed to load audio: {str(e)}")
            if audio_data is None or len(audio_data) == 0:
                raise ValueError("No audio data could be extracted")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                tmp_path = tmpfile.name
            try:
                sf.write(tmp_path, audio_data, sample_rate)
                result = whisper_model.transcribe(tmp_path)
                if not result or 'segments' not in result:
                    raise ValueError("Whisper returned empty results")
                segments = result.get('segments', [])
                if not segments:
                    print_translated("No speech was detected in the audio.")
                    return get_translated_input(prompt)
                try:
                    confidences = [seg.get('confidence', 0) for seg in segments]
                    avg_confidence = sum(confidences)/len(confidences)
                    transcribed_text = ' '.join(seg.get('text', '').strip() for seg in segments).strip()
                    if not transcribed_text:
                        print_translated("Audio processed but no text was recognized.")
                        return get_translated_input(prompt)
                except (TypeError, ZeroDivisionError) as e:
                    print_translated("There was a problem interpreting the results.")
                    return get_translated_input(prompt)
                if avg_confidence < CONFIDENCE_THRESHOLD:
                    print_translated(f"Warning: Low confidence ({int(avg_confidence*100)}%). Please speak clearly or say 'text' to switch.")
                    print_translated(f"I heard: '{transcribed_text}'. Is this correct? (yes/no)")
                    confirmation = input().strip().lower()
                    if confirmation not in ['yes', 'y', 'हाँ', 'oui', 'sí']:
                        print_translated("Audio processing failed. Please try typing instead.")
                        return get_translated_input(prompt)
                detected_lang = result.get('language', current_language)
                if handle_language_switch(detected_lang, 'audio'):
                    translated_prompt = translate_from_english(prompt)
                    print(f"{translated_prompt}")
                print(translate_from_english(f"Detected: {transcribed_text}"))
                return translate_to_english(transcribed_text)
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        except ValueError as e:
            error_msg = str(e).lower()
            if "unsupported" in error_msg or "format" in error_msg:
                print(translate_from_english("Unsupported audio format. Please use WAV, MP3, FLAC or OPUS."))
            elif "corrupt" in error_msg or "invalid" in error_msg:
                print(translate_from_english("The audio file appears corrupt or invalid."))
            else:
                print(translate_from_english(f"Error processing audio: {error_msg}"))
            return get_translated_input(prompt)
        except Exception as e:
            print(translate_from_english(f"An unexpected error occurred: {str(e)}"))
            print_translated("Audio processing failed. Please try typing instead.")
            current_input_method = 'text'
            return get_translated_input(prompt)
          
def print_translated(text, end="\n"):
    translated_text = translate_from_english(text)
    lang_indicator = f"[{current_language.upper()}]"
    print(f"Chatbot ({lang_indicator}): {translated_text}", end=end)
    if enable_tts:
        audio_file = text_to_speech(translated_text, current_language)
        if audio_file:
            display(Audio(audio_file, autoplay=False))

def select_language():
    global current_language
    print("Please select your preferred language / कृपया अपनी पसंदीदा भाषा चुनें / Veuillez sélectionner votre langue préférée / Por favor seleccione su idioma preferido:")
    for i, lang in enumerate(SUPPORTED_LANGUAGES.keys(), 1):
        print(f"{i}. {lang.capitalize()}")
    while True:
        choice = input("Enter choice (1-4): ")
        if choice.isdigit() and 1 <= int(choice) <= 4:
            selected_lang = list(SUPPORTED_LANGUAGES.keys())[int(choice)-1]
            current_language = SUPPORTED_LANGUAGES[selected_lang]
            print_translated(f"Language set to {selected_lang}")
            return
        print("Invalid choice. Please enter a number between 1 and 4.")

def select_input_method():
    global current_input_method
    print_translated("Please select your preferred input method:")
    print("1. Text")
    print("2. Audio upload")
    while True:
        choice = input("Enter choice (1-2): ")
        if choice == '1':
            current_input_method = 'text'
            print_translated("Text input selected")
            return
        elif choice == '2':
            current_input_method = 'audio'
            print_translated("Audio input selected")
            print_translated("Note: Please upload audio files in WAV, MP3, or FLAC format")
            return
        print_translated("Invalid choice. Please enter 1 or 2.")

def ask_to_enable_tts():
    global enable_tts
    print("Would you like to enable text-to-speech output? (yes/no)")
    choice = input().strip().lower()
    enable_tts = choice in ['yes', 'y']
    if enable_tts:
        print("Text-to-speech enabled. Audio buttons will appear with responses.")

def text_to_speech(text, language):
    try:
        temp_file = f"temp_{hash(text) % 1000}.mp3"
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(temp_file)
        return temp_file
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return None

def play_audio(audio_file):
    if audio_file and os.path.exists(audio_file):
        display(Audio(audio_file, autoplay=False))
        return True
    return False

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
patterns = {
    "RESCHEDULE_APPOINTMENT": [
        [{"LOWER": "reschedule"}, {"OP": "*"}, {"LOWER": {"IN": ["appointment", "meeting", "session"]}}]
    ],
    "BOOK_APPOINTMENT": [
        [{"LOWER": {"IN": ["book", "schedule"]}}, {"OP": "*"}, {"LOWER": {"IN": ["appointment", "meeting", "session"]}}]
    ],
    "CANCEL_APPOINTMENT": [
        [{"LOWER": "cancel"}, {"OP": "*"}, {"LOWER": {"IN": ["appointment", "meeting", "session"]}}]
    ],
    "ATTENDING_APPOINTMENT": [
        [{"LOWER": {"IN": ["attend", "attending", "here"]}}, {"OP": "*"}, {"LOWER": {"IN": ["appointment", "meeting", "session"]}}],
        [{"LOWER": "meet"}, {"OP": "*"}]
    ]
}
for label, pattern_list in patterns.items():
    matcher.add(label, pattern_list)

def extract_key_elements(text):
    doc = nlp(text)
    result = {}
    matches = matcher(doc)
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        result["task"] = label.replace("_", " ").lower()
        break
    persons = []
    for ent in doc.ents:
        if ent.label_ in ["DATE", "TIME"]:
            result[ent.label_.lower()] = ent.text
        elif ent.label_ == "PERSON":
            persons.append(ent.text)
    if persons:
        if len(persons) == 1:
            result["person"] = {"name": persons[0]}
        else:
            result["person"] = [{"name": p} for p in persons]
    meeting_mode = "offline"
    online_indicators = {"online", "virtual"}
    for token in doc:
        if token.lower_ in online_indicators:
            meeting_mode = "online"
            break
    result["meeting_mode"] = meeting_mode

    return result

def loadfile():
    try:
        with open("data.json", "r") as file:
            data = json.load(file)
        print_translated("Data Successfully Loaded.")
        return data
    except FileNotFoundError:
        print_translated("Data Not Found. Please Ensure 'data.json' Exists.")
    except json.JSONDecodeError:
        print_translated("Error Decoding The Data File.")

def save_data(data):
    try:
        with open("data.json", "w") as file:
            json.dump(data, file, indent=4)
        print_translated("Data successfully saved to data.json.")
    except Exception as e:
        print_translated(f"Error saving data to data.json: {e}")

def getip(prompt, options):
    while True:
        print_translated(prompt)
        ip = get_translated_input("")
        for option in options:
            if ip.lower() in option.lower():
                return option
        print_translated("Invalid Input. Please Try Again.")

def gettype(data):
    while True:
        ut = getip("Staff Member / Student / Visitor", data["Type"].keys())
        print_translated(f"What category of {ut}s do you belong to?")
        ust = getip(" / ".join(data["Type"][ut]), data["Type"][ut])
        return ut, ust

def validate_phone():
    while True:
        print_translated("Please enter your 10-digit phone number:")
        phone = get_translated_input("")
        if re.fullmatch(r"\d{10}", phone):
            return phone
        print_translated("Invalid phone number. Please enter exactly 10 digits.")

def validate_email():
    while True:
        print_translated("Please enter your email address:")
        email = get_translated_input("")
        if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            return email
        print_translated("Invalid email format. Please enter a valid email address.")

def generate_meet_link():
    return f"https://meet.google.com/{''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=10))}"

def appointment1(prof_slots):
    while True:
        time = f"{random.randint(10, 17)}:{random.choice(['00', '30'])}"
        if time not in prof_slots:
            return time

def confirm_appointment(prof_slots):
    while True:
        time = appointment1(prof_slots)
        print_translated(f"Your appointment is scheduled at {time}. Is this fine? (yes/no)")
        confirm = get_translated_input("")
        if confirm.lower() in ["yes", "y"]:
            return time
        elif confirm.lower() in ["no", "n"]:
            print_translated("Please enter your preferred time (HH:MM):")
            preferred_time = get_translated_input("")
            if preferred_time not in prof_slots:
                return preferred_time
            else:
                print_translated("Sorry, that slot is already taken. Please choose another time.")

def appointment2(data, name, phone, email, role, subrole, db_entry):
    while True:
        print_translated("Which professor are/were you to meet?")
        nip = get_translated_input("")
        for i in ["mr.", "ms.", "mrs.", "mr", "ms", "mrs", "mister", "miss", "sir", "ma'am", "maam", "prof", "professor"]:
            if nip.lower().startswith(i):
                nip = nip[len(i):].strip()
            elif nip.lower().endswith(i):
                nip = nip[:-len(i)].strip()
        prof = None
        for p in data["Appointment"].keys():
            if nip.lower() in p.lower():
                prof = p
                break
        if prof:
            venue = data["Appointment"][prof]["Venue"]
            if venue is None:
                venue = {"Building": "N/A", "Room": "N/A"}
            print_translated(f"{prof} is available in {venue['Building']}, Room {venue['Room']}.")
            print_translated("Do you already have an appointment? (yes/no)")
            action = get_translated_input("")
            if action.lower() in ["yes", "y"]:
                print_translated(f"Your appointment is at {venue['Building']}, Room {venue['Room']}.")
                print_translated("Would you like to reschedule or cancel your appointment? (reschedule/cancel/no)")
                action = get_translated_input("")
                if action.lower() == "reschedule":
                    print_translated("Please enter the time of your current appointment to reschedule (HH:MM):")
                    old_time = get_translated_input("")
                    if old_time in data["Appointment"][prof]["Slots"]:
                        data["Appointment"][prof]["Slots"].remove(old_time)
                        new_time = confirm_appointment(data["Appointment"][prof]["Slots"])
                        data["Appointment"][prof]["Slots"].append(new_time)
                        print_translated(f"Your appointment with {prof} has been rescheduled to {new_time} in {venue['Building']}, Room {venue['Room']}.")
                        createmeetingfile(name, phone, email, role, subrole, prof, new_time, venue, None, False, action="reschedule")
                        save_data(data)
                    else:
                        print_translated(f"No appointment found for {prof} at {old_time}.")
                elif action.lower() == "cancel":
                    print_translated("Please enter the time of your appointment to cancel (HH:MM):")
                    time = get_translated_input("")
                    if time in data["Appointment"][prof]["Slots"]:
                        data["Appointment"][prof]["Slots"].remove(time)
                        print_translated(f"Your appointment with {prof} at {time} has been canceled.")
                        createmeetingfile(name, phone, email, role, subrole, prof, None, {"Building": "N/A", "Room": "N/A"}, None, False, action="cancel")
                        save_data(data)
                    else:
                        print_translated(f"No appointment found for {prof} at {time}.")
                return
            time = confirm_appointment(data["Appointment"][prof]["Slots"])
            data["Appointment"][prof]["Slots"].append(time)
            print_translated("Would you prefer an online meeting? (yes/no)")
            meet_option = get_translated_input("")
            is_online = False
            meet_link = None
            if meet_option.lower() in ["yes", "y"]:
                meet_link = generate_meet_link()
                print_translated(f"Your appointment with {prof} is scheduled at {time}. Meet link: {meet_link}")
                is_online = True
            else:
                print_translated(f"Your appointment with {prof} is scheduled at {time} in {venue['Building']}, Room {venue['Room']}.")
            db_entry["interacrtion details"] = {"prof": prof, "time": time, "venue": venue, "meet_link": meet_link, "is_online": is_online}
            createmeetingfile(name, phone, email, role, subrole, prof, time, venue, meet_link, is_online)
            save_data(data)
            return
        else:
            print_translated("Professor not found. Please try again.")
            break

def createmeetingfile(name, phone, email, role, subrole, prof, time, venue, meet_link, is_online, action=None):
    file_number = 1
    target_file = None
    target_pdf = None
    if action == "reschedule":
        for fname in os.listdir():
            if fname.startswith(f"{name}_appointmentdetails_") and fname.endswith(".txt"):
                with open(fname, "r") as f:
                    if prof in f.read():
                        file_number = int(fname.split("_")[-1].split(".")[0])
                        target_file = fname
                        target_pdf = fname.replace("appointmentdetails_", "report_").replace(".txt", ".pdf")
                        break
    if not target_file:
        while True:
            target_file = f"{name}_appointmentdetails_{file_number}.txt"
            target_pdf = f"{name}_report_{file_number}.pdf"
            if not os.path.exists(target_file) and not os.path.exists(target_pdf):
                break
            file_number += 1
    appointment_data = f"Name: {name}\nPhone: {phone}\nEmail: {email}\nRole of user: {role}\n"
    appointment_data += f"Name of prof: {prof}\nAppointment timings: {time}\n"
    if is_online:
        appointment_data += f"Venue: N/A\nMeet link: {meet_link}\n\n"
    else:
        appointment_data += f"Venue: {venue['Building']}, Room {venue['Room']}\nMeet link: N/A\n\n"
    conn = create_connection()
    if not conn:
        print_translated("Warning: Could not connect to database")
    else:
        try:
            encrypted_phone = encrypt_data(phone)
            encrypted_email = encrypt_data(email)
            if action == "reschedule":
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM appointments WHERE name=? AND professor=?", (name, prof))
                appt_id = cursor.fetchone()
                if appt_id:
                    update_appointment(
                        conn, appt_id[0],
                        time,
                        venue['Building'] if not is_online else None,
                        venue['Room'] if not is_online else None,
                        meet_link if is_online else None,
                        is_online
                    )
            elif action == "cancel":
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM appointments WHERE name=? AND professor=?", (name, prof))
                appt_id = cursor.fetchone()
                if appt_id:
                    delete_appointment(conn, appt_id[0])
            else:
                add_appointment(
                    conn,
                    (
                        name, encrypted_phone, encrypted_email, role, subrole,
                        prof, time,
                        venue['Building'] if not is_online else None,
                        venue['Room'] if not is_online else None,
                        meet_link if is_online else None,
                        is_online
                    )
                )
        except Error as e:
            print_translated(f"Database operation failed: {e}")
        finally:
            conn.close()
    if action == "reschedule":
        with open(target_file, "w") as file:
            file.write(appointment_data)
        print_translated(f"Your appointment with {prof} has been rescheduled. Details updated in {target_file}.")
        receptionist_details = {"Name": name, "Phone": phone, "Email": email, "Role": role}
        appointment_details = [{
            "Professor": prof,
            "Time": time,
            "Venue": f"{venue['Building']}, Room {venue['Room']}" if not is_online else "N/A",
            "Meet Link": meet_link if is_online else "N/A"
        }]
        generate_report(receptionist_details, appointment_details, target_pdf)
    elif action == "cancel":
        if os.path.exists(target_file):
            os.remove(target_file)
            print_translated(f"Deleted {target_file}")
        if os.path.exists(target_pdf):
            os.remove(target_pdf)
            print_translated(f"Deleted {target_pdf}")
    else:
        with open(target_file, "w") as file:
            file.write(appointment_data)
        print_translated(f"Meeting details have been saved to {target_file}.")

        receptionist_details = {"Name": name, "Phone": phone, "Email": email, "Role": role}
        appointment_details = [{
            "Professor": prof,
            "Time": time,
            "Venue": f"{venue['Building']}, Room {venue['Room']}" if not is_online else "N/A",
            "Meet Link": meet_link if is_online else "N/A"
        }]
        generate_report(receptionist_details, appointment_details, target_pdf)

def generate_report(receptionist_details, appointment_details, filename):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", style='B', size=16)
    pdf.image('style1colorlarge.png', x=10, y=8, w=30)
    pdf.cell(200, 10, "RECEPTIONIST REPORT", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Helvetica", style='B', size=14)
    pdf.cell(200, 10, "Receptionist Details", ln=True, border='B')
    pdf.set_font("Helvetica", size=12)
    pdf.ln(5)
    col_width = 95
    row_height = 10
    for key, value in receptionist_details.items():
        pdf.cell(col_width, row_height, key, border=1)
        pdf.cell(col_width, row_height, str(value), border=1, ln=True)
    pdf.ln(10)
    pdf.set_font("Helvetica", style='B', size=14)
    pdf.cell(200, 10, "Appointment Details", ln=True, border='B')
    pdf.set_font("Helvetica", size=12)
    pdf.ln(5)
    col_widths = [40, 50, 50, 50]
    headers = ["Professor", "Appointment Time", "Venue", "Meet Link"]
    row_height = 10
    pdf.set_fill_color(3, 156, 143)
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], row_height, header, border=1, fill=True, align='C')
    pdf.ln(row_height)
    pdf.set_fill_color(140, 140, 140)
    for entry in appointment_details:
        pdf.cell(col_widths[0], row_height, entry["Professor"], border=1)
        pdf.cell(col_widths[1], row_height, entry["Time"], border=1)
        pdf.cell(col_widths[2], row_height, entry["Venue"], border=1)
        meet_link = entry.get("Meet Link", "N/A")
        if meet_link != "N/A":
            pdf.multi_cell(col_widths[3], row_height, meet_link, border=1, align='C')
        else:
            pdf.cell(col_widths[3], row_height, meet_link, border=1, ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Helvetica", style='I', size=10)
    pdf.cell(200, 10, "Generated by Receptionist Chatbot", ln=True, align='C')
    pdf.output(filename)
    print_translated(f"Receptionist report successfully saved as {filename}.")

DB_NAME = "receptionist_db.sqlite"
KEY_FILE = "encryption.key"

def generate_key():
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as key_file:
        key_file.write(key)

def load_key():
    try:
        return open(KEY_FILE, "rb").read()
    except FileNotFoundError:
        generate_key()
        return open(KEY_FILE, "rb").read()

cipher_suite = Fernet(load_key())

def encrypt_data(data):
    if data is None:
        return None
    return cipher_suite.encrypt(data.encode()).decode()

def decrypt_data(data):
    if data is None:
        return None
    return cipher_suite.decrypt(data.encode()).decode()

def create_connection():
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        return conn
    except Error as e:
        print_translated(f"Database error: {e}")
    return conn

def create_tables(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin BOOLEAN NOT NULL DEFAULT 0
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encrypted_phone TEXT NOT NULL,
            encrypted_email TEXT NOT NULL,
            role TEXT NOT NULL,
            subrole TEXT NOT NULL,
            professor TEXT NOT NULL,
            time TEXT,
            building TEXT,
            room TEXT,
            meet_link TEXT,
            is_online BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
    except Error as e:
        print_translated(f"Error creating tables: {e}")

def initialize_database():
    conn = create_connection()
    if conn:
        create_tables(conn)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username='admin'")
        if not cursor.fetchone():
            admin_pass = "admin123"
            hashed_pass = encrypt_data(admin_pass)
            cursor.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, 1)",
                          ("admin", hashed_pass))
            conn.commit()
        conn.close()

def add_appointment(conn, appointment_data):
    try:
        sql = """INSERT INTO appointments(
                    name, encrypted_phone, encrypted_email, role, subrole,
                    professor, time, building, room, meet_link, is_online)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)"""
        cursor = conn.cursor()
        cursor.execute(sql, appointment_data)
        conn.commit()
        return cursor.lastrowid
    except Error as e:
        print_translated(f"Error adding appointment: {e}")
        return None

def update_appointment(conn, appointment_id, new_time, new_building, new_room, new_meet_link, new_is_online):
    try:
        sql = """UPDATE appointments
                SET time=?, building=?, room=?, meet_link=?, is_online=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=?"""
        cursor = conn.cursor()
        cursor.execute(sql, (new_time, new_building, new_room, new_meet_link, new_is_online, appointment_id))
        conn.commit()
        return True
    except Error as e:
        print_translated(f"Error updating appointment: {e}")
        return False

def delete_appointment(conn, appointment_id):
    try:
        sql = "DELETE FROM appointments WHERE id=?"
        cursor = conn.cursor()
        cursor.execute(sql, (appointment_id,))
        conn.commit()
        return True
    except Error as e:
        print_translated(f"Error deleting appointment: {e}")
        return False

def find_appointments(conn, search_criteria, is_admin=False):
    try:
        base_query = "SELECT id, name, encrypted_phone, encrypted_email, role, subrole, professor, time, building, room, meet_link, is_online FROM appointments WHERE "
        conditions = []
        params = []
        if "name" in search_criteria:
            conditions.append("name LIKE ?")
            params.append(f"%{search_criteria['name']}%")
        if "professor" in search_criteria:
            conditions.append("professor LIKE ?")
            params.append(f"%{search_criteria['professor']}%")
        if "time" in search_criteria:
            conditions.append("time = ?")
            params.append(search_criteria["time"])
        if not conditions:
            return None
        query = base_query + " AND ".join(conditions)
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        results = []
        for row in rows:
            if is_admin:
                decrypted_phone = decrypt_data(row[2])
                decrypted_email = decrypt_data(row[3])
            else:
                decrypted_phone = "******" + row[2][-4:] if row[2] else "N/A"
                decrypted_email = "******" + row[3][-4:] if row[3] else "N/A"
            results.append({
                "id": row[0],
                "name": row[1],
                "phone": decrypted_phone,
                "email": decrypted_email,
                "role": row[4],
                "subrole": row[5],
                "professor": row[6],
                "time": row[7],
                "building": row[8],
                "room": row[9],
                "meet_link": row[10],
                "is_online": bool(row[11])
            })
        return results
    except Error as e:
        print_translated(f"Error searching appointments: {e}")
        return None

def create_user(conn, username, password, is_admin=False):
    try:
        encrypted_pass = encrypt_data(password)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                      (username, encrypted_pass, is_admin))
        conn.commit()
        return True
    except Error as e:
        print_translated(f"Error creating user: {e}")
        return False

def authenticate_user(conn, username, password):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT password, is_admin FROM users WHERE username=?", (username,))
        result = cursor.fetchone()
        if result:
            stored_pass = decrypt_data(result[0])
            if password == stored_pass:
                return True, bool(result[1])
        return False, False
    except Error as e:
        print_translated(f"Authentication error: {e}")
        return False, False

def view_all_appointments(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, name, encrypted_phone, encrypted_email, role, subrole,
                   professor, time, building, room, meet_link, is_online
            FROM appointments
            ORDER BY time
        """)
        rows = cursor.fetchall()
        if not rows:
            print_translated("No appointments found in database.")
            return
        print_translated("\nALL APPOINTMENTS IN DATABASE")
        print_translated("-" * 50)
        for row in rows:
            print_translated(f"\nAppointment ID: {row[0]}")
            print_translated(f"Name: {row[1]}")
            print_translated(f"Phone: {decrypt_data(row[2])}")
            print_translated(f"Email: {decrypt_data(row[3])}")
            print_translated(f"Role: {row[4]} ({row[5]})")
            print_translated(f"Professor: {row[6]}")
            print_translated(f"Time: {row[7]}")
            if row[11]:
                print_translated(f"Meeting Link: {row[10]}")
            else:
                print_translated(f"Location: {row[8]}, Room {row[9]}")
            print_translated("-" * 50)
        print_translated(f"\nTotal appointments: {len(rows)}")
    except Error as e:
        print_translated(f"Error fetching appointments: {e}")

def database_interface():
    conn = create_connection()
    if not conn:
        print_translated("Error connecting to database")
        return
    print_translated("\nDatabase Access Login")
    username = input("Username: ")
    password = getpass.getpass("Password: ")
    authenticated, is_admin = authenticate_user(conn, username, password)
    if not authenticated:
        print_translated("Invalid credentials")
        conn.close()
        return
    print_translated(f"\nWelcome {'Admin' if is_admin else 'User'}!")
    while True:
        print_translated("\nOptions:")
        print_translated("1. Search appointments")
        if is_admin:
            print_translated("2. View all appointments")
            print_translated("3. Add new user")
            print_translated("4. Exit")
            choice = input("Select option (1-4): ")
        else:
            print_translated("2. Exit")
            choice = input("Select option (1-2): ")
        if choice == "1":
            search_appointments(conn, is_admin)
        elif choice == "2" and is_admin:
            view_all_appointments(conn)
        elif choice == "3" and is_admin:
            manage_users(conn)
        else:
            break
    conn.close()

def search_appointments(conn, is_admin):
    print_translated("\nSearch Appointments")
    print_translated("Available search criteria:")
    print_translated("1. By name")
    print_translated("2. By professor")
    print_translated("3. By time")
    print_translated("4. Back")
    criteria = {}
    while True:
        choice = input("Select search criteria (1-5): ")
        if choice == "1":
            criteria["name"] = input("Enter name to search: ")
        elif choice == "2":
            criteria["professor"] = input("Enter professor name: ")
        elif choice == "3":
            criteria["time"] = input("Enter time (HH:MM): ")
        elif choice == "4":
            return
        results = find_appointments(conn, criteria, is_admin)
        if results:
            print_translated("\nSearch Results:")
            for idx, appt in enumerate(results, 1):
                print_translated(f"\nAppointment {idx}:")
                print_translated(f"Name: {appt['name']}")
                print_translated(f"Phone: {appt['phone']}")
                print_translated(f"Email: {appt['email']}")
                print_translated(f"Role: {appt['role']} ({appt['subrole']})")
                print_translated(f"Professor: {appt['professor']}")
                print_translated(f"Time: {appt['time']}")
                if appt['is_online']:
                    print_translated(f"Meeting Link: {appt['meet_link']}")
                else:
                    print_translated(f"Location: {appt['building']}, Room {appt['room']}")
        else:
            print_translated("No appointments found matching criteria")
        if input("\nSearch again? (y/n): ").lower() != 'y':
            break

def manage_users(conn):
    print_translated("\nUser Management")
    print_translated("1. Add new user")
    print_translated("2. Back")
    choice = input("Select option (1-2): ")
    if choice == "1":
        username = input("Enter new username: ")
        password = getpass.getpass("Enter password: ")
        is_admin = input("Is this an admin user? (y/n): ").lower() == 'y'
        if create_user(conn, username, password, is_admin):
            print_translated("User created successfully")
        else:
            print_translated("Failed to create user")

def f():
    db_entry = {}
    select_language()
    select_input_method()
    ask_to_enable_tts()
    data = loadfile()
    if not data:
        return
    initialize_database()
    print_translated("Hi! I'm the receptionist chatbot. What would you like to do?")
    print_translated("1. Appointment Management")
    print_translated("2. Access Database")
    choice = get_translated_input("")
    if choice == "2":
        database_interface()  # admin admin123
        return
    print_translated("What's your name?")
    name = get_translated_input("")
    phone = validate_phone()
    email = validate_email()
    db_entry["DemographicData"] = {"name": name, "phone": phone, "email": email}
    print_translated(f"Nice to meet you {name}! What are you in relation to this institution?")
    while True:
        ut, ust = gettype(data)
        print_translated(f"Based on your input, you are a {ut} ({ust}).")
        while True:
            print_translated("How may I help you today?")
            req = get_translated_input("")
            if any(word in req.lower() for word in ["appointment", "meet", "talk", "reschedule", "cancel", "discuss"]):
                appointment2(data, name, phone, email, ut, ust, db_entry)
            elif not req or req.lower() in ["umm...", "umm","um","uhh..","uhh","uh","hmm...","hmm","hm","..."]:
                print_translated("Did you mean:\n1. Book Appointment\n2. Cancel Appointment\n3. Reschedule Appointment")
            else:
                print_translated("Sorry! I cannot assist with that yet.")
            print_translated("Would you like to exit? (yes/no)")
            ch = get_translated_input("")
            if ch.lower() in ["yes", "y"]:
                print_translated("See You Later!")
                return
