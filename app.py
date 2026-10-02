import cv2
import os
import shutil
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Attendance System", layout="wide")

st.title("🎓 Face Recognition Attendance System")
st.markdown("---")

DATASET_FOLDER = "dataset"
STUDENT_CSV = "students.csv"
ATTENDANCE_CSV = "attendance.csv"
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_FILE = "haarcascade_frontalface_default.xml"

if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)

if not os.path.exists(CASCADE_FILE):
    urllib.request.urlretrieve(CASCADE_URL, CASCADE_FILE)

if not os.path.exists(STUDENT_CSV):
    df_init = pd.DataFrame(columns=["RollNo", "Name", "Class"])
    df_init.to_csv(STUDENT_CSV, index=False)

def load_cascade():
    return cv2.CascadeClassifier(CASCADE_FILE)

def update_attendance(roll, name):
    date_today = datetime.now().strftime('%Y-%m-%d')
    time_now = datetime.now().strftime('%H:%M:%S')
    
    if not os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, 'w') as f:
            f.write('RollNo,Name,Time,Date')
            
    with open(ATTENDANCE_CSV, 'r+') as f:
        lines = f.readlines()
        marked = False
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 4 and parts[0] == str(roll) and parts[3] == date_today:
                marked = True
                break
        
        if not marked:
            f.writelines(f"\n{roll},{name},{time_now},{date_today}")

def train_model():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        return None, {}

    faces = []
    labels = []
    label_dict = {}
    current_id = 0
    
    if os.path.exists(DATASET_FOLDER):
        for folder in os.listdir(DATASET_FOLDER):
            folder_path = os.path.join(DATASET_FOLDER, folder)
            if os.path.isdir(folder_path):
                tokens = folder.split("_")
                if len(tokens) >= 2:
                    r_no, s_name = tokens[0], tokens[1]
                    label_dict[current_id] = (r_no, s_name)
                    
                    for file in os.listdir(folder_path):
                        img_path = os.path.join(folder_path, file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            faces.append(img)
                            labels.append(current_id)
                    current_id += 1

    if len(faces) > 0:
        recognizer.train(faces, np.array(labels))
        return recognizer, label_dict
        
    return None, {}

nav = st.sidebar.selectbox("Menu", [
    "📊 View Attendance", 
    "📋 All Students Directory",
    "📝 Register Student", 
    "🔴 Mark Attendance"
])

if nav == "📊 View Attendance":
    st.subheader("Class Attendance Dashboard")
    
    if os.path.exists(STUDENT_CSV):
        students_df = pd.read_csv(STUDENT_CSV)
        if "Class" not in students_df.columns:
            students_df["Class"] = ""
    else:
        students_df = pd.DataFrame(columns=["RollNo", "Name", "Class"])
        
    today = datetime.now().strftime('%Y-%m-%d')
    att_dict = {}
    
    if os.path.exists(ATTENDANCE_CSV):
        att_df = pd.read_csv(ATTENDANCE_CSV)
        if {'Date', 'RollNo', 'Time'}.issubset(att_df.columns):
            today_df = att_df[att_df['Date'] == today]
            for _, row in today_df.iterrows():
                att_dict[str(row['RollNo'])] = row['Time']
            
    if not students_df.empty:
        status_col = []
        time_col = []
        
        for roll in students_df['RollNo'].astype(str):
            if roll in att_dict:
                status_col.append("Present")
                time_col.append(att_dict[roll])
            else:
                status_col.append("Absent")
                time_col.append("-")
                
        students_df['Status'] = status_col
        students_df['Time'] = time_col
        
        st.write(f"Date: {today}")
        st.dataframe(students_df, use_container_width=True, hide_index=True)
    else:
        st.info("No students found in database.")

elif nav == "📋 All Students Directory":
    st.subheader("Complete Student Database, Search & Management")
    
    if os.path.exists(STUDENT_CSV):
        students_df = pd.read_csv(STUDENT_CSV)
        if "Class" not in students_df.columns:
            students_df["Class"] = ""
    else:
        students_df = pd.DataFrame(columns=["RollNo", "Name", "Class"])
        
    if not students_df.empty:
        search_query = st.text_input("🔍 Search Student (by Name, Roll No, or Class)").strip().lower()
        
        if search_query:
            filtered_df = students_df[
                students_df['RollNo'].astype(str).str.lower().str.contains(search_query) |
                students_df['Name'].astype(str).str.lower().str.contains(search_query) |
                students_df['Class'].astype(str).str.lower().str.contains(search_query)
            ]
        else:
            filtered_df = students_df
            
        st.write(f"Total Registered Students: {len(students_df)}")
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("✏️ Edit or Delete Student Record")
        
        student_list = [f"{row['RollNo']} - {row['Name']} ({row['Class']})" for _, row in students_df.iterrows()]
        selected = st.selectbox("Select Student to Modify", student_list)
        
        if selected:
            old_roll = selected.split(" - ")[0]
            old_name = selected.split(" - ")[1].split(" (")[0]
            
            row_data = students_df[students_df['RollNo'].astype(str) == str(old_roll)]
            old_class = row_data.iloc[0]['Class'] if not row_data.empty else ""

            with st.form("edit_form"):
                up_roll = st.text_input("Roll Number", value=old_roll)
                up_name = st.text_input("Name", value=old_name)
                up_class = st.text_input("Class", value=old_class)
                
                c1, c2 = st.columns(2)
                update_click = c1.form_submit_button("Update Details")
                delete_click = c2.form_submit_button("Delete Student")
                
            if update_click:
                if up_roll and up_name and up_class:
                    old_path = os.path.join(DATASET_FOLDER, f"{old_roll}_{old_name}")
                    new_path = os.path.join(DATASET_FOLDER, f"{up_roll}_{up_name}")
                    
                    if os.path.exists(old_path) and old_path != new_path:
                        os.rename(old_path, new_path)
                    elif not os.path.exists(old_path):
                        os.makedirs(new_path, exist_ok=True)
                        
                    students_df.loc[students_df['RollNo'].astype(str) == str(old_roll), ['RollNo', 'Name', 'Class']] = [up_roll, up_name, up_class]
                    students_df.to_csv(STUDENT_CSV, index=False)
                    st.success("Updated successfully!")
                    st.rerun()
                else:
                    st.error("Fields cannot be empty.")
                    
            if delete_click:
                target_path = os.path.join(DATASET_FOLDER, f"{old_roll}_{old_name}")
                if os.path.exists(target_path):
                    shutil.rmtree(target_path)
                    
                students_df = students_df[students_df['RollNo'].astype(str) != str(old_roll)]
                students_df.to_csv(STUDENT_CSV, index=False)
                st.success("Deleted successfully!")
                st.rerun()
    else:
        st.info("No students registered in the directory yet.")

elif nav == "📝 Register Student":
    st.subheader("New Student Registration")
    
    with st.form("reg_form", clear_on_submit=True):
        roll_input = st.text_input("Roll Number")
        name_input = st.text_input("Student Name")
        class_input = st.text_input("Class / Section (Enter any custom class)")
        submit_reg = st.form_submit_button("Capture & Register")
        
    if submit_reg:
        if roll_input and name_input and class_input:
            folder_name = os.path.join(DATASET_FOLDER, f"{roll_input}_{name_input}")
            os.makedirs(folder_name, exist_ok=True)
                
            students_df = pd.read_csv(STUDENT_CSV)
            if "Class" not in students_df.columns:
                students_df["Class"] = ""
                
            if str(roll_input) not in students_df['RollNo'].astype(str).values:
                new_row = pd.DataFrame({"RollNo": [roll_input], "Name": [name_input], "Class": [class_input]})
                students_df = pd.concat([students_df, new_row], ignore_index=True)
                students_df.to_csv(STUDENT_CSV, index=False)
                
            cap = cv2.VideoCapture(0)
            detector = load_cascade()
            count = 0
            
            st.warning("Camera started. Look at the camera and move slightly to capture 30 photos...")
            frame_slot = st.image([])
            
            while count < 30:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    count += 1
                    face_roi = gray[y:y+h, x:x+w]
                    cv2.imwrite(f"{folder_name}/img_{count}.jpg", face_roi)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if count >= 30:
                    break
            cap.release()
            st.success("Student registered successfully!")
        else:
            st.error("Please fill all fields.")

elif nav == "🔴 Mark Attendance":
    st.subheader("Live Attendance Camera")
    cam_on = st.checkbox("Turn On Camera")
    frame_slot = st.image([])
    
    model, labels_map = train_model()
    if model is None:
        st.error("Model training error or dataset is empty.")
        cam_on = False

    detector = load_cascade()
    cap = cv2.VideoCapture(0)
    
    while cam_on:
        ret, frame = cap.read()
        if not ret:
            st.warning("Webcam not accessible.")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5)
        
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            text = "Unknown"
            
            if model is not None and len(labels_map) > 0:
                label, confidence = model.predict(roi)
                if confidence < 75:
                    r_val, s_val = labels_map[label]
                    text = f"{s_val} ({r_val})"
                    update_attendance(r_val, s_val)
                else:
                    text = "Unknown"
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()