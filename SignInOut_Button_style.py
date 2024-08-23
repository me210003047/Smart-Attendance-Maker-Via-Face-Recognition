import streamlit as st
import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd
import zipfile
from PIL import Image,ExifTags
from streamlit_option_menu import option_menu
import base64
import time
import shutil
import pickle
from concurrent.futures import ProcessPoolExecutor
import io
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import tempfile

# Initialize session state
if 'cache_key' not in st.session_state:
    st.session_state.cache_key = time.time()
if 'current_processed_files' not in st.session_state:
    st.session_state.current_processed_files = {}

# Base path to the folder containing all reference images subfolders
base_path = 'Ref Folder (structure)'

# Function to refresh the cache key
def refresh_cache():
    st.session_state.cache_key = time.time()
    st.rerun()

# Google Drive API scope
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly', 'https://www.googleapis.com/auth/drive.file']

# Authenticate and create the Drive service
def authenticate_user():
    flow = InstalledAppFlow.from_client_secrets_file('client_secret_228454635738-otma3d2n2e8ghak360o00j7mo6a294lu.apps.googleusercontent.com.json', SCOPES)
    creds = flow.run_local_server(port=0)
    return creds

def create_drive_service(creds):
    return build('drive', 'v3', credentials=creds)

def list_drive_folders(drive_service):
    folders = {}
    page_token = None
    while True:
        response = drive_service.files().list(
            q="mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields="nextPageToken, files(id, name)",
            pageToken=page_token
        ).execute()

        for item in response.get('files', []):
            folders[item['name']] = item['id']

        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break

    return folders

def upload_to_drive(drive_service, file_path, folder_id=None):
    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': [folder_id] if folder_id else []
    }
    media = MediaFileUpload(file_path, mimetype='text/csv')
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file.get('id')


# Use Streamlit's built-in caching with the cache key
@st.cache_data(show_spinner=False)
def get_subfolders(base_path, cache_key):
    return [f.name for f in os.scandir(base_path) if f.is_dir()]

def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def process_image(args):
    image_path, name, roll_no = args
    image = load_image(image_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        return encodings[0], (name, roll_no)
    return None, None

@st.cache_data(show_spinner=False)
def load_images_and_encodings(folder, cache_key):
    path = os.path.join(base_path, folder)
    encoding_file = os.path.join(path, 'encodings.pkl')

    if os.path.exists(encoding_file):
        with open(encoding_file, 'rb') as f:
            return pickle.load(f)

    encodeList = []
    personInfo = []
    args_list = []

    for person_folder in os.listdir(path):
        person_path = os.path.join(path, person_folder)
        if os.path.isdir(person_path):
            name, roll_no = person_folder.rsplit('_', 1)
            for image_file in os.listdir(person_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_path, image_file)
                    args_list.append((image_path, name, roll_no))

    with ProcessPoolExecutor() as executor:
        results = executor.map(process_image, args_list)

        for encoding, info in results:
            if encoding is not None:
                encodeList.append(encoding)
                if info not in personInfo:
                    personInfo.append(info)

    # Save encodings to file
    with open(encoding_file, 'wb') as f:
        pickle.dump((encodeList, personInfo), f)

    return encodeList, personInfo

def update_encodings(folder):
    path = os.path.join(base_path, folder)
    encoding_file = os.path.join(path, 'encodings.pkl')

    encodeList = []
    personInfo = []
    args_list = []

    for person_folder in os.listdir(path):
        person_path = os.path.join(path, person_folder)
        if os.path.isdir(person_path):
            name, roll_no = person_folder.rsplit('_', 1)
            for image_file in os.listdir(person_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_path, image_file)
                    args_list.append((image_path, name, roll_no))

    with ProcessPoolExecutor() as executor:
        results = executor.map(process_image, args_list)

        for encoding, info in results:
            if encoding is not None:
                encodeList.append(encoding)
                if info not in personInfo:
                    personInfo.append(info)

    # Save updated encodings to file
    with open(encoding_file, 'wb') as f:
        pickle.dump((encodeList, personInfo), f)

    st.success("Encodings updated successfully.")

def logo_add1(image_path):
    def load_image(image_path):
        with open(image_path, 'rb') as file:
            return file.read()

    image = load_image(image_path)
    encoded_image = base64.b64encode(image).decode()

    st.markdown(
        f"""
        <div style="
            position: absolute;
            top: -150px;
            right: -200px;
            width: 175px;
            height: auto;">
            <img src="data:image/png;base64,{encoded_image}" style="width: 100%;">
        </div>
        """,
        unsafe_allow_html=True
    )
def logo_add2(image_path):
    def load_image(image_path):
        with open(image_path, 'rb') as file:
            return file.read()

    image = load_image(image_path)
    encoded_image = base64.b64encode(image).decode()

    st.markdown(
        f"""
        <div style="
            position: absolute;
            top: -120px;
            right: -200px;
            width: 175px;
            height: 500;">
            <img src="data:image/png;base64,{encoded_image}" style="width: 100%;">
        </div>
        """,
        unsafe_allow_html=True
    )

def process_images(images, encodeListKnown, personInfo):
    attendance = []
    processed_images = []
    for img in images:
        img_array = np.array(img)
        imgC = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        facesCurFrame = face_recognition.face_locations(imgC)
        encodesCurFrame = face_recognition.face_encodings(imgC, facesCurFrame)

        img_with_boxes = img_array.copy()
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.47)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name, roll_no = personInfo[matchIndex]
                attendance.append((name.upper(), roll_no))

                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img_with_boxes, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img_with_boxes, f"{name.upper()} ({roll_no})", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX,
                            0.5, (0, 0, 0), 1)

        processed_images.append(img_with_boxes)

    return attendance, processed_images

def extract_date_from_image(image):
    try:
        exif_data = image._getexif()
        if exif_data:
            for tag, value in exif_data.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                if decoded == "DateTimeOriginal":
                    return datetime.strptime(value, '%Y:%m:%d %H:%M:%S').strftime('%Y-%m-%d')
    except Exception as e:
        st.error(f"Error extracting date from image: {e}")
    return None

def create_class_list_from_images(image_folder):
    class_list = []
    for person_folder in os.listdir(image_folder):
        person_path = os.path.join(image_folder, person_folder)
        if os.path.isdir(person_path):
            name, roll_no = person_folder.rsplit('_', 1)
            class_list.append([name, roll_no])
    return pd.DataFrame(class_list, columns=['Name', 'Roll No']).drop_duplicates()


def combine_attendance(attendance_data, image_folder):
    class_list_df = create_class_list_from_images(image_folder)
    all_roll_numbers = class_list_df['Roll No'].tolist()

    attendance_df = class_list_df.copy()

    # Sort dates to ensure earlier dates are on the left
    sorted_dates = sorted(attendance_data.keys())

    for date in sorted_dates:
        present_roll_numbers = attendance_data[date]['Roll Number'].astype(str).tolist()
        attendance_df[date] = attendance_df['Roll No'].apply(lambda x: 'P' if str(x) in present_roll_numbers else 'A')

    # Calculate attendance percentage
    total_days = len(sorted_dates)
    attendance_df['Attendance %'] = attendance_df[sorted_dates].apply(
        lambda row: (row == 'P').sum() / total_days * 100, axis=1
    ).round(2)

    # Reorder columns to have dates from left to right, with Attendance % at the end
    date_columns = sorted_dates
    final_columns = ['Name', 'Roll No'] + date_columns + ['Attendance %']
    attendance_df = attendance_df[final_columns]

    return attendance_df

def main():
    # Streamlit app structure
    # Streamlit app structure
    st.markdown("""
        <style>
        .stButton > button:first-child {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton > button:first-child:hover {
            background-color: #45a049;
        }
        .stButton > button[kind="secondary"] {
            background-color: #008CBA;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton > button[kind="secondary"]:hover {
            background-color: #007B9E;
        }
        </style>
        """, unsafe_allow_html=True)

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        if st.sidebar.button("Sign In to Google Drive",type="secondary"):
            creds = authenticate_user()
            st.session_state.creds = creds
            st.session_state.drive_service = create_drive_service(creds)
            st.session_state.folders = list_drive_folders(st.session_state.drive_service)
            st.session_state.authenticated = True
            st.experimental_rerun()

    with st.sidebar:
        page = option_menu(
            menu_title="Main Menu",
            options=["Home", "Image Attendance", "Add New Reference Folder", "Explore Subfolders"],
            icons=["house", "image", "file-earmark-arrow-up", "file-earmark-person"],
            menu_icon="cast"
        )

        # Add the "Sign Out" button in the sidebar
        if 'authenticated' in st.session_state and st.session_state.authenticated:
            if st.button("Sign Out"):
                st.session_state.authenticated = False
                st.session_state.creds = None
                st.session_state.drive_service = None
                st.session_state.folders = {}
                st.success("Signed out successfully!")
                st.experimental_rerun()  # Force rerun to reset the UI

        if st.button("Refresh App"):
            refresh_cache()


    subfolders = get_subfolders(base_path, st.session_state.cache_key)

    if page == "Home":
        st.title("Face Recognition Attendance System")
        logo_add1('Attendance_APP Logo.png')

        selected_folder = st.selectbox("Select Reference Image Folder", subfolders)

        def markAttendance(name, roll_no, fileName):
            if os.path.exists(fileName):
                attendance_df = pd.read_csv(fileName, converters={'Roll Number': str})
            else:
                attendance_df = pd.DataFrame(columns=['Name', 'Roll Number', 'Time'])

            if not ((attendance_df['Name'] == name) & (attendance_df['Roll Number'] == roll_no)).any():
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                new_entry = pd.DataFrame([[name, roll_no, dtString]], columns=['Name', 'Roll Number', 'Time'])
                attendance_df = pd.concat([attendance_df, new_entry], ignore_index=True)
                attendance_df.to_csv(fileName, index=False)

        def clearAttendance(fileName):
            if os.path.exists(fileName):
                os.remove(fileName)

        encodeListKnown, personInfo = load_images_and_encodings(selected_folder, st.session_state.cache_key)
        st.sidebar.write('Encoding Complete')

        start_attendance = st.button("START ATTENDANCE",key="start_attendance")
        take_attendance = st.button("TAKE ATTENDANCE",key="take_attendance",type="secondary")
        stop_attendance = st.button("STOP ATTENDANCE")

        frame_placeholder = st.empty()

        # Initialize variables
        # Initialize session state variables if they do not exist
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
            st.session_state.folders = {}

        if not st.session_state.authenticated:
            if st.button("Sign In"):
                creds = authenticate_user()
                st.session_state.creds = creds
                st.session_state.drive_service = create_drive_service(creds)
                st.session_state.folders = list_drive_folders(
                    st.session_state.drive_service)  # Fetch folders after authentication
                st.session_state.authenticated = True
                st.experimental_rerun()  # Force rerun to update the display
        else:
            drive_service = st.session_state.drive_service

            # List available folders in Google Drive from session state
            folders = st.session_state.folders

            if folders:
                folder_name = st.selectbox("Choose a folder to upload the CSV file to:", list(folders.keys()))
                folder_id = folders[folder_name]
            else:
                st.warning("No folders found in your Google Drive.")

            # Check if the attendance CSV file exists
            dateStr = datetime.now().strftime('%d-%m-%Y')
            fileName = f'Attendance_{dateStr}.csv'

            if os.path.exists(fileName):
                st.write("Attendance CSV file found. Uploading this file.")
                if st.button("Upload to Google Drive"):
                    file_id = upload_to_drive(drive_service, fileName, folder_id)
                    st.success(f"CSV file uploaded successfully to {folder_name}! File ID: {file_id}")
            else:
                st.write("No attendance CSV file found. You can upload a CSV file manually.")
                uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

                if uploaded_file is not None:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    # Upload to Google Drive
                    if st.button("Upload to Google Drive"):
                        file_id = upload_to_drive(drive_service, tmp_file_path, folder_id)
                        st.success(f"CSV file uploaded successfully to {folder_name}! File ID: {file_id}")

                    # Remove the temporary file after use
                    os.remove(tmp_file_path)
        if 'cap' not in st.session_state:
            st.session_state['cap'] = None
            st.session_state['show_live'] = False

        if start_attendance:
            st.session_state['cap'] = cv2.VideoCapture(0)
            st.session_state['show_live'] = True
            st.sidebar.write('Webcam started. Click "TAKE ATTENDANCE" to take attendance.')

        while st.session_state['show_live'] and st.session_state['cap']:
            # Check if the "STOP ATTENDANCE" button is pressed
            if stop_attendance:
                if st.session_state['cap']:
                    cap = st.session_state['cap']
                    cap.release()
                    st.sidebar.write("Webcam released")
                    st.session_state['cap'] = None
                    st.session_state['show_live'] = False
                    # Clear attendance file
                    dateStr = datetime.now().strftime('%d-%m-%Y')
                    fileName = f'Attendance_{dateStr}.csv'
                    clearAttendance(fileName)
                    st.sidebar.write("Attendance list cleared.")
                break

            cap = st.session_state['cap']
            success, img = cap.read()
            if success:
                imgC = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                facesCurFrame = face_recognition.face_locations(imgC)

                # Display the current frame with bounding boxes
                img_with_boxes = img.copy()
                for faceLoc in facesCurFrame:
                    y1, x2, y2, x1 = faceLoc
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                frame_placeholder.image(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB), channels="RGB")
            else:
                st.sidebar.write("Failed to capture image")
                break

            if take_attendance:
                facesCurFrame = face_recognition.face_locations(imgC)
                encodesCurFrame = face_recognition.face_encodings(imgC, facesCurFrame)

                dateStr = datetime.now().strftime('%d-%m-%Y')
                fileName = f'Attendance_{dateStr}.csv'

                for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.5)
                    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                    matchIndex = np.argmin(faceDis)

                    if matches[matchIndex]:
                        name, roll_no = personInfo[matchIndex]
                        y1, x2, y2, x1 = faceLoc
                        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(img_with_boxes, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img_with_boxes, f"{name.upper()} ({roll_no})", (x1 + 6, y2 - 6),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
                        markAttendance(name.upper(), roll_no, fileName)

                # Display the captured frame with bounding boxes
                frame_placeholder.image(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB), channels="RGB")

                # Pause the display for 4 seconds
                time.sleep(4)

                # Load the updated attendance file into DataFrame and display it
                attendance_df = pd.read_csv(fileName, converters={'Roll Number': str})
                attendance_df.index = attendance_df.index + 1  # Start index from 1
                st.write("Attendance for today:")
                st.dataframe(attendance_df)

                take_attendance = False  # Reset the button state to allow continuous attendance taking

    elif page == "Image Attendance":
        st.title("Image Attendance")
        logo_add2('ATTTENDANCE APP(Image Attendance Symbol ).png')

        subfolders = get_subfolders(base_path, st.session_state.cache_key)
        selected_folder = st.selectbox("Select Reference Image Folder", subfolders)

        # Store combined attendance in session state
        if 'combined_attendance' not in st.session_state:
            st.session_state.combined_attendance = None

        # Add the COMBINE ATTENDANCE button at the top
        if st.button("COMBINE ATTENDANCE"):
            if st.session_state.current_processed_files:
                st.write("Combining all attendance records...")
                combined_attendance = combine_attendance(st.session_state.current_processed_files,
                                                         os.path.join(base_path, selected_folder))
                st.session_state.combined_attendance = combined_attendance
                st.write("Combined Attendance:")

            else:
                st.warning("No attendance records to combine. Please process attendance first.")

        # Display combined attendance if it exists
        if st.session_state.combined_attendance is not None:
            st.write("Combined Attendance:")
            st.dataframe(st.session_state.combined_attendance)

            # Add Google Drive upload functionality
            if st.session_state.authenticated:
                drive_service = st.session_state.drive_service
                folders = st.session_state.folders

                if folders:
                    folder_name = st.selectbox("Choose a folder to upload the CSV file to:", list(folders.keys()))
                    folder_id = folders[folder_name]

                    if st.button("Upload Combined Attendance to Google Drive"):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                            st.session_state.combined_attendance.to_csv(tmp_file.name, index=False)
                            tmp_file_path = tmp_file.name

                        file_id = upload_to_drive(drive_service, tmp_file_path, folder_id)
                        st.success(
                            f"Combined Attendance CSV file uploaded successfully to {folder_name}! File ID: {file_id}")

                        # Remove the temporary file after use
                        os.remove(tmp_file_path)
                else:
                    st.warning("No folders found in your Google Drive.")
            else:
                st.info("Please sign in to Google Drive using the button in the sidebar to enable uploads.")

        st.write("---")  # Add a separator

        encodeListKnown, uniqueIdentifiers = load_images_and_encodings(selected_folder, st.session_state.cache_key)

        uploaded_files = st.file_uploader("Upload images for attendance", type=["jpg", "jpeg", "png"],
                                          accept_multiple_files=True)

        if uploaded_files and st.button("Process Attendance"):
            # Clear previous processed data
            st.session_state.current_processed_files = {}

            uploaded_images = [Image.open(file) for file in uploaded_files]
            upload_dates = [extract_date_from_image(img) for img in uploaded_images]

            for i, date in enumerate(upload_dates):
                if date is None:
                    st.warning(
                        f"Date could not be extracted for image {uploaded_files[i].name}. Using current date instead.")
                    upload_dates[i] = datetime.now().strftime('%Y-%m-%d')

            attendance_data = {}
            processed_images_data = {}

            for img, date in zip(uploaded_images, upload_dates):
                if date not in attendance_data:
                    attendance_data[date] = []
                    processed_images_data[date] = []

                attendance, processed_images = process_images([img], encodeListKnown, uniqueIdentifiers)
                attendance_data[date].extend(attendance)
                processed_images_data[date].extend(processed_images)

            for date, attendance in attendance_data.items():
                attendance_df = pd.DataFrame(attendance, columns=['Name', 'Roll Number'])
                attendance_df['Time'] = datetime.now().strftime('%H:%M:%S')
                attendance_df = attendance_df.drop_duplicates(subset=['Name', 'Roll Number'])

                st.write(f"Attendance for {date} processed successfully:")
                st.dataframe(attendance_df)

                # Store attendance data in session state
                st.session_state.current_processed_files[date] = attendance_df

                # Prepare CSV for download
                csv_buffer = io.StringIO()
                attendance_df.to_csv(csv_buffer, index=False)
                csv_string = csv_buffer.getvalue()

                st.download_button(
                    label=f"Download Attendance CSV for {date}",
                    data=csv_string,
                    file_name=f'Attendance_{date}.csv',
                    mime='text/csv',
                )

                st.write(f"Processed Images with Face Detection for {date}:")
                for i, img in enumerate(processed_images_data[date]):
                    st.image(img, caption=f"Processed Image {i + 1} ({date})", use_column_width=True)

    elif page == "Add New Reference Folder":
        st.title("Add New Reference Folder")

        # Get the name for the new folder
        new_folder_name = st.text_input("Enter the name of the new reference folder:")

        # Number of images to upload manually
        num_images_upload = st.number_input("Number of images to upload", min_value=0, step=1)

        # Store uploaded images and their respective names
        image_files = []
        image_names = []

        for i in range(num_images_upload):
            uploaded_file = st.file_uploader(f"Upload Image {i + 1}", type=["jpg", "jpeg", "png"],
                                             key=f"file_uploader_{i}")
            if uploaded_file is not None:
                image_files.append(uploaded_file)
                image_name = st.text_input(f"Name for Image {i + 1}", key=f"text_input_{i}")
                image_names.append(image_name)

        # Option to upload a zip file
        zip_file = st.file_uploader("Upload Zip Folder", type="zip")

        if st.button("Create"):
            if new_folder_name:
                new_folder_path = os.path.join(base_path, new_folder_name)
                os.makedirs(new_folder_path, exist_ok=True)

                # Save manually uploaded images
                for i, (file, name) in enumerate(zip(image_files, image_names)):
                    if file is not None and name:
                        file_extension = os.path.splitext(file.name)[1]
                        file_path = os.path.join(new_folder_path, f"{name}{file_extension}")
                        with open(file_path, "wb") as f:
                            f.write(file.read())
                        st.success(f"Image '{name}' uploaded successfully.")

                # Extract images from the uploaded zip file
                if zip_file is not None:
                    zip_folder_path = os.path.join(new_folder_path, zip_file.name)
                    with open(zip_folder_path, "wb") as f:
                        f.write(zip_file.read())

                    with zipfile.ZipFile(zip_folder_path, 'r') as zip_ref:
                        zip_ref.extractall(new_folder_path)
                    os.remove(zip_folder_path)
                    st.success("Images from the zip folder added successfully.")

                # Update the encodings for the new folder
                update_encodings(new_folder_name)
                refresh_cache()
                st.success(f"Reference folder '{new_folder_name}' created and encodings updated successfully.")
            else:
                st.warning("Please enter a valid name for the new reference folder.")

    elif page == "Explore Subfolders":
        st.title("Explore Subfolders")

        if not subfolders:
            st.write("No subfolders available.")
        else:
            selected_folder = st.selectbox("Select a Folder to Explore", subfolders)

            if selected_folder:
                folder_path = os.path.join(base_path, selected_folder)

                # Rename folder
                with st.form(key='rename_form'):
                    new_folder_name = st.text_input("Enter new name for the folder:")
                    rename_submit = st.form_submit_button("Rename Folder")
                    if rename_submit and new_folder_name:
                        new_folder_path = os.path.join(base_path, new_folder_name)
                        os.rename(folder_path, new_folder_path)
                        st.success(f"Folder renamed to {new_folder_name}.")
                        update_encodings(new_folder_name)  # Update encodings for the renamed folder
                        refresh_cache()

                # List person folders
                person_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
                selected_person = st.selectbox("Select a person to explore", person_folders)

                if selected_person:
                    person_path = os.path.join(folder_path, selected_person)

                    # Display images
                    st.write(f"Images for {selected_person}:")
                    image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                    if image_files:
                        for image_file in image_files:
                            image_path = os.path.join(person_path, image_file)
                            image = Image.open(image_path)
                            st.image(image, caption=image_file, use_column_width=True)
                    else:
                        st.write("No images available for this person.")

                    # Add new image
                    with st.form(key='add_image_form'):
                        st.write("Add New Image")
                        new_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
                        new_image_name = st.text_input("Enter image name (without extension)")
                        add_image_submit = st.form_submit_button("Add Image")

                        if add_image_submit:
                            if new_image is not None and new_image_name:
                                file_path = os.path.join(person_path, f"{new_image_name}.jpg")
                                with open(file_path, "wb") as f:
                                    f.write(new_image.getbuffer())
                                st.success(f"Image '{new_image_name}' added successfully.")
                                update_encodings(selected_folder)  # Update encodings after adding a new image
                                refresh_cache()
                            else:
                                st.warning("Please provide both an image file and a name.")

                    # Delete image
                    if image_files:
                        with st.form(key='delete_image_form'):
                            image_to_delete = st.selectbox("Select an image to delete", image_files)
                            delete_image_submit = st.form_submit_button("Delete Image")
                            if delete_image_submit and image_to_delete:
                                os.remove(os.path.join(person_path, image_to_delete))
                                st.success(f"Image '{image_to_delete}' deleted successfully.")
                                update_encodings(selected_folder)  # Update encodings after deleting an image
                                refresh_cache()

                # Add new person folder
                with st.form(key='add_person_form'):
                    st.write("Add New Person")
                    new_person_name = st.text_input("Enter person's name")
                    new_person_roll = st.text_input("Enter person's roll number")
                    add_person_submit = st.form_submit_button("Add Person")

                    if add_person_submit:
                        if new_person_name and new_person_roll:
                            new_person_folder = f"{new_person_name}_{new_person_roll}"
                            new_person_path = os.path.join(folder_path, new_person_folder)
                            os.makedirs(new_person_path, exist_ok=True)
                            st.success(f"Person folder '{new_person_folder}' created successfully.")
                            refresh_cache()
                        else:
                            st.warning("Please provide both name and roll number.")

                # Delete person folder
                with st.form(key='delete_person_form'):
                    st.write("Delete Person Folder")
                    person_to_delete = st.selectbox("Select a person folder to delete", person_folders)
                    confirm_delete = st.checkbox("Confirm deletion (This action is irreversible)")
                    delete_person_submit = st.form_submit_button("Delete Person Folder")
                    if delete_person_submit and confirm_delete:
                        try:
                            shutil.rmtree(os.path.join(folder_path, person_to_delete))
                            st.success(f"Person folder '{person_to_delete}' deleted successfully.")
                            update_encodings(selected_folder)  # Update encodings after deleting a person
                            refresh_cache()
                        except Exception as e:
                            st.error(f"Error deleting person folder: {str(e)}")

                # Delete subfolder
                with st.form(key='delete_subfolder_form'):
                    st.write("Delete Subfolder")
                    confirm_delete_subfolder = st.checkbox("Confirm subfolder deletion (This action is irreversible)")
                    delete_subfolder_submit = st.form_submit_button("Delete Subfolder")
                    if delete_subfolder_submit and confirm_delete_subfolder:
                        try:
                            shutil.rmtree(folder_path)
                            st.success(f"Subfolder '{selected_folder}' deleted successfully.")
                            refresh_cache()
                        except Exception as e:
                            st.error(f"Error deleting subfolder: {str(e)}")


if __name__ == "__main__":
        main()