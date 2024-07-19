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

# Initialize session state
if 'cache_key' not in st.session_state:
    st.session_state.cache_key = time.time()
if 'current_processed_files' not in st.session_state:
    st.session_state.current_processed_files = {}

# Base path to the folder containing all reference images subfolders
base_path = 'Reference Folders'

# Function to refresh the cache key
def refresh_cache():
    st.session_state.cache_key = time.time()
    st.rerun()


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

    image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    args_list = []
    for image_file in image_files:
        name, roll_no = os.path.splitext(image_file)[0].rsplit('_', 2)[0:2]  # Split by last two underscores
        args_list.append((os.path.join(path, image_file), name, roll_no))

    encodeList = []
    personInfo = []

    with ProcessPoolExecutor() as executor:
        results = executor.map(process_image, args_list)

        for encoding, info in results:
            if encoding is not None:
                encodeList.append(encoding)
                personInfo.append(info)

    # Save encodings to file
    with open(encoding_file, 'wb') as f:
        pickle.dump((encodeList, personInfo), f)

    return encodeList, personInfo


def update_encodings(folder):
    path = os.path.join(base_path, folder)
    encoding_file = os.path.join(path, 'encodings.pkl')

    image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    args_list = []
    for image_file in image_files:
        name, roll_no = os.path.splitext(image_file)[0].rsplit('_', 2)[0:2]
        args_list.append((os.path.join(path, image_file), name, roll_no))

    encodeList = []
    personInfo = []

    with ProcessPoolExecutor() as executor:
        results = executor.map(process_image, args_list)

        for encoding, info in results:
            if encoding is not None:
                encodeList.append(encoding)
                personInfo.append(info)

    # Save updated encodings to file
    with open(encoding_file, 'wb') as f:
        pickle.dump((encodeList, personInfo), f)

    st.success("Encodings updated successfully.")


# Function to add the logo to the top right corner
def logo_add(image_path):
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
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.5)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name, roll_no = personInfo[matchIndex]
                attendance.append((name.upper(), roll_no))

                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img_with_boxes, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img_with_boxes, f"{name.upper()} ({roll_no})", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX,
                            0.5, (255, 255, 255), 1)

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
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            name_roll = os.path.splitext(filename)[0].rsplit('_', 2)[0:2]  # Split by last two underscores
            name, roll_no = name_roll
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
    with st.sidebar:
        page = option_menu(
            menu_title="Main Menu",
            options=["Home", "Image Attendance", "Add New Reference Folder", "Explore Subfolders"],
            icons=["house", "image", "file-earmark-arrow-up", "file-earmark-person"],
            menu_icon="cast"
        )
        if st.button("Refresh App"):
            refresh_cache()

    subfolders = get_subfolders(base_path, st.session_state.cache_key)

    if page == "Home":
        st.title("Face Recognition Attendance System")
        logo_add('Attendance_APP Logo.png')

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

        start_attendance = st.button("START ATTENDANCE")
        take_attendance = st.button("TAKE ATTENDANCE")
        stop_attendance = st.button("STOP ATTENDANCE")

        frame_placeholder = st.empty()

        # Initialize variables
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
        logo_add('Attendance_APP Logo.png')

        subfolders = get_subfolders(base_path, st.session_state.cache_key)
        selected_folder = st.selectbox("Select Reference Image Folder", subfolders)

        # Add the COMBINE ATTENDANCE button at the top
        if st.button("COMBINE ATTENDANCE"):
            if st.session_state.current_processed_files:
                st.write("Combining all attendance records...")
                combined_attendance = combine_attendance(st.session_state.current_processed_files,
                                                         os.path.join(base_path, selected_folder))
                st.write("Combined Attendance:")
                st.dataframe(combined_attendance)

                # Prepare combined attendance for download
                combined_csv = combined_attendance.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Combined Attendance CSV",
                    data=combined_csv,
                    file_name='combined_attendance.csv',
                    mime='text/csv',
                )
            else:
                st.warning("No attendance records to combine. Please process attendance first.")

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
        new_folder_name = st.text_input("Enter the name of the new reference folder:")
        num_images_upload = st.number_input("Number of images to upload", min_value=0, step=1)
        image_files = []
        image_names = []

        for i in range(num_images_upload):
            uploaded_file = st.file_uploader(f"Upload Image {i + 1}", type=["jpg", "jpeg", "png"],
                                             key=f"file_uploader_{i}")
            if uploaded_file is not None:
                image_files.append(uploaded_file)
                image_names.append(st.text_input(f"Name of Image {i + 1}", key=f"text_input_{i}"))

        zip_file = st.file_uploader("Upload Zip Folder", type="zip")

        if st.button("Create"):
            if new_folder_name:
                new_folder_path = os.path.join(base_path, new_folder_name)
                os.makedirs(new_folder_path, exist_ok=True)

                for i, (file, name) in enumerate(zip(image_files, image_names)):
                    if file is not None and name:
                        file_path = os.path.join(new_folder_path, f"{name}.jpg")
                        with open(file_path, "wb") as f:
                            f.write(file.read())

                        if zip_file is not None:
                            zip_file_path = os.path.join(new_folder_path, zip_file.name)
                            with open(zip_file_path, "wb") as f:
                                f.write(zip_file.read())

                            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                                zip_ref.extractall(new_folder_path)
                            os.remove(zip_file_path)
                            st.success("Images from zip folder added successfully.")

                        st.success(f"Reference folder '{new_folder_name}' created successfully.")
                        update_encodings(new_folder_name)  # Update encodings for the new folder
                        refresh_cache()
                    else:
                        st.error("Please enter a valid folder name.")

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

                # Simplified Add Image
                with st.form(key='add_image_form'):
                    st.write("Add New Image")
                    new_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
                    new_image_name = st.text_input("Enter image name (without extension)")
                    add_image_submit = st.form_submit_button("Add Image")

                    if add_image_submit:
                        if new_image is not None and new_image_name:
                            file_path = os.path.join(folder_path, f"{new_image_name}.jpg")
                            with open(file_path, "wb") as f:
                                f.write(new_image.getbuffer())
                            st.success(f"Image '{new_image_name}' added successfully.")
                            update_encodings(selected_folder)  # Update encodings after adding a new image
                            refresh_cache()
                        else:
                            st.warning("Please provide both an image file and a name.")

                # Delete image
                image_files = [f for f in os.listdir(folder_path) if
                               os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(
                                   ('.png', '.jpg', '.jpeg'))]
                if image_files:
                    with st.form(key='delete_image_form'):
                        image_to_delete = st.selectbox("Select an image to delete", image_files)
                        delete_image_submit = st.form_submit_button("Delete Image")
                        if delete_image_submit and image_to_delete:
                            os.remove(os.path.join(folder_path, image_to_delete))
                            st.success(f"Image '{image_to_delete}' deleted successfully.")
                            update_encodings(selected_folder)  # Update encodings after deleting an image
                            refresh_cache()
                else:
                    st.write("No images available to delete.")

                # Display images
                st.write(f"Images in {selected_folder}:")
                for image_file in image_files:
                    image_path = os.path.join(folder_path, image_file)
                    image = Image.open(image_path)
                    st.image(image, caption=image_file)

                # Delete subfolder
                with st.form(key='delete_subfolder_form'):
                    st.write("Delete Subfolder")
                    confirm_delete = st.checkbox("Confirm deletion (This action is irreversible)")
                    delete_subfolder_submit = st.form_submit_button("Delete Subfolder")
                    if delete_subfolder_submit and confirm_delete:
                        try:
                            shutil.rmtree(folder_path)
                            st.success(f"Subfolder '{selected_folder}' deleted successfully.")
                            refresh_cache()
                        except Exception as e:
                            st.error(f"Error deleting subfolder: {str(e)}")


if __name__ == "__main__":
        main()