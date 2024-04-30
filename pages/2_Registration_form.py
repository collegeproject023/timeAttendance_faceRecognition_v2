import streamlit as st
from Home import face_rec
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av


# st.set_page_config(page_title='Registration Form')
st.subheader('Registration Form')

## init registration form
registration_form = face_rec.RegistrationForm()

# Step-1: Collect person name and role
# form
person_name = st.text_input(label='Name',placeholder='First & Last Name')
role = st.selectbox(label='Select your Role',options=('Student','Teacher'))
courselevel = st.selectbox(label='Select Course & Year Level',options=('BSIT-1','BSIT-2','BSIT-3'))
sect = st.selectbox(label='Select Section',options=('A','B','C'))
scol_year = st.selectbox(label='Select Year',options=('2024','2025','2026'))
address = st.text_input(label='Address',placeholder='Sunflower Street County')
contact_no = st.text_input(label='Contact #',placeholder='0987654321')
email_add = st.text_input(label='Email Address',placeholder='youremail@mail.com')
professor = "none"

if role == "Student":
    # If "student" is selected, display the textbox
    professor = st.text_input(label='Teacher Name',placeholder='Full Name')


# step-2: Collect facial embedding of that person
def video_callback_func(frame):
    img = frame.to_ndarray(format='bgr24') # 3d array bgr
    reg_img, embedding = registration_form.get_embedding(img)
    # two step process
    # 1st step save data into local computer txt
    if embedding is not None:
        with open('face_embedding.txt',mode='ab') as f:
            np.savetxt(f,embedding)
    
    return av.VideoFrame.from_ndarray(reg_img,format='bgr24')

webrtc_streamer(key='registration',video_frame_callback=video_callback_func,
rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)


# step-3: save the data in redis database


if st.button('Submit'):
    return_val = registration_form.save_data_in_redis_db(person_name,role,courselevel,sect,scol_year,address,contact_no,email_add,professor)
    if return_val == True:
        st.success(f"{person_name} registered sucessfully")
    elif return_val == 'name_false':
        st.error('Please enter the name: Name cannot be empty or spaces')
        
    elif return_val == 'file_false':
        st.error('face_embedding.txt is not found. Please refresh the page and execute again.')

# if st.button('clear_database'):
#     registration_form.delete_all_data()
        
# if st.button('clear logs'):
#     registration_form.delete_data()
