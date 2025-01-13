import streamlit as st
from Home import face_rec
#import time
from streamlit_webrtc import webrtc_streamer
import av

st.set_page_config(page_title='Predictions',layout='centered')
st.subheader('Real-Time Attendance System')


# Retrieve the data from Redis Database
with st.spinner("Retrieving Data from Redis DB..."):
    redis_face_db = face_rec.retrieve_data(name='academy:register')
    st.dataframe(redis_face_db)

st.success("Data retrieved successfully from Redis")



#time
# waitTime = 30   # in seconds
# setTime = face_rec.time.time() 
# realtimepred = face_rec.RealTimePred()


# Real Time Prediction
#streamlit webrtc

#callback function for video
def video_frame_callback(frame):
    # global setTime
    img = frame.to_ndarray(format="bgr24") # 3D array (numpy)
    pred_img = face_rec.face_prediction(img,redis_face_db,'facial_features',['Name','Role'],thresh=0.5)
    # timenow = face_rec.time.time()
    # difftime = timenow-setTime  
    # if difftime >= waitTime:
    #     realtimepred.saveLogs_redis()
    #     setTime = face_rec.time.time() # reset time
    #     print("Save Data to redis database")



    return av.VideoFrame.from_ndarray(pred_img,format="bgr24")


webrtc_streamer(key="realtimePrediction", video_frame_callback=video_frame_callback)
