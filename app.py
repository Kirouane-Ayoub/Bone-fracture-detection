from ultralytics import YOLO
import streamlit as st
import cv2
import math 
import cvzone
import uuid

with st.sidebar : 
    st.image("icon.png" , width=300)
    save = st.selectbox("Do you want to save the Result ? " , ["YES" , "NO"]) 
    conf = st.slider("Select threshold confidence value : " , min_value=0.1 , max_value=1.0 , value=0.25)
    iou = st.slider("Select Intersection over union (iou) value : " , min_value=0.1 , max_value=1.0 , value=0.5)
uuidOne = uuid.uuid1()
class_names = ["angle", "fracture", "line", "messed_up_angle"]
model = YOLO("bone_frac_v8v2.pt")
tab0 , tab1 = st.tabs(["Home" , "Detection"])

with tab0 : 
    st.header("About This Project : ")
    st.write("""   This system utilizes deep learning algorithms to detect bone fractures in X-ray images. The YOLO model is trained on a labeled dataset of X-ray images with annotated bounding boxes around the fractures.

When a new X-ray image is fed into the system, the YOLO model analyzes the image and predicts the location of any fractures present in the image. This information can be displayed to a radiologist or other medical professional to assist in diagnosing the fracture accurately and quickly.

The system has the potential to reduce the time and effort required for manual analysis of X-ray images, which can help to improve patient outcomes and increase the efficiency of medical practices. However, it's important to note that this system should not be used as a replacement for a trained medical professional's diagnosis, but rather as a tool to aid in their diagnosis.
""")
    st.image("img_home/home_img.jpg")
    st.header("About the Dataset")
    st.write(""" This dataset was originally created by Jason Zhang, Caden Li. To see the current project, which may have been updated since this version,
please go here: https://universe.roboflow.com/science-research/science-research-2022:-bone-fracture-detection.""")
    st.write("""This dataset is part of RF100, an Intel-sponsored initiative to create a new object detection benchmark for model generalizability.
Access the RF100 Github repo: https://github.com/roboflow-ai/roboflow-100-benchmark """)


with tab1:
    file_uploader = st.file_uploader("Select your file : " , type=["jpg" , "png"])
    if file_uploader : 
        name = file_uploader.name
        img = cv2.imread(name)
        results = model.predict(source=name, conf=conf , iou=iou) # , save=True
        for result in  results: 
            bboxs = result.boxes
            for box in bboxs : 
                # bboxes
                x1  , y1 , x2 , y2 = box.xyxy[0]
                x1  , y1 , x2 , y2 = int(x1)  , int(y1) , int(x2) , int(y2)
                clsi = int(box.cls[0])
                conf = math.ceil((box.conf[0] * 100 ))/100
                w,h = int(x2 - x1) , int(y2 - y1)
                cvzone.cornerRect(img , (x1 , y1 , w , h) , l=7 , rt=1)
                #(wt, ht), _ = cv2.getTextSize(f"class:{clsi} {conf}", cv2.FONT_HERSHEY_PLAIN,1, 1)
                #cv2.rectangle(img, (x1, y1 - 18), (x1 + wt, y1), (0,0,255), -1)
                #cv2.putText(img, f"class:{clsi} {conf}", (max(0,x1),max(20 , y1)),cv2.FONT_HERSHEY_PLAIN,1,(255, 255, 255), 1)
                cvzone.putTextRect(img , f"{class_names[clsi]} {conf}" , (max(0,x1) , max(20 , y1)),thickness=1 , colorR=(0,0,255) , scale=0.9 , offset=3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            st.image(img)
            if save == "YES" :
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                cv2.imwrite(f"results/{uuidOne}.jpg", img)
            else : 
                pass

            