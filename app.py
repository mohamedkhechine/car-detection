import streamlit as st
from ultralytics import YOLO
from collections import Counter 
from PIL import Image    

@st.cache_resource
def modelload():
    model=YOLO('yolo26m.pt')
    return model

def detect(model,image):
    results=model(source=image,
                  conf=0.2,
                  classes=[0, 1, 2, 3, 5, 7],   # pedestrians, bicycles, cars, motorcycles, buses, trucks
                  imgsz=2048,
                  iou=0.3,
                  max_det=1000
                  )
    return results

def visualization(results):
    st.image(results[0].plot(), channels="BGR")

def counted(results,model):
    names = [model.names[int(c)] for c in results[0].boxes.cls]
    counts = Counter(names)

    for name, n in counts.items():
        st.text(f"{name}: {n}")
    st.text(f"total: {len(names)}")

def stremlitload():
    st.title('TRAFFIC DETECTION')
    st.header("enter the image and well see how many cars,motorcycles,pedestrians are there")
    uploaded=st.file_uploader("enter your image")
    if uploaded is not None:     
        model=modelload()
        with st.spinner("Loading........"):
            image = Image.open(uploaded).convert("RGB")
            results=detect(model,image)
            visualization(results)
            counted(results,model)

stremlitload()