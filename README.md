Traffic Detection

A web app that counts the vehicles and pedestrians in a traffic photo. Upload an image and it draws a box around every car, truck, bus, motorcycle, bicycle and person it finds, then tells you how many of each there are.

How it works

It uses YOLO26, the latest pretrained YOLO model from Ultralytics, trained on the COCO dataset. I didn't train anything myself: COCO already includes the classes I needed (person, bicycle, car, motorcycle, bus, truck), so I filter the detections down to those and count them by class.

The app runs the model on the uploaded image, shows the annotated result, and lists a count per type plus a total.

Things I ran into

Small, far-away objects get missed. In my first tests the model only caught the cars in the front of the image and missed almost everything further back. The reason is that YOLO resizes the image before detecting, and at the default size (640) distant cars shrink to just a few pixels. Raising the inference size to 1280 fixed most of it. Image resolution matters too: a low-res photo can't be fixed by upscaling, there's simply no detail to find.

Colors came out wrong at first. The annotated image from YOLO is in BGR order (OpenCV style), while Streamlit and matplotlib expect RGB, so everything looked blue. Telling Streamlit the image is BGR fixed it.

Model size is a tradeoff. Locally I run the medium model on my GPU. The deployed version uses a smaller one because the hosting runs on CPU.

Some labels are approximate. COCO has no "van" class, so minivans sometimes get labeled as trucks.

Running it
pip install -r requirements.txt
streamlit run app.py

The YOLO weights download automatically the first time you run it.

Live demo

](https://car-detection-5rvin8hzbhojldgsdsqxtn.streamlit.app/)

Built with

Python, Ultralytics YOLO26, Streamlit, Pillow. Developed in WSL2 (Ubuntu) with PyTorch on an RTX 4050.
