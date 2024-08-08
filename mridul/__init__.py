from flask import Blueprint,request,redirect,url_for,send_from_directory,jsonify
from utils.ApiResponse import ApiResponse
from utils.RenderResponse import RenderResponse
from flask_jwt_extended import jwt_required,current_user,get_jwt_identity
from constants.https_status_codes import *
from torchvision import models,transforms as T
from config import cache,socket
from ultralytics import YOLO
import torch
from PIL import Image
import cv2
import os
import torch
import random
from models import User


mridul=Blueprint("mridul",__name__,url_prefix="/api/v1/mridul")

UPLOAD_FOLDER = 'images/uploads'
ANNOTATED_FOLDER = 'images/results'

def load_model():
    print("Loading the model")
    # model=models.detection.fasterrcnn_resnet50_fpn(weights=None)
        
    # # Load the model state dictionary
    # state_dict = torch.load("path/to/model.pth", map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)
    # model.eval()

    coco_names = ["person" , "bicycle" , "car" , "motorcycle" , "airplane" , "bus" , "train" , "truck" , "boat" , "traffic light" , "fire hydrant" , "street sign" , "stop sign" , "parking meter" , "bench" , "bird" , "cat" , "dog" , "horse" , "sheep" , "cow" , "elephant" , "bear" , "zebra" , "giraffe" , "hat" , "backpack" , "umbrella" , "shoe" , "eye glasses" , "handbag" , "tie" , "suitcase" ,
    "frisbee" , "skis" , "snowboard" , "sports ball" , "kite" , "baseball bat" ,
    "baseball glove" , "skateboard" , "surfboard" , "tennis racket" , "bottle" ,
    "plate" , "wine glass" , "cup" , "fork" , "knife" , "spoon" , "bowl" ,
    "banana" , "apple" , "sandwich" , "orange" , "broccoli" , "carrot" , "hot dog" ,
    "pizza" , "donut" , "cake" , "chair" , "couch" , "potted plant" , "bed" ,
    "mirror" , "dining table" , "window" , "desk" , "toilet" , "door" , "tv" ,
    "laptop" , "mouse" , "remote" , "keyboard" , "cell phone" , "microwave" ,
    "oven" , "toaster" , "sink" , "refrigerator" , "blender" , "book" ,
    "clock" , "vase" , "scissors" , "teddy bear" , "hair drier" , "toothbrush" , "hair brush"]


    # font=cv2.FONT_HERSHEY_SIMPLEX
    # cache.set('model',model)
    # cache.set('coco_names',coco_names)
    # cache.set('font',font)
    model=YOLO('yolov8n.pt')
    ALLOWED_EXTENSIONS={'png','jpg','jpeg','gif'}


    
def currentUser():
    current_user = get_jwt_identity()
    var = User.query.filter_by(id= current_user).one_or_none()
    return var

@mridul.route("/",methods=['GET'])
@jwt_required()
def index():
    return redirect(url_for("mridul.dashboard"))


@mridul.route('/images/<path:filename>')
def serve_image(filename):

    return send_from_directory(ANNOTATED_FOLDER, filename)

# TODO: YOLO IMAGE
def predict_img(data):
    print("Predicting.....")
    
    file_path = os.path.join("images/uploads", data.filename)
    data.save(file_path)
    model = YOLO('weights/yolov8n.pt')
    results = model(file_path)
    # results = yolo(image, save=True)
    res_plotted = results[0].plot()
    result_dir = "images/results"
    output_path = os.path.join(result_dir, "result.jpg")
    cv2.imwrite(output_path, res_plotted) 
    # results.save("images/results")
    annotated_video_url= url_for('mridul.serve_image', filename="result.jpg")
    return annotated_video_url


    
# TODO: YOLO VIDEO
def predict_video(data):
    model=cache.get('model')
    coco=cache.get('coco_names')
    font=cache.get('font')

    upload_path = os.path.join(UPLOAD_FOLDER, data.filename)
    print(upload_path)
    data.save(upload_path)
    file_extension = f.filename.rsplit('.', 1)[1].lower()

    if file_extension == 'mp4':
        video_path = upload_path
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        output_path = os.path.join('static', f.filename)
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))
        frames = [] 
        i=0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            i=i+1
            if i<10:
                continue
            i=0
            results = model(frame, save=False)
            res_plotted = results[0].plot()

            # Append processed frame to list
            frames.append(res_plotted)

            if cv2.waitKey(1) == ord('q'):
                break

        # Write all frames to output video
        for frame in frames:
            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    annotated_video_url= url_for('mridul.serve_image', filename="annotated_" + data.filename)

    return annotated_video_url

# TODO: DEEP FAKE MODEL
def predict_DeepFake(data):
    return "DEEPFAKE"


@mridul.route("/dashboard",methods=['GET','POST'])
@jwt_required()
def dashboard():
    var=currentUser()
    data={
    "message":"Model has been Loaded",
    "username":var.username,
    }
    if 'predict_img' in request.args:
        img=request.files['img']
        try:
            annotated_img = predict_img(img)
            data['resultImage'] = annotated_img
        except Exception as e:
            data['message'] = str(e)

    elif 'predict_video' in request.args:
        img=request.files['img']
        try:
            annotated_img = predict_video(img)
            print(annotated_img)
            data['resultImage'] = annotated_img
        except Exception as e:
            data['message'] = str(e)

    elif 'predict_df' in request.args:
        img=request.files['img']
        try:
            annotated_img = predict_DeepFake(img)
            print(annotated_img)
            data['resultImage'] = annotated_img
        except Exception as e:
            data['message'] = str(e)
        
    return RenderResponse("dashboard.html",HTTP_200_OK,context=data)
