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
    print(data.filename)

    model=cache.get('model')
    coco_names=cache.get('coco_names')
    font=cache.get('font')
    
    # yield "data: {}\n\n".format("Model and Font loaded")

    upload_path = os.path.join(UPLOAD_FOLDER, data.filename)
    print(upload_path)
    data.save(upload_path)
    
    img=Image.open('images/uploads/'+data.filename)

    yield "data: {}\n\n".format("Image loaded")
    # socket.emit('progress',{'data':"Image loaded "})
    # socket.start_background_task(predict_img)

    transform= T.ToTensor()
    ig=transform(img)
    with torch.no_grad():
        _pred=model([ig])
    bboxes,labels,scores=_pred[0]['boxes'],_pred[0]['labels'],_pred[0]['scores']
    num=torch.argwhere(scores>0.9).shape[0]

    # socket.emit('progress',{'data':"Predicted image formed"})
    # socket.start_background_task(predict_img)
        
    yield "data: {}\n\n".format("Image Annotated")


    igg=cv2.imread(upload_path)
    print(igg)
    for i in range(num):
        x1,y1,x2,y2=bboxes[i].numpy().astype("int")
        class_name=coco_names[labels.numpy()[i]-1]
        igg=cv2.rectangle(igg,(x1,y1),(x2,y2),(0,255,0),2)
        igg=cv2.putText(igg,class_name,(x1,y1-10),font,2,(255,0,0),2,cv2.LINE_AA)
    
    # socket.emit('progress',{'data':"Annotations completed"})
    # socket.start_background_task(predict_img)
        
    yield "data: {}\n\n".format("Process completed")

    annotated_image_path = os.path.join("images/results", "annotated_" + data.filename)
    if cv2.imwrite(annotated_image_path, igg):
        # Construct the URL for the annotated image
        annotated_image_url = url_for('mridul.serve_image', filename="annotated_" + data.filename)
        # socket.emit('progress',{'data':"Annotations completed"})
        return annotated_image_url
        # return RenderResponse("resultImage.html", HTTP_200_OK, {'resultImage': annotated_image_url})
    
    # TODO: remove image after some time
    
    else:
        return ApiResponse("Error saving annotated image", HTTP_500_INTERNAL_SERVER_ERROR)
    
# TODO: YOLO VIDEO
def predict_video(data):
    # model = cache.get('model')
    # coco = cache.get('coco_names')
    # font = cache.get('font')
    model=YOLO('weights/yolov8n.pt')
    print(model)
    upload_path = os.path.join(UPLOAD_FOLDER, data.filename)
    data.save(upload_path)
    file_extension = data.filename.rsplit('.', 1)[-1].lower()
    
     # Check file extension
    if file_extension == 'mp4':
        print("Video Path:", upload_path)
        print("File Extension:", file_extension)
        video_path = upload_path
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return ApiResponse("Error opening video file", HTTP_400_BAD_REQUEST)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for mp4 files
        output_path = os.path.join('images/results', "annotated_" + data.filename)
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))
        
        print("Starting video processing...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed or end of video.")
                break
            
            print("Processing frame...")
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Perform object detection
            results = model(rgb_frame)
            # Annotate frame
            annotated_frame = results[0].plot()
            
            # Convert RGB to BGR for OpenCV
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            
            # Write frame to output video
            out.write(annotated_frame)

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Return URL for the annotated video
        annotated_video_url = url_for('mridul.serve_image', filename="annotated_" + data.filename)
        return annotated_video_url
    else:
        return ApiResponse("Unsupported file type", HTTP_400_BAD_REQUEST)


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
