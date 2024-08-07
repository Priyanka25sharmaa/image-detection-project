from flask import Blueprint,request
from utils.ApiResponse import ApiResponse
from utils.RenderResponse import RenderResponse
from flask_jwt_extended import jwt_required,current_user,get_jwt_identity
from constants.https_status_codes import *
from torchvision import models,transforms as T
from config import cache,socket
import torch
from PIL import Image
import cv2
import torch
from models import User


mridul=Blueprint("mridul",__name__,url_prefix="/api/v1/mridul")

@mridul.before_app_request
def load_model():
    model=models.detection.fasterrcnn_resnet50_fpn(weights=None)
        
    # Load the model state dictionary
    state_dict = torch.load("path/to/model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

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


    font=cv2.FONT_HERSHEY_SIMPLEX
    cache.set('model',model)
    cache.set('coco_names',coco_names)
    cache.set('font',font)
    
    socket.emit('progress',{'data':"Model Loaded"})

# @mridul.route("/",methods=['GET'])
# @jwt_required
# def loading():
#     RenderResponse("loading.html",HTTP_200_OK)

@mridul.route("/",methods=['GET'])
@jwt_required()

def load_model():
    
    current_user = get_jwt_identity()
    
    var = User.query.filter_by(id= current_user).one_or_none()
    
    return RenderResponse("dashboard.html",HTTP_200_OK,context={'message':"Model has been loaded","username":var.username})

@mridul.route("/predict",methods=['POST'])
def predict_img():
    model=cache.get('model')
    coco_names=cache.get('coco_names')
    font=cache.get('font')


    data=request.files['img']
    data.save(data.filename)
    
    img=Image.open(data.filename)

    socket.emit('progress',{'data':"Image loaded "})
    socket.start_background_task(predict_img)

    transform= T.ToTensor()
    ig=transform(img)
    with torch.no_grad():
        _pred=model([ig])
    bboxes,labels,scores=_pred[0]['boxes'],_pred[0]['labels'],_pred[0]['scores']
    num=torch.argwhere(scores>0.9).shape[0]

    socket.emit('progress',{'data':"Predicted image formed"})
    socket.start_background_task(predict_img)


    igg=cv2.imread(data.filename)
    for i in range(num):
        x1,y1,x2,y2=bboxes[i].numpy().astype("int")
        class_name=coco_names[labels.numpy()[i]-1]
        igg=cv2.rectangle(igg,(x1,y1),(x2,y2),(0,255,0),2)
        igg=cv2.putText(igg,class_name,(x1,y1-10),font,2,(255,0,0),2,cv2.LINE_AA)
    
    socket.emit('progress',{'data':"Annotations completed"})
    socket.start_background_task(predict_img)

    annotated_image_path = "annotated_" + data.filename
    cv2.imwrite(annotated_image_path, igg)

    return ApiResponse("Model has been loaded!",HTTP_200_OK,{"annotate_image":annotated_image_path})
