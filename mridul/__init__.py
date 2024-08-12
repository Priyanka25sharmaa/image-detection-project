from flask import Blueprint,request,redirect,url_for,send_from_directory,jsonify
from utils.ApiResponse import ApiResponse
from utils.ApiError import ApiError
from utils.RenderResponse import RenderResponse
from flask_jwt_extended import jwt_required,current_user,get_jwt_identity
from constants.https_status_codes import *
from torchvision import models,transforms as T
from facenet_pytorch import MTCNN,InceptionResnetV1
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from config import cache
# from events import status
from flask_sse import sse
from ultralytics import YOLO
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import os
import numpy
import torch
import random
from models import User


mridul=Blueprint("mridul",__name__,url_prefix="/api/v1/mridul")

UPLOAD_FOLDER = 'images/uploads'
ANNOTATED_FOLDER = 'images/results'

def load_model():
    print("Loading the model")

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

    model=YOLO('weights/yolov8n.pt')
    svm=torch.load('path/to/svm.pth')
    DEVICE='cude:0' if torch.cuda.is_available() else 'cpu'
    mtcnn=MTCNN(
        select_largest=False,
        post_process=False,
        device=DEVICE
    ).to(DEVICE).eval()
    inceptmodel=InceptionResnetV1(
        pretrained="vggface2",
        classify=True,
        num_classes=1,
    )
    # checkpoint=torch.load("resnetinceptionv1.epoch_32.pth",map_location=torch.device('cpu'))
    # inceptmodel.load_state_dict(checkpoint['load_state_dict'])
    inceptmodel.to(DEVICE)
    inceptmodel.eval()

    ALLOWED_EXTENSIONS={'png','jpg','jpeg','gif'}

    font=cv2.FONT_HERSHEY_SIMPLEX
    cache.set('model',model)
    cache.set('svm',svm)
    cache.set('mtcnn',mtcnn)
    cache.set('inceptmodel',inceptmodel)

    cache.set('coco_names',coco_names)
    cache.set('font',font)
    cache.set('DEVICE',DEVICE)

    print("All Models Loaded")
    
def currentUser():
    current_user = get_jwt_identity()
    var = User.query.filter_by(id= current_user).one_or_none()
    return var

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f"Folder {folder_path} does not exist.")

@mridul.route("/",methods=['GET'])
@jwt_required()
def index():
    load_model()
    return redirect(url_for("mridul.dashboard"))


@mridul.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(ANNOTATED_FOLDER, filename)

@mridul.route('/video/<path:filename>')
def serve_video(filename):
    return send_from_directory(ANNOTATED_FOLDER, filename, mimetype='video/mp4')


def annotate_image_with_prediction(upload_path, annotated_path, prediction_text):
    # Read the image from the upload path
    image = cv2.imread(upload_path)

    if image is None:
        raise ValueError("Image not found at the specified upload path.")

    # Set the font, size, and color for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5
    color = (255, 0, 0)  # Blue color in BGR
    thickness = 2
    position = (50, 70)  # Position where the text will be placed

    # Add the prediction text to the image
    cv2.putText(image, prediction_text, position, font, font_scale, color, thickness, cv2.LINE_AA)

    # Save the annotated image to the specified path
    cv2.imwrite(annotated_path, image)

# TODO: YOLO IMAGE
def predict_img(data):
    print("Initiating Image Predict")

    file_path = os.path.join(UPLOAD_FOLDER, data.filename)
    data.save(file_path)

    print("Image Uploaded!")

    model = cache.get('model')
    results = model(file_path)

    print("Model Running!")


    res_plotted = results[0].plot()
    output_path = os.path.join(ANNOTATED_FOLDER,"annotated_"+data.filename)
    cv2.imwrite(output_path, res_plotted) 

    print("Annotated Image Saved")

    annotated_video_url= url_for('mridul.serve_image', filename="annotated_"+data.filename)
    return annotated_video_url

    
# TODO: YOLO VIDEO
def predict_video(data):
    model = cache.get('model')
    coco = cache.get('coco_names')
    # font = cache.get('font')


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
        output_path = os.path.join(ANNOTATED_FOLDER, "annotated_" + data.filename)
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

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes.numpy():
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)

                        confidence = box.conf[0]
                        class_id = int(box.cls[0])
                        label = f"{coco[class_id]}: {confidence:.2f}"
                        
                        # Filter results based on a confidence threshold (e.g., 0.5)
                        if confidence > 0.5:
                            # Draw bounding box and label on the frame
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
            print("Confidence Calculated.")
                    # Write frame to output video
            out.write(frame)
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Return URL for the annotated video
        annotated_video_url = url_for('mridul.serve_video', filename="annotated_" + data.filename)
        print("Annotated Video Loaded",annotated_video_url)
        return annotated_video_url
    else:
        return ApiResponse("Unsupported file type", HTTP_400_BAD_REQUEST)


# TODO: DEEP FAKE MODEL
def predict_DeepFake(data):
    mtcnn=cache.get('mtcnn')
    model=cache.get('inceptmodel')
    DEVICE=cache.get('DEVICE')

    print("DeepFake Detection initiated")

    upload_path = os.path.join(UPLOAD_FOLDER, data.filename)
    data.save(upload_path)

    print("File Uploaded")
    img=Image.open(upload_path)
    face=mtcnn(img)
    
    print("Face detected")
    if face is None:
        return ApiError("No Face Detected!",HTTP_400_BAD_REQUEST)
    face=face.unsqueeze(0) 
    print("add the batch dimension")
    face =F.interpolate(face,size=(256,256),mode='bilinear',align_corners=False)

    print("convert the face into a numpy array to be able to plot it")
    prev_face=face.squeeze(0).permute(1,2,0).cpu().detach().int().numpy()
    prev_face=prev_face.astype('uint8')
    face=face.to(DEVICE)
    face=face.to(torch.float32)
    face=face/255.0


    face_img_to_plot=face.squeeze(0).permute(1,2,0).cpu().detach().int().numpy()
    print("Face dimensions setup")

    target_layers=[model.block8.branch1[-1]]
    use_cuda=True if torch.cuda.is_available() else False
    cam=GradCAM(model=model,target_layers=target_layers,use_cuda=use_cuda)
    print("cam done")
    targets=[ClassifierOutputTarget(0)]

    print("targets identified")

    grayscale_cam=cam(input_tensor=face,targets=targets,eigen_smooth=True)
    grayscale_cam=grayscale_cam[0,:]
    visualization=show_cam_on_image(face_img_to_plot,grayscale_cam,use_rgb=True)
    face_with_mask=cv2.addWeighted(prev_face,1,visualization,0.5,0)

    print("DeepFake Prediction Begin")
    with torch.no_grad():
        output=torch.sigmoid(model(face).squeeze(0))
        prediction="real" if output.item() > 0.5 else "fake"

        real_prediction=1-output.item()
        fake_prediction=output.item()

        confidences={
            'real':real_prediction,
            'fake':fake_prediction
        }
    print("Prediction Saving...")
    annotate_image_with_prediction(upload_path,annotated_path=ANNOTATED_FOLDER+'/annotated_'+data.filename,prediction_text=prediction)
    annotated_video_url = url_for('mridul.serve_image', filename="annotated_" + data.filename)
    return annotated_video_url
    
def predict_DeepFake_Video(data):
    upload_path = os.path.join(UPLOAD_FOLDER, data.filename)
    data.save(upload_path)

    annotated_video_url = url_for('mridul.serve_video', filename="annotated_" + data.filename)
    return annotated_video_url

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
            data['resultVideo'] = annotated_img
        except Exception as e:
            data['message'] = str(e)

    elif 'predict_df' in request.args:
        img=request.files['img']
        try:
            print("DEEPFAKE")
            annotated_img = predict_DeepFake(img)
            data['resultImage'] = annotated_img
        except Exception as e:
            data['message'] = str(e)
    
    elif 'predict_df_video' in request.args:
        img=request.files['img']
        try:
            print("DEEPFAKE Video")
            annotated_img = predict_DeepFake_Video(img)
            data['resultVideo'] = annotated_img
        except Exception as e:
            data['message'] = str(e)
    
    else:
        clear_folder(UPLOAD_FOLDER)
        # clear_folder(ANNOTATED_FOLDER)
    return RenderResponse("dashboard.html",HTTP_200_OK,context=data)
