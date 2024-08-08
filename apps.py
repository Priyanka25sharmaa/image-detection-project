from flask import Flask,request,make_response,redirect,url_for
from config import app,db
from mridul import mridul
from utils.RenderResponse import RenderResponse
from constants.https_status_codes import *
from flask_sse import sse
from auth import auth
from priyanka import priyanka

app.register_blueprint(priyanka)
app.register_blueprint(mridul)
app.register_blueprint(auth)
app.register_blueprint(sse,url_prefix="/stream")

@app.route("/",methods=['GET'])
def register_login():
    accessToken = request.cookies.get('accessToken')
    registration = 0 if accessToken else 1
    if request.args.get('login'):
        registration = 0
    if request.args.get('register'):
        registration=1
    return RenderResponse("register.html",HTTP_200_OK,context={'registration':registration})

if __name__=="__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)