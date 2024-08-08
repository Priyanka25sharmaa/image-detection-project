from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
from flask_jwt_extended import JWTManager
from flask_socketio import SocketIO
import os

app=Flask(__name__)
CORS(app)

app.config["SQLALCHEMY_DATABASE_URI"]="sqlite:///projectdb.db"
app.config['SECRET_KEY']='SUPER-SECRET-KEY'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"]=False
app.config["JWT_COOKIE_SECURE"] = False
app.config["JWT_SECRET_KEY"] = "super-secret"
app.config['CACHE_TYPE']='simple'
app.config['MODEL_PATH']='path/to'
app.config["REDIS_URL"] = "redis://localhost:6379/0"
app.config['JWT_COOKIE_CSRF_PROTECT']=False
app.config['JWT_TOKEN_LOCATION'] = ["headers", "cookies", "json", "query_string"]

db=SQLAlchemy(app)
jwt=JWTManager(app)
cache = Cache(app)
socket=SocketIO(app,cors_allowed_origins="*")