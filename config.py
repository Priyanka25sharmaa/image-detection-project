from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
import os

app=Flask(__name__)
CORS(app)

app.config["SQLALCHEMY_DATABASE_URI"]="sqlite:///projectdb.db"
app.config['SECRET_KEY']='SUPER-SECRET-KEY'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"]=False
app.config["JWT_COOKIE_SECURE"] = False
app.config["JWT_SECRET_KEY"] = "super-secret"
app.config['JWT_TOKEN_LOCATION'] = ["headers", "cookies", "json", "query_string"]

db=SQLAlchemy(app)
jwt=JWTManager(app)