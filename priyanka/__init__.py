from flask import Blueprint
from utils.RenderResponse import RenderResponse
from constants.https_status_codes import *

priyanka=Blueprint("priyanka",__name__,url_prefix="/api/v1/priyanka")

@priyanka.route("/",methods=['GET'])
def load_html():
    return RenderResponse("layout.html",HTTP_200_OK,context={"content":"Hellow"})