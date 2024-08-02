from flask import Flask
from config import app,db
from mridul import mridul
from auth import auth
from priyanka import priyanka

app.register_blueprint(priyanka)
app.register_blueprint(mridul)
app.register_blueprint(auth)

if __name__=="__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)