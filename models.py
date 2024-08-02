from config import db

class User(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    first_name=db.Column(db.String(80),unique=False,nullable=False)
    last_name=db.Column(db.String(80),unique=False,nullable=False)
    username=db.Column(db.String(80),unique=True,nullable=False)
    email=db.Column(db.String(120),unique=True,nullable=False)
    password=db.Column(db.String(80),unique=False,nullable=False)

    def to_json(self):
        return{
            "id":self.id,
            "firstName":self.first_name,
            "lastName":self.last_name,
            "userName":self.username,
            "email":self.email,
            "password":self.password,
        }
