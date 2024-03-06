from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from flask_mail import Mail, Message
import random
import string
from werkzeug.utils import secure_filename
import os
import plotly.express as px
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np,pandas as pd
import os
import csv
from dotenv import load_dotenv
import pdfkit
from reportlab.pdfgen import canvas
from io import BytesIO
from flask.helpers import send_file
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from io import BytesIO

app = Flask(__name__)
mail = Mail(app)
load_dotenv()

app.secret_key = 'MYSECRETKEY'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT') or 465)
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')
mail = Mail(app)
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), nullable=False)
    password = db.Column(db.String(120), nullable=False)
  
    type_of_doctor = db.Column(db.String(50))

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    blood_group = db.Column(db.String(10), nullable=False)
    time_slot = db.Column(db.String(50), nullable=False)
    phone_number = db.Column(db.String(15), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    type_of_doctor = db.Column(db.String(50))
    status = db.Column(db.String(20), default='Pending')
    prescription_file = db.Column(db.String(255))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('appointments', lazy=True))

def create_tables():
    with app.app_context():
        db.create_all()

def generate_random_string(length=10):
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))

def send_mail(subject, recipient, body):
    msg = Message(subject, recipients=[recipient])
    msg.body = body
    mail.send(msg)
    
# Set the path to the directory containing text files
text_files_dir = os.path.join(os.path.dirname(__file__), 'static/prescriptions')

# Set the path to the directory where PDFs will be saved
pdf_output_dir = os.path.join(os.path.dirname(__file__), 'static/pdfs')

# Function to convert text file to PDF
def convert_to_pdf(file_path, output_path):
    with open(file_path, 'r') as file:
        content = file.read()

    pdfkit.from_string(content, output_path, {'title': 'PDF Conversion', 'footer-center': '[page]/[topage]'})
    
# ============================================================ model ============================================================ 


data = pd.read_csv(os.path.join("static","Data", "Training.csv"))
df = pd.DataFrame(data)
cols = df.columns
cols = cols[:-1]
x = df[cols]
y = df['prognosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

dt = DecisionTreeClassifier()
clf_dt=dt.fit(x_train,y_train)

indices = [i for i in range(132)]
symptoms = df.columns.values[:-1]

dictionary = dict(zip(symptoms,indices))

def predict(symptom):
    user_input_symptoms = symptom
    user_input_label = [0 for i in range(132)]
    for i in user_input_symptoms:
        idx = dictionary[i]
        user_input_label[idx] = 1

    user_input_label = np.array(user_input_label)
    user_input_label = user_input_label.reshape((-1, 1)).transpose()

    predicted_disease = dt.predict(user_input_label)[0]
    confidence_score = np.max(dt.predict_proba(user_input_label)) * 100  # Assuming decision tree has predict_proba method

    return predicted_disease, confidence_score

with open('static/Data/Testing.csv', newline='') as f:
        reader = csv.reader(f)
        symptoms = next(reader)
        symptoms = symptoms[:len(symptoms)-1]
        
# ============================================================ routes ============================================================ 

@app.route('/', methods=['GET', 'POST'])
def index():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        if user.type_of_doctor:
            appointments = Appointment.query.filter_by(type_of_doctor=user.type_of_doctor).all()
            return render_template('doctor-dashboard.html', username=username, appointments=appointments)
            
        else:
            user_appointments = user.appointments
            return render_template('patient-dashboard.html', username=username, user_appointments=user_appointments)
            
    return render_template('index.html')


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        Email = user.email 
        user_appointments = user.appointments
        return render_template('patient-profile.html', username=username,Email=Email, user_appointments=user_appointments)
    return render_template('index')

@app.route('/patient-register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        try:
            user = User(username=username, email=email, password=password)
            db.session.add(user)
            db.session.commit()
            session['user_id'] = user.id
            return redirect(url_for('index'))
        except IntegrityError:
            db.session.rollback()
            flash('Username already exists. Please choose a different username.', 'error')

    return render_template('patient-register.html')

@app.route('/doctor-register', methods=['GET', 'POST'])
def doctor_register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        type_of_doctor = request.form['type_of_doctor']
        user = User(username=username,email=email, password=password, type_of_doctor=type_of_doctor)
        db.session.add(user)
        db.session.commit()
        session['user_id'] = user.id
        return redirect(url_for('index'))
    return render_template('doctor-register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user_id'] = user.id
            return redirect(url_for('index'))
        else:
            flash('Wrong username or password. Please try again.', 'error')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/book-appointment', methods=['GET', 'POST'])
def book_appointment():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    username = None
    
    user = User.query.get(session['user_id'])
    username = user.username
    
    # Fetch distinct types of doctors from the database
    doctor_types = db.session.query(User.type_of_doctor).distinct().all()
    doctor_types = [doctor[0] for doctor in doctor_types]

    if request.method == 'POST':
        name = request.form['name']
        age = int(request.form['age'])
        blood_group = request.form['blood_group']
        time_slot = request.form['time_slot']
        phone_number = request.form['phone_number']
        email = request.form['email']
        type_of_doctor = request.form['type_of_doctor']

        appointment = Appointment(
            name=name,
            age=age,
            blood_group=blood_group,
            time_slot=time_slot,
            phone_number=phone_number,
            email=email,
            type_of_doctor=type_of_doctor,
            user=user
        )

        db.session.add(appointment)
        db.session.commit()

        # Notify the doctor via email
        doctor_email = User.query.filter_by(type_of_doctor=type_of_doctor).first().email
        subject = 'New Appointment Request'
        body = f'Hello Doctor,\n\nYou have a new appointment request. Please log in to the system to approve or reject it.'
        send_mail(subject, doctor_email, body)

        return redirect(url_for('index'))

    return render_template('book-appointment.html',doctor_types=doctor_types,username=username)

@app.route('/approve-appointment/<int:appointment_id>')
def approve_appointment(appointment_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    doctor = User.query.get(session['user_id'])
    appointment = Appointment.query.get(appointment_id)

    if appointment.type_of_doctor != doctor.type_of_doctor:
        return redirect(url_for('index'))

    appointment.status = 'Approved'
    db.session.commit()

    # Notify the patient via email
    subject = 'Appointment Approved'
    body = f'Hello {appointment.name},\n\nYour appointment has been approved. Please log in to the system to view the details.'
    send_mail(subject, appointment.email, body)

    return redirect(url_for('index'))

@app.route('/policy')
def policy():
    return render_template('privacy-policy.html')

@app.route('/Transforming_Healthcare')
def Transforming_Healthcare():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('blog_Transforming Healthcare.html',username=username)
    return render_template('index.html')

@app.route('/Holistic_Health')
def Holistic_Health():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('blog_Holistic Health.html',username=username)
    return render_template('index.html')

@app.route('/Nourishing_Body')
def Nourishing_Body():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('blog_Nourishing_Body.html',username=username)
    return render_template('index.html')

@app.route('/Importance_of_Games')
def Importance_of_Games():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('blog_Importance_of_Games.html',username=username)
    return render_template('index.html')

@app.route('/admin')
def admin():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('admin.html',username=username)
    return render_template('index.html')

@app.route('/videocall')
def videocall():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('videocall.html',username=username)
    
    return render_template('index.html')

@app.route('/doctor-patients')
def doctor_patients():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username

        doctor = User.query.get(session['user_id'])

        if not doctor.type_of_doctor:
            return redirect(url_for('index'))

        # Fetch appointments assigned to the doctor
        appointments = Appointment.query.filter_by(type_of_doctor=doctor.type_of_doctor).all()
        file_list = os.listdir(text_files_dir)

        return render_template('doctor-patients.html', doctor=doctor, appointments=appointments,username=username,file_list=file_list)
    return render_template('index.html')

@app.route('/prescribe-medicine/<int:appointment_id>', methods=['GET', 'POST'])
def prescribe_medicine(appointment_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    doctor = User.query.get(session['user_id'])
    appointment = Appointment.query.get(appointment_id)

    if appointment.type_of_doctor != doctor.type_of_doctor:
        return redirect(url_for('index'))

    available_medicines = ["Medicine 1", "Medicine 2", "Medicine 3"]  # Update this with your list of medicines

    if request.method == 'POST':
        selected_medicines = request.form.getlist('medicines[]')

        # Create a PDF document using ReportLab
        buffer = BytesIO()
        pdf = SimpleDocTemplate(buffer, pagesize=letter)

        # Define styles for the header and footer
        styles = getSampleStyleSheet()
        header_style = ParagraphStyle(
            'Header1',
            parent=styles['Heading1'],
            fontName='monospace',
            fontSize=18,
            spaceAfter=12,
            textColor=colors.green,
        )

        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.gray,
        )

        # Create content for the PDF
        content = []

        # Add Jansevak header with green color
        jansevak_header = Paragraph("<font color='green' size='24'><b>Jansevak: We Care for Your Health</b></font>", header_style)
        content.append(jansevak_header)

        # Add space after Jansevak header
        content.append(Spacer(1, 12))

        # Add patient details
        patient_details = (
            f"<b>Patient Details:</b><br/>"
            f"Name: {appointment.name}<br/>"
            f"Age: {appointment.age}<br/>"
            f"Blood Group: {appointment.blood_group}<br/>"
            f"Phone Number: {appointment.phone_number}"
        )
        content.append(Paragraph(patient_details, styles['Normal']))

        # Add space after patient details
        content.append(Spacer(1, 12))

        # Add prescribed medicines
        prescribed_meds = "<b>Prescribed Medicines:</b><br/>"
        for medicine in selected_medicines:
            prescribed_meds += f"- {medicine}<br/>"
        content.append(Paragraph(prescribed_meds, styles['Normal']))

        # Add space after prescribed medicines
        content.append(Spacer(1, 12))

        # Add doctor details and footer
        doctor_details = (
            f"<b>Prescribed by Dr. {doctor.username} ({doctor.type_of_doctor})</b><br/>"
            "Thank you for choosing Jansevak! We wish you good health."
        )
        content.append(Paragraph(doctor_details, styles['Normal']))

        # Add space after doctor details
        content.append(Spacer(1, 12))

        # Build the PDF
        pdf.build(content)

        # Save the PDF to the file
        pdf_filename = f"prescription_{appointment_id}.pdf"
        pdf_filepath = os.path.join("static", "prescriptions", pdf_filename)
        buffer.seek(0)
        with open(pdf_filepath, 'wb') as pdf_file:
            pdf_file.write(buffer.read())

        buffer.close()

        # Update appointment status to 'Prescribed'
        appointment.status = 'Prescribed'
        appointment.prescription_file = pdf_filepath
        db.session.commit()

        return redirect(url_for('doctor_patients'))

    return render_template('prescribe-medicine.html', appointment=appointment, available_medicines=available_medicines)
    
@app.route('/view-prescription/<int:appointment_id>')
def view_prescription(appointment_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    doctor = User.query.get(session['user_id'])
    appointment = Appointment.query.get(appointment_id)

    if appointment.type_of_doctor != doctor.type_of_doctor or appointment.status != 'Prescribed':
        return redirect(url_for('index'))

    prescription_filepath = appointment.prescription_file

    return send_file(prescription_filepath, as_attachment=True)

@app.route('/view-prescription-patient/<int:appointment_id>')
def view_prescription_patient(appointment_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    appointment = Appointment.query.get(appointment_id)

    if not user or not appointment or appointment.user_id != user.id or appointment.status != 'Prescribed':
        return redirect(url_for('profile'))  # Change this line to redirect to the patient's profile instead of index

    # Read prescription text from the file
    prescription_filepath = appointment.prescription_file
    

    return send_file(prescription_filepath, as_attachment=True)

# ============================================================ scans ============================================================ 
    
@app.route('/braintumor', methods=['GET', 'POST'])
def braintumor():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('brain-tumor.html',username=username)
    else:
        return render_template('index.html')
    

@app.route('/disease_predict', methods=['GET', 'POST'])
def disease_predict():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        chart_data={}
        if request.method == 'POST':
            selected_symptoms = []
            if(request.form['Symptom1']!="") and (request.form['Symptom1'] not in selected_symptoms):
                selected_symptoms.append(request.form['Symptom1'])
            if(request.form['Symptom2']!="") and (request.form['Symptom2'] not in selected_symptoms):
                selected_symptoms.append(request.form['Symptom2'])
            if(request.form['Symptom3']!="") and (request.form['Symptom3'] not in selected_symptoms):
                selected_symptoms.append(request.form['Symptom3'])
            if(request.form['Symptom4']!="") and (request.form['Symptom4'] not in selected_symptoms):
                selected_symptoms.append(request.form['Symptom4'])
            if(request.form['Symptom5']!="") and (request.form['Symptom5'] not in selected_symptoms):
                selected_symptoms.append(request.form['Symptom5'])
            disease, confidence_score = predict(selected_symptoms)
            
            chart_data = {
            'disease': disease,
            'confidence_score': confidence_score
            }
            return render_template('disease_predict.html',symptoms=symptoms,disease=disease, chart_data=chart_data,confidence_score=confidence_score,username=username)
            
        return render_template('disease_predict.html',symptoms=symptoms,username=username,chart_data=chart_data)
    else:
        return render_template('index.html')

@app.route('/lung')
def lung():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('lung.html',username=username)
    else:
        return render_template('index.html')

@app.route('/cataract')
def cataract():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('cataract.html',username=username)
    return render_template('index.html')


# if __name__ == '__main__':
#     create_tables()
#     app.run(debug=True)