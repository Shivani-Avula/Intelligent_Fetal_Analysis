import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, send_file,send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_wtf import FlaskForm
from wtforms import PasswordField, StringField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo, ValidationError, Regexp
#from forms import RegistrationForm
#from your_app.models import User  # Adjust this import as needed
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[
        DataRequired(),
        Length(min=3, max=20, message="Username must be between 3 and 20 characters")
    ])
    
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=8, message="Password must be at least 8 characters long."),
        Regexp(
             r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).+$',
            message="Password must contain at least one uppercase letter, one lowercase letter, and one number."
        )    
    ])
    
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(),
        EqualTo('password', message="Passwords must match.")
    ])
    
    submit = SubmitField('Sign Up')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Username already exists. Please choose a different one.')
# Initialize Flask app, database, and bcrypt
app = Flask(__name__)
app.config['SECRET_KEY'] = '132861'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Folder to store uploaded files
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Initialize Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Redirect to login page if not logged in

# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

# Result model to store the user's scan results
class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(100), nullable=False)
    prediction = db.Column(db.String(100), nullable=False)

# Registration form
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=150)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6, max=150)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

# Load user by ID for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
# Load the trained CNN model
model = load_model('your_model.h5')  # Replace with the actual path to your model
# Home route (Welcome page after login)
@app.route('/welcome')
@login_required
def welcome():
    username= current_user.username
    return render_template('welcome.html', username=username)
# Home route (Upload and Compare Forms)
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    username= current_user.username
    user_id = current_user.id
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                flash('No selected file', 'danger')
                return redirect(url_for('upload_scan'))

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the image and predict the class label
            img = image.load_img(file_path, target_size=(224, 224))  # Adjust size as per your model
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Predict the class using your CNN model
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]

            # Map the predicted class to a label (adjust as per your classes)
            class_labels = {0: 'Trans-thalamic', 1: 'Trans-cerebellum', 2: 'Trans-ventricular', 3: 'Other'}
            result_label = class_labels.get(predicted_class, 'Unknown')

            # Save the result in the database
            result = Result(user_id=current_user.id, filename=filename, prediction=result_label)
            db.session.add(result)
            db.session.commit()

            flash(f'Scan uploaded and classified as: {result_label}', 'success')

        # Get previous results for comparison dropdown
        previous_results = Result.query.filter_by(user_id=current_user.id).all()
        return render_template('upload_scan.html', user_id=user_id, username=username, previous_results=previous_results)

    # Get previous results for comparison dropdown
    previous_results = Result.query.filter_by(user_id=current_user.id).all()
    return render_template('upload_scan.html', user_id=user_id, username=username, previous_results=previous_results)

from matplotlib import pyplot as plt
from skimage import feature

@app.route('/compare', methods=['GET', 'POST'])
@login_required
def compare():
    user_id = current_user.id
    username = current_user.username
    previous_results = Result.query.filter_by(user_id=user_id).all()

    if request.method == 'POST':
        # Get scan IDs from the form
        scan_id_1 = request.form.get('scan_1')
        scan_id_2 = request.form.get('scan_2')

        # Fetch scan details from the database
        result_1 = Result.query.get_or_404(scan_id_1)
        result_2 = Result.query.get_or_404(scan_id_2)

        # Load images for comparison
        img1 = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], result_1.filename))
        img2 = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], result_2.filename))

        # Convert images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Resize the second image to match the first image's dimensions
        height, width = gray1.shape  # Get dimensions of the first image
        gray2_resized = cv2.resize(gray2, (width, height))  # Resize the second image

        # Initialize ORB detector
        orb = cv2.ORB_create()

        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2_resized, None)  # Use resized image

        # Use Brute Force Matcher to find matches
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw matches
        img_matches = cv2.drawMatches(
            img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesThickness=2)

        # Save the image with matches drawn
        matches_path = os.path.join(app.config['UPLOAD_FOLDER'], 'matches.png')
        cv2.imwrite(matches_path, img_matches)

        # Calculate Structural Similarity Index (SSIM) for grayscale images
        similarity_index, diff_map = ssim(gray1, gray2_resized, full=True)  # Use resized image
        diff_map = (diff_map * 255).astype(np.uint8)

        # Save heatmap of differences
        plt.figure(figsize=(8, 8))
        plt.imshow(diff_map, cmap='hot', interpolation='nearest')
        plt.title('Difference Heatmap')
        plt.axis('off')
        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], 'heatmap.png')
        plt.savefig(heatmap_path)
        plt.close()

        # Edge-based difference visualization
        edges_img1 = feature.canny(gray1)
        edges_img2 = feature.canny(gray2_resized)  # Use resized image
        edges_diff = np.abs(edges_img1.astype(int) - edges_img2.astype(int))

        plt.figure(figsize=(8, 8))
        plt.imshow(edges_diff, cmap='gray')
        plt.title('Edge Differences')
        plt.axis('off')
        edge_diff_path = os.path.join(app.config['UPLOAD_FOLDER'], 'edge_diff.png')
        plt.savefig(edge_diff_path)
        plt.close()

        # Detect keypoint changes and provide insights
        keypoints_summary = analyze_keypoints(kp1, kp2,img1,img2)

        # Determine improvement based on SSIM
        improvement = "Improvement detected" if similarity_index > 0.5 else "No significant improvement detected"

        # Pass results to the comparison page
        report_data = {
        'result_1': result_1,
        'result_2': result_2,
        'similarity_index': f"{similarity_index:.4f}",
        'improvement': improvement,
        'matches_path': matches_path,
        'heatmap_path': heatmap_path,
        'edge_diff_path': edge_diff_path,
        'keypoints_summary': keypoints_summary,
        'username': username,  # Passing username
        'user_id': user_id     # Passing user_id
        }
 
        return render_template('compare.html', **report_data)
    return render_template('compare_scan.html', user_id=user_id, username=username, previous_results=previous_results)

def analyze_keypoints(kp1, kp2, img1, img2):
    """
    Analyze keypoint matches and provide insights based on dynamically detected regions of interest (ROIs)
    in the input images. This method adapts to the content of the images and provides a concise result.
    """
    insights = []

    # Compare the number of matching and non-matching keypoints
    matching_keypoints = len(kp1)
    non_matching_keypoints = len(kp2) - matching_keypoints
    insights.append(f"Number of matching keypoints: {matching_keypoints}")
    
    # Basic analysis based on matching keypoints
    if matching_keypoints > non_matching_keypoints:
        insights.append("Keypoints in matching areas suggest a positive developmental change.")
    else:
        insights.append("Mismatch in keypoints suggests potential structural differences between the two images.")
    
    # Dynamically identify regions of interest (ROIs) based on edge detection or segmentation
    # Here, using edge detection as an example, but this can be replaced with more sophisticated techniques
    edges_img1 = cv2.Canny(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 100, 200)
    edges_img2 = cv2.Canny(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 100, 200)

    contours_img1, _ = cv2.findContours(edges_img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_img2, _ = cv2.findContours(edges_img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Track the number of significant matching keypoints
    total_matching_in_regions = 0

    for contour1, contour2 in zip(contours_img1, contours_img2):
        # Use bounding boxes to determine the region
        x1, y1, w1, h1 = cv2.boundingRect(contour1)
        x2, y2, w2, h2 = cv2.boundingRect(contour2)

        # Count how many keypoints fall within each contour region (bounding box)
        kp1_in_region = [kp for kp in kp1 if x1 < kp.pt[0] < x1 + w1 and y1 < kp.pt[1] < y1 + h1]
        kp2_in_region = [kp for kp in kp2 if x2 < kp.pt[0] < x2 + w2 and y2 < kp.pt[1] < y2 + h2]

        # Skip regions with very few matching keypoints
        if len(kp1_in_region) > 0 or len(kp2_in_region) > 0:
            total_matching_in_regions += len(kp1_in_region) + len(kp2_in_region)
        
        # Only output for significant regions (e.g., more than 1 keypoint match in either image)
        #if len(kp1_in_region) > 1 and len(kp2_in_region) > 1:
         #   if len(kp1_in_region) > len(kp2_in_region):
          #      insights.append(f"More keypoints in this region in Image 1, indicating stronger feature matching.")
         #   elif len(kp1_in_region) < len(kp2_in_region):
           #     insights.append(f"More keypoints in this region in Image 2, suggesting a developmental change.")
           # else:
            #    insights.append("Keypoints in this region are similar in both images, indicating balanced development.")

    # Summary of overall result
    if total_matching_in_regions > 100:
        insights.append("Overall, keypoint analysis suggests a significant matching trend.")
    else:
        insights.append("No significant matching keypoints were found across regions.")
    return "<br>".join(insights)
@app.route('/generate_report/<int:user_id>', methods=['GET', 'POST'])
@login_required
def generate_comparison_report(user_id):
    if request.method == 'POST':
        # Get scan IDs from the form
        scan_1_id = request.form.get('scan_1')
        scan_2_id = request.form.get('scan_2')
        
        # Fetch scan details from the database
        result_1 = Result.query.get_or_404(scan_1_id)
        result_2 = Result.query.get_or_404(scan_2_id)
        
        # Read and convert the images to grayscale
        img1 = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], result_1.filename))
        img2 = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], result_2.filename))
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Resize second image to match the first image's dimensions
        height, width = gray1.shape  
        gray2_resized = cv2.resize(gray2, (width, height))

        # Compute SSIM (Structural Similarity Index)
        similarity_index, _ = ssim(gray1, gray2_resized, full=True)
        if similarity_index is None:
            similarity_index = 0.0  # Default value if None

        # Curability assessment logic
        if similarity_index > 0.65:
            curability_message = "The comparison suggests a high level of similarity and indicates positive prognosis trends."
        elif 0.35 <= similarity_index <= 0.65:
            curability_message = "The comparison suggests moderate similarity. Follow-up analysis might be required."
        else:
            curability_message = "The scans show low similarity. Immediate medical attention may be necessary."

        # Paths for generated images
        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], 'heatmap.png')
        edge_diff_path = os.path.join(app.config['UPLOAD_FOLDER'], 'edge_diff.png')
        matches_path = os.path.join(app.config['UPLOAD_FOLDER'], 'matches.png')

        # Create PDF report
        pdf_file = BytesIO()
        c = canvas.Canvas(pdf_file, pagesize=letter)

        # Title: Fetal Health Analysis
        c.setFont("Helvetica-Bold", 16)
        c.drawString(200, 780, "Fetal Health Analysis")

        # Section: Comparison Report
        c.setFont("Helvetica-Bold", 14)
        c.drawString(30, 750, "Comparison Report")

        c.setFont("Helvetica", 12)
        c.drawString(30, 730, f"User: {current_user.username} (ID: {user_id})")
        c.drawString(30, 710, f"Scan 1: {result_1.filename} - {result_1.prediction}")
        c.drawString(30, 690, f"Scan 2: {result_2.filename} - {result_2.prediction}")

        # Section: Structural Similarity Index
        c.setFont("Helvetica-Bold", 14)
        c.drawString(30, 670, "Structural Similarity Index (SSIM):")

        c.setFont("Helvetica", 12)
        c.drawString(30, 650, f"{similarity_index:.4f}")

        # Section: Curability Assessment
        c.setFont("Helvetica-Bold", 14)
        c.drawString(30, 630, "Curability Assessment:")

        c.setFont("Helvetica", 12)
        c.drawString(30, 610, curability_message)

        # Insert images into PDF report
        page_width, page_height = letter  # Standard letter size 8.5 x 11 inches
        img_width, img_height = 200, 150
        center_x = (page_width - img_width) / 2
        img_y = 400  

        try:
            if os.path.exists(matches_path):
                c.drawImage(matches_path, center_x, img_y, width=img_width, height=img_height)
                c.drawString(center_x, img_y - 15, "Matches Visualization")
                img_y -= 180

            if os.path.exists(heatmap_path):
                c.drawImage(heatmap_path, center_x, img_y, width=img_width, height=img_height)
                c.drawString(center_x, img_y - 15, "Difference Heatmap")
                img_y -= 180

            if os.path.exists(edge_diff_path):
                c.drawImage(edge_diff_path, center_x, img_y, width=img_width, height=img_height)
                c.drawString(center_x, img_y - 10, "Edge Difference")

        except Exception as e:
            print(f"Error inserting images: {e}")

        # Save the PDF to memory
        c.save()
        pdf_file.seek(0)

        # Send the PDF as a response
        return send_file(
            pdf_file,
            as_attachment=True,
            download_name="comparison_report.pdf",
            mimetype="application/pdf",
        )
    
    else:
        return redirect(url_for("compare"))

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        # Check if user already exists
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            flash('Username already exists. Please choose a different one.', 'danger')
            return redirect(url_for('register'))

        # Hash password and create a new user
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('login'))

    # Show individual validation errors
    for field, errors in form.errors.items():
        for error in errors:
            flash(f"{field.capitalize()}: {error}", 'danger')

    return render_template('register.html', form=form)


# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('welcome'))  # Redirect to welcome page if already logged in
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful', 'success')
            return redirect(url_for('welcome'))  # Redirect to the welcome page after login
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
            return redirect(url_for('login'))
    return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))
# Run the app
if __name__ == '__main__':
    app.run(debug=True)
