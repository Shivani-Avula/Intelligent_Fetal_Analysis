\# 🩺 Intelligent Fetal Analysis



An AI-powered fetal health monitoring and prediction system that uses deep learning models to analyze fetal data and predict health conditions.  

The system supports secure web-based access for doctors and patients, allowing for easy data uploading, automated predictions, and report generation.



---



\## 📌 Features

\- \*\*Deep Learning Model\*\* – Trained with medical datasets for fetal health prediction.

\- \*\*Multiple Model Support\*\* – Includes different architectures (`best\_model.h5`, `your\_model.h5`).

\- \*\*Web Interface\*\* – Built with Flask for real-time interaction.

\- \*\*File Upload\*\* – Supports uploading CSV data for prediction.

\- \*\*Automated Analysis\*\* – Generates prediction reports (`predictions.csv`).

\- \*\*Database Integration\*\* – Stores patient data in `site.db`.

\- \*\*Responsive UI\*\* – HTML/CSS templates for desktop \& mobile access.



---



\## 🛠 Tech Stack

\*\*Backend\*\*: Python, Flask  

\*\*Frontend\*\*: HTML, CSS, JavaScript  

\*\*Machine Learning\*\*: TensorFlow/Keras, NumPy, Pandas, scikit-learn  

\*\*Database\*\*: SQLite (`site.db`)  

\*\*Deployment\*\*: GitHub, (optional: Render/Heroku)



---



\## 📂 Project Structure

├── app.py # Main Flask application

├── model.py # Model loading and prediction logic

├── best\_model.h5 # Best trained deep learning model

├── your\_model.h5 # Alternative trained model

├── dataset/ # Training dataset (if provided)

├── predictions.csv # Output predictions

├── templates/ # HTML templates

├── static/ # CSS, JS, images

├── uploads/ # Uploaded CSV files

├── site.db # SQLite database

├── requirements.txt # Dependencies

└── venv/ # Virtual environment (ignored in Git)





---



\## 🚀 How to Run Locally



1\. \*\*Clone the repository\*\*

&nbsp;  ```bash

&nbsp;  git clone https://github.com/Shivani-Avula/Intelligent\_Fetal\_Analysis.git

&nbsp;  cd Intelligent\_Fetal\_Analysis



2\. Create virtual environment \& install dependencies

python -m venv venv

source venv/bin/activate      # For Linux/Mac

venv\\Scripts\\activate         # For Windows



pip install -r requirements.txt



3\. Run the application

python app.py



4\. Access in browser

http://127.0.0.1:5000





