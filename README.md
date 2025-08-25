# Cardiovascular and Obesity Disease Predictor

## Introduction

This project is a machine learning application designed to predict the likelihood of cardiovascular and obesity-related diseases based on a patient's health metrics. It provides a web-based graphical user interface for user-friendly interaction and a command-line tool for direct predictions.

## Key Features

* **Multi-Label Classification**: The model is capable of predicting multiple health conditions simultaneously for a single set of inputs.
* **Web Interface**: An intuitive web-based user interface allows for easy input of patient data and clear presentation of results.
* **Command-Line Interface (CLI)**: A secondary script is provided to run predictions directly from the terminal.
* **Data Preprocessing Pipeline**: The model training process includes robust steps for data cleaning, feature engineering, and handling class imbalance to ensure model reliability.

---

## Technology Stack

* **Backend**: Python, Flask
* **Machine Learning**: Scikit-learn, Pandas, Imblearn
* **Frontend**: HTML, Tailwind CSS
* **Environment**: The application is designed to be run locally using the Flask development server.

---

## Project Structure

The project directory is organized as follows:

├── app.py                       # Flask application for the web interface and API endpoint.
├── train_model.py               # Script for data preprocessing and model training.
├── Prediction_hackathon.py      # Command-line script for making predictions.
├── templates/
│   └── index.html               # Frontend HTML file for the web application.
├── cardio_train_with_disease_prediction.csv # The raw dataset for training the model.
├── blood_model.pkl              # Serialized, trained machine learning model.
├── label_binarizer.pkl          # Serialized label binarizer for target variable encoding.
└── README.md                    # Project documentation.

---

## Setup and Installation

Follow these steps to set up the project on your local machine.

### 1. Project Directory

Ensure all project files are located in a single root directory.

### 2. Create and Activate a Virtual Environment

Using a virtual environment is highly recommended to manage project-specific dependencies.

* **macOS / Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

* **Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
    Once activated, your terminal prompt will be prefixed with `(venv)`.

### 3. Install Dependencies

Install the required Python packages from the command line.

```bash
pip install Flask pandas scikit-learn "imblearn"



