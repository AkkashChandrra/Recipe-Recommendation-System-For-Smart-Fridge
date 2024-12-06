# Recipe Recommendations System for Smart Fridge

This project provides a **smart fridge system** that detects ingredients from an uploaded image of fridge contents and recommends recipes based on the detected items. The system combines **object detection** with **recipe generation** using a modern graphical interface.

---

## **Table of Contents**

1. [Features](#features)  
2. [Installation](#installation)  
3. [Usage](#usage)  
4. [Dependencies](#dependencies)  
5. [Directory Structure](#directory-structure)  
6. [Screenshots](#screenshots)  
7. [Acknowledgments](#acknowledgments)  
8. [License](#license)

---

## **Features**
- **Modern User Interface**: A sleek, intuitive PyQt5-based application.
- **Ingredient Detection**: Uses Roboflow’s object detection API to identify items from fridge images.
- **Recipe Generation**: Suggests recipes based on detected ingredients using GPT-2 text generation.
- **Dynamic Image Upload**: Upload and analyze fridge images with one click.

---

## **Installation**

### **Prerequisites**
1. Python 3.8+  
2. Install required Python packages:
   ```bash
   pip install PyQt5 transformers roboflow-sdk


## **Clone REPO**

git clone https://github.com/your-username/recipe-recommendation-smart-fridge.git
cd recipe-recommendation-smart-fridge


## **Connecting to Roboflow**
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="your_api_key"
)




python app.py


