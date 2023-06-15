#!/usr/bin/env python
# coding: utf-8

# ..............................................................................FLASK App for AutoInspect..................................................................
# FLASK app with a restful API that can detects defects in surgical instruments as well as capable for camera calibrations
# It calls multiple models i.e., 3 in a sequence

# Step 1 :- The first model detects whether there is a surgical instrument or not . If there exists no surgical instrument in the picture , it terminates the execution returning a message and a specific HTTP response code (Object Detection)
# Step 2 :- The second in the pipeline classifies the instruments as whther faulty or non-faulty (DEFECT-NET v2)
# Step 3 :- The third model in the model localizes the instruments classified as faulty in the previous step along with indicating the type of fault (YOLOv5)



#Loading libraries necessary for the predictions
import os
from flask import Flask,request,jsonify,render_template
from flask_cors import CORS
import os
import pickle
import pandas as pd



# Configure upload folder for Flask application
 

app = Flask(__name__, template_folder='templates', static_folder='staticFiles')
CORS(app)

@app.route('/')
def data():
    return render_template('index.html')

@app.route('/' ,methods=['POST'])
def car_price_prediction():
    name = request.form.get('name')
    company = request.form.get('company')
    year = request.form.get('year')
    kms_driven = request.form.get('kms_driven')
    fuel_type = request.form.get('fuel_type')

    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, 'LinearRegressionModel.pkl')

    model = pickle.load(open(file_path,'rb'))
    data = pd.DataFrame({'name': [name],
                         'company': [company],
                         'year': [year],
                         'kms_driven': [kms_driven],
                         'fuel_type': [fuel_type]})

    # Make the price prediction using the loaded model
    price_prediction = model.predict(data)

    return str(price_prediction[0])


# @app.route('/uploader/' ,methods=['POST','GET'])
# def default():
#   return render_template('index.html')

# @app.route('/inference/' ,methods=['POST','GET'])
# def inference():
#     uploaded_img = request.files['uploaded-file']
#     # # Extracting uploaded data file name
#     basepath = os.path.dirname(__file__)
#     file_path = os.path.join(basepath, 'uploads', 'image.jpg')
#     uploaded_img.save(file_path)

#     file_path='/home/saqib0494/AutoInspect/uploads/image.jpg'
#     rf = Roboflow(api_key="jg7DuFlunhwlrYBQaMkX")
#     project = rf.workspace().project("defect_detection-geufo")
#     model = project.version(7).model
#     response=model.predict(file_path,confidence=40, overlap=30).json()
#     all_defects=response['predictions']
#     count=0
#     for x in range(len(all_defects)):
#         if all_defects[x]['class']=='Defects':
#             count+=1
#     return str(count)

if __name__ == "__main__":
    app.run()


# In[ ]:




