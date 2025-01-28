# way to upload image: endpoint
# way to save the image
# function to make prediction on the image
# show the results
import os
import cv2
from flask import Flask
from flask import request
from flask import render_template
#from tensorflow.keras.models import load_model
from random import randint
from tensorflow.keras.models import load_model


app = Flask(__name__)
UPLOAD_FOLDER = "static/"

model = load_model("model.h5")


import cv2

def preprocess_image(image_path):
    
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Unable to read image")
        return None
    
   
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('static/preprocess.jpg',gray_image)
    
    
def segmented(im):
    import cv2


    image = cv2.imread(im)


    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    _, segmented_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite('static/seg.jpg',segmented_image)

    


    
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    value=""
    tt=""
    nrows = 150
    ncolumns  = 150
    channels = 3
    import numpy as np
    from PIL import Image
    import image_dehazer
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)
           
            print(f"path:{image_file.filename}")
            preprocess_image('static/'+image_file.filename)

            segmented('static/'+image_file.filename)
            import numpy as np
            import cv2
            from sklearn.svm import SVR
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error

            
            hazy_img = cv2.imread('static/'+image_file.filename)
            clear_img = cv2.imread('static/'+image_file.filename)

           
            hazy_gray = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2GRAY)
            clear_gray = cv2.cvtColor(clear_img, cv2.COLOR_BGR2GRAY)

            
            hazy_flat = hazy_gray.flatten().reshape(-1, 1)
            clear_flat = clear_gray.flatten()

            
            X_train, X_test, y_train, y_test = train_test_split(hazy_flat, clear_flat, test_size=0.2, random_state=42)
            svm = SVR(kernel='linear')
            svm.fit(X_train, y_train)
            HazeImg = svm.predict(X_test)                        
            image = Image.open(image_location).convert("RGB")
            print("patho"+os.path.dirname('preprocess.jpg'))
            pre = "static/preprocess.jpg"
            seg = "static/seg.jpg"
            print(pre)
            print(image_location)
            HazeImg = cv2.imread(image_location)						
            HazeCorrectedImg, haze_map = image_dehazer.remove_haze(HazeImg, showHazeTransmissionMap=False)
            # cv2.imshow('haze_map', haze_map);# display the original hazy image
            # cv2.imshow('enhanced_image', HazeCorrectedImg);			
            # cv2.waitKey(0)
            cv2.imwrite("static/result.jpg", HazeCorrectedImg)
            enhanced = "static/result.jpg"
            image=cv2.imread('static/result.jpg')
            from sklearn import linear_model
            clf = linear_model.Lasso(alpha=0.1)
            clf.fit(X_train, y_train)
            fog = clf.predict(X_test)
            fog_color=(255, 255, 255)
            fog_density=0.9
            fog_mask = cv2.GaussianBlur(image, (17, 17), 0)

  
            fogged_image = cv2.addWeighted(image, 1 - fog_density, fog_mask, fog_density, 0)
            
            cv2.imwrite("static/lasso.jpg", fogged_image)
            im="static/lasso.jpg"
      


                
            return render_template("result.html", img_path=enhanced,pre_im=pre,seg_im=seg,lasso=im)
        else:
                return render_template('result1.html',img_path=image_location)
    return render_template("index.html")
    
if __name__ == "__main__":
    app.run(port=12000, debug=True)
    