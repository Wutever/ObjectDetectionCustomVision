from azure.cognitiveservices.vision.customvision.training import training_api
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry, Region
from azure.cognitiveservices.vision.customvision.prediction import prediction_endpoint
from azure.cognitiveservices.vision.customvision.prediction.prediction_endpoint import models
import time
import csv
import os.path
from glob import glob
import chardet
import numpy as np


# Replace with a valid key
training_key = "7411bfbbf68a4249a906a54f6cd93da2"
prediction_key = "9281639b5f9045ceba9ea7990ed3f58d"

#bounding box reference size
Rwidth = 1080
Rheight = 750

#training file location declearation
imageLoc = "test/Michael Wanfang Yuan"
imagebboxLoc="test/Michael Wanfang Yuan/AI Data"
projectName = "TNC test project11"

trainer = training_api.TrainingApi(training_key)

# Find the object detection domain
obj_detection_domain = next(domain for domain in trainer.get_domains() if domain.type == "ObjectDetection")

# Create a new project
print ("Creating project...")
project = trainer.create_project(projectName, domain_id=obj_detection_domain.id)

#get csv files from the folder
files = []
start_dir = os.getcwd()
pattern   = "*.csv"

for dir,_,_ in os.walk(start_dir):
    files.extend(glob(os.path.join(dir,pattern))) 
photo_dict= {}
species = []
i = 0
for file in files:
    with open(file, "rt",encoding='GB2312') as f:
        #print (i)
        if i == 8 :
            i=i+1
            continue
        reader1 = csv.reader(f, delimiter = ',')
        for row in reader1:
            if row[10] == '':
                row[10]='empty'
            photo_dict[row[0]] = row[10]
            species.append(row[10])
        i=i+1

uniquespecies =np.unique(species)
uniquespecies= uniquespecies.tolist()


lista = []
for specie in uniquespecies :
    lista.append (trainer.create_tag(project.id,specie))

filelist = os.listdir(imageLoc)
#according to the bounding box data, put the according image and its bbox infomation into the custom vision dataframe
tagged_images_with_regions = []
animal_image_regoins = {}
filelisttxt = os.listdir(imagebboxLoc)
i = 0 
for element in filelisttxt:
     if( ".txt" in element):
        #print (element[:-4])
        if element[:-4]+".JPG" in filelist:
            file = open(imagebboxLoc+"/"+element, "r")
            a=file.readlines()
            firstline=a[0]
            if(firstline[:-1] == '0'):
                left = 0
                top = 0
                width = 0
                height = 0
            else:
                x=a[1]
                x=x[:-1]
                x=[int(i) for i in x.split()]
                left  = x[0]/Rwidth
                top = x[1]/Rheight
                width = (x[2]-x[0]) /Rwidth
                height = (x[3]-x[1])/Rheight
            animal_image_regoins[element[:-4]] =[left,top,width,height]
            x,y,w,h = animal_image_regoins[element[:-4]]
            print(element)

            if (element[:-4] in photo_dict):
                regions = [ Region(tag_id=lista[uniquespecies.index(photo_dict[element[:-4]])].id, left=x,top=y,width=w,height=h) ]
                with open(imageLoc+"/" + element[:-4] + ".jpg", mode="rb") as image_contents:                   
                    tagged_images_with_regions.append(ImageFileCreateEntry(name=element[:-4], contents=image_contents.read(), regions=regions))
                    i=i+1   
            if(i%50==0 and i !=0):
                trainer.create_images_from_files(project.id, images=tagged_images_with_regions)
                tagged_images_with_regions=[]


#the following part can be done in the web
print ("Training...")
iteration = trainer.train_project(project.id)
while (iteration.status != "Completed"):
    iteration = trainer.get_iteration(project.id, iteration.id)
    print ("Training status: " + iteration.status)
    time.sleep(1)

# The iteration is now trained. Make it the default project endpoint
trainer.update_iteration(project.id, iteration.id, is_default=True)
print ("Done!")


predictor = prediction_endpoint.PredictionEndpoint(prediction_key)

# Open the sample image and get back the prediction results.
with open("images/Test/test_od_image.jpg", mode="rb") as test_data:
    results = predictor.predict_image(project.id, test_data, iteration.id)

# Display the results.
for prediction in results.predictions:
    print ("\t" + prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100), prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height)
