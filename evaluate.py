from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import decimal

json_file = open('dogbreedmodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("dogbreedweights.h5")

beagle_test_data_folder_path = 'data/test/beagle'
goldenretriever_test_data_folder_path = 'data/test/goldenretriever'
rottweiler_test_data_folder_path = 'data/test/rottweiler'
samoyed_test_data_folder_path = 'data/test/samoyed'
siberianhusky_test_data_folder_path = 'data/test/siberianhusky'

beaglematched = 0
beaglenotmatched = 0
goldenretrievermatched = 0
goldenretrievernotmatched = 0
rottweilermatched = 0
rottweilernotmatched = 0
samoyedmatched = 0
samoyednotmached = 0
siberianhuskymatched = 0
siberianhuskynotmatched = 0

for dogimg in os.listdir(beagle_test_data_folder_path):
    if not dogimg.startswith('.'):
        img_path = os.path.join(beagle_test_data_folder_path,dogimg)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        if(np.argmax(preds) == 0):
            beaglematched+=1
        else: beaglenotmatched+=1

for dogimg in os.listdir(goldenretriever_test_data_folder_path):
    if not dogimg.startswith('.'):
        img_path = os.path.join(goldenretriever_test_data_folder_path,dogimg)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        if(np.argmax(preds) == 1):
            goldenretrievermatched+=1
        else: goldenretrievernotmatched+=1

for dogimg in os.listdir(rottweiler_test_data_folder_path):
    if not dogimg.startswith('.'):
        img_path = os.path.join(rottweiler_test_data_folder_path,dogimg)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        if(np.argmax(preds) == 2):
            rottweilermatched+=1
        else: rottweilernotmatched+=1

for dogimg in os.listdir(samoyed_test_data_folder_path):
    if not dogimg.startswith('.'):
        img_path = os.path.join(samoyed_test_data_folder_path,dogimg)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        if(np.argmax(preds) == 3):
            samoyedmatched+=1
        else: samoyednotmached+=1

for dogimg in os.listdir(siberianhusky_test_data_folder_path):
    if not dogimg.startswith('.'):
        img_path = os.path.join(siberianhusky_test_data_folder_path,dogimg)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        if(np.argmax(preds) == 4):
            siberianhuskymatched+=1
        else: siberianhuskynotmatched+=1
    
beagleaccuracy = beaglematched/(beaglematched+beaglenotmatched)
goldenretrieveraccuracy = goldenretrievermatched/(goldenretrievermatched+goldenretrievernotmatched)
rottweileraccuracy = rottweilermatched/(rottweilermatched+rottweilernotmatched)
samoyedaccuracy = samoyedmatched/(samoyedmatched+samoyednotmached)
siberianhuskyaccuracy = siberianhuskymatched/(siberianhuskymatched+siberianhuskynotmatched)
allclassaccuracy = (beaglematched+goldenretrievermatched+rottweilermatched+samoyedmatched+siberianhuskymatched)/(beaglematched+beaglenotmatched+goldenretrievermatched+goldenretrievernotmatched+rottweilermatched+rottweilernotmatched+samoyedmatched+samoyednotmached+siberianhuskymatched+siberianhuskynotmatched)

print('Beagle class accuracy = {:0.2f}'.format(beagleaccuracy*100))
print('Golden Retriever class accuracy = {:0.2f}'.format(goldenretrieveraccuracy*100))
print('Rottweiler class accuracy = {:0.2f}'.format(rottweileraccuracy*100))
print('Samoyed class accuracy = {:0.2f}'.format(samoyedaccuracy*100))
print('Siberian Husky class accuracy = {:0.2f}'.format(siberianhuskyaccuracy*100))
print('All class accuracy = {:0.2f}'.format(allclassaccuracy*100))