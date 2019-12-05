from keras.models import load_model
from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
from keras.applications.resnet50 import preprocess_input
import argparse

def main(input):
    json_file = open('dogbreedmodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("dogbreedweights.h5")

    img_path = input
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)

    if(np.argmax(preds) == 0):
        print('It is Beagle')
    elif(np.argmax(preds) == 1):
        print('It is Golden Retriever')
    elif(np.argmax(preds) == 2):
        print('It is Rottweiler')
    elif(np.argmax(preds) == 3):
        print('It is Samoyed')
    elif(np.argmax(preds) == 4):
        print('It is Siberian Husky')   
    else: print('Something went wrong')

    # print('Predicted:', decode_predictions(preds, top=3)[0])
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    args = parser.parse_args()
    main(
        input = args.input_dir
    )