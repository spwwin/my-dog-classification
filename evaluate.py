from keras.models import load_model
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

json_file = open('dogbreedmodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("dogbreedweights.h5")

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

output = model.evaluate_generator(test_generator)
print(output)