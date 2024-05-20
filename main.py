import os
from keras import Sequential
from keras.applications import VGG16
from keras.layers import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.src.utils import load_img, img_to_array
from mapper import mapper

from tensorflow.lite.python.schema_py_generated import np

directory = "train/train"

Name=[]
for file in os.listdir(directory):
    Name+=[file]

fruit_map = dict(zip(Name, [t for t in range(len(Name))]))
print(fruit_map)
r_fruit_map = dict(zip([t for t in range(len(Name))], Name))

img_datagen = ImageDataGenerator(rescale=1./255,
                                vertical_flip=True,
                                horizontal_flip=True,
                                rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                zoom_range=0.1,
                                validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = img_datagen.flow_from_directory(directory,
                                                   shuffle=True,
                                                   batch_size=32,
                                                   subset='training',
                                                   target_size=(100, 100))

valid_generator = img_datagen.flow_from_directory(directory,
                                                   shuffle=True,
                                                   batch_size=16,
                                                   subset='validation',
                                                   target_size=(100, 100))

# Load pre-trained VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# Freeze layers to prevent training
for layer in vgg_model.layers:
    layer.trainable = False

# Add custom layers for classification
model = Sequential()
model.add(vgg_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(fruit_map), activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, validation_data=valid_generator,
                    steps_per_epoch=train_generator.n // train_generator.batch_size,
                    validation_steps=valid_generator.n // valid_generator.batch_size,
                    epochs=10)
image=load_img("test/test/0030.jpg",target_size=(100,100))

image=img_to_array(image)
image=image/255.0
prediction_image = np.array(image)
prediction_image = np.expand_dims(image, axis=0)

prediction=model.predict(prediction_image)
value=np.argmax(prediction)
move_name = r_fruit_map[value]
print("Prediction is {}.".format(move_name))
