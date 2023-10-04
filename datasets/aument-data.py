import os
import numpy as np
from keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
    save_img,
)

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)


input_dir = "chips-simple-original"
output_dir = "chips-simple-aumented"
num_augmented_per_image = 8  # Number of augmented images to generate per original image

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for class_name in os.listdir(input_dir):
    class_dir = os.path.join(input_dir, class_name)
    augmented_class_dir = os.path.join(output_dir, class_name)

    if not os.path.exists(augmented_class_dir):
        os.makedirs(augmented_class_dir)

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = load_img(img_path)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        i = 0
        for batch in datagen.flow(
            img_array,
            batch_size=1,
            save_to_dir=augmented_class_dir,
            save_prefix="aug",
            save_format="jpeg",
        ):
            i += 1
            if i >= num_augmented_per_image:
                break
