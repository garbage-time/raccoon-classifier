import tensorflow as tf
import numpy as np
import streamlit as st

from PIL import Image

def load_and_preprocess_image(image):
    img = tf.keras.preprocessing.image.img_to_array(image)
    img = tf.image.resize(img, (180, 180))

    #Apply RGB to grayscale images to prevent crash:
    if img.shape[-1] == 1:
        img = tf.image.grayscale_to_rgb(img)

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Reshape to add batch dimension
    return img_array

def load_model(model_path='raccoon.tflite'):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_image_class(image, interpreter = load_model()):
    class_names = ['a cat :cat:', 'a dog :dog:', 'a possum', 'a raccoon :raccoon:', 'something else']
    img_array = load_and_preprocess_image(image)

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    score = tf.nn.softmax(predictions[0])
    result = "I'm {:.2f}% certain that's {}.".format(100 * np.max(score), class_names[np.argmax(score)])
    return result

def main():
    st.title("Raccoon Image Classifier")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            st.image(image, caption="Uploaded Image.", use_column_width=True)

            # Make prediction
            prediction = predict_image_class(img_array)

            # Display the prediction
            st.write(prediction)
        except:
            st.write("I couldn't parse this image :(")

if __name__ == "__main__":
    main()