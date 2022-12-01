import numpy as np
import streamlit as st
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img

st.set_page_config(page_title="Detector de Neumonia",
 page_icon="random",
)
st.title("Detector de Neumonia")
def cnn(image):
    with open('Red.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("Red.h5")
    # evaluate loaded model on test data
    test_image = load_img(image, target_size = (64, 64))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = loaded_model.predict(test_image)
    return 'Neumonia' if result[0][0] == 1 else 'Sano'

file = st.file_uploader("Suba una imagen de rayos X en el pecho",type=['png','jpeg','jpg'])
if file is not None:
    st.image(file, width=400)
    result = cnn(file)
    st.subheader(f"Diagnostico: {result}")
