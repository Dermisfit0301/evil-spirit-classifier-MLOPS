import streamlit as st
import pickle
import pandas as pd
from scipy.sparse import hstack
from PIL import Image

# Load the pickled model
model = pickle.load(open('spirit.sav', 'rb'))

st.title('Evil spirit identification')

color_mapping = {0:'clear', 1:'green', 2:'black', 3:'white', 4:'blue', 5:'blood'}
spirit_mapping = {0:'Jinnat', 1:'Preta', 2:'Bhoot'}
def get_user_input():
    color_options = list(color_mapping.values())
    color = st.sidebar.selectbox('Color', color_options)
    color_code = list(color_mapping.keys())[list(color_mapping.values()).index(color)]
    bone_length = st.sidebar.slider('bone_length', 0.0000, 1.0000, 0.0001)
    rotting_flesh = st.sidebar.slider('rotting_flesh', 0.0000, 1.0000, 0.0001)
    hair_length = st.sidebar.slider('hair_length', 0.0000, 1.000, 0.0001)
    has_soul = st.sidebar.slider('has_soul', 0.0000, 1.0000, 0.0001)

    
    features = [bone_length, rotting_flesh, hair_length, has_soul, color_code]
    return features
# Get user input and make prediction
user_input = get_user_input()




# Select only the first 5 features of user_input, since the SVC model only uses 5 features
user_input = user_input[:5]

prediction = model.predict([user_input])[0]



# Display the prediction and relevant image
if prediction == 'Jinnat':
    st.write('<span style="font-size:50px; color:red;">The evil spirit is a Jinnat</span>', unsafe_allow_html=True)
    st.write('<span style="font-size:22px; color:black;">According to islamic lore, Jinnat are otherworldly beings who live alongside humans in a different dimension. They are made of fire and possess extraordinary powers! </span>', unsafe_allow_html=True)
    jinnat_image = Image.open('jinnat.jpg')
    st.image(jinnat_image, caption='Jinnat')
    
elif prediction == 'Preta':
    st.write('<span style="font-size:50px; color:red;">The evil spirit is a Preta</span>', unsafe_allow_html=True)
    st.write('<span style="font-size:22px; color:black;">According to Hindu scriptures, pretas are the spirit of greedy humans who are unable to let go of their glutton and hence are condemned to roam as ever hungry spirits in the afterlife!  </span>', unsafe_allow_html=True)
    preta_image = Image.open('preta.jpg')
    st.image(preta_image, caption='Preta')
    
elif prediction == 'Bhoot':
    st.write('<span style="font-size:50px; color:red;">The evil spirit is a Bhoot</span>', unsafe_allow_html=True)
    st.write('<span style="font-size:22px; color:black;">The person who dies an untimely death or due to some unnatural incidents become a bhoot. They roam the place where they died and try to possess people to fulfill their unfullfilled wishes!  </span>', unsafe_allow_html=True)
    bhoot_image = Image.open('bhoot.jpg')
    st.image(bhoot_image, caption='Bhoot')
   
else:
    st.write('<span style="font-size:32px; color:red;">Error: invalid prediction</span>', unsafe_allow_html=True)