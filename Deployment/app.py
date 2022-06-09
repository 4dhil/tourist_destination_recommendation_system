import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
import pickle
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Recomendation Exploration",
    page_icon=":round_pushpin:",
    layout="wide",
    initial_sidebar_state="expanded")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("""
<style>
res{background-color:#000000;
        font-size:80px;
        color: #FFFFFF;
        border: 2px solid;
        padding: 20px 20px 20px 70px; */
        padding: 5% 5% 5% 10%;
    text-align: center}
    
b{ background: #3F4F4F;
   padding: 25px 25px 25px 25px;
   border-radius: 10px;
   box-shadow: 0 0 20px 0 rgba(0, 0, 0, 0.2), 0 5px 5px 0 rgba(0, 0, 0, 0.24);
   color: #888
   margin: 5px 0 0 0;
   font-size: 70px;
   line-height: 0.80;
   width: 900px;
   text-align: center;
}
desc{
    font-size: 30px;
}
</style>
<b> :airplane: GYOTA (Guide Your Travel Agent) :airplane: </b>
""", unsafe_allow_html=True)

st.write(" ")
st.write(" ")

model = keras.models.load_model('my_model.h5')

rating = pd.read_csv('tourism_rating.csv')
tourism = pd.read_pickle('cluster.pkl')
tourism_cat = tourism[['Place_Id','Category', 'cluster']]
merge = pd.merge(rating, tourism_cat, on="Place_Id")

user = st.number_input('User Id', min_value=rating['User_Id'].min(), max_value=rating['User_Id'].max()) # Number Input
query_user = merge[merge['User_Id']==user][['User_Id', 'Place_Id', 'cluster']].groupby(['cluster'])['cluster'].count()
highest_cat = query_user.idxmax()
place_visited_by_user = rating[rating.User_Id == user]
place_visited = tourism[tourism['Place_Id'].isin(place_visited_by_user.Place_Id)][['Place_Id','Place_Name','cluster', 'Price', 'City']]
place_not_visited = tourism[~tourism['Place_Id'].isin(place_visited_by_user.Place_Id)][['Place_Id', 'cluster']]

st.header(highest_cat)
if highest_cat == 'others':
    place_list = place_not_visited[place_not_visited['cluster'] == 'others']['Place_Id'].values
    user_1 = np.array([user for i in range(len(place_list))])
    
    pred = model.predict([user_1,place_list])
    ids= (-pred).flatten().argsort()
    place_id = place_list[ids][:5]

elif highest_cat == 'Hiburan Keluarga dan Wisata Air':
    place_list = place_not_visited[place_not_visited['cluster'] == 'Hiburan Keluarga dan Wisata Air']['Place_Id'].values
    user_1 = np.array([user for i in range(len(place_list))])
    #model = keras.models.load_model('my_model.h5')
    pred = model.predict([user_1,place_list])

    ids= (-pred).flatten().argsort()
    place_id = place_list[ids][:5]

elif highest_cat == 'Desa dan Perkebunan':
    place_list = place_not_visited[place_not_visited['cluster'] == 'Desa dan Perkebunan']['Place_Id'].values
    user_1 = np.array([user for i in range(len(place_list))])
    #model = keras.models.load_model('my_model.h5')
    pred = model.predict([user_1,place_list])

    ids= (-pred).flatten().argsort()
    place_id = place_list[ids][:5]

elif highest_cat == 'Nilai Budaya dan Sejarah':
    place_list = place_not_visited[place_not_visited['cluster'] == 'Nilai Budaya dan Sejarah']['Place_Id'].values
    user_1 = np.array([user for i in range(len(place_list))])
    #model = keras.models.load_model('my_model.h5')
    pred = model.predict([user_1,place_list])

    ids= (-pred).flatten().argsort()
    place_id = place_list[ids][:5]

elif highest_cat == 'Wisata Bahari':
    place_list = place_not_visited[place_not_visited['cluster'] == 'Wisata Bahari']['Place_Id'].values
    user_1 = np.array([user for i in range(len(place_list))])
    #model = keras.models.load_model('my_model.h5')
    pred = model.predict([user_1,place_list])

    ids= (-pred).flatten().argsort()
    place_id = place_list[ids][:5]

elif highest_cat == 'Museum':
    place_list = place_not_visited[place_not_visited['cluster'] == 'Museum']['Place_Id'].values
    user_1 = np.array([user for i in range(len(place_list))])
    #model = keras.models.load_model('my_model.h5')
    pred = model.predict([user_1,place_list])

    ids= (-pred).flatten().argsort()
    place_id = place_list[ids][:5]

elif highest_cat == 'Tempat Ibadah dan Edukasi':
    place_list = place_not_visited[place_not_visited['cluster'] == 'Tempat Ibadah dan Edukasi']['Place_Id'].values
    user_1 = np.array([user for i in range(len(place_list))])
    #model = keras.models.load_model('my_model.h5')
    pred = model.predict([user_1,place_list])

    ids= (-pred).flatten().argsort()
    place_id = place_list[ids][:5]

elif highest_cat == 'Taman Hiburan':
    place_list = place_not_visited[place_not_visited['cluster'] == 'Taman Hiburan']['Place_Id'].values
    user_1 = np.array([user for i in range(len(place_list))])
    #model = keras.models.load_model('my_model.h5')
    pred = model.predict([user_1,place_list])

    ids= (-pred).flatten().argsort()
    place_id = place_list[ids][:5]

elif highest_cat == 'Wisata Alam':
    place_list = place_not_visited[place_not_visited['cluster'] == 'Wisata Alam']['Place_Id'].values
    user_1 = np.array([user for i in range(len(place_list))])
    #model = keras.models.load_model('my_model.h5')
    pred = model.predict([user_1,place_list])

    ids= (-pred).flatten().argsort()
    place_id = place_list[ids][:5]


else:
    st.write('in construction')

st.subheader('Tempat yang pernah user kunjungi')
st.write(place_visited)

st.subheader('Rekomendasi tujuan wisata')
place_df = pd.DataFrame({'Reccomended_Id': place_id})
reccomended = tourism[tourism['Place_Id'].isin(place_df.Reccomended_Id)][['Place_Id','Place_Name','cluster', 'Price', 'City']]
st.write(reccomended)