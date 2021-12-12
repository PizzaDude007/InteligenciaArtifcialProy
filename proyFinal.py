import streamlit as st
import pandas as pd
import numpy as np

#imports de otras paginas
import apriori
import metricas
import clustering
import clasificacion
import arboles
import footer as f

PAGES = {"Algoritmo Apriori": apriori, "Metricas de distancia": metricas, 
        "Módulo Clustering":clustering, "Clasificación (R.Logística)":clasificacion,
        "Módulo Árboles":arboles}

st.set_page_config(page_title='Proyecto IA', page_icon='🤓',layout='centered',initial_sidebar_state='expanded')
#st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Proyecto Inteligencia Artificial')
#st.caption('Presiona la flecha de la izquierda para acceder al menú de Navegación')
st.sidebar.title('Navegación')
empty = st.sidebar.empty()
navbarra = empty.selectbox("Selección de Algoritmo", list(PAGES.keys()))

visualizacion = st.sidebar.selectbox('Modo de visualización', ['Datos Precargados','Modo Avanzado'])
selection = navbarra

# col1, col2, col3, col4, col5 = st.columns(5)
# with col1:
#     if st.button('Algoritmo Apriori'):
#         navbarra = empty.selectbox("Go to", list(PAGES.keys()), index=0)
#         selection = "Algoritmo Apriori"
# with col2:
#     if st.button('Metricas de distancia'):
#         navbarra = empty.selectbox("Go to", list(PAGES.keys()), index=1)
#         selection = "Metricas de distancia"
# with col3:
#     if st.button("Módulo Clustering"):
#         navbarra = empty.selectbox("Go to", list(PAGES.keys()), index=2)
#         selection = "Módulo Clustering"
# with col4:
#     if st.button("Clasificación (R.Logística)"):
#         navbarra = empty.selectbox("Go to", list(PAGES.keys()), index=3)
#         selection = "Clasificación (R.Logística)"
# with col5:
#     if st.button("Módulo Árboles"):
#         navbarra = empty.selectbox("Go to", list(PAGES.keys()), index=4)
#         selection = "Módulo Árboles"

#if selection is None:
#    selection = navbarra

f.footer()

page = PAGES[selection]
#st.balloons()
page.app(visualizacion)
