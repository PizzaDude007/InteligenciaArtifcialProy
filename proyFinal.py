import streamlit as st
import pandas as pd
import numpy as np

#imports de otras paginas
import apriori
import metricas
import clustering
import clasificacion
import arboles

PAGES = {"Algoritmo Apriori": apriori, "Metricas de distancia": metricas, 
        "Módulo Clustering":clustering, "Clasificación (R.Logística)":clasificacion,
        "Módulo Árboles":arboles}

st.title('Proyecto Inteligencia Artificial')
st.sidebar.title('Menu chido')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
