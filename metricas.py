import pandas as pd                       #Para la mainupación y análisis de datos
import numpy as np                        #Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt           #Para generar gráficas a partir de los datos
from scipy.spatial.distance import cdist  #Para el cálculo de distancias
from scipy.spatial import distance
import streamlit as st

def app(visualizacion):
    st.subheader("Métricas de Distancia")

    if visualizacion == 'Modo Avanzado':
        st.image('https://images.pexels.com/photos/590041/pexels-photo-590041.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940', width=700)
        archivo = st.file_uploader("Importar archivo CSV", type=["csv"])
        if archivo is not None: nombreArchivo = archivo.name 
    else:
        st.image('https://images.pexels.com/photos/162500/measurement-millimeter-centimeter-meter-162500.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940', width=700)
        st.caption('Se busca comparar mediante distintas métricas la similitud entre dos elementos de un conjunto de datos, '+
                    'en este caso se tiene personas que buscan realizar una hipoteca a una casa en un plazo de 30 años')
        nombreArchivo = 'Hipoteca.csv'

    if visualizacion == 'Datos Precargados' or archivo is not None:
        """
        tipo_archivo = {"filename":archivo.name, "filetype":archivo.type,
                        "filesize":archivo.size}
        """

        Datos = pd.read_csv(nombreArchivo)
        with st.expander("Datos"):
            st.write(Datos)
        
        #Datos = pd.read_csv(archivo.name, header=None)
        with st.expander("Matriz de Distancia"):
            st.info("Seleccionar método de distancia")

            col1, col2, col3  = st.columns([3,3,1])

            opcion = ''

            with col1:
                if st.button('Distancia Euclideana'):
                    opcion = 'euclidean'
            with col2:
                if st.button('Chebyshev'):
                    opcion = 'chebyshev'
            with col3:
                if st.button("Manhattan"):
                    opcion = 'cityblock'

            if opcion != '':
                dist = cdist(Datos, Datos, metric=opcion)
                matriz = pd.DataFrame(dist)
                st.write(matriz)
        
        with st.expander('Distancia entre dos objetos',expanded=True):
            with st.form(key='Seleccion de objetos'):
                objeto1 = st.number_input(label='Objeto 1',min_value=0, max_value=len(Datos), step=1)
                objeto2 = st.number_input(label='Objeto 2',min_value=0, max_value=len(Datos), step=1)
                listo = st.form_submit_button('Seleccionar')

            if listo:
                st.info('Datos Cargados')

            st.write("Seleccionar método de distancia")

            col1, col2, col3 = st.columns(3)

            opcion = ''
            distancia2 = None

            with col1:
                if st.button('Metrica Euclideana'):
                    distancia2 = distance.euclidean(Datos.iloc[objeto1],Datos.iloc[objeto2])
            with col2:
                if st.button('Metrica Chebyshev'):
                    distancia2 = distance.chebyshev(Datos.iloc[objeto1],Datos.iloc[objeto2])
            with col3:
                if st.button("Metrica Manhattan"):
                    distancia2 = distance.cityblock(Datos.iloc[objeto1],Datos.iloc[objeto2])

            if distancia2 is not None:
                st.success('El resultado es: '+str(distancia2))

