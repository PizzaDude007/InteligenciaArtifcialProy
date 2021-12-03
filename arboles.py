import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection
import streamlit as st

def app():
    st.title("Arboles de decisión")

    archivo = st.file_uploader("Importar archivo CSV", type=["csv"])

    if archivo is not None:
        """
        tipo_archivo = {"filename":archivo.name, "filetype":archivo.type,
                        "filesize":archivo.size}
        """

        Datos = pd.read_csv(archivo.name)
        header = list(Datos.columns)
        with st.expander("Datos"):
            st.write(Datos)

    
    #Division de los  datos
        with st.expander('Pronostico'):
            DatosA = None
            with st.form('Clasificaion'):
                col1, col2 = st.columns([3, 1])
                VPredictoras = col1.multiselect('Seleccionar Variables Predictoras', header)
                VPronostico = col2.selectbox('Variable a Pronosticar', header)
                listo = col2.form_submit_button('Enviar')

            if listo:
                X = np.array(Datos[VPredictoras])
                Y = np.array(Datos[VPronostico])

                X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 1234, 
                                                                    shuffle = True)
                PronosticoAD = DecisionTreeRegressor()
                PronosticoAD