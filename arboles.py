import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt
from streamlit.elements.dataframe_selector import _use_arrow   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection
import graphviz
from sklearn.tree import export_graphviz
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

        with st.expander('Mapa de Calor'):
            a = st.empty()
            fig, ax = plt.subplots(figsize=(14,10))
            MatrizInf = np.triu(Datos.corr())
            sns.heatmap(Datos.corr(), cmap='RdBu_r', annot=True, mask=MatrizInf)
            a.info('Cargando Mapa de Calor')
            a.pyplot(fig)

    
    #Division de los  datos
        with st.expander('Pronostico'):
            DatosA = None
            with st.form('Selección Pronostico'):
                col1, col2 = st.columns([3, 1])
                VPredictoras = col1.multiselect('Seleccionar Variables Predictoras', header)
                VPronostico = col2.selectbox('Variable a Pronosticar', header)
                listo = col1.form_submit_button('Enviar')

            if listo or (len(VPredictoras)!=0 and VPronostico is not None):
                X = np.array(Datos[VPredictoras])
                Y = np.array(Datos[VPronostico])

                X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 1234, 
                                                                    shuffle = True)
                PronosticoAD = DecisionTreeRegressor(max_depth=8, min_samples_split=2, min_samples_leaf=8)
                PronosticoAD.fit(X_train, Y_train)

                Y_Pronostico = PronosticoAD.predict(X_test)
                Valores = pd.DataFrame(Y_test, Y_Pronostico)

                Elementos = export_graphviz(PronosticoAD, feature_names=VPredictoras)
                st.graphviz_chart(Elementos, use_container_width=True)

                pronostico = {}
                with st.form('Pronostico'):
                    count = 0
                    col1, col2, col3 = st.columns(3)
                    for x in VPredictoras:
                        aux = []
                        if count%3 == 1:
                            aux.append(col2.number_input(label=x, min_value=0.0, format='%f'))
                            pronostico[x] = aux
                        elif count%3 == 2:
                            aux.append(col3.number_input(label=x, min_value=0.0, format='%f'))
                            pronostico[x] = aux
                        else:
                            aux.append(col1.number_input(label=x, min_value=0.0, format='%f'))
                            pronostico[x] = aux
                        count+=1
                    listo = col1.form_submit_button('Enviar')

                if listo:
                    #st.write(pronostico)
                    caso = pd.DataFrame(pronostico)
                    resultado = PronosticoAD.predict(caso)
                    st.write('El resultado de la predicción fue: ', resultado)


        #Clasificacion
        with st.expander('Clasificacion'):
            DatosA = None
            with st.form('Selección Clasificacion'):
                col1, col2 = st.columns([3, 1])
                VPredictoras = col1.multiselect('Seleccionar Variables Predictoras', header)
                VPronostico = col2.selectbox('Variable a Pronosticar', header)
                listo = col1.form_submit_button('Enviar')

            if listo or (len(VPredictoras)!=0 and VPronostico is not None):
                X = np.array(Datos[VPredictoras])
                Y = np.array(Datos[VPronostico])

                X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 1234, 
                                                                    shuffle = True)
                PronosticoAD = DecisionTreeRegressor(max_depth=8, min_samples_split=2, min_samples_leaf=8)
                PronosticoAD.fit(X_train, Y_train)

                Y_Pronostico = PronosticoAD.predict(X_test)
                Valores = pd.DataFrame(Y_test, Y_Pronostico)

                Elementos = export_graphviz(PronosticoAD, feature_names=VPredictoras)
                st.graphviz_chart(Elementos, use_container_width=True)

                pronostico = {}
                with st.form('Pronostico'):
                    count = 0
                    col1, col2, col3 = st.columns(3)
                    for x in VPredictoras:
                        aux = []
                        if count%3 == 1:
                            aux.append(col2.number_input(label=x, min_value=0.0, format='%f'))
                            pronostico[x] = aux
                        elif count%3 == 2:
                            aux.append(col3.number_input(label=x, min_value=0.0, format='%f'))
                            pronostico[x] = aux
                        else:
                            aux.append(col1.number_input(label=x, min_value=0.0, format='%f'))
                            pronostico[x] = aux
                        count+=1
                    listo = col1.form_submit_button('Enviar')

                if listo:
                    #st.write(pronostico)
                    caso = pd.DataFrame(pronostico)
                    resultado = PronosticoAD.predict(caso)
                    st.write('El resultado de la predicción fue: ', resultado)

        


