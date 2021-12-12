import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#import funciones as f
import streamlit as st

def app(visualizacion):
    st.header("Clasificación Regresión Logística")
    variablesPredefinidas = []

    if visualizacion == 'Modo Avanzado':
        st.image('https://images.pexels.com/photos/7512859/pexels-photo-7512859.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940', use_column_width=True)
        archivo = st.file_uploader("Importar archivo CSV", type=["csv"])
        if archivo is not None: nombreArchivo = archivo.name 
    else:
        st.image('https://images.pexels.com/photos/356040/pexels-photo-356040.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940',use_column_width=True)
        st.caption('Se utiliza como ejemplo el diagnóstico de un tumor de mama, para conocer si es maligno o benigno.')
        nombreArchivo = 'WDBCOriginal.csv'
        variablesPredefinidas = ['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension']

    if visualizacion == 'Datos Precargados' or archivo is not None:
        """
        tipo_archivo = {"filename":archivo.name, "filetype":archivo.type,
                        "filesize":archivo.size}
        """

        Datos = pd.read_csv(nombreArchivo)
        header = list(Datos.columns)
        #corrDatos = Datos.corr(method='pearson')
        with st.expander("Datos"):
            st.write(Datos)

        with st.expander('Matriz de dispersion'):
            #color = None
            if st.checkbox('Colorear',value=True if visualizacion=='Datos Precargados' else False):
                color = st.selectbox('Seleccione una variable para colorear grafica', header,index=1)
                fig = sns.pairplot(Datos, hue=color)
                st.pyplot(fig)
            else:
                fig = sns.pairplot(Datos)
                st.pyplot(fig)
        
        with st.expander('Mapa de Calor'):
            fig, ax = plt.subplots(figsize=(14,10))
            MatrizInf = np.triu(Datos.corr())
            sns.heatmap(Datos.corr(), cmap='RdBu_r', annot=True, mask=MatrizInf)
            st.pyplot(fig)

        #Division de los  datos
        with st.expander('Aplicación Algoritmo', expanded = False if visualizacion=='Datos Precargados' else True):
            with st.form('Selección Variables'):
                col1, col2 = st.columns([3, 1])
                VPredictoras = col1.multiselect('Seleccionar Variables Predictoras', header,default=variablesPredefinidas)
                VPronostico = col2.selectbox('Variable a Pronosticar', header,index=1)
                listo = col2.form_submit_button('Enviar')

            if listo or (len(VPredictoras)!= 0 and VPronostico is not None):
                X = np.array(Datos[VPredictoras])
                #pd.Dataframe(X)
                Y = np.array(Datos[VPronostico])
                #pd.Dataframe(Y)

                if visualizacion == 'Modo Avanzado':
                    with st.form('Selección split'):
                        col1, col2, col3 = st.columns(3)
                        col1.write('Modificar Parámetros de entrada')
                        test = col2.number_input(label='test_size', min_value=0.0, max_value=1.0, value=0.2, step=0.0025, format='%f')
                        semilla = col3.number_input(label='semilla', min_value=0, value=1234, step=1, format='%d')
                        listoA = col1.form_submit_button('Cambiar')

                    if listoA or (test!=0 and semilla is not None):
                        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                        test_size = test, 
                                                                        random_state = semilla, 
                                                                        shuffle = True)
                        Clasificacion = linear_model.LogisticRegression()
                        Clasificacion.fit(X_train, Y_train)

                        exactitud = Clasificacion.score(X_validation, Y_validation)
                        exactitud = str(exactitud*100)[:str(exactitud*100).index('.')+4]
                        st.success('El porcentaje de exactitud es: '+exactitud+'%')
                        col1, col2, col3 = st.columns([3, 2, 4])
                        with col1:
                            st.write('Probabilidad')
                            Probabilidad = Clasificacion.predict_proba(X_validation)
                            st.write(Probabilidad)
                        with col2:
                            st.write('Predicciones')
                            Predicciones = Clasificacion.predict(X_validation)
                            st.dataframe(Predicciones)
                        with col3:
                            Y_Clasificacion = Clasificacion.predict(X_validation)
                            Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                                            Y_Clasificacion, 
                                                            rownames=['Reales'], 
                                                            colnames=['Clasificación']) 
                            Matriz_Clasificacion
                            st.write('Matriz de Clasificación')
                            st.dataframe(Matriz_Clasificacion)
                            st.write('Reporte de Clasificación')
                            st.code(classification_report(Y_validation, Y_Clasificacion))
                else:
                    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                test_size = 0.2, 
                                                                random_state = 1234, 
                                                                shuffle = True)
                    #pd.Dataframe(X_train)
                    Clasificacion = linear_model.LogisticRegression()
                    Clasificacion.fit(X_train, Y_train)
                        
                
        with st.expander('Realizar Clasificación',expanded=True if visualizacion=='Datos Precargados' else False):
            predicciones = {}
            with st.form('Clasificación'):
                count = 0
                col1, col2, col3 = st.columns(3)
                for x in VPredictoras:
                    aux = []
                    if count%3 == 1:
                        aux.append(col2.number_input(label=x, min_value=0.0, format='%f'))
                        predicciones[x] = aux
                    elif count%3 == 2:
                        aux.append(col3.number_input(label=x, min_value=0.0, format='%f'))
                        predicciones[x] = aux
                    else:
                        aux.append(col1.number_input(label=x, min_value=0.0, format='%f'))
                        predicciones[x] = aux
                    count+=1
                listo = col1.form_submit_button('Enviar')

            if listo:
                #st.write(predicciones)
                caso = pd.DataFrame(predicciones)
                resultado = Clasificacion.predict(caso)
                if visualizacion == 'Modo Avanzado':
                    st.info('El resultado de la predicción fue: '+str(resultado[0][0]))
                    #st.write(resultado)
                else:
                    if resultado[0][0] == 'B':
                        st.info('El resultado de la predicción fue Benigno')
                        st.balloons()
                    elif resultado[0][0] == 'M':
                        st.info('El resultado de la predicción fue Maligno')


            
            

