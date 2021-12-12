from matplotlib.colors import hexColorPattern
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt
from streamlit.elements.dataframe_selector import _use_arrow   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection
import graphviz
from sklearn.tree import export_graphviz

#import funciones as f
import streamlit as st

def app(visualizacion):
    st.header("Arboles de decisión")

    variablesClasificacion = []
    variablesPronostico = []

    if visualizacion == 'Modo Avanzado':
        st.image('https://images.pexels.com/photos/1632790/pexels-photo-1632790.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940', use_column_width=True)
        archivo = st.file_uploader("Importar archivo CSV", type=["csv"])
        if archivo is not None: nombreArchivo = archivo.name 
    else:
        st.image('https://images.pexels.com/photos/7088521/pexels-photo-7088521.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940',use_column_width=True)
        st.caption('Se trabaja con un conjunto de datos para pacientes con diagnósticos de tumor de mama.')
        nombreArchivo = 'WDBCOriginal.csv'
        variablesClasificacion = ['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension']
        variablesPronostico = ['Texture', 'Perimeter', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension']


    if visualizacion == 'Datos Precargados' or archivo is not None:
        """
        tipo_archivo = {"filename":archivo.name, "filetype":archivo.type,
                        "filesize":archivo.size}
        """

        Datos = pd.read_csv(nombreArchivo)
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

        st.subheader('Pronóstico')
        #Division de los  datos
        with st.expander('Pronostico', expanded=True if visualizacion=='Modo Avanzado' else False):
            DatosA = None
            with st.form('Selección Pronostico'):
                col1, col2 = st.columns([3, 1])
                VPredictorasP = col1.multiselect('Seleccionar Variables Predictoras', header, 
                                                default=variablesPronostico if visualizacion=='Datos Precargados' else None)
                VPronosticoP = col2.selectbox('Variable a Pronosticar', header, index=5)
                listoP = col1.form_submit_button('Enviar')

            if listoP or (len(VPredictorasP)!=0 and VPronosticoP is not None):
                X = np.array(Datos[VPredictorasP])
                Y = np.array(Datos[VPronosticoP])

                if visualizacion=='Modo Avanzado':
                    with st.form('Selección split P'):
                        st.info('Modificar Parámetros de entrada')
                        col1, col2 = st.columns(2)
                        grafica = col1.checkbox('Mostrar árbol')
                        test = col1.number_input(label='test_size', min_value=0.0, max_value=1.0, value=0.2, step=0.0025, format='%f')
                        semilla = col1.number_input(label='semilla', min_value=0, value=1234, step=1, format='%d')
                        #col2.write('0 = Default (None)')
                        profundidad = col2.number_input(label='Profundidad máxima', min_value=0, value=0, step=1, format='%d')
                        min_split = col2.number_input(label='Mínimo split', min_value=2, value=2, step=1, format='%d')
                        min_leaf = col2.number_input(label='Mínimo por hoja', min_value=1, value=1, step=1, format='%d')
                        listoA = col1.form_submit_button('Cambiar')

                    if listoA or (test!=0 and semilla is not None):
                        if profundidad == 0: profundidad = None

                        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                        test_size = test, 
                                                                        random_state = semilla, 
                                                                        shuffle = True)
                        #DecisionTreeRegressor()
                        PronosticoAD = DecisionTreeRegressor(max_depth=profundidad, min_samples_split=min_split, min_samples_leaf=min_leaf)
                        PronosticoAD.fit(X_train, Y_train)

                        Y_Pronostico = PronosticoAD.predict(X_test)
                        ValoresP = pd.DataFrame(Y_test, Y_Pronostico)

                        exactitud = r2_score(Y_test, Y_Pronostico)
                        exactitud = str(exactitud*100)[:str(exactitud*100).index('.')+4]
                        st.success('El porcentaje de exactitud es: '+exactitud+'%')

                        col1, col2, col3 = st.columns([2, 1, 2])
                        with col1:
                            st.write('Datos Reales y Pronosticados',ValoresP)
                        with col2:
                            #st.write('Importancia variables: \n', PronosticoAD.feature_importances_)
                            st.write('Parámetros del Modelo')
                            st.text('MAE: %.4f' % mean_absolute_error(Y_test, Y_Pronostico))
                            st.text("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
                            st.text("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
                            st.text('Score: %.4f' % r2_score(Y_test, Y_Pronostico))
                        with col3:
                            Importancia = pd.DataFrame({'Variable':VPredictorasP,
                                                    'Importancia':PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
                            st.dataframe(Importancia)
                        
                        fig, ax = plt.subplots(figsize=(14,8))
                        ax.plot(Y_test, color = 'green', marker='o', label='Y_test')
                        ax.plot(Y_Pronostico, color='red', marker='o', label='Y_Pronostico')
                        ax.set_title('Datos '+nombreArchivo[:-4])
                        ax.legend()
                        st.pyplot(fig)

                        if grafica:
                            #Elementos = export_graphviz(PronosticoAD, feature_names=VPredictorasP)
                            #st.graphviz_chart(Elementos, use_container_width=True)
                            fig, ax = plt.subplots(figsize=(14,8))
                            plot_tree(PronosticoAD, feature_names = VPredictorasP)
                            st.pyplot(fig)
                      
                else:
                    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                        test_size = 0.2, 
                                                                        random_state = 1234, 
                                                                        shuffle = True)
                    PronosticoAD = DecisionTreeRegressor(max_depth=8, min_samples_split=2, min_samples_leaf=8)
                    PronosticoAD.fit(X_train, Y_train)

                    Y_Pronostico = PronosticoAD.predict(X_test)
                    #Valores = pd.DataFrame(Y_test, Y_Pronostico)
                    
                    #Elementos = export_graphviz(PronosticoAD, feature_names=VPredictoras)
                    #st.graphviz_chart(Elementos, use_container_width=True)

                pronosticoP = {}
                with st.form('P'):
                    count = 0
                    col1, col2, col3 = st.columns(3)
                    for x in VPredictorasP:
                        auxP = []
                        if count%3 == 1:
                            auxP.append(col2.number_input(label=x, min_value=0.0, format='%f'))
                            pronosticoP[x] = auxP
                        elif count%3 == 2:
                            auxP.append(col3.number_input(label=x, min_value=0.0, format='%f'))
                            pronosticoP[x] = auxP
                        else:
                            auxP.append(col1.number_input(label=x, min_value=0.0, format='%f'))
                            pronosticoP[x] = auxP
                        count+=1
                    listoP = col1.form_submit_button('Enviar')

                if listoP:
                    #st.write(pronosticoP)
                    casoP = pd.DataFrame(pronosticoP)
                    resultadoP = PronosticoAD.predict(casoP)
                    #st.write(resultado)
                    st.success('El resultado del pronóstico fue un '+str(VPronosticoP).casefold()+' de: '+str(resultadoP[0])[:str(resultadoP[0]).index('.')+3])

        st.subheader('Clasificación')
        #Clasificacion
        with st.expander('Clasificacion', expanded=True if visualizacion=='Modo Avanzado' else False):
            with st.form('Selección Clasificacion'):
                col1, col2 = st.columns([3, 1])
                VPredictorasC = col1.multiselect('Seleccionar Variables Predictoras', header,
                                                 default=variablesClasificacion if visualizacion=='Datos Precargados' else None)
                VPronosticoC = col2.selectbox('Variable a Pronosticar', header, index=1)
                listoC = col1.form_submit_button('Enviar')

            if listoC or (len(VPredictorasC)!=0 and VPronosticoC is not None):
                XC = np.array(Datos[VPredictorasC])
                YC = np.array(Datos[VPronosticoC])
                
                if visualizacion=='Modo Avanzado':
                    with st.form('Selección split C'):
                        st.info('Modificar Parámetros de entrada')
                        col1, col2 = st.columns(2)
                        graficaC = col1.checkbox('Mostrar árbol')
                        testC = col1.number_input(label='test_size', min_value=0.0, max_value=1.0, value=0.2, step=0.0025, format='%f')
                        semillaC = col1.number_input(label='semilla', min_value=0, value=1234, step=1, format='%d')
                        #col2.write('0 = Default (None)')
                        profundidadC = col2.number_input(label='Profundidad máxima', min_value=0, value=0, step=1, format='%d')
                        min_splitC = col2.number_input(label='Mínimo split', min_value=2, value=2, step=1, format='%d')
                        min_leafC = col2.number_input(label='Mínimo por hoja', min_value=1, value=1, step=1, format='%d')
                        listoAC = col1.form_submit_button('Cambiar')

                    if listoAC or (testC!=0 and semillaC is not None):
                        if profundidadC == 0: profundidadC = None
                        XC_train, XC_validation, YC_train, YC_validation = model_selection.train_test_split(XC, YC, 
                                                                        test_size = testC, 
                                                                        random_state = semillaC, 
                                                                        shuffle = True)
                        ClasificacionAD = DecisionTreeClassifier(max_depth=profundidadC, 
                                                                min_samples_split=min_splitC, 
                                                                min_samples_leaf=min_leafC)
                        ClasificacionAD.fit(XC_train, YC_train)

                        YC_Clasificacion = ClasificacionAD.predict(XC_validation)
                        ValoresC = pd.DataFrame(YC_validation, YC_Clasificacion)

                        exactitudC = ClasificacionAD.score(XC_validation, YC_validation)
                        exactitudC = str(exactitudC*100)[:str(exactitudC*100).index('.')+4]
                        st.success('El porcentaje de exactitud es: '+exactitudC+'%')
                        col1, col2 = st.columns([1,3])
                        with col1:
                            st.write('Datos Reales y Pronosticados',ValoresC)
                        with col2:
                            YC_Clasificacion = ClasificacionAD.predict(XC_validation)
                            Matriz_Clasificacion = pd.crosstab(YC_validation.ravel(), 
                                                            YC_Clasificacion, 
                                                            rownames=['Reales'], 
                                                            colnames=['Clasificación']) 
                            Matriz_Clasificacion
                            st.write('Matriz de Clasificación')
                            st.dataframe(Matriz_Clasificacion)
                            st.write('Reporte de Clasificación')
                            st.code(classification_report(YC_validation, YC_Clasificacion))
                        
                        if graficaC:
                            fig, ax = plt.subplots(figsize=(14,8))
                            plot_tree(ClasificacionAD, feature_names = VPredictorasC)
                            st.pyplot(fig)

                else:
                    XC_train, XC_validation, YC_train, YC_validation = model_selection.train_test_split(XC, YC, 
                                                                        test_size = 0.2, 
                                                                        random_state = 1234, 
                                                                        shuffle = True)
                    ClasificacionAD = DecisionTreeClassifier(max_depth=8, min_samples_split=2, min_samples_leaf=8)
                    ClasificacionAD.fit(XC_train, YC_train)

                    YC_Clasificacion = ClasificacionAD.predict(XC_validation)
                    #ValoresC = pd.DataFrame(YC_validation, YC_Clasificacion)

                pronosticoC = {}
                with st.form('Clasificacion'):
                    count = 0
                    col1, col2, col3 = st.columns(3)
                    for x in VPredictorasC:
                        auxC = []
                        if count%3 == 1:
                            auxC.append(col2.number_input(label=x, min_value=0.0, format='%f'))
                            pronosticoC[x] = auxC
                        elif count%3 == 2:
                            auxC.append(col3.number_input(label=x, min_value=0.0, format='%f'))
                            pronosticoC[x] = auxC
                        else:
                            auxC.append(col1.number_input(label=x, min_value=0.0, format='%f'))
                            pronosticoC[x] = auxC
                        count+=1
                    listoC2 = col1.form_submit_button('Enviar')

                if listoC2:
                    #st.write(pronostico)
                    casoC = pd.DataFrame(pronosticoC)
                    resultadoC = ClasificacionAD.predict(casoC)
                    #st.write(resultadoC)
                    st.info('El resultado de la clasificación fue un '+str(VPronosticoC).casefold()+' de tipo: '+str(resultadoC[0]))

        


