import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import streamlit as st

def app():
    st.title("Clasificación R.Logística")

    archivo = st.file_uploader("Importar archivo CSV", type=["csv"])

    if archivo is not None:
        """
        tipo_archivo = {"filename":archivo.name, "filetype":archivo.type,
                        "filesize":archivo.size}
        """

        Datos = pd.read_csv(archivo.name)
        header = list(Datos.columns)
        corrDatos = Datos.corr(method='pearson')
        with st.expander("Datos"):
            st.write(Datos)

        with st.expander('Matriz de dispersion'):
            #color = None
            if st.checkbox('Colorear'):
                color = st.selectbox('Seleccione una variable para colorear grafica', header)
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
        with st.expander('Aplicación Algoritmo'):
            with st.form('Selección Variables'):
                col1, col2 = st.columns([3, 1])
                VPredictoras = col1.multiselect('Seleccionar Variables Predictoras', header)
                VPronostico = col2.selectbox('Variable a Pronosticar', header)
                listo = col2.form_submit_button('Enviar')
                    

            if listo or (len(VPredictoras)!= 0 and VPronostico is not None):
                X = np.array(Datos[VPredictoras])
                #pd.Dataframe(X)
                Y = np.array(Datos[VPronostico])
                #pd.Dataframe(Y)

                X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 1234, 
                                                                    shuffle = True)
                #pd.Dataframe(X_train)
                Clasificacion = linear_model.LogisticRegression()
                Clasificacion.fit(X_train, Y_train)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write('Probabilidad')
                    Probabilidad = Clasificacion.predict_proba(X_validation)
                    st.write(Probabilidad)
                with col2:
                    st.write('Predicciones')
                    Predicciones = Clasificacion.predict(X_validation)
                    st.write(Predicciones)
                #Clasificacion.score(X_validation, Y_validation)
                
        with st.expander('Realizar Clasificación'):
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
                st.write('El resultado de la predicción fue: ', resultado)


        
        

