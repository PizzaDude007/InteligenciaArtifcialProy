import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
import streamlit as st

def app():
   st.title("Clustering") 

   archivo = st.file_uploader("Importar archivo CSV", type=["csv"])
   
   if archivo is not None:
      """
      tipo_archivo = {"filename":archivo.name, "filetype":archivo.type,
                     "filesize":archivo.size}
      """

      Datos = pd.read_csv(archivo.name)
      corrDatos = Datos.corr(method='pearson')
      header = list(Datos.columns)

      with st.expander("Datos"):
         st.write(Datos)
      
      with st.expander('Matriz de dispersion'):
         #color = None
         if st.checkbox('Colorear'):
            color = st.selectbox('Seleccione una variable para colorear grafica', header)
            fig = sns.pairplot(Datos, hue=color)
            a = st.empty()
            a.info('Cargando Gráfica')
            a.pyplot(fig)
         else:
            fig = sns.pairplot(Datos)
            a = st.empty()
            a.info('Cargando Gráfica')
            a.pyplot(fig)
            

      with st.expander('Mapa de Calor'):
         a = st.empty()
         fig, ax = plt.subplots(figsize=(14,10))
         MatrizInf = np.triu(corrDatos)
         sns.heatmap(corrDatos, cmap='RdBu_r', annot=True, mask=MatrizInf)
         a.info('Cargando Mapa de Calor')
         a.pyplot(fig)

      with st.expander("Clustering Jerarquico"):
         DatosJ=None
         with st.form('Seleccionar Variables J'):
            col1, col2 = st.columns([3, 1])
            seleccionJ = col1.multiselect('Seleccion Múltiple de Variables', header)
            mostrar = col2.checkbox('Mostrar árbol')
            num_clusterJ=col2.number_input('Num Clusters',min_value=0,value=7,step=1)
            metrica = col1.selectbox('Seleccionar métrica de distancia', 
                                    ['euclidean','chebyshev','cityblock'])  
            listo = col2.form_submit_button('Enviar')
         
         if listo or (len(seleccionJ)!=0 and metrica!=None):
            a = st.empty()
            a.info('Cargando Resultado')
            seleccionDatos = np.array(Datos[seleccionJ])

            #with st.container():
            #st.write('Matriz Ajustada')
            #st.write(seleccionDatos)

            estandarizarJ = StandardScaler()
            MJEstandarizada = estandarizarJ.fit_transform(seleccionDatos)        
            
            if mostrar:
               b = st.empty()
               b.info('Cargando Árbol')
               fig, ax = plt.subplots(figsize=(14,10))
               ax.set_xlabel(archivo.name[:-4])
               ax.set_ylabel('Distancia')
               shc.dendrogram(shc.linkage(MJEstandarizada, method='complete', metric=metrica))
               b.pyplot(fig)
         
            MJerarquico = AgglomerativeClustering(n_clusters=num_clusterJ, linkage='complete', affinity=metrica)
            MJerarquico.fit_predict(MJEstandarizada)
            DatosJ = Datos
            DatosJ['cluster'] = MJerarquico.labels_
            a.empty()
            st.write('Matriz con Clústeres')
            st.write(DatosJ)

         if DatosJ is not None:
            valorCluster = st.slider('Mostrar Clúster', min_value=0, max_value=(num_clusterJ-1), step=1)
            st.write(DatosJ[DatosJ.cluster == valorCluster])

      with st.expander("Clustering Particional"):
         DatosP=None
         with st.form('Seleccionar Variables P'):
            col1, col2 = st.columns(2)
            seleccionP = st.multiselect('Seleccion Múltiple de Variables', header)
            usar_codo = col2.checkbox('Usar método de codo',value=True)
            codo1 = col1.number_input('Rango Codo ini',min_value=0,value=2,step=1)
            codo2 = col1.number_input('Rango Codo fin',min_value=0,value=12,step=1)
            num_clusterP=col2.number_input('Num Clusters (si no se usa codo)',min_value=0,step=1)
            listo = col2.form_submit_button('Enviar')
         
         if listo or (len(seleccionP)!=0 and codo1>=0 and codo2>=0):
            a = st.empty()
            a.info('Cargando Resultado')
            seleccionDatos = np.array(Datos[seleccionP])
            estandarizarP = StandardScaler()
            MPEstandarizada = estandarizarP.fit_transform(seleccionDatos)

            if usar_codo:
               SSE = []
               for i in range(codo1,codo2):
                  km = KMeans(n_clusters=i, random_state=0)
                  km.fit(MPEstandarizada)
                  SSE.append(km.inertia_)

               k1 = KneeLocator(range(codo1,codo2), SSE, curve='convex', direction='decreasing')
               a.empty()
               fig, ax = plt.subplots()
               plt.style.use('ggplot')
               ax.plot(range(2, 12), SSE, marker='o')
               ax.set_xlabel('Cantidad de clusters *k*')
               ax.set_ylabel('SSE')
               ax.axvline(x=k1.elbow, color='black', linestyle='--')
               k1.plot_knee()
               st.pyplot(fig=fig, clear_figure=None)
               #st.write(plot_figure(codo1, codo2, k1, all_knees=SSE))

               MParticional = KMeans(n_clusters=k1.elbow, random_state=0).fit(MPEstandarizada)  
               num_clusterP=k1.elbow
            else:
               a.empty()
               MParticional = KMeans(n_clusters=num_clusterP, random_state=0).fit(MPEstandarizada)

            MParticional.predict(MPEstandarizada)
            DatosP=Datos
            DatosP['cluster'] = MParticional.labels_
            st.write(DatosP)

         if DatosP is not None:
            if usar_codo:
               valorCluster = st.slider('Mostrar Clúster', min_value=0, max_value=int(k1.elbow-1), step=1)
            else:
               valorCluster = st.slider('Mostrar Clúster', min_value=0, max_value=int(num_clusterP-1), step=1)
            st.write(DatosP[DatosP.cluster == valorCluster])
            
