import streamlit as st
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from apyori import apriori


def app():
    st.title('Apriori')

    archivo = st.file_uploader("Importar archivo CSV", type=["csv"])
    #colg1, colg2 = st.columns([2, 3])

    if archivo is not None:
        """
        tipo_archivo = {"filename":archivo.name, "filetype":archivo.type,
                        "filesize":archivo.size}
        """

        Datos = pd.read_csv(archivo.name)
        with st.expander("Datos"):
            #st.write('Datos Peliculas')
            st.write(Datos)
        
        Datos = pd.read_csv(archivo.name, header=None)
        #st.write('Datos Peliculas del 1 al 10')
        #st.write(DatosMovies.head(10))

        Datos.shape

        Transacciones = Datos.values.reshape(-1).tolist()
        ListaM = pd.DataFrame(Transacciones)
        #st.write('Transacciones')
        #st.write(ListaM)

        ListaM['Frecuencia'] = 0
        ListaM

        #Se agrupa los elementos
        ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
        ListaM['Porcentaje'] = (ListaM['Frecuencia'] / ListaM['Frecuencia'].sum()) #Porcentaje
        ListaM = ListaM.rename(columns={0 : 'Item'})
        
        
        # Se genera un gráfico de barras
        fig, ax = plt.subplots(figsize=(16,20), dpi=300)
        ax.set_ylabel('Item')
        ax.set_xlabel('Frecuencia')
        ax.barh(ListaM['Item'], width=ListaM['Frecuencia'], color='blue')
        #ax.set_alpha(0)
        with st.expander("Gráfico"):
            #st.write("Grafico de ejemplo")
            a = st.empty()
            a.info('Cargando Gráfico')
            a.pyplot(fig, clear_figure=True)

        #Se crea una lista de listas a partir del dataframe y se remueven los 'NaN'
        #level=0 especifica desde el primer índice
        TransaccionesLista = Datos.stack().groupby(level=0).apply(list).tolist()
        TransaccionesLista 
        
        #Aplicacion del algoritmo 
        with st.expander("Reglas"):
            with st.form('Selección de componentes'):
                col1, col2, col3 = st.columns(3)
                with col1:
                    soporte = st.number_input("Soporte", min_value=0.000, max_value=1.000, value=0.0000, step=0.001, format='%f')
                with col2:
                    confianza = st.number_input("Confianza", min_value=0.0, max_value=1.0, value=0.0000, step=0.01, format='%f')
                with col3:
                    elevacion = st.number_input("Elevación", min_value=0.0, step=0.01, value=0.0000, format='%f')
                listo =st.form_submit_button('Enviar')

            if listo or (soporte != 0 and confianza != 0 and elevacion != 0):
                e = st.empty()
                e.info('Cargando Datos')

                Reglas = apriori(TransaccionesLista, min_support=soporte, min_confidence=confianza,
                                min_lift=elevacion)
                Resultados = list(Reglas)

                #st.write(pd.DataFrame(Reglas))
                
                col1, col2 = st.columns(2)
                for item in Resultados:
                    with e.container():
                        #El primer índice de la lista
                        Emparejar = item[0]
                        items = [x for x in Emparejar]
                        col1.write("Regla: ")
                        col2.write(item[0])

                        #El segundo índice de la lista
                        col1.write("Soporte: ")
                        col2.write(item[1])

                        #El tercer índice de la lista
                        col1.write("Confianza: ")
                        col2.write(item[2][0][2])
                        col1.write("Lift: ")
                        col2.write(item[2][0][3]) 
                        #st.write("=====================================") 