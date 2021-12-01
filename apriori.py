import streamlit as st
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from apyori import apriori


def app():
    st.title('Apriori')

    archivo = st.file_uploader("Importar archivo CSV", type=["csv"])

    if archivo is not None:
        """
        tipo_archivo = {"filename":archivo.name, "filetype":archivo.type,
                        "filesize":archivo.size}
        """

        Datos = pd.read_csv(archivo.name)
        st.write('Datos Peliculas')
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
        st.write("Grafico de ejemplo")
        st.pyplot(fig)

        #Se crea una lista de listas a partir del dataframe y se remueven los 'NaN'
        #level=0 especifica desde el primer índice
        TransaccionesLista = Datos.stack().groupby(level=0).apply(list).tolist()
        TransaccionesLista 
        
        #Aplicacion del algoritmo
        soporte = st.number_input("Soporte", min_value=0.000, max_value=1.000, value=0.0, step=0.001)
        confianza = st.number_input("Confianza", min_value=0.0, max_value=1.0, value=0.0, step=0.001)
        elevacion = st.number_input("Elevación", min_value=0.0, value=0.0, step=0.001)

        if (soporte != 0 and confianza != 0 and elevacion != 0):

            Reglas = apriori(TransaccionesLista, min_support=soporte, min_confidence=confianza,
                            min_lift=elevacion)
            Resultados = list(Reglas)


            for item in Resultados:
                #El primer índice de la lista
                Emparejar = item[0]
                items = [x for x in Emparejar]
                st.write("Regla: " + str(item[0]))

                #El segundo índice de la lista
                st.write("Soporte: " + str(item[1]))

                #El tercer índice de la lista
                st.write("Confianza: " + str(item[2][0][2]))
                st.write("Lift: " + str(item[2][0][3])) 
                st.write("=====================================") 