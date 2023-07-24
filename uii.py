import streamlit as st
import pandas as pd
import tensorflow as tf
#from tensorflow.keras import models, layer
#tf.keras.models
from streamlit_lottie import st_lottie
import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import roc_curve, roc_auc_score
from PIL import Image
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
import plotly.express as px
######################################################  FUNCIONES ######################################################
# Personalizar la apariencia visual de la interfaz
# Vincular archivo CSS
def plotly_EDA(df):
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    cat_features = ["Condition", "Vehicle_model", "Fuel_type", "Drive", "Transmission", "Type", "First_owner"]
    st.title(f"Distribución Variables Categóricas :green[{predictions}]")
    # Create the subplot grid based on the number of categorical features
    rows = 2  # Two rows
    cols = 4  # We want four columns to spread the bar charts horizontally

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=cat_features)

    for i, columna in enumerate(cat_features):
        # Obtener el conteo de cada categoría y ordenar por frecuencia descendente
        model_counts = df[columna].value_counts().sort_values(ascending=False)

        # Calculate the row and column for the current subplot
        row = i // cols + 1
        col = i % cols + 1

        # Create the bar chart for the current categorical feature
        fig.add_trace(go.Bar(x=model_counts.index, y=model_counts.values, name=f'Tipo de {columna}'), row=row, col=col)

        # Configurar el título y las etiquetas del eje x
        #fig.update_xaxes(tickangle=45, title_text=f'Tipo de {columna}', row=row, col=col)
        fig.update_yaxes(title_text='Frecuencia', row=row, col=col)

    # Update layout of the subplots
    fig.update_layout(showlegend=False, title='', 
                    height=350*rows, width=900, template='plotly_dark')

    # Mostrar la figura
    st.plotly_chart(fig)

def relacion_1_1(y_test,y_pred):
    y_true = y_test
    y_pred = y_pred
    # Calcular los errores
    errors = y_true - y_pred

    # Graficar la relación 1:1
    plt.scatter(y_true, y_pred, color='green')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='r', linestyle='--')

    # Configurar etiquetas y título del gráfico
    plt.xlabel('Etiquetas Reales')
    plt.ylabel('Predicciones')
    plt.title('Relación 1:1')

    # Mostrar el gráfico
    plt.grid(True)
    plt.savefig('relacion1a1.png', dpi=300)
    st.image('relacion1a1.png', caption='Relacción 1:1')

def plot_features_importance(model,cat_features,num_features,marca):
    # Obtener el modelo XGBoost Regressor del pipeline
    model1 = model.named_steps['regressor']
    plt.clf()
    # Obtener las características de importancia
    importance = model1.feature_importances_

    # Obtener los nombres de las características desde el transformador OneHotEncoder
    onehot_encoder = model.named_steps['preprocessor'].transformers_[1][1].named_steps['encoder']
    feature_names = onehot_encoder.get_feature_names_out(input_features=cat_features)
    test = np.concatenate((num_features,feature_names),axis = None)
    # Ordenar las características por importancia descendente
    sorted_indices = np.argsort(importance)[::-1]
    sorted_importance = importance[sorted_indices]
    sorted_feature_names = test[sorted_indices]

    subset = pd.DataFrame([test,importance])
    subset = subset.T.sort_values(by = 1,ascending = False)
    subset.rename(columns = {0:"Features",1:"Importance"},inplace = True)
    subset = subset.head(15)
    plt.title(f"FEAUTRES IMPORTANCE {marca}")
    plt.grid()
    sns.barplot(data = subset, y="Features",x = "Importance",color= "blue")
    #plt.xticks(rotation = 33)
    plt.savefig('importance.png', dpi=300)
    st.image('importance.png', caption='Features importance')
    
def grafica_curva_validacion(X_Train,y_train,model,marca):
    
    # Calcular la curva de aprendizaje
    train_sizes, train_scores, test_scores = learning_curve(model, X_Train, y_train, cv=5,scoring = "r2")

    # Calcular las medias y desviaciones estándar de los puntajes
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Graficar la curva de aprendizaje
    plt.figure(figsize=(10, 6))
    plt.title(f"Curva de Aprendizaje {marca}")
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel("R2")
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="R2 de entrenamiento")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="R2 validación")
    plt.legend(loc="best")
    plt.savefig('validation_curve.png', dpi=300)
    st.image('validation_curve.png', caption='Curva_Aprendizaje')
    plt.clf()
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
     
def imprime_metricas(model,df,marca):
    
    X = df[df.columns[1:]].copy()
    y = df["Price"].copy()
    X_Train, X_Test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   
    y_pred = model.predict(X_Test)
    categoricos_features = ["Condition", "Vehicle_brand", "Vehicle_model", "Fuel_type", "Drive", "Transmission", "Type", "First_owner"]

# Lista de características numéricas
    nummericos_features = ["Mileage_km", "Power_HP", "Displacement_cm3"]      
    # Calcular métricas de regresión
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write("---")
    st.title(f"Evaluación del modelo  :green[XGB_{predictions}.pkl]")
    with st.container():
            left,right = st.columns(2)
            with left:
                
                st.write('Mean Squared Error (MSE):', mse)
                st.write('Root M Squared Error (RMSE):', rmse)
                st.write('Mean Absolute Error (MAE):', mae)
                st.write('R^2:', r2)
            with right:
                
                lottie_url = "https://lottie.host/9ae7670e-04c5-45b9-9ce1-e7c283130108/clFCe7tOCA.json"
                lottie_json = load_lottieurl(lottie_url,)
                st_lottie(lottie_json,height=200)
    
def plotly_price(df,predictions):
    st.title(f"Distribución Variables Numéricas :green[{predictions}]")
    top_10_models = df['Vehicle_model'].value_counts().nlargest(15).index.tolist()
    df_top_10 = df[df['Vehicle_model'].isin(top_10_models)]
    st.subheader(f'Distribución del Precio por Modelo  :green[{predictions}](Top 10)')
    # Crear el box plot de la distribución del precio para los 10 modelos más frecuentes
    fig = px.box(df_top_10, x='Vehicle_model', y='Price', 
            color='Vehicle_model', points="all")

    # Personalizar el gráfico
    fig.update_traces(marker=dict(size=3), line=dict(width=1.5))
    st.plotly_chart(fig)
    st.write("---")
    st.subheader(f'Distribución Power_HP por Modelo  :green[{predictions}](Top 10)')
    fig = px.box(df_top_10, x='Vehicle_model', y='Power_HP', 
            color='Vehicle_model', points="all")

    # Personalizar el gráfico
    fig.update_traces(marker=dict(size=3), line=dict(width=1.5))
    st.plotly_chart(fig)
    st.write("---")
    st.subheader(f'Distribución Displacement_cm3 Modelo  :green[{predictions}](Top 10)')
    fig = px.box(df_top_10, x='Vehicle_model', y='Displacement_cm3', 
            color='Vehicle_model', points="all")

    # Personalizar el gráfico
    fig.update_traces(marker=dict(size=3), line=dict(width=1.5))
    st.plotly_chart(fig)

    

def get_unique_values(df,marca,columna):
    subset = df[df["Vehicle_brand"] == marca].copy()
    
    return subset[columna].unique()

# Función para cargar y mostrar la imagen 
def cargar_imagen():
    
    with st.sidebar.container():
    
    
        st.sidebar.write("[Linkelid ](https://www.linkedin.com/in/rafael-ca%C3%B1ada-abolafia-68692450/)")
    #st.sidebar.write("[GitHub ](https://github.com/falin23333)")

    
    
    uploaded_file = st.sidebar.file_uploader(":green[Seleccione imagen] ", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, use_column_width=True)
        return uploaded_file
def make_predictions(cat_input,num_input):
    user_input = {}
    user_input.update(cat_input)
    user_input.update(num_input)
    precio_predicho = model.predict(pd.DataFrame(user_input, index=[0]))
    #st.write(user_input)
    return int(precio_predicho),user_input
# Función para mostrar los features
def mostrar_features():
    cat_input = {}
    num_input = {}
    categoricos_features = ["Condition", "Vehicle_brand", "Vehicle_model", "Fuel_type", "Drive", "Transmission", "Type", "First_owner"]
    nummericos_features = ["Mileage_km", "Power_HP", "Displacement_cm3"]  
    df = pd.read_csv("csv_final.csv")
    df = df[df["Vehicle_brand"] == predictions]
    with open(f'models\XGB_{predictions}.pkl', 'rb') as f:
            model = pickle.load(f)
    
    
    st.markdown('<link rel="stylesheet" href="style.css">', unsafe_allow_html=True)
    st.sidebar.write(':red[Características del Vehículo]')
    # Mostrar características 
    for feature in nummericos_features:
        value = st.sidebar.slider(f'Selecciona {feature}', df[feature].min(), df[feature].max())
        num_input[feature] = value
        #
    for feature in categoricos_features:
        value = st.sidebar.selectbox(f'Selecciona {feature}', df[feature].unique())
        
        cat_input[feature] = value
    # Mostrar características numéricas
    if st.sidebar.button("Realizar Predicción"):
        marca = df["Vehicle_brand"].unique()
        st.container()
        left,right = st.columns(2)
        price, inputs = make_predictions(cat_input,num_input)
        with left:
            st.title( 
            f":red[PRECIO  ESTIMADO:] :green[{price}€]")
            st.write("---")
            lottie_url = "https://lottie.host/9b1d760a-d152-4817-9a7c-d5dce70d0f96/65tWrCArzp.json"
            lottie_json = load_lottieurl(lottie_url)
            st_lottie(lottie_json,height=200)
            
        with right:
            #st.write(pd.DataFrame(inputs.values()))
            
            st.write(f":red[Condition:] :green[{inputs['Condition']}]")
            st.write(f":red[Vehicle_brand:] :green[{inputs['Vehicle_brand']}]")
            st.write(f":red[Vehicle_model:] :green[{inputs['Vehicle_model']}]")
            st.write(f":red[Fuel_type:] :green[{inputs['Fuel_type']}]")
            st.write(f":red[Drive:] :green[{inputs['Drive']}]")
            st.write(f":red[Transmission:] :green[{inputs['Transmission']}]")
            st.write(f":red[Type:] :green[{inputs['Type']}]")
            st.write(f":red[First_owner:] :green[{inputs['First_owner']}]")
            st.write(f":red[Mileage_km:] :green[{inputs['Mileage_km']}]")
            st.write(f":red[Power_HP:] :green[{inputs['Power_HP']}]")
            st.write(f":red[Displacement_cm3:] :green[{inputs['Displacement_cm3']}]")
            
           
        imprime_metricas(model,df,predictions)
        X = df[df.columns[1:]].copy()
        y = df["Price"].copy()
        X_Train, X_Test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   
        y_pred = model.predict(X_Test)
        relacion_1_1(y_test,y_pred)
        plot_features_importance(model,categoricos_features,nummericos_features,marca)
        grafica_curva_validacion(X_Train,y_train,model,marca)
        st.write("---")
        plotly_price(df,predictions)
        st.write("---")
        plotly_EDA(df)
        # Crear el histograma de la distribución del precio por marca
        # Crear el box plot de la distribución del precio por marca
        
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#################################################################################################################################
import json
import requests
local_css("style.css")
with st.container():
    st.subheader(":red[Proyecto Final Data Science and Machine Learning] :wave:")
    left,rigth = st.columns(2)
    with left:
        lottie_url = "https://lottie.host/d27c410d-c34e-494c-826b-47d37805e1e1/VkSmAWhA8B.json"
        lottie_json = load_lottieurl(lottie_url)
        st_lottie(lottie_json,height=400)
    with rigth:
        st.title(":blue[PREDICT MY CAR!]")
        
        st.write("""Reconocimiento y Predicción de Coches

Bienvenido a nuestra aplicación de Inteligencia Artificial para el reconocimiento y predicción del precio. Utilizando una red neuronal convolucional, nuestra IA puede identificar la marca de un coche a partir de una imagen y, a partir de sus características, realizar una predicción del precio estimado.

Carga una imagen de un coche y descubre cómo nuestra tecnología de vanguardia desvela la marca del vehículo y estima su valor. 
""")


categoricos_features = ["Condition", "Vehicle_brand", "Vehicle_model", "Fuel_type", "Drive", "Transmission", "Type", "First_owner"]

# Lista de características numéricas
nummericos_features = ["Mileage_km", "Power_HP", "Displacement_cm3"]      
# Cargar imagen
imagen = cargar_imagen()

# Mostrar features y realizar predicción
if imagen is not None:
    import os
    import tf.keras.models

# Obtiene la ruta absoluta del archivo modelo2capasDROPOUT.h5
    model_path = os.path.abspath('models/modelo2capasDROPOUT.h5')

# Carga el modelo usando la ruta absoluta
    model = tf.keras.models.load_model(model_path)
    class_names = ['BMW', 'Ford', 'Mercedes-Benz', 'Nissan', 'Toyota', 'Volkswagen']
    model = tf.keras.models.load_model('models\modelo2capasDROPOUT.h5')
    image = Image.open(imagen)
    # Preprocesamiento de la imagen (ajústalo según las necesidades de tu modelo)
    image = image.resize((256, 256))
    #image = image.convert('RGB')
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)
    #image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    
    batch_predictions = model.predict(image)
    
    predictions = class_names[np.argmax(batch_predictions)]
    with st.container():
        left,right = st.columns(2)
        with left:
            #st.write(f'Predicción: {class_names[batch_predictions]}')
            st.title(f':red[Predicción: :green[{predictions}]]')
            agree = st.checkbox(':red[El modelo acierta?]')
        with right:
            lottie_url = " https://lottie.host/0caf0142-8476-4755-bef1-4a166b67f6a3/vNFugFbq57.json"
            lottie_json = load_lottieurl(lottie_url)
            st_lottie(lottie_json,height=400)
            

    if agree:
        
        # Cargamos el modelo
        #st.write(f"<span style='font-size:24px;'>:green[XGB_{predictions}.pkl]</span>", unsafe_allow_html=True)
        with open(f'models\XGB_{predictions}.pkl', 'rb') as f:
            model = pickle.load(f)
        
        
        mostrar_features()
        
    





















