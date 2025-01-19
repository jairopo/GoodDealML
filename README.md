# GoodDealML 🤖💸

Proyecto que desarrolla un modelo para detectar si una oferta es buena, regular o mala.

![Logo](https://github.com/jairopo/GoodDealML/blob/main/img/logo.jpeg?raw=true)

## 📑 Índice

1. [Autores](#autores)
2. [Descripción](#descripción)
3. [Metodología](#metodología)
    - [1. Scraping y Preparación de Datos](#1-scraping-y-preparación-de-datos)
    - [2. Entrenamiento del Modelo de ML](#2-entrenamiento-del-modelo-de-ml)
    - [3. Despliegue del Modelo con Streamlit](#3-despliegue-del-modelo-con-streamlit)
4. [Tecnologías Utilizadas](#tecnologías-utilizadas)

## 👥 Autores

- [Marcos García Estévez](https://warcos.dev)
- [Darío Nievas López](https://github.com/Darnielop)
- [Jairo Andrades Bueno](https://github.com/jairopo)

## 📖 Descripción

**GoodDealML** tiene como objetivo desarrollar un modelo de Machine Learning capaz de **determinar si una oferta es buena, regular o mala**.

## 🛠 Metodología

### 1. Scraping y Preparación de Datos 🕸️📊

- **Obtención de Datos:** Realizamos web scraping para recopilar información de ofertas de diferentes sitios web.
- **Limpieza y Preprocesamiento:** Se procesan los datos para eliminar duplicados, manejar valores faltantes y preparar el conjunto para el análisis.

### 2. Entrenamiento del Modelo de ML 📈🤓

- **Simplificación de Títulos:** Utilizamos el modelo Llama 3.3 70b para simplificar los títulos de las ofertas, manteniendo la información clave.
- **Vectorización:** Empleamos BGE-M3 para transformar los títulos simplificados en vectores de características.
- **Ingeniería de Características:** Creamos nuevas variables como diferencia de precios, porcentaje de descuento y ratio de precios para mejorar el rendimiento del modelo.
- **Modelo de Predicción:** Implementamos XGBoost para clasificar las ofertas en buena, regular o mala.
- **Balanceo de Clases:** Aplicamos SMOTE para equilibrar las clases y evitar sesgos.
- **Optimización y Evaluación:** Ajustamos los umbrales para maximizar el F1-score y analizamos la importancia de las variables para entender las principales influencias en las predicciones.

### 3. Despliegue del Modelo con Streamlit 🚀🖥️

- **Interfaz Web Interactiva:** Utilizamos **Streamlit** para crear una aplicación web donde los usuarios pueden ingresar detalles de un producto y obtener una predicción sobre la oferta.
- **Opciones de Entrada:**
    - **FotoScan 📷:** Los usuarios pueden subir una imagen del producto y el modelo **Gemini 2.0 Flash** extraerá automáticamente el título, la empresa y los precios.
    - **Manual 📝:** Los usuarios pueden ingresar manualmente los detalles del producto.
- **Resultados en Tiempo Real:** Al hacer clic en "Calcular oferta", la aplicación muestra si la oferta es buena, regular o mala, además de la diferencia de precios y el porcentaje de descuento.

## 🧰 Tecnologías Utilizadas

- **Lenguajes y Librerías:** Python, Streamlit, XGBoost
- **Modelos de Lenguaje:** Llama 3.3 70b, BGE-M3, Gemini 2.0 Flash
- **Técnicas de Procesamiento:** Web Scraping, SMOTE, Ingeniería de Características
