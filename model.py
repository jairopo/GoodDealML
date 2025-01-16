from FlagEmbedding import BGEM3FlagModel
import pandas as pd
import numpy as np
import joblib
import sys

class GoodDealModel:

    def __init__(self):
        """Inicializa el modelo de clasificación y escalador."""
        self._modelo = None
        self._scaler = None
        self._embedding_model = None
        self._lista_empresas = self._getListaEmpresas()
        self._cargaModelo()

    def _cargaModelo(self):
        """Carga el modelo y el escalador"""
        print("Cargando el modelo y el escalador...")
        try:
            self._modelo = joblib.load('./model/modelo_xgboost_clasificacion.joblib')
            self._scaler = joblib.load('./model/scaler.joblib')
        except FileNotFoundError:
            print("Error: No se encontraron los archivos 'modelo_xgboost_clasificacion.joblib' o 'scaler.joblib'. Asegúrate de que existen en el directorio actual.")
            sys.exit(1)
        except Exception as e:
            print(f"Error al cargar los archivos: {e}")
            sys.exit(1)

        # Inicialización del modelo de embeddings
        print("Cargando el modelo de embeddings...")
        try:
            self._embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        except Exception as e:
            print(f"Error al inicializar el modelo de embeddings: {e}")
            sys.exit(1)
    
    @staticmethod
    def _getListaEmpresas():
        """Devuelve la lista de empresas"""
        return ['empresa_AliExpress', 'empresa_Amazon', 'empresa_Asos', 'empresa_Carrefour', 'empresa_Decathlon',
                'empresa_El Corte Inglés', 'empresa_Fnac', 'empresa_Game', 'empresa_Lidl', 'empresa_MediaMarkt',
                'empresa_Miravia', 'empresa_Outlet PC', 'empresa_PcComponentes', 'empresa_Privalia',
                'empresa_Privé by Zalando', 'empresa_Sports Direct', "empresa_Women'secret", 'empresa_Xiaomi',
                'empresa_Zalando', 'empresa_adidas', 'empresa_otras'
                ]
    
    def _getEmpresaSeleccionada(self, empresa_input):
        """Devuelve la empresa seleccionada"""
        empresa_feature = 'empresa_otras'  # Valor por defecto
        # Compara de forma insensible a mayúsculas y espacios
        empresa_input_normalizada = empresa_input.lower().replace(" ", "").replace("_", "")
        # Recorre la lista de empresas para ver a cuál corresponde
        for empresa in self._lista_empresas:
            nombre_empresa = empresa.replace('empresa_', '').lower().replace(" ", "").replace("_", "")
            if empresa_input_normalizada == nombre_empresa:
                empresa_feature = empresa
                break
        # Devuelve la empresa correspondiente
        return empresa_feature

    def _getTituloEmbedding(self, titulo):
        """Devuelve el embedding generado para el título del producto"""
        sentences = [titulo]
        try:
            embeddings = self._embedding_model.encode(
                sentences, 
                batch_size=12, 
                max_length=8192
            )['dense_vecs']
            return embeddings[0]
        except Exception as e:
            print(f"Error al generar el embedding: {e}")
            sys.exit(1)

    def _getEmpresaFeature(self, empresa_feature):
        """Devuelve las características de la empresa en one-hot encoding"""
        empresa_features = {empresa:0 for empresa in self._lista_empresas}
        empresa_features[empresa_feature] = 1
        return empresa_features
    
    def _getClasePredicha(self, y_pred_proba):
        """Devuelve la clase predicha según los umbrales óptimos establecidos"""
        # Aplicación de umbrales óptimos para las 3 clases
        umbrales_optimos = {
            0: 0.30,  # Clase 0: Mala oferta
            1: 0.55,  # Clase 1: Oferta regular
            2: 0.35   # Clase 2: Buena oferta
        }
        print(y_pred_proba)
        # Asignación de clase basada en umbrales
        # Paso 1: Identificar qué clases superan sus umbrales
        clases_superan_umbral = []
        for clase, umbral in umbrales_optimos.items():
            if y_pred_proba[clase] >= umbral:
                clases_superan_umbral.append(clase)
        
        # Paso 2: Asignar la clase con mayor probabilidad entre las que superan umbral
        if len(clases_superan_umbral) == 1:
            clase_predicha = clases_superan_umbral[0]
        elif len(clases_superan_umbral) > 1:
            # Asignar la clase con mayor probabilidad entre las que superan el umbral
            clase_predicha = clases_superan_umbral[np.argmax([y_pred_proba[clase] for clase in clases_superan_umbral])]
        else:
            # Si ninguna clase supera su umbral, asignar la clase con mayor probabilidad
            clase_predicha = np.argmax(y_pred_proba)
        # Definición de los nombres de las clases
        nombres_clases = {0: 'Mala😢', 1: 'Regular😐', 2: 'Buena😎'}
        # Devuelve el nombre de la clase predicha
        return nombres_clases[clase_predicha]


    def predict(self, titulo, empresa, precio_anterior, precio_actual):
        """
        Predice el tipo de oferta según los datos proporcionados. 
        Devuelve el tipo de oferta, la diferencia de precios, el porcentaje de descuento y el ratio de precios
        """
        # Recoge la empresa seleccionada y el embedding del titulo
        empresa_seleccionada = self._getEmpresaSeleccionada(empresa)
        embedding_vector = self._getTituloEmbedding(titulo)
        # Preparación de las características de la empresa (one-hot encoding)
        empresa_features = self._getEmpresaFeature(empresa_seleccionada)
        # Cálculo de características derivadas
        diferencia_precios = precio_anterior - precio_actual
        porcentaje_descuento = ((diferencia_precios) / precio_anterior) * 100
        ratio_precios = precio_actual / precio_anterior
        # Preparación del vector de características
        features = {
            'Precio': precio_actual,
            'Precio Antiguo': precio_anterior,
            'Diferencia de precios': diferencia_precios,
            'Porcentaje de descuento': porcentaje_descuento,
            'Ratio de precios': ratio_precios
        }
        # Añadir características de la empresa
        features.update(empresa_features)
        # Añadir características de embedding
        for i in range(1024):
            features[f'embedding_{i}'] = embedding_vector[i]
        # Conversión a DataFrame
        input_df = pd.DataFrame([features])
        # Escalado de características
        try:
            X_scaled = self._scaler.transform(input_df)
        except Exception as e:
            print(f"Error al escalar las características: {e}")
            sys.exit(1)
        # Predicción de probabilidades
        try:
            y_pred_proba = self._modelo.predict_proba(X_scaled)[0]
        except Exception as e:
            print(f"Error al realizar la predicción: {e}")
            sys.exit(1)
        # Recoge el nombre de la clase predicha
        nombre_clase_predicha = self._getClasePredicha(y_pred_proba)
        # Devuelve el tipo de oferta, la diferencia de precios, el porcentaje de descuento y el ratio de precios
        return nombre_clase_predicha, diferencia_precios, porcentaje_descuento, ratio_precios
    
        

    

