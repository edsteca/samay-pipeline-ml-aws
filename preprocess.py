import os
import pandas as pd
import numpy as np
import argparse
import boto3
import pathlib
from time import gmtime, strftime
import io

BASE_DIR = "ml/processing"

#Function to transform the dataset for the model
def transform_data(data):

    df=data
    genero = {'F': 1,'M': 2}
    medi_especi = {
            'Calcipotriol mas betametasona': 0,
                              'Metotrexate': 1,
                              'Secukinumab': 2,
                               'Expectante': 3,
                     'Adalimumab ( humira)': 4,
                   'Adalimumab ( amgevita)': 5,
                               'Guselkumab': 6,
                        'Fotoquimioterapia': 7,
                               'Ixekizumab': 8,
                               'Apremilast': 9,
                               'Clobetasol': 10,
                             'Betametasona': 11,
                              'Ustekinumab': 12,
                             'Risankizumab': 13,
                               'Infliximab': 14,
                                'Golimumab': 15,
                               'Acitretina': 16,
                            'Isotretinoina': 17,
                               'Etanercept': 18,
 'Fluorato De Mometasona /Acido Salicilico': 19,
                 'Propionato de clobetasol': 20,
                             'Certolizumab': 21,
                    'Furoato de mometasona': 22,
             'Dipropionato de betametasona': 23,
                             'Prednisolona': 24,
                             'Fexofenadina': 25,
               'Trimetoprim/Sulfametoxazol': 26,
                            'Desloratadina': 27,
                              'Tofacitinib': 28,
                               'Cetirizina': 29,
                               'Tacrolimus': 30,
                                 'Desonida': 31,
                               'Loratadina': 32,
                             'Ciclosporina': 33,
           'Betametasona/ acido salicilico': 34,
                                  'Dapsona': 35,
                        'Quimiofototerapia': 36,
                           'Hidrocortisona': 37
    }
    tipo_terapia = { 'Terapia tópica': 0, 'Terapia sistématica de 1a linea': 1, 'Terapia biológica': 2 }
    terapia_previa = { 'Si': 0, 'No': 1 }
    adherencia = { 'No Adherente': 0, 'Adherente': 1 }
    df = df[df['IMC'] <= 100]
    df = df.drop(['Fecha_Registro', 'Pasi_fecha_inicial', 'Pasi_fecha_final', 'Dlqi_fecha_inicial', 'Dlqi_fecha_final'], axis=1)
    df = df.drop(['Zona', 'Ciudad', 'Departamento'], axis=1)
    df = df.drop(['Num_Morbilidades', 'Multimorbido', 'Num_Medicamentos','Polimedicado'], axis=1)
    df= df.dropna()
    df['Genero'] = df['Genero'].apply(lambda x: genero[x])
    df['Medicamento_especifico'] = df['Medicamento_especifico'].apply(lambda x: medi_especi[x])
    df['Tipo_terapia'] = df['Tipo_terapia'].apply(lambda x: tipo_terapia[x])
    df['Terapias_Previas'] = df['Terapias_Previas'].apply(lambda x: terapia_previa[x])
    df['Clasificacion_Adherencia'] = df['Clasificacion_Adherencia'].apply(lambda x: adherencia[x])
    return df
    
    
#Function to extract the patients for apply the model    
def extract_patient(data):
    df=data
    df = df[df['Fracaso_o_exito'].isnull()]
    df=df.drop(['Fracaso_o_exito'], axis=1)
    print("Patients to predict extracted")
    df = transform_data(df) 
    return df




if __name__ == "__main__":
    
    print('**************************** Inicio del proceso ****************************')
    print('Fecha del proceso: {}'.format(strftime("%d-%m-%Y", gmtime())))

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    args = parser.parse_args()
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    print('**************************** Carga y formateo de datos ****************************')
    pathlib.Path(f"{BASE_DIR}/input").mkdir(parents=True, exist_ok=True)
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = pd.read_parquet(io.BytesIO(obj['Body'].read()))
    print("Total de datos en el dataframe: ", len(data))

    data_result=transform_data(data)
    data_to_predict=extract_patient(data)

    output_directory = os.path.join("/opt/ml/processing/output", "data_to_train.csv")
    output_directory_to_predict = os.path.join("/opt/ml/processing/output_p", "data_to_predict.csv")
    
    print("Total de datos resultantes (DataSet Entrenamiento): ", len(data_result))
    print("Total de datos resultantes (Data para predecir): ", len(data_to_predict))


    print("Guardando datos en {}".format(output_directory))
    data_result.to_csv(output_directory, sep=';')
    print("Guardando datos en {}".format(output_directory_to_predict))
    data_to_predict.to_csv(output_directory_to_predict, sep=';')

    print('**************************** Fin del proceso ****************************')
