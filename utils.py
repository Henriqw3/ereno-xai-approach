import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
#from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score)
#from metrics import Metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import re

def load_small_data(sample_size=0.2, random_seed=42, save_path=None):
    # Carregamento de dados completo
    train_df = pd.read_csv('data/test_gray.csv', sep=',')
    test_df = pd.read_csv('data/test_gray.csv', sep=',')

    # Remoção de colunas enriquecidas ou com NaN
    train_df = train_df.dropna(axis=1)
    test_df = test_df.dropna(axis=1)

    # Realizar amostragem estratificada para manter a distribuição das classes
    X_train, _, y_train, _ = train_test_split(train_df.drop(columns=['@class@']), train_df['@class@'],
                                              test_size=1 - sample_size, stratify=train_df['@class@'],
                                              random_state=random_seed)

    X_test, _, y_test, _ = train_test_split(test_df.drop(columns=['@class@']), test_df['@class@'],
                                            test_size=1 - sample_size, stratify=test_df['@class@'],
                                            random_state=random_seed)

    # Salvar partes estratificadas como CSV, se o caminho de salvamento for fornecido
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        X_train.to_csv(os.path.join(save_path, 'X_train_stratified.csv'), index=False)
        pd.DataFrame(y_train, columns=['@class@']).to_csv(os.path.join(save_path, 'y_train_stratified.csv'), index=False)
        X_test.to_csv(os.path.join(save_path, 'X_test_stratified.csv'), index=False)
        pd.DataFrame(y_test, columns=['@class@']).to_csv(os.path.join(save_path, 'y_test_stratified.csv'), index=False)

    # Retorna amostra estratificada
    return X_train, y_train, X_test, y_test



def load_data():
    # Carregamento de dados
    train_df = pd.read_csv('data/test_gray.csv', sep=',')
    test_df = pd.read_csv('data/test_gray.csv', sep=',')

    # Colunas enriquecidas para remover
    columns_to_remove = ['stDiff', 'sqDiff', 'gooseLenghtDiff', 'cbStatusDiff', 'apduSizeDiff',
                         'frameLengthDiff', 'timestampDiff', 'tDiff', 'timeFromLastChange',
                         'delay', 'isbARms', 'isbBRms', 'isbCRms', 'ismARms', 'ismBRms', 'ismCRms',
                         'ismARmsValue', 'ismBRmsValue', 'ismCRmsValue', 'csbArms', 'csvBRms',
                         'csbCRms', 'vsmARms', 'vsmBRms', 'vsmCRms', 'isbARmsValue', 'isbBRmsValue',
                         'isbCRmsValue', 'vsbARmsValue', 'vsbBRmsValue', 'vsbCRmsValue',
                         'vsmARmsValue', 'vsmBRmsValue', 'vsmCRmsValue', 'isbATrapAreaSum',
                         'isbBTrapAreaSum', 'isbCTrapAreaSum', 'ismATrapAreaSum', 'ismBTrapAreaSum',
                         'ismCTrapAreaSum', 'csvATrapAreaSum', 'csvBTrapAreaSum', 'vsbATrapAreaSum',
                         'vsbBTrapAreaSum', 'vsbCTrapAreaSum', 'vsmATrapAreaSum', 'vsmBTrapAreaSum',
                         'vsmCTrapAreaSum', 'gooseLengthDiff']

    # Remoção de colunas enriquecidas ou com NaN
    train_df = train_df.dropna(axis=1)  # .drop(columns=columns_to_remove, errors='ignore')
    test_df = test_df.dropna(axis=1)  # .drop(columns=columns_to_remove, errors='ignore')

    # Separação de features e labels
    X_train = train_df.drop(columns=['@class@'])
    y_train = train_df['@class@']
    X_test = test_df.drop(columns=['@class@'])
    y_test = test_df['@class@']
    return X_train, y_train, X_test, y_test


def preprocess_data(X_train, y_train, X_test, y_test):
    # Identificar colunas numéricas
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    # Utilizar StandardScaler para normalizar os dados numéricos
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Inicializar listas vazias para armazenar os nomes das colunas categóricas
    cat_column_names = []

    # Utilizar OneHotEncoder para colunas categóricas
    if cat_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_train_cat = encoder.fit_transform(X_train[cat_cols])
        X_test_cat = encoder.transform(X_test[cat_cols])

        # Recuperar os nomes das colunas após a transformação OneHotEncoder
        for col, categories in zip(cat_cols, encoder.categories_):
            cat_column_names.extend([f"{col}_{category}" for category in categories])

        # Criar DataFrames para os dados categóricos
        X_train_cat_df = pd.DataFrame(X_train_cat, columns=cat_column_names)
        X_test_cat_df = pd.DataFrame(X_test_cat, columns=cat_column_names)

        # Transformar em DataFrames do Pandas
        X_train_num_df = pd.DataFrame(X_train[num_cols], columns=num_cols)
        X_test_num_df = pd.DataFrame(X_test[num_cols], columns=num_cols)

        # Concatenar dados numéricos e categóricos
        X_train = pd.concat([X_train_num_df, X_train_cat_df], axis=1)
        X_test = pd.concat([X_test_num_df, X_test_cat_df], axis=1)

    # Inicializar o LabelEncoder para os rótulos
    le = LabelEncoder()

    # Transformar y_train e y_test para numérico
    if y_train.dtype == 'object':
        y_train = le.fit_transform(y_train)
    if y_test.dtype == 'object':
        y_test = le.transform(y_test)  # usar o mesmo encoder para garantir uma codificação consistente

    # Salvar os dados processados em CSV, se fornecido o caminho de saída
    processed_data = pd.concat([pd.Series(y_train),X_train], axis=1)
    processed_data.to_csv(os.path.join("C:\\Projetos\\codes_tcc\\data_samples", 'preprocess_train.csv'), index=False)
    processed_data = pd.concat([pd.Series(y_train),X_train], axis=1)
    processed_data.to_csv(os.path.join("C:\\Projetos\\codes_tcc\\data_samples", 'preprocess_test.csv'), index=False)

    # Retornar os dados em objetos pandas e o LabelEncoder
    return pd.Series(y_train), pd.Series(y_test), X_train, X_test, le


def preprocess_small_data(X_train, y_train, X_test, y_test):
    # Identificar colunas numéricas e categóricas
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    # Tratar valores ausentes (NaN) com SimpleImputer para dados numéricos
    imputer_num = SimpleImputer(strategy='mean')
    X_train[num_cols] = imputer_num.fit_transform(X_train[num_cols])
    X_test[num_cols] = imputer_num.transform(X_test[num_cols])

    # Tratar valores ausentes (NaN) com SimpleImputer para dados categóricos
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X_train[cat_cols] = imputer_cat.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = imputer_cat.transform(X_test[cat_cols])

    # Utilizar StandardScaler para normalizar os dados numéricos
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Utilizar OneHotEncoder para colunas categóricas
    if cat_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_train_cat = encoder.fit_transform(X_train[cat_cols])
        X_test_cat = encoder.transform(X_test[cat_cols])

        # Recuperar os nomes das colunas após a transformação OneHotEncoder
        cat_column_names = [f"{str(col).strip()}_{str(category).strip()}" for col in cat_cols for category in encoder.categories_[0]]

        # Criar DataFrames para os dados categóricos
        X_train_cat_df = pd.DataFrame(X_train_cat, columns=cat_column_names)
        X_test_cat_df = pd.DataFrame(X_test_cat, columns=cat_column_names)

        # Transformar em DataFrames do Pandas
        X_train_num_df = pd.DataFrame(X_train[num_cols], columns=num_cols)
        X_test_num_df = pd.DataFrame(X_test[num_cols], columns=num_cols)

        # Concatenar dados numéricos e categóricos
        X_train = pd.concat([X_train_num_df, X_train_cat_df], axis=1)
        X_test = pd.concat([X_test_num_df, X_test_cat_df], axis=1)

    # Inicializar o LabelEncoder para os rótulos
    le = LabelEncoder()

    # Transformar y_train e y_test para numérico
    if y_train.dtype == 'object':
        y_train = le.fit_transform(y_train)
    if y_test.dtype == 'object':
        y_test = le.transform(y_test)  # usar o mesmo encoder para garantir uma codificação consistente


    # Salvar os dados processados em CSV, se fornecido o caminho de saída
    processed_data = pd.concat([pd.Series(y_train),X_train], axis=1)
    processed_data.to_csv(os.path.join("C:\\Projetos\\codes_tcc\\data_samples", 'preprocess_small_train.csv'), index=False)
    processed_data = pd.concat([pd.Series(y_train),X_train], axis=1)
    processed_data.to_csv(os.path.join("C:\\Projetos\\codes_tcc\\data_samples", 'preprocess_small_test.csv'), index=False)


    # Retornar os dados em objetos pandas e o LabelEncoder
    return pd.Series(y_train), pd.Series(y_test), X_train, X_test, le









def clean_variable_name(nm_var):
    # Substituir caracteres não aceitos por "_"
    cl_var = re.sub(r'[^a-zA-Z0-9_.-]', '_', nm_var)
    return cl_var
