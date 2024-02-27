import os
import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import shap
import time
from xgboost import XGBClassifier


if __name__ == '__main__':

    path_save_graphics = os.getcwd() + "\\graphics\\"
    path_save_data = os.getcwd() + "\\data_samples\\"

    tempo_inicial = time.time() # tempo em segundos

    #---------- Carrega e pré-processa os dados --------------#

    # Carrega todo dataset
    X_train, y_train, X_test, y_test = utils.load_data()
    
    # Carregar amostra representativa
    #X_train, y_train, X_test, y_test = utils.load_small_data(sample_size=0.2, random_seed=42, save_path=path_save_data)

    tempo_final = time.time()
    print(f">> Carregou os dados em: {(tempo_final - tempo_inicial):.2f} segundos")
    
    #debug load
    print("\n\ndebug load:\n",X_train.isnull().sum())
    pd.set_option('display.max_columns', None)
    print("X_train Head: \n", X_train.head())
    print("y_train Head: \n", y_train.head())
    print("\n")

    # Normaliza o dataset inteiro
    y_train, y_test, X_train, X_test, le = utils.preprocess_data(X_train, y_train, X_test, y_test, )

    # Normaliza o dataset estratificado
    #y_train, y_test, X_train, X_test, le = utils.preprocess_small_data(X_train, y_train, X_test, y_test, )
    
    #debug preprocess
    print("\n\ndebug preprocess:\n",X_train.isnull().sum())
    pd.set_option('display.max_columns', None)
    print("X_train Head: \n", X_train.head())
    print("y_train Head: \n", y_train.head())
    print("\n")


    tempo_final = time.time()
    print(f">> Normalizou os dados em: {(tempo_final - tempo_inicial):.2f} segundos")


    #-------------- Treinamento de modelo --------------------#

    #model = RandomForestClassifier(n_estimators=100)
    #model.fit(X_train, y_train)

    model = XGBClassifier(objective="multi:softprob", num_class=2 , random_state=42)
    model.fit(X_train, y_train)

    tempo_final = time.time()
    print(f">> Treinou o modelo em: {(tempo_final - tempo_inicial):.2f} segundos")

    #------------ Geração da Explicabilidade -----------------#
    shap.initjs()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    #shap_values_array = np.array(shap_values)

    tempo_final = time.time()
    print(f">> Gerou explicação em: {(tempo_final - tempo_inicial):.2f} segundos")

    #------ Plotagem e armazenamento de gráficos de dependência, importância global e força ---------#
    
    variable_focus = ['cbStatus','timeFromLastChange', 'timestampDiff', 'SqNum', 'isbBRmsValue']
    variable_interaction = ['cbStatus','timeFromLastChange', 'timestampDiff', 'SqNum', 'isbBRmsValue']

    for class_atack in range(len(np.unique(y_train))):
        for var_focus in variable_focus:
            for var_itc in variable_interaction:
                if var_focus == var_itc:
                    continue
                shap.dependence_plot(var_focus, shap_values[class_atack], X_train, interaction_index= var_itc, show=False)
                plt.savefig(os.path.join(path_save_graphics, f"{class_atack}_dependence_plot_{var_focus}X{var_itc}.png"))
                plt.close()

    # Geração do gráfico de força SHAP (summary plot)
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig(os.path.join(path_save_graphics, "summary_plot.png"))
    plt.close()
    

    shap.force_plot(explainer.expected_value, shap_values[0, :], X_train.iloc[0, :], show=False)
    shap.save_html(os.path.join(path_save_graphics, "force_plot.html"))

    # Geração do gráfico waterfall SHAP
    #shap_values.values=shap_values.values[:,:,1]
    #shap_values.base_values=shap_values.base_values[:,1]
    shap.plots.waterfall(shap_values, max_display=10, show=True)
    shap.plots._waterfall.waterfall_legacy(shap_values[0], max_display = 4, show = True)
    plt.savefig(os.path.join(path_save_graphics, "waterfall_plot.png"))
    plt.close()

    # Geração do gráfico de dispersão scatter SHAP
    # Converte shap_values para uma matriz NumPy
    # Converta shap_values para uma matriz NumPy
    shap.plots.scatter(shap_values[:, shap_values.abs.mean(0).argsort[-1]])
    plt.savefig(os.path.join(path_save_graphics, "cb_status_scatter_plot.png"))
    plt.close()
    
    tempo_final = time.time()
    print(f">> Finalizou em: {(tempo_final - tempo_inicial):.2f} segundos")