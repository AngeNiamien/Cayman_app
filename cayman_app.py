import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from streamlit_extras.row import row
from streamlit_extras.colored_header import colored_header
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import pointbiserialr
import numpy as np
# from streamlit.locale import gettext as _
from PIL import Image
from scipy.stats import f_oneway
import scipy.stats as stats
from matplotlib.ticker import FuncFormatter
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import SelectFromModel
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix,accuracy_score , classification_report
from scipy.stats import percentileofscore
from scipy.optimize import minimize
from patsy import dmatrices
from scipy.spatial.distance import mahalanobis
from scipy.special import expit
import jenkspy
from io import BytesIO
from scipy.stats import beta,binom
import plotly.graph_objects as go
import os
# from weasyprint import HTML
import pdfkit




st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.image(Image.open("logo1.png"))
# Page d'accueil
def home():
    # Chemin vers le logo (remplacez "chemin_vers_logo.png" par le chemin réel)
    # logo_path ="logo.png"

    # Affichage du logo à l'aide de la balise HTML <img>
    # st.image(logo_path, caption='', use_column_width=True)
    # st.markdown(("APPLICATION DE DESCRIPTION DE DONNEES - CAYMAN CONSULTING"))

    st.title("APPLICATION DE DESCRIPTION DE DONNEES - CAYMAN CONSULTING")
                                                                                            # ,"4-🎚️discrimimant"
    st.sidebar.header("Pages")
    page = st.sidebar.selectbox( "selectionner une page",["1-🏚️Accueil", "2-🧾Manquants", "3-📉Graphiques","4-🎚️discrimimant","5-🤖modeles","6-🏋🏾Performance","7-🧮segmentation"])
    
    if page == "1-🏚️Accueil":
        Accueil()
    elif page == "2-🧾Manquants":
        Manquants()
    elif page == "3-📉Graphiques":
        graphiques_page()
    elif page == "4-🎚️discrimimant":
        discriminant()
    elif page == "5-🤖modeles":
        modeles()
    elif page == "6-🏋🏾Performance":
         performance()
    elif page == "7-🧮segmentation":
         segmentation()
   

# Page d'Accueil 
def Accueil():
    st.sidebar.header('Cayman-App `version 1`')
    st.header("🏚️Accueil")

    colored_header(
    label="Base de donnees",
    description="Afficher un echantillon de 10 observations du jeu de donnee ",
    color_name="light-blue-70",)
    
    # Sélectionner le type de fichier à importer
    file_type = st.sidebar.radio("Sélectionner le type de fichier", ["CSV", "TXT", "XLSX"])
    

    # if hasattr(st.session_state, 'df') and st.session_state.df is not None:
    #     df = st.session_state.df 
    # Sélectionner le délimiteur pour les fichiers CSV et TXT
    delimiter = ","
    if file_type in ["CSV", "TXT"]:
        delimiter = st.sidebar.selectbox("choisir le delimiteur",(";",","))
    
    file = st.sidebar.file_uploader("Importer le fichier", type=[file_type.lower()])
    # df= pd.read_csv("classe.csv", sep=";")
    if file is not None:
        st.success("Fichier importé avec succès!") 
        if file_type == "CSV":
            df = pd.read_csv(file, delimiter=delimiter)
        elif file_type == "TXT":
            df = pd.read_csv(file, delimiter=delimiter)
        elif file_type == "XLSX":
            df = pd.read_excel(file, engine='openpyxl')
        st.session_state.df = df
    else:
        st.error("Aucun fichier n'a été importé.")
        st.info("Selectionner un Jeu de donnees dans la barre latterale , en specifiant les differents parametre")
        st.stop()

    if hasattr(st.session_state, 'df') and st.session_state.df is not None:
        df = st.session_state.df 
    # df = pd.read_csv('classe.csv',sep=";")
    if st.checkbox("Afficher la data",False):
        st.subheader("")
        st.write(df)
    # st.table(df.head(10))
    numeric_variables = df.select_dtypes(include=['number']).columns
    categorical_variables = df.select_dtypes(include=['object']).columns
    col1, col2 , col3, col4  = st.columns(4)
    with col1:
            st.write("Nombres total de ligne:", df.shape[0])
    with col2:
            st.write("Nombres de variables:", df.shape[1])
    with col3:
            st.write("Nombres de variables Quantitatives:", len(numeric_variables))
    with col4:
            st.write("Nombres de variables Qualitatives:", len(categorical_variables))
   
    row1 = row(2, vertical_align="center")
    colored_header(
    label="Analyse Descriptive des variables Quantitatives",
    description="Resumé statistique des variables ",
    color_name="light-blue-70",)
    desc_stats = df.describe(percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])

    # missing_values = df.isnull().sum()   
      # Calcul du nombre de manquant
    manquant = df.isnull().sum()
    desc_stats.loc['manquant'] = manquant   
    st.table(desc_stats)  
     

    colored_header(
    label="Analyse Descriptive des variables Qualitatives",
    description="Description des Fréquences des modalités ",
    color_name="light-blue-70",)
    qual_vars = df.select_dtypes(include='object')
    # Initialiser un DataFrame pour stocker les fréquences et les pourcentages
    all_freq_df = pd.DataFrame()
    for column in qual_vars.columns:
            # missing_percent = (qual_vars[column].isnull().sum() / len(qual_vars) * 100)
            st.write(f"Variable: {column}")
            freq_table = qual_vars[column].value_counts()
                # Calculer les pourcentages avec format
            percent_table = (freq_table / len(qual_vars) * 100).map("{:.2f}%".format)
             # Calculer le nombre de données manquantes
            missing_count = qual_vars[column].isnull().sum()
            freq_df = pd.DataFrame({'Effectif': freq_table, 'Pourcentage': percent_table})
            freq_df.loc['Données manquantes'] = [missing_count, f'{(missing_count / len(qual_vars) * 100):.2f}%']
            # Ajouter les données de cette variable au DataFrame global
            # all_freq_df = pd.concat([all_freq_df, freq_df])
            st.table(freq_df )   

    st.sidebar.markdown('''
---
Created by [Cayman consulting](Cayman-consulting.com).
''')    
    st.session_state.df.shape[0] = df.shape[0]      

# Page des Manquants
def Manquants():
    colored_header(
    label="🧾Manquants",
    description="Valeur manquantes des variables du jeu de donnees(nombre et %)  ",
    color_name="red-70",)

    if hasattr(st.session_state, 'df') and st.session_state.df is not None:
        df = st.session_state.df
        
        # Calculer les valeurs manquantes
        missing_values = df.isnull().sum()
        missing_percent = (missing_values / len(df)) * 100
        
        missing_data = pd.DataFrame({'Valeurs Manquantes': missing_values, 'Pourcentage': missing_percent.apply(lambda x: f"{x:.2f}%")
                                     })
        missing_data = missing_data[missing_data['Valeurs Manquantes'] > 0]

        st.dataframe(missing_data)

               

                # Sélectionner les variables quantitatives et qualitatives
        numeric_vars = df.select_dtypes(include='number').columns
        categorical_vars = df.select_dtypes(include='object').columns
        
        # Sélectionner les variables quantitatives avec des données manquantes
        numeric_vars_with_missing = [var for var in numeric_vars if df[var].isnull().any()]
                # Sélectionner les variables qualitatives avec des données manquantes
        categorical_vars_with_missing = [var for var in categorical_vars if df[var].isnull().any()]
        
        if numeric_vars_with_missing :
             # Créer un select input pour les variables quantitatives avec des manquants
            var_num_missing = st.sidebar.multiselect("Sélectionnez une variable quantitative a traiter :", ["------"] + numeric_vars_with_missing)

        elif categorical_vars_with_missing :
             # Créer un select input pour les variables qualitatives avec des manquants
            var_cat_missing = st.sidebar.selectbox("Sélectionnez une variable qualitative a traiter :", ["------"] + categorical_vars_with_missing)

         # Méthodes d'imputation
        st.sidebar.header('choix de methode d\'imputation')
        imputation_method = st.sidebar.selectbox("Choisir une méthode d'imputation", ["------","Supprimer", "Moyenne"])

        # condition pour l'execution de la methode d'imputation
        data_filled = None   
        if imputation_method == "------":
            # df_cleaned = df.dropna()
            st.write("Données avant suppression des valeurs manquantes:")
            desc = df.describe()
            manquant = df.isnull().sum()
            desc.loc['manquant'] = manquant   
            st.dataframe(desc)
             
        elif imputation_method == "Supprimer":
            if var_num_missing != "------":
                data_filled = df.dropna()
                st.write("Données après suppression des valeurs manquantes:")
                desc = data_filled.describe()
                manquant = data_filled.isnull().sum()
                desc.loc['manquant'] = manquant   
                st.dataframe(desc)
        elif imputation_method == "Moyenne":
                    if var_num_missing != "------": 
                        # Imputation par la moyenne (uniquement pour les variables numériques)
                            data_filled = df.copy() # Créez une copie des données d'origine

                            if any(var in numeric_vars for var in var_num_missing):
                                for var in var_num_missing:
                                    if var in numeric_vars:
                                        data_filled[var].fillna(data_filled[var].mean(), inplace=True)
                            # Afficher les résultats
                            if data_filled is not None:
                                st.write("Données après imputation par la moyenne:")
                                desc = data_filled.describe()
                                manquant = data_filled.isnull().sum()
                                desc.loc['manquant'] = manquant
                                st.dataframe(desc)
                    else:
                        st.warning("La méthode d'imputation par la moyenne est disponible uniquement pour les variables quantitatives.")
        # Imputation par la Médiane
        elif imputation_method == "Médiane":    
                        if var_cat_missing in categorical_vars:  
                            mode_value = df[var_cat_missing].mode()[0]
                            data_filled = df.fillna({var_cat_missing: mode_value})
                            st.write(f"Données après imputation par le mode pour la variable {var_cat_missing}:")
                            desc = data_filled.describe()
                            manquant = data_filled.isnull().sum()
                            desc.loc['manquant'] = manquant
                            st.dataframe(desc)
                        else:
                            st.warning("La méthode d'imputation par le mode est disponible uniquement pour les variables qualitatives.")
        st.session_state.data_filled = data_filled
        if hasattr(st.session_state, 'data_filled') and st.session_state.data_filled is not None:
                data_filled = st.session_state.data_filled

 

# Page de graphiques
def graphiques_page():
    colored_header(
    label="📉Analyse Univarié ",
    description="Visualisation univarié des Variables Qualitatives(Bar),quantitative(histogram) ",
    color_name="blue-green-70",)
    if hasattr(st.session_state, 'df') and st.session_state.df is not None:
        df = st.session_state.df
        
         # df = pd.read_csv('classe.csv',sep=";")     
        columns = df.columns.tolist()

                # Séparer les variables qualitatives et quantitatives
        qualitative_vars = [col for col in columns if df[col].dtype == 'object']
        quantitative_vars = [col for col in columns if df[col].dtype != 'object']

        st.sidebar.header("option")
        quali = st.sidebar.selectbox("choisir une variable qualitative",(qualitative_vars))

        quanti = st.sidebar.selectbox("choisir une variable quantitative",(quantitative_vars))
      
        st.sidebar.header("la variable cible")
        target = st.sidebar.selectbox("Pour l'analyse Bivarié selectionner la variables cible",(columns))
        

        c1,c2 = st.columns((12,12))
        with c1 :  
                
                # Calculer les pourcentages
                percentages = (df[quali].value_counts(normalize=True) * 100).reset_index()
                # Renommer les colonnes pour les rendre plus explicites
                percentages.columns = [quali, 'Pourcentage']

                st.write(f"Diagramme à Barre pour {quali} (en pourcentage)")
                # Créer un diagramme à barres avec les pourcentages
                plt.figure(figsize=(10, 8))
                ax = sns.barplot(data=percentages, x=quali, y='Pourcentage', palette='Set2')

                # Formatez l'axe des ordonnées en pourcentage
                def percentage_formatter(x, pos):
                    return f"{x:.0f}%"

                ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

                plt.title(f"Diagramme à Barre pour {quali} (en pourcentage)")
                plt.ylabel('Pourcentage')
                plt.xlabel(quali)
                plt.xticks(rotation=45)  
                st.pyplot()

             
        with c2 :
                # Créez un histogramme en pourcentage
                st.write(f"{quanti} (en pourcentage)")
                plt.figure(figsize=(10, 8))
                ax = sns.histplot(data=df, x=quanti, bins=10, kde=True, stat="percent", color='skyblue', edgecolor='black', linewidth=1.2)

                # Formatez l'axe des ordonnées en pourcentage
                def percentage_formatter(x, pos):
                    return f"{x:.0f}%"

                ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

                plt.title(f"{quanti} (en pourcentage)")
                plt.xlabel(quanti)
                plt.ylabel("Pourcentage")
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(rotation=45)
                plt.title(f"Histogramme pour {quanti}")
                st.pyplot()
        

        # st.header("")
        colored_header(
        label="📉Analyse bivarié",
        description="Visualisation bivarié (variable cible vs qualitative ||  variable cible vs Quantitative) ",
        color_name="blue-green-70",)

        c3,c4 = st.columns((7,7))
        with c3 :
                # Calculer les pourcentages pour chaque catégorie de la variable qualitative
                percentages = (df.groupby([quali, target]).size() / df.groupby(quali).size()) * 100
                percentages = percentages.reset_index(name='Pourcentage')
                # percentages = percentages.sort_values(by='Pourcentage', ascending=True)
# 
                # Créer un graphique à barres empilées (barplot avec hue) avec les pourcentages
                st.write(f"{quali} vs {target} (en pourcentage)")
                plt.figure(figsize=(10, 8))
                ax = sns.barplot(x=quali, y='Pourcentage', hue=target, data=percentages, palette='Set2')

                # Formatez l'axe des ordonnées en pourcentage
                def percentage_formatter(x, pos):
                    return f"{x:.0f}%"

                ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

                plt.title(f"Graphique en bâton de la variable {quali} vs {target}")
                plt.xlabel(quali)
                plt.ylabel('Pourcentage')
                plt.xticks(rotation=45)  
                plt.legend(title=target)
                st.pyplot()

        with c4 :
                st.write(f"histogramme de {quanti} vs {target} en pourcentage")
                plt.figure(figsize=(10, 8))
                sns.displot(df, x=quanti, hue=target,kind="kde")
                st.pyplot() 

def discriminant():
    colored_header(
    label="🎚️Discrétisation ",
    description="Analyse de pouvoir Discriminant",
    color_name="blue-green-70",)
    st.sidebar.header("Options")
    
    if hasattr(st.session_state, 'data_filled') and st.session_state.data_filled is not None:
        data_filled = st.session_state.data_filled 

        columns = data_filled.columns.tolist() 
    ###########################################################################################""
        # Sélectione les variables quantitatives
    # quantitative_columns = data_filled.select_dtypes(include=["int64", "float64"]).columns.tolist()


    # Discrétiser les variables quantitatives
    st.sidebar.subheader("Discrétisation des variables quantitatives")
     # Vérifiez si la variable cible a été sélectionnée
    if "target_variable" not in st.session_state:
        st.warning("Sélectionnez une variable cible dans la barre latérale a gauche.")
        st.success(
        "Pour commencer, suivez les étapes suivantes pour le pouvoir discriminant et correlation entre les variables!\n"
        "Étapes à suivre :\n"
        "1. Sélectionnez la variable cible pour le pouvoir discriminant.\n"
        "2. Ensuite, sélectionnez le nombre de classes pour la discrétisation (par défaut, 6 classes).\n"
        "3. Calculez la corrélation entre les variables deux à deux.\n"
    )
        target_variable = st.sidebar.selectbox("Preciser la variable cible", data_filled.columns)
        # Stockez la variable cible dans la session
        st.session_state.target_variable = target_variable
        
    else:
        target_variable = st.session_state.target_variable

        target_variable = st.sidebar.selectbox("Preciser la variable cible", data_filled.columns)
          # Convertissez la variable cible en une variable catégorique
        data_filled[target_variable] = data_filled[target_variable].astype('category')
        data_filled2 = data_filled.copy()
        quantitative_columns = data_filled.select_dtypes(include=["int64", "float64"]).columns.tolist()

        # st.write(target_variable)
        # st.write(data_filled[target_variable].dtype)
    
        num_classes = st.sidebar.number_input("Nombre de classe", min_value=2, max_value=10, value=5)
        for col in quantitative_columns:
                # Use qcut with the desired number of classes directly
                discretized_col = pd.qcut(data_filled[col], q=num_classes, labels=False, duplicates='drop') + 1
                data_filled[col] = discretized_col

                
        st.session_state.num_classes = num_classes

        # for col in data_filled.columns:
        #     unique_values = data_filled[col].unique()
        #     st.write(f'Colonnes {col} - Valeurs uniques : {unique_values}')
        #     print(f'Colonnes {col} - Valeurs uniques : {unique_values}')

       # Sélectionnez les variables explicatives (toutes sauf la variable cible)
        explanatory_variables = data_filled.columns.tolist()
        explanatory_variables.remove(target_variable)


        # Créez un tableau pour stocker les résultats du V-Cramer
        v_cramer_results = []

         # Calculer le V-Cramer entre la variable cible et les autres variables explicatives
        for col in explanatory_variables:
            # Créez une table de contingence entre la variable qualitative et la variable cible
            contingency_table = pd.crosstab(data_filled[col], data_filled[target_variable])

            # Calculez le V-Cramer - 1ere methode 
            chi2, _, _, _ = stats.chi2_contingency(contingency_table)
            n = contingency_table.sum().sum()
            r, k = contingency_table.shape
            v_cramer = np.sqrt(chi2 / (n * (min(k, r) - 1)))

            # Ajoutez les résultats au tableau
            v_cramer_results.append((col, v_cramer))

        # Trier les résultats par ordre décroissant du V-Cramer
        v_cramer_results.sort(key=lambda x: x[1], reverse=True)

        # Afficher les résultats de la corrélation entre la variable cible et les autres variables
        st.write("Analyse du pouvoir discriminant entre la variable cible et les variables explicatives (V-Cramer)")
        result_df = pd.DataFrame(v_cramer_results, columns=["Variable Explicative", "V-Cramer"])
        cramer_corr_matrix = pd.DataFrame(index=explanatory_variables, columns=explanatory_variables)

        for var1 in explanatory_variables:
            for var2 in explanatory_variables:
                if var1 == var2:
                    contingency_table = pd.crosstab(data_filled[var1], data_filled[var2])
                    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
                    n = contingency_table.sum().sum()
                    r, k = contingency_table.shape
                    v_cramer = np.sqrt(chi2 / (n * (min(k, r) - 1)))
                    cramer_corr_matrix.loc[var1, var2] = v_cramer
   
                elif var1 != var2:
                    contingency_table = pd.crosstab(data_filled[var1], data_filled[var2])
                    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
                    n = contingency_table.sum().sum()
                    r, k = contingency_table.shape
                    v_cramer = np.sqrt(chi2 / (n * (min(k, r) - 1)))
                    cramer_corr_matrix.loc[var1, var2] = v_cramer
                     
        #         # Afficher la matrice de corrélation de Cramer's V entre les variables explicatives catégorielles
        # st.write("Matrice de Corrélation de Cramer's V (Variables Explicatives Catégorielles)")
        # st.write(cramer_corr_matrix)

                # Afficher la matrice de corrélation de Cramer's V sous forme de heatmap

        col1, col2  = st.columns(2)
        with col1:
                st.write(result_df)
        with col2:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cramer_corr_matrix.astype(float), annot=True, cmap="coolwarm", fmt=".2f")
            st.pyplot(plt)
        
        #######################------##############################################################################
        if st.sidebar.checkbox("Personaliser la discretisation",False):
            
                # Créer un tableau pour stocker les variables déjà discrétisées
                quantitative_vars = [col for col in columns if data_filled2[col].dtype != 'object']
                variables_deja_discretisees = []
                variable_variable = st.sidebar.selectbox("Sélectionner la variable à discrétiser", quantitative_vars)

                num_classe = st.sidebar.number_input("Choisir le nombre de classe", min_value=2, max_value=10, value=5)
                # Créer un discretiseur
                kbins = KBinsDiscretizer(n_bins=num_classe, encode='ordinal', strategy='uniform')
                    # Créer un tableau pour stocker les résultats du V de Cramer
                resultats_vcramer = []
                
                # Bouton pour lancer la discrétisation
                if st.sidebar.button("discrétiser"):
                     
                    if variable_variable not in variables_deja_discretisees:
                        # Discrétiser la variable choisie
                                        data_filled2[variable_variable + '_discret'] = kbins.fit_transform(data_filled2[[variable_variable]])
                                        # Ajouter la variable à la liste des variables déjà discrétisées
                                        variables_deja_discretisees.append(variable_variable)

                                        # Créer une table de contingence
                                        contingency_table = pd.crosstab(data_filled2[variable_variable+'_discret'], data_filled2[target_variable])
                                        chi2, _, _, _ = stats.chi2_contingency(contingency_table)
                                        n = contingency_table.sum().sum()
                                        r, k = contingency_table.shape
                                        v_cramer = np.sqrt(chi2 / (n * (min(k, r) - 1)))

                                        # Ajouter le résultat au tableau
                                        resultats_vcramer.append((variable_variable, v_cramer))
                                        # Afficher les résultats dans Streamlit
                                        st.write("Résultats du V de Cramer des variable personalisé :")
                                        resultat = pd.DataFrame(resultats_vcramer, columns=["Variable Explicative", "V-Cramer"])
                                        st.write(resultat)
                    else :
                        st.warning("Cette variable a déjà été discrétisée. Veuillez en choisir une autre.")

        ######################---------------#################################################################
        colored_header(
        label="⛓️Correlation ",
        description="Analyse de la correlation",
        color_name="blue-green-70",)

                # Identifiez les colonnes catégoriques
        categorical_columns = data_filled.select_dtypes(include=['object']).columns

            # Créez une instance de LabelEncoder
        label_encoder = LabelEncoder()
                # Appliquez LabelEncoder sur les colonnes catégorielles
        for column in categorical_columns:
            data_filled[column] = label_encoder.fit_transform(data_filled[column])
        
        ###data apres discretisation en 5 classes 
        st.write(data_filled)
        

    # target_variable = st.sidebar.selectbox("Preciser la variable cible", data_filled.columns)
    # # Stockez la variable cible dans la session
    st.session_state.data_filled = data_filled
    st.session_state.target_variable = target_variable
    

def modeles():
    colored_header(
        label="⛓️Modelisation ",
        description="Modelisation et selection de caractéristique",
        color_name="blue-green-70",
    )
    
    df = st.session_state.df
    data_filled = st.session_state.data_filled
    num_classes = st.session_state.num_classes 
    target_variable = st.session_state.target_variable 

    columns = data_filled.columns.tolist()

    # Affiche le message d'instructions si la variable cible n'est pas encore définie
    if "target_variable" not in st.session_state:
        st.warning("Sélection de caractéristiques avec Sequential Feature Selector (Stepwise)")
        st.success(
            "Pour commencer, suivez les étapes suivantes pour le Stepwise!\n"
            "Étapes à suivre :\n"
            "1. Sélectionnez la variable cible pour la sélection.\n"
            "2. Ensuite, sélectionnez la taille du jeu de test.\n"
        )

        # Permettre à l'utilisateur de sélectionner la variable cible
        # target_variable = st.sidebar.selectbox("Choisissez la variable cible", data_filled.columns)
        # st.session_state.target_variable = target_variable  # Stocke la variable cible dans la session
    else:
        # Récupère la variable cible depuis la session
        # target_variable = st.session_state.target_variable

        data_filled_2 = df.copy()
        
        # target_variable = st.sidebar.selectbox("Choisissez la variable cible", data_filled.columns)
        data_filled[target_variable] = data_filled[target_variable].astype('category')

        quantitative_columns = data_filled.select_dtypes(include=["int64", "float64"]).columns.tolist()

        for col in quantitative_columns:
            # Discrétisation des colonnes quantitatives
            discretized_col = pd.qcut(data_filled[col], q=num_classes, labels=False, duplicates='drop') + 1
            discretized_col_2 = pd.qcut(data_filled_2[col], q=num_classes, labels=False, duplicates='drop') + 1
            data_filled[col] = discretized_col
            data_filled_2[col] = discretized_col_2

        data_filled_2[quantitative_columns] = data_filled_2[quantitative_columns].astype('object')

        # Sélection de la taille du jeu de test
        test_size = st.sidebar.number_input("Taille du jeu de test (0.2 à 0.4)", min_value=0.2, max_value=0.4, step=0.01, value=0.2)

        # Séparation de la variable cible et des caractéristiques
        Y = data_filled[target_variable]
        X = data_filled.drop([target_variable], axis=1)

        seed = 123        
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed, stratify=Y)

        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)

        # Calcul des pourcentages pour les classes dans les ensembles d'entraînement et de test
        pourcentage_non_defaut_train = (y_train.value_counts()[0] / X_train.shape[0]) * 100
        pourcentage_defaut_train = (y_train.value_counts()[1] / X_train.shape[0]) * 100

        pourcentage_non_defaut_test = (y_test.value_counts()[0] / X_test.shape[0]) * 100
        pourcentage_defaut_test = (y_test.value_counts()[1] / X_test.shape[0]) * 100

        # Affichage des pourcentages dans un DataFrame
        df_matrice = pd.DataFrame({
            '': ['Train', 'Test'],
            'Non-Defaut': [pourcentage_non_defaut_train, pourcentage_non_defaut_test],
            'Defaut': [pourcentage_defaut_train, pourcentage_defaut_test]
        })

        st.text("Formes des sous-ensembles test et train après découpage des données ")
        st.dataframe(df_matrice)

        # Sélection de caractéristiques SFS
        sfs = SFS(LogisticRegression(), k_features='best', forward=True, floating=True, verbose=4, scoring='roc_auc', cv=5)
        sfs = sfs.fit(X_train, y_train)
        
        # Récupération des caractéristiques sélectionnées
        selected_features = list(sfs.k_feature_names_)
        st.write("Caractéristiques sélectionnées après le Stepwise:", selected_features)
        
        # Création et entraînement du modèle de régression logistique
        clf_sfs = LogisticRegression()
        clf_sfs.fit(X_train[selected_features], y_train)

        model2 = sm.Logit(y_train, X_train_const[selected_features])
        result2 = model2.fit()

        y_pred = result2.predict(X_test_const[selected_features])
        y_pred_binary = (y_pred > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred_binary)
        auc_sfs = roc_auc_score(y_test, clf_sfs.predict_proba(X_test[selected_features])[:, 1])

        st.write(f"AUC du modèle avec sélection de caractéristiques (Stepwise): {auc_sfs:.4f}")

        # Stockage des données dans la session
        st.session_state.target_variable = target_variable
        st.session_state.data_filled = data_filled
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.selected_features = selected_features
        st.session_state.df = df
        st.session_state.data_filled_2 = data_filled_2


def performance():
    colored_header(
    label="6-🏋🏾Performance",
    description="Statistique et performance du modele",
    color_name="blue-green-70")

    # Récupérer les variables de la session State
    target_variable = st.session_state.target_variable
    data_filled = st.session_state.data_filled
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    df_not_filled = st.session_state.df
    selected_features = st.session_state.selected_features
    data_filled_2 = st.session_state.data_filled_2

    # st.write(data_filled)
    
            # Ajoutez une constante à la matrice de caractéristiques
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    # Fonction pour le V de Cramer
    def calculate_cramer_v(data_filled, target_variable, explanatory_variables):
        v_cramer_results = []

        for col in explanatory_variables:
                contingency_table = pd.crosstab(data_filled[col], data_filled[target_variable])
                chi2, _, _, _ = stats.chi2_contingency(contingency_table)
                n = contingency_table.sum().sum()
                r, k = contingency_table.shape
                v_cramer = np.sqrt(chi2 / (n * (min(k, r) - 1)))
                v_cramer_results.append((col, v_cramer))

        v_cramer_results.sort(key=lambda x: x[1], reverse=True)

        return pd.DataFrame(v_cramer_results, columns=["Variable Explicative", "V-Cramer"])
        
       # Fonction pour le Heatmap de la matrice de corrélation de Cramer's V
    def plot_cramer_corr_heatmap(data_filled, explanatory_variables):
            cramer_corr_matrix = pd.DataFrame(index=explanatory_variables, columns=explanatory_variables)

            for var1 in explanatory_variables:
                for var2 in explanatory_variables:
                    if var1 == var2:
                        contingency_table = pd.crosstab(data_filled[var1], data_filled[var2])
                        chi2, _, _, _ = stats.chi2_contingency(contingency_table)
                        n = contingency_table.sum().sum()
                        r, k = contingency_table.shape
                        v_cramer = np.sqrt(chi2 / (n * (min(k, r) - 1)))
                        cramer_corr_matrix.loc[var1, var2] = v_cramer
                    elif var1 != var2:
                        contingency_table = pd.crosstab(data_filled[var1], data_filled[var2])
                        chi2, _, _, _ = stats.chi2_contingency(contingency_table)
                        n = contingency_table.sum().sum()
                        r, k = contingency_table.shape
                        v_cramer = np.sqrt(chi2 / (n * (min(k, r) - 1)))
                        cramer_corr_matrix.loc[var1, var2] = v_cramer

            plt.figure(figsize=(10, 8))
            sns.heatmap(cramer_corr_matrix.astype(float), annot=True, cmap="coolwarm", fmt=".2f")
            st.pyplot(plt)

    st.write("Tableau de V-Cramer correlation des variables selectionnées")
    c1,c2 = st.columns((12,12))
    with c1 :
            cramer_results = calculate_cramer_v(data_filled, target_variable, selected_features)
            st.write(cramer_results)
    with c2:
                
            plot_cramer_corr_heatmap(data_filled, selected_features)
    
             # Fonction pour tracer la courbe ROC
    def plot_roc_curve(model, X, y):
            # Plot ROC curve
            fpr, tpr, _ = roc_curve(y, model.predict_proba(X)[:, 1])
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('Taux de Faux Positive')
            plt.ylabel('Taux de Vrai Positive')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            st.pyplot()

    def plot_confusion_matrix(y_test,X_test):
         y_pred = clf_last_model.predict(X_test[selected_variables])
         y_pred_binary = (y_pred > 0.5).astype(int)

                    # Calculer la matrice de confusion
         conf_matrix = confusion_matrix(y_test, y_pred)

                    # Calculer les pourcentages
         cm_percentage = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
         sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues", cbar=False)

                    # Afficher la matrice de confusion sous forme de graphique avec seaborn
         plt.xlabel('Prédits')
         plt.ylabel('Réels')
         plt.title('Matrice de Confusion')
         st.pyplot(plt)


    #################################################### Test Final###########################################
        # Widget pour la sélection multiple des variables pour le dernier modèle
    selected_variables = st.multiselect("Sélectionnez les variables pour le dernier modèle", selected_features)

    bouton_demarre = False
        # Bouton pour démarrer l'entraînement du dernier modèle
    if st.button("Démarrer"):
                 bouton_demarre = True
    if bouton_demarre:
        if selected_variables:
            st.write("Entraînement du dernier modèle avec les caractéristiques sélectionnées manuellement")

                # Créer et entraîner le dernier modèle de régression logistique
            clf_last_model = LogisticRegression()
            clf_last_model.fit(X_train[selected_variables], y_train)

            # Utilisation du modèle pour prédire les probabilités
            y_proba = clf_last_model.predict_proba(X_test[selected_variables])
            # Séparation des probabilités pour les sains et les défaillants
            proba_defaillants = y_proba[:, 1]  # Probabilités de la classe 1 ( défaillants )
            proba_sains = y_proba[:, 0]  # Probabilités de la classe 0 (sains)

            ######################################## Distance de Mahalanobis ################################
            # Calcul des scores de log-odds
            score_sains_ = np.log((1 - proba_sains) / proba_sains)
            score_defaillants_ = np.log((1 - proba_defaillants) / proba_defaillants)

            # Calcul du score normalisé pour chaque observation
            min_score = min(score_sains_)  
            max_score = max(score_sains_)  
            score_sains = [(max_score - score) / (max_score - min_score) * 20 for score in score_sains_]

            # Calcul du score normalisé pour chaque observation
            min_score_2 = min(score_defaillants_) 
            max_score_2 = max(score_defaillants_)  
            score_defaillants = [(max_score_2 - score) / (max_score_2 - min_score_2) * 20 for score in score_defaillants_]

            # Calcul de la moyenne et de la variance pour les sains
            mean_sains = np.mean(score_sains)
            var_sains = np.var(score_sains)

            # Calcul de la moyenne et de la variance pour les défaillants
            mean_defaillants = np.mean(score_defaillants)
            var_defaillants = np.var(score_defaillants)

            # Calcul du nombre d'observations pour les sains et les défaillants
            n_sains = len(score_sains)
            n_defaillants = len(score_defaillants)

            n_total = n_sains + n_defaillants

            # Calcul des termes de la formule de Mahalanobis
            numerator = mean_sains - mean_defaillants
            denominator = np.sqrt(((n_sains * var_sains) + (n_defaillants * var_defaillants)) / (n_total))
            D = numerator / denominator
            Mahalanobis = expit(D)
            ######################################## FIN Distance de Mahalanobis ################################


            ################ Indice de Robustesse #######################################
            y_train_proba = clf_last_model.predict_proba(X_train[selected_variables])[:, 1]
            y_test_proba = clf_last_model.predict_proba(X_test[selected_variables])[:, 1]

            gini_train = 2 * roc_auc_score(y_train, y_train_proba) - 1
            gini_test = 2 * roc_auc_score(y_test, y_test_proba) - 1

            counts_train = y_train.value_counts()
            tx_defun_train = counts_train[1] / len(y_train)         

            counts_test = y_test.value_counts()
            tx_defun_test = counts_test[1] / len(y_test)
            IR = 1 - abs(gini_train - gini_test) * (max(tx_defun_train, 1 - tx_defun_train) / tx_defun_test)

            # IR = 1 - (abs(gini_train - gini_test) / (max(tx_defun_train, 1 - tx_defun_train)))
            ################################    Fin IR ###############################################
            
            #####################Calcule de GINI et AUC ##############################################
            # Évaluer le dernier modèle en utilisant l'AUC
            auc_last_model = roc_auc_score(y_test, clf_last_model.predict_proba(X_test[selected_variables])[:, 1])
            gini_last_model = 2 * auc_last_model - 1
            #####################"Fin calcule GINI et AUC #######################################"

            st.title("Performance du modèle")
            
            tab1, tab2 ,tab3 ,tab4 ,tab5 = st.tabs(["Statistique du Modele","Poids & Importance des variables","roc_curve","confusion_matrix","Densités conditionnelles"])

            with tab1:
                model2 = sm.Logit(y_train, X_train_const[selected_variables])

                    # Ajustez le modèle
                result2 = model2.fit()

                # Prédictions sur l'ensemble de test
                y_pred = result2.predict(X_test_const[selected_variables])
                y_pred_binary = (y_pred > 0.5).astype(int)
 
                    # Affichez les résultats
                    # Évaluez les performances du modèle
                accuracy = accuracy_score(y_test, y_pred_binary)

                ###############test avec Dmatrices ###############""""
                # Créer la formule pour le modèle
                # 'target_variable' ~ . signifie que toutes les autres variables seront utilisées comme prédicteurs
                # Construire la formule pour le modèle

                
                formula = f"{target_variable} ~ " + " + ".join(selected_variables)

                # Créer les matrices de design
                y, X = dmatrices(formula, data=data_filled_2, return_type='dataframe')

                # y = y[target_variable + '[1]']

                # Ajuster le modèle
                model = sm.Logit(y, X)
                result = model.fit()


                            # Obtenir les poids (coefficients) des variables
                coefficients = result2.params

                # Calculer les valeurs maximales pour chaque variable
                max_values = data_filled[coefficients.index].max()

                                # Remplacer chaque valeur maximale par 1
                max_values = pd.Series(1, index=coefficients.index)

                # Créer un DataFrame avec les poids et autres colonnes calculées
                feature_weights = pd.DataFrame({
                    'Variable': coefficients.index,
                    'coef': coefficients.values,
                    'Max value': max_values.values,
                    'prod': coefficients.values * max_values.values,
                    'abs prod': abs(coefficients.values * max_values.values)
                })

                # Calculer le poids comme pourcentage du total de abs prod
                total_abs_prod = feature_weights['abs prod'].sum()
                feature_weights['poids'] = (feature_weights['abs prod'] / total_abs_prod) * 100
                # Sélectionner uniquement les colonnes souhaitées
                feature_weights = feature_weights[['Variable', 'coef', 'poids']]

                st.header("Statistiques du modele")

                # col1, col2 = st.columns(2)
                # with col1:
                #      st.write("Caractéristiques sélectionnées pour le dernier modèle:", selected_variables)
                # with col2:
                     # Afficher les poids des variables dans Streamlit
               
                # st.dataframe(data_filled)

                col1, col2 , col3 = st.columns(3)

                with col1:
                    st.write("AUC :", round(auc_last_model,2))
                with col2:
                    st.write("Gini:", round(gini_last_model, 2))
                with col3:
                    st.write("AIC:", "{:.2f}" .format(result2.aic))
                            # Affichez les performances du modèle

                col1, col2 , col3 = st.columns(3)   

                with col1:
                    st.write("BIC:", "{:.2f}" .format(result2.bic))
                with col2:   
                    # Afficher les résultats mahalanobis
                    st.write(f"Distance de Mahalanobis  :", "{:.2f}".format(Mahalanobis) )
                with col3:
                    st.write("Indice de Robustesse  :", "{:.2f}".format(IR))
                    
                # st.write(result2.summary())
                # Afficher le résumé
                st.write(result.summary())

            with tab2:
                  st.write('Poids des variables dans le modele')
                  st.dataframe(feature_weights)
                 
            
            with tab3:
                st.header("roc_curve")
                plot_roc_curve(clf_last_model, X_test[selected_variables], y_test)    

            with tab4:
                st.header("confusion_matrix")

                plot_confusion_matrix(y_test,X_test)         

            with tab5:
                 st.header("Densité Conditionnelles")
                    # Créer un DataFrame avec les scores normalisés et les modalités prédites
                 data = pd.DataFrame({
                        'Score_sains': score_sains,
                        'Score_defaillants': score_defaillants,
                        'Modalite_predite': clf_last_model.predict(X_test[selected_variables])
                    })
                 data_classe_0 = data[data['Modalite_predite'] == 0]
                 data_classe_1 = data[data['Modalite_predite'] == 1]


                 plt.figure(figsize=(8, 6))
                    # Tracer le graphique de densité conditionnelle
                 sns.kdeplot(data=data_classe_0, x='Score_sains', hue='Modalite_predite', fill=True, label='Sains' )
                 sns.kdeplot(data=data_classe_1, x='Score_defaillants', hue='Modalite_predite',  fill=True, label='Defaillants', palette="husl")

                    # Ajouter un titre et des étiquettes d'axe
                 plt.title('Graphe de densité conditionnelle')
                 plt.xlabel('Score normalisé')
                 plt.ylabel('Densité')

                    # Afficher la légende
                 plt.legend()

                    # Afficher le graphique
                 st.pyplot(plt)
        else:
            st.warning("Sélectionnez au moins une variable pour le dernier modèle.")

        st.session_state.clf_last_model = clf_last_model
        st.session_state.selected_variables = selected_variables

def segmentation():
     
     colored_header(
     label="7-🧮segmentation",
     description="",
     color_name="blue-green-70")


     target_variable = st.session_state.target_variable
     data_not_filled = st.session_state.df
     data_filled = st.session_state.data_filled
     X_train = st.session_state.X_train
     y_train = st.session_state.y_train
     X_test = st.session_state.X_test
     y_test = st.session_state.y_test
     selected_features = st.session_state.selected_features
     clf_last_model = st.session_state.clf_last_model
     selected_variables = st.session_state.selected_variables

      
     n_classes = st.sidebar.number_input(
          label="Selectionnez le nombre de classe de risque souhaité",
          min_value=1,
          max_value=10,
     )

     if n_classes :
       ############################## Classe de score avec Methode de Jinks######################
            # Ajouter la colonne PD (Probabilité de Défaut) à data_filled
            y_proba = clf_last_model.predict_proba(data_filled[selected_variables])[:, 1]
            data_filled['PD'] = y_proba

            # Calculer les scores normalisés entre 0 et 20
            scores = -np.log(y_proba)
            score_min = scores.min()
            score_max = scores.max()
            scores_norm = 20 * (scores - score_min) / (score_max - score_min)
            data_filled['Score'] = scores_norm

            # Appliquer la méthode de Jenks pour créer les classes de risque
            n_classes = n_classes
            breaks = jenkspy.jenks_breaks(scores_norm, n_classes)


            data_filled['Classe de Risque'] = np.digitize(scores_norm, breaks, right=True)

            # Ajouter 1 pour que les classes commencent à 1
            data_filled['Classe de Risque'].replace(0, 1, inplace=True)

            # Inverser les classes
            max_classe = data_filled['Classe de Risque'].max()
            data_filled['Classe de Risque'] = max_classe - data_filled['Classe de Risque'] + 1


            # Convertir les classes en format lisible
            data_filled['Classe de Risque'] = data_filled['Classe de Risque'].apply(lambda x: f'CR {x}')            
            data_filled_classe_risque = data_filled

            

            def generate_excel():
                """Generates an Excel file from the DataFrame and returns the buffer."""
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    data_filled_classe_risque.to_excel(writer, index=False)
                buffer.seek(0)
                return buffer

            # Création du bouton de téléchargement
            buffer = generate_excel()
            st.download_button(
                label="Télécharger Excel", 
                data=buffer, 
                file_name="data_filled_classe_risque.xlsx", 
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.write(data_filled_classe_risque.head())

            colored_header(
            label="Statistiques sur les classes de risques ",
            description=" ",
            color_name="light-blue-70",)

            def calculate_gini(auc):
                return 2 * auc - 1
            
                    # Fonction pour calculer le nombre d'observations
            def calculate_nombre_observations(df):
                return df.groupby('Classe de Risque')['Classe de Risque'].size()
            

                # Fonction pour calculer le pourcentage d'observations
            def calculate_pourcentage_observations(df, total_observations):
                    nombre_observations = calculate_nombre_observations(df)
                    pourcentage = (nombre_observations / total_observations) * 100
                    return pourcentage.apply(lambda x: f"{x:.2f}%") 


                # Fonction pour calculer le score minimum
            def calculate_score_min(df):
                return df.groupby('Classe de Risque')['Score'].min()

                # Fonction pour calculer le score maximum
            def calculate_score_max(df):
                return df.groupby('Classe de Risque')['Score'].max()

                # Fonction pour calculer le taux de défaut
            def calculate_taux_defaut(df):
                return df.groupby('Classe de Risque')['PD'].mean() * 100
            
            # Fonction pour calculer l'indice HHI
            def calculate_hhi(df):
                nombre_observations = calculate_nombre_observations(df)
                total_observations = len(df)
                pourcentage_observations = (nombre_observations / total_observations)
                hhi = (pourcentage_observations**2).sum() * 100  # Multiplier par 100 pour que le HHI soit sur une échelle de 0 à 10000
                return hhi
            
                        # Fonction pour calculer le nombre de défauts
            def calculate_nombre_defauts(df):
                taux_defaut = calculate_taux_defaut(df) / 100  # Convertir en proportion
                nombre_observations = calculate_nombre_observations(df)
                nombre_defauts = (taux_defaut * nombre_observations).round().astype(int)
                return nombre_defauts
            
            # Fonction pour calculer la PD calibrée par la loi bêta
            def calculate_pd_beta(nombre_defauts, nombre_observations, niveau_str=0.95):
                alpha = nombre_defauts + 0.5
                beta_param = nombre_observations - nombre_defauts + 0.5
                pd_estimated = beta.ppf(niveau_str, alpha, beta_param)
                return pd_estimated
            
            # Fonction pour calculer la PD calibrée par la loi binomiale
            def calculate_pd_binomial(niveau_str, nombre_observations, taux_defaut):
                return binom.ppf(niveau_str, nombre_observations, taux_defaut) / nombre_observations

                # Calculer les statistiques
            total_observations = len(data_filled_classe_risque)
            nombre_observations = calculate_nombre_observations(data_filled_classe_risque)
            pourcentage_observations = calculate_pourcentage_observations(data_filled_classe_risque, total_observations)
            score_min = calculate_score_min(data_filled_classe_risque)
            score_max = calculate_score_max(data_filled_classe_risque)
            taux_defaut = calculate_taux_defaut(data_filled_classe_risque).apply(lambda x: f"{x:.2f}%")
            HHi = calculate_hhi(data_filled_classe_risque)
            nombre_defauts = calculate_nombre_defauts(data_filled_classe_risque)

                        # Obtenir les poids (coefficients) des variables
            weights = clf_last_model.coef_[0]


                # Organiser les résultats dans un DataFrame
            stat = pd.DataFrame({
                    'Nombre_observations': nombre_observations,
                    'Pourcentage_observations': pourcentage_observations,
                    'Score_min': score_min,
                    'Score_max': score_max,
                    'Nombre_defauts' : nombre_defauts,
                    'Taux_defaut': taux_defaut,

                    # 'Taux de Défaut Estimé (Beta)': data_filled_classe_risque['Taux de Défaut Estimé (Beta)']
                })
                        
                        # Ajouter la sélection pour le type de calibration
            calibration_type = st.sidebar.selectbox('Choisissez le type de calibration (Binomiale par défaut)', ['Binomiale','Beta'])

            # Ajouter la sélection pour le niveau_str si la calibration binomiale est choisie
            if calibration_type == 'Binomiale':
                niveau_str = st.sidebar.slider('Choisissez le niveau de confiance(95% par defaut)', 0.90, 1.0, 0.95)
            elif calibration_type == 'Beta':
                niveau_str = st.sidebar.slider('Choisissez le niveau de confiance(95% par defaut)', 0.90, 1.0, 0.95)
            else:
                niveau_str = 0.95  # Par défaut pour la calibration Beta

            # Calculer et afficher le tableau en fonction de la sélection
            if calibration_type == 'Beta':
                stat['PD calibrée (Beta)'] = stat.apply(
                    lambda row: calculate_pd_beta(row['Nombre_defauts'], row['Nombre_observations'], niveau_str), axis=1
                )
                stat['PD calibrée (Beta)'] = stat['PD calibrée (Beta)'].apply(lambda x: f"{x * 100:.2f}%")
            else:
                stat['PD calibrée (Binomiale)'] = stat.apply(
                    lambda row: calculate_pd_binomial(niveau_str, row['Nombre_observations'], row['Nombre_defauts']/row['Nombre_observations']), axis=1
                )
                stat['PD calibrée (Binomiale)'] = stat['PD calibrée (Binomiale)'].apply(lambda x: f"{x * 100:.2f}%")
                        

            color = "green" if HHi <= 40 else "red"

            # Utiliser st.write avec une chaîne HTML et les backticks

            # Calculer AUC et Gini
            auc = roc_auc_score(data_filled_classe_risque['loan_status'], data_filled_classe_risque['PD'])
            gini = calculate_gini(auc)

            tab1, tab2 , tab3 = st.tabs(["Statistiques & Classe de risque","Courbes PD vs PD calibrée","Tests homogénéité & hétérogénéité"])

            with tab1:
                 col1, col2 , col3 = st.columns(3)
                 with col1:
                    st.write(f"AUC: {auc:.2f}")
                 with col2:
                    st.write(f"Gini: {gini:.2f}")
                 with col3:
                        st.write(f'Indice de Concentration HHI : <span style="color:{color};">{HHi:.2f}</span>', unsafe_allow_html=True)  

                    # st.write(f"Indice de Concentration HHI :`{HHi:.2f}`")

                        # Transposer le DataFrame
                 stats_transposed = stat

                            # Affichage du tableau des statistiques dans Streamlit
                 st.write("Statistiques par classe de risque")
                 st.dataframe(stats_transposed)        

            with tab2:
                 
                    # # Supprimer le format de pourcentage et reconvertir en valeurs numériques
                if calibration_type == 'Beta':
                    stat['PD calibrée (Beta)'] = stat['PD calibrée (Beta)'].str.rstrip('%').astype('float') / 100
                    stat['Taux_defaut'] = stat['Taux_defaut'].str.rstrip('%').astype('float') / 100
                else:
                    stat['PD calibrée (Binomiale)'] = stat['PD calibrée (Binomiale)'].str.rstrip('%').astype('float') / 100
                    stat['Taux_defaut'] = stat['Taux_defaut'].str.rstrip('%').astype('float') / 100

            # # Tracer les courbes d'évolution
            # plt.figure(figsize=(10, 6))
            # plt.plot(stat.index, stat['Taux_defaut'], marker='o', label='Taux de Défaut')

            # if calibration_type == 'Beta':
            #     plt.plot(stat.index, stat['PD calibrée (Beta)'], marker='o', label='PD Calibrée (Beta)')
            # else:
            #     plt.plot(stat.index, stat['PD calibrée (Binomiale)'], marker='o', label='PD Calibrée (Binomiale)')

            # # Ajouter des labels et un titre
            # plt.xlabel('Classe de Risque')
            # plt.ylabel('Valeur')
            # plt.title(f'Évolution du Taux de Défaut et de la PD Calibrée ({calibration_type}) en fonction des Classes de Risque')
            # plt.legend()

            # # Limiter l'axe des ordonnées de 0 à 1
            # plt.ylim(0, 0.7)

            # # Afficher le graphique
            # plt.grid(True)
            # st.pyplot(plt)

                        ################graphique Plotly####################
                # Créer une figure avec Plotly
                fig = go.Figure()

                # Ajouter la courbe du Taux de Défaut
                fig.add_trace(go.Scatter(
                    x=stat.index,  # Classe de Risque devrait être l'index
                    y=stat['Taux_defaut'],
                    mode='lines+markers',
                    name='Taux de Défaut'
                ))

                # Ajouter la courbe de la PD calibrée en fonction du type de calibration
                if calibration_type == 'Beta':
                    fig.add_trace(go.Scatter(
                        x=stat.index,
                        y=stat['PD calibrée (Beta)'],
                        mode='lines+markers',
                        name='PD Calibrée (Beta)'
                    ))
                elif calibration_type == 'Binomiale':
                    fig.add_trace(go.Scatter(
                        x=stat.index,
                        y=stat['PD calibrée (Binomiale)'],
                        mode='lines+markers',
                        name='PD Calibrée (Binomiale)'
                    ))

                # Mettre à jour les labels et le titre
                fig.update_layout(
                    title=f'Évolution du Taux de Défaut et de la PD Calibrée ({calibration_type}) en fonction des Classes de Risque',
                    xaxis_title='Classe de Risque',
                    yaxis_title='Valeur',
                    yaxis=dict(range=[0, 0.70]),  # Limiter l'axe Y entre 0 et 1
                    hovermode='closest'
                )

                # Afficher le graphique dans Streamlit
                st.plotly_chart(fig)
                 
                 

            
            with tab3:
                data_not_filled['Classe de Risque'] = data_filled_classe_risque['Classe de Risque']
                data_not_filled['Score'] = data_filled_classe_risque['Score']

                # Filtrer les variables quantitatives et exclure la variable cible
                # quantitative_columns = [col for col in data_not_filled.columns if pd.api.types.is_numeric_dtype(data_not_filled[col]) and col != target_variable]

                # Exclure la variable cible et la variable Score des colonnes quantitatives
                quantitative_columns = [col for col in data_not_filled.columns if pd.api.types.is_numeric_dtype(data_not_filled[col]) and col not in [target_variable, 'Score']]

                # Permettre à l'utilisateur de choisir la variable pour former les sous-groupes
                variable_selection = st.sidebar.selectbox('Choisissez la variable pour créer les sous-groupes', quantitative_columns)

                # Initialiser un DataFrame pour stocker les résultats homogénéité
                results = pd.DataFrame(columns=['Classe de Risque','t_stat', 'p_value_t', 'w_stat', 'p_value_w'])

                # Boucle pour créer une variable pour chaque classe de risque
                classes_risque = data_not_filled['Classe de Risque'].unique()
                class_data = {}
                for classe in classes_risque:
                    class_data[classe] = data_not_filled[data_not_filled['Classe de Risque'] == classe]

                # Trier les classes de risque en ordre croissant
                sorted_risk_classes = sorted(class_data.keys())

                # Boucle pour comparer chaque classe de risque sélectionnée homoge
                for classe in sorted_risk_classes:
                    data = class_data[classe]
                    # st.write(f"Classe de Risque: {classe}")
                    # st.write(data)

                    # Calculer la médiane pour séparer en sous-groupes
                    median_value = data[variable_selection].median()

                    group_inf_median = data[data[variable_selection] < median_value]['Score']
                    group_sup_median = data[data[variable_selection] >= median_value]['Score']

                            # Effectuer les tests de Student et Wilcoxon entre les sous-groupes
                    t_stat, p_value_t = stats.ttest_ind(group_inf_median, group_sup_median, equal_var=False)
                    w_stat, p_value_w = stats.ranksums(group_inf_median, group_sup_median)

                    result = pd.DataFrame({
                        'Classe de Risque': [classe],
                        't_stat': [t_stat],
                        'p_value_t': [p_value_t],
                        'w_stat': [w_stat],
                        'p_value_w': [p_value_w]
                    })
                    results = pd.concat([results, result])

                # Afficher les résultats
                # st.write("Résultats des tests d'homogénéité entre les sous-groupes de", variable_selection, "de chaque classe de risque:")
                st.write(f'Résultats des tests d homogénéité entre les sous-groupes de : <span style="color:green;">{variable_selection}</span> de chaque classe de risque', unsafe_allow_html=True)
                st.dataframe(results)

                # Initialiser un DataFrame pour stocker les résultats pour d'hétérogénéité 
                result1 = pd.DataFrame(columns=['Classe 1', 'Classe 2', 't_stat', 'p_value_t'])

                # Boucle pour comparer chaque classe de risque avec les classes suivantes
                for i in range(len(sorted_risk_classes)):
                    for j in range(i + 1, len(sorted_risk_classes)):

                        classe_1 = sorted_risk_classes[i]
                        classe_2 = sorted_risk_classes[j]


                        # Sélectionner les données pour les classes de risque
                        data_1 = class_data[classe_1]
                        data_2 = class_data[classe_2]


                        # Effectuer le test de Welch entre les classes de risque
                        t_stat, p_value_t = stats.ttest_ind(data_1['Score'], data_2['Score'], equal_var=False)


                        result2 = pd.DataFrame({
                        'Classe 1': [classe_1],
                        'Classe 2': [classe_2],
                        't_stat': [t_stat],
                        'p_value_t': [p_value_t]
                        })

                        result1 = pd.concat([result1, result2]) 

                # Afficher les résultats
                st.write("Résultats des tests d'hétérogénéité de Welch entre chaque paire de classes de risque:")
                st.dataframe(result1)

                                                                            # Chemins
                # Chemins
                RAPPORT_DIR = "Rapport"  # Dossier contenant vos fichiers HTML
                OUTPUT_PDF = os.path.join(os.getcwd(), "rapport_final.pdf")  # Sauvegarde à la racine du projet

                st.title("Génération et Téléchargement du Rapport PDF")
                st.write("Cliquez sur le bouton ci-dessous pour générer et télécharger le rapport PDF à partir des pages HTML.")

                # Configuration pour wkhtmltopdf
                config = pdfkit.configuration(wkhtmltopdf=r"C:\Users\angeK\wkhtmltopdf\bin\wkhtmltopdf.exe")

                # Fonction pour générer le PDF à partir de fichiers HTML
                def generate_pdf_from_html(directory, output_pdf):
                    # Lire tous les fichiers HTML dans le dossier et les trier par ordre alphabétique
                    html_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.html')])

                    if not html_files:
                        raise FileNotFoundError("Aucun fichier HTML trouvé dans le dossier spécifié.")

                    # Générer le PDF avec PDFKit (en passant la liste des fichiers HTML)
                    pdfkit.from_file(html_files, output_pdf, configuration=config)
                    return output_pdf

                # Bouton Streamlit pour générer et télécharger le PDF
                if st.button("Generer le Rapport"):
                    try:
                        # Appeler la fonction pour générer le PDF
                        pdf_path = generate_pdf_from_html(RAPPORT_DIR, OUTPUT_PDF)
                        st.success(f"Le rapport a été généré avec succès : rapport_final.pdf")

                        # Bouton de téléchargement pour le PDF généré
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label="Télécharger le en PDF",
                                data=pdf_file,
                                file_name="rapport_final.pdf",
                                mime="application/pdf",
                            )
                    except Exception as e:
                        st.error(f"Erreur lors de la génération du rapport : {e}")
                 

            


     else:
          st.warning("Veuillez sélectionner le nombre de classes de risque souhaité.")
    

    
     
    


     


     
     
     

     

     
     
     

     
    
    


















          
def main():
    home()

if __name__ == "__main__":
    main()
