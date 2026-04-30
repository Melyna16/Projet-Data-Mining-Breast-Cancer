# pyright: reportMissingImports=false
# pylint: disable=import-error
#!/usr/bin/env python
# coding: utf-8

# importation

# Partie 1 – Chargement et exploration initiale

# In[60]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import os


# Question 1.1  — Chargement du Dataset 

# In[61]:


df = pd.read_csv('wdbc.data', header=None)
col_names = ['id', 'diagnosis',
             'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
             'smoothness_mean', 'compactness_mean', 'concavity_mean',
             'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
             'radius_se', 'texture_se', 'perimeter_se', 'area_se',
             'smoothness_se', 'compactness_se', 'concavity_se',
             'concave points_se', 'symmetry_se', 'fractal_dimension_se',
             'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
             'smoothness_worst', 'compactness_worst', 'concavity_worst',
             'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
df.columns = col_names
df.drop('id', axis=1, inplace=True)
df.head(10)


# Exploration initiale (qst 1.2 et 1.3)

# In[62]:


print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df['diagnosis'].value_counts(normalize=True) * 100)


# 1.4 Statistiques Descriptives Globales : 

# In[63]:


print(df.describe())
num_cols = df.select_dtypes(include=['float64']).columns
for col in list(num_cols)[:5]:
    print(f"{col}: skew={skew(df[col]):.3f}, kurtosis={kurtosis(df[col]):.3f}")


# 1.5 Analyse Comparative par Classe : 

# In[64]:


grouped = df.groupby('diagnosis').mean()
diff = abs(grouped.loc['M'] - grouped.loc['B']).sort_values(ascending=False)
print(diff.head(5))


# PARTIE 2 – Nettoyage des Données : 

# 2.1 Correction des Catégories Invalides 
# 

# 2.2 Détection des Outliers (Méthode IQR) : 

# In[65]:


def iqr_outliers(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - 1.5*IQR
    high = Q3 + 1.5*IQR
    out = data[(data[col] < low) | (data[col] > high)]
    return len(out), len(out)/len(data)*100

os.makedirs('graphs', exist_ok=True)
for col in ['radius_mean', 'area_mean', 'smoothness_mean']:
    n, p = iqr_outliers(df, col)
    print(f"{col}: {n} outliers ({p:.2f}%)")
    plt.figure()
    sns.boxplot(x='diagnosis', y=col, data=df)
    plt.title(f'Boxplot {col}')
    plt.savefig(f'graphs/boxplot_{col}.png', dpi=300, bbox_inches='tight')
    plt.show()


# 2.3 Traitement des Variables Asymétriques : 

# In[66]:


skew_vals = df[num_cols].apply(lambda x: skew(x.dropna()))
high_skew = skew_vals[abs(skew_vals) > 2].index.tolist()
for var in high_skew:
    df[f'{var}_log'] = np.log1p(df[var])
    print(f"{var}: skew avant {skew(df[var]):.3f} → après {skew(df[f'{var}_log']):.3f}")
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    ax1.hist(df[var], bins=30)
    ax1.set_title(f'Original: {var}')
    ax2.hist(df[f'{var}_log'], bins=30)
    ax2.set_title(f'Log: {var}')
    plt.savefig(f'graphs/log_{var}.png', dpi=300, bbox_inches='tight')
    plt.show()


# 2.4 Vérification Finale du Nettoyage : 

# In[67]:


df.to_csv('breast_cancer_cleaned.csv', index=False)
print("Fichier sauvegardé : breast_cancer_cleaned.csv")


# In[68]:


# Cellule : Chargement du dataset nettoyé
df = pd.read_csv('breast_cancer_cleaned.csv')
print("Dimensions du dataset :", df.shape)
df.head()


# PARTIE 3 – FEATURE ENGINEERING
# 

# Question 3.1 – Création d’une variable de tendance

# In[69]:


# Cellule 3.1.1 : Création de la variable de tendance 'area_trend'
df['area_trend'] = df['area_worst'] - df['area_mean']

# Statistiques par classe
trend_stats = df.groupby('diagnosis')['area_trend'].agg(['mean', 'median', 'std'])
print(trend_stats)


# In[70]:


# Cellule 3.1.2 : Histogramme + KDE de area_trend par classe
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='area_trend', hue='diagnosis', kde=True, bins=30, alpha=0.6)
plt.title('Tendance de la surface tumorale (worst - mean) par diagnostic')
plt.xlabel('Area trend')
plt.savefig('graphs/area_trend_hist.png', dpi=300, bbox_inches='tight')
plt.show()


# Question 3.2 – Création de ratios / variables dérivées

# In[71]:


# Cellule 3.2.1 : Création des ratios
df['ratio_perimeter_area'] = df['perimeter_mean'] / (df['area_mean'] + 1e-6)
df['ratio_concavity'] = df['concavity_mean'] / (df['compactness_mean'] + 1e-6)

# Statistiques par classe
ratios = ['ratio_perimeter_area', 'ratio_concavity']
for r in ratios:
    print(f"\n{r} :")
    print(df.groupby('diagnosis')[r].agg(['mean', 'median', 'std']))


# In[72]:


# Cellule 3.2.2 : Boxplots comparatifs des ratios
fig, axes = plt.subplots(1, 2, figsize=(12,5))
for ax, r in zip(axes, ratios):
    sns.boxplot(data=df, x='diagnosis', y=r, hue='diagnosis', palette='Set2', legend=False, ax=ax)
    ax.set_title(f'Boxplot de {r}')
plt.tight_layout()
plt.savefig('graphs/ratios_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()


# Question 3.3 – Agrégations multi-colonnes

# In[73]:


# Cellule 3.3.1 : Compter le nombre de features worst au-dessus du 75e percentile
worst_cols = [col for col in df.columns if col.endswith('_worst')]
thresholds = df[worst_cols].quantile(0.75)
df['n_severe_features'] = (df[worst_cols] > thresholds).sum(axis=1)

# Distribution par classe
print(df.groupby('diagnosis')['n_severe_features'].describe())


# In[74]:


# Cellule 3.3.2 : Scatter plot n_severe_features vs cible (avec jitter)
plt.figure(figsize=(8,5))
sns.stripplot(data=df, x='diagnosis', y='n_severe_features', jitter=True, alpha=0.5)
plt.title('Nombre de caractéristiques sévères par diagnostic')
plt.savefig('graphs/n_severe_features.png', dpi=300, bbox_inches='tight')
plt.show()


# Question 3.4 – Transformations logarithmiques (complément)

# In[75]:


# Cellule 3.4.1 - Création des transformations log pour les variables asymétriques
import pandas as pd
import numpy as np
from scipy.stats import skew

# Chargement du dataset
df = pd.read_csv('breast_cancer_cleaned.csv')   

# 1. Sélectionner les variables numériques (sauf la cible et les colonnes déjà log)
exclude = ['diagnosis', 'target']  # target sera créée plus tard
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Enlever les colonnes qui contiennent déjà '_log' pour ne pas les re-loguer
numeric_cols = [col for col in numeric_cols if '_log' not in col and col not in exclude]

# 2. Calculer le skewness initial
skewness_initial = {col: skew(df[col].dropna()) for col in numeric_cols}

# 3. Identifier celles avec |skewness| > 2
high_skew = [col for col in numeric_cols if abs(skewness_initial[col]) > 2]
print(f"Variables avec |skewness| > 2 : {high_skew}")

# 4. Créer les colonnes log (si elles n'existent pas déjà)
for col in high_skew:
    log_col = col + '_log'
    if log_col not in df.columns:
        df[log_col] = np.log1p(df[col])   # log(1+x) pour gérer les zéros
        print(f"Création de {log_col} (skewness initial : {skewness_initial[col]:.2f})")

# 5. Calculer le nouveau skewness après transformation
skewness_after = {}
for col in high_skew:
    log_col = col + '_log'
    skewness_after[col] = skew(df[log_col].dropna())

# 6. Tableau comparatif
comparison = pd.DataFrame({
    'Variable': high_skew,
    'Skewness AVANT': [skewness_initial[col] for col in high_skew],
    'Skewness APRÈS': [skewness_after[col] for col in high_skew],
    'Amélioration': [skewness_initial[col] - skewness_after[col] for col in high_skew]
})
print("\nTableau comparatif :")
print(comparison)


# In[76]:


# Cellule autonome pour 3.4 : transformations log et graphique
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
import os

# Charger le dataset
df = pd.read_csv('breast_cancer_cleaned.csv')
# Créer le dossier graphs
os.makedirs('graphs', exist_ok=True)

# Sélectionner les colonnes numériques (sauf la cible diagnosis)
exclude = ['diagnosis']
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if '_log' not in col and col not in exclude]

# Calcul du skewness initial
skewness_initial = {}
for col in numeric_cols:
    skewness_initial[col] = skew(df[col].dropna())

# Identifier les variables avec |skew| > 2
high_skew = [col for col in numeric_cols if abs(skewness_initial[col]) > 2]
print(f"Variables avec |skew| > 2 : {high_skew}")

# Créer les colonnes log si nécessaire
for col in high_skew:
    log_col = col + '_log'
    if log_col not in df.columns:
        df[log_col] = np.log1p(df[col])

# Sélectionner une variable pour le graphique (par exemple la première de high_skew)
if high_skew:
    var = high_skew[0]
    log_var = var + '_log'
else:
    # Fallback : utiliser une variable manuelle (par exemple area_mean si elle existe)
    var = 'area_mean'
    log_var = var + '_log'
    if log_var not in df.columns:
        df[log_var] = np.log1p(df[var])

print(f"Génération du graphique pour : {var}")

# Graphique côte à côte
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(df[var].dropna(), bins=30, edgecolor='black', alpha=0.7)
ax1.set_title(f'Distribution originale : {var}')
ax1.set_xlabel(var)
ax1.set_ylabel('Fréquence')

ax2.hist(df[log_var].dropna(), bins=30, edgecolor='black', alpha=0.7, color='orange')
ax2.set_title(f'Distribution après log : {log_var}')
ax2.set_xlabel(log_var)
ax2.set_ylabel('Fréquence')

plt.tight_layout()
plt.savefig(f'graphs/log_{var}.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Graphique sauvegardé : graphs/log_{var}.png")


# Question 3.5 – Bilan du Feature Engineering
# 

# In[77]:


# Cellule 3.5.1 : Compter les nouvelles variables créées
original_features = set(pd.read_csv('breast_cancer_cleaned.csv').columns)  # avant FE
current_features = set(df.columns)
new_features = current_features - original_features
print(f"Nouvelles variables créées : {len(new_features)}")
print("Liste :", list(new_features))

# Dimensions avant / après
print(f"Dimensions avant FE : {pd.read_csv('breast_cancer_cleaned.csv').shape}")
print(f"Dimensions après FE : {df.shape}")


# PARTIE 4 – ANALYSE EXPLORATOIRE AVANCÉE (EDA)

# Question 4.1 – Matrice de corrélation

# In[78]:


# Cellule 4.1 - Matrice de corrélation et paires avec |r| > 0.9
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement du dataset (adapter le nom si nécessaire)
df = pd.read_csv('breast_cancer_featured.csv')

# Sélection des colonnes numériques
num_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[num_cols].corr()

# --- Heatmap triangulaire ---
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
plt.figure(figsize=(14,12))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdYlGn', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink":0.8})
plt.title('Matrice de corrélation triangulaire')
plt.savefig('graphs/corr_matrix_triangular.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Liste des paires avec |r| > 0.9 ---
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.9:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], r))

print("\nPaires de variables avec |corrélation| > 0.9 :")
print(f"{'Variable 1':<30} {'Variable 2':<30} {'r':>8}")
print("-" * 70)
for v1, v2, r in high_corr_pairs:
    print(f"{v1:<30} {v2:<30} {r:>8.3f}")

# Sauvegarde des paires (optionnel)
with open('graphs/high_corr_pairs.txt', 'w') as f:
    f.write("Paires avec |r| > 0.9 :\n")
    for v1, v2, r in high_corr_pairs:
        f.write(f"{v1} & {v2} : {r:.3f}\n")


# Question 4.2 – Corrélations avec la variable cible

# In[79]:


# Cellule 4.2.1 : Corrélations avec diagnosis (transformée en numérique : M=1, B=0)

# Création de la cible numérique
df['target'] = (df['diagnosis'] == 'M').astype(int)

# Sélection des colonnes numériques (exclure 'target' pour éviter l'autocorrélation)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [col for col in num_cols if col != 'target']

# Calcul des corrélations avec la cible
target_corr = df[num_cols].corrwith(df['target']).sort_values(ascending=False)

# Affichage des 10 premières
print("Top 10 des variables les plus corrélées avec la cible :")
print(target_corr.head(10))

# Heatmap verticale triée (version corrigée sans warning)
plt.figure(figsize=(10, 8))
sns.barplot(x=target_corr.values, y=target_corr.index, 
            hue=target_corr.index, palette='coolwarm', legend=False)
plt.title('Corrélation avec la cible (malin=1, bénin=0)')
plt.xlabel('Corrélation')
plt.tight_layout()
plt.savefig('graphs/target_corr_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()


# Question 4.3 – Distributions univariées

# In[80]:


# Cellule 4.3.1 : Grille de sous-graphiques pour les 6 variables les plus importantes
top_vars = target_corr.head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, var in enumerate(top_vars):
    sns.histplot(data=df, x=var, hue='diagnosis', kde=True, bins=30, ax=axes[i], alpha=0.6)
    axes[i].set_title(var)

plt.tight_layout()
plt.savefig('graphs/univariate_distributions.png', dpi=300, bbox_inches='tight')
plt.show()


# In[81]:


# Cellule 4.3.2 : Tableau récapitulatif (forme, outliers, interprétation)
summary = []
for var in top_vars:
    skew_val = skew(df[var])
    # Détection simple d'outliers (IQR)
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[var] < Q1 - 1.5*IQR) | (df[var] > Q3 + 1.5*IQR)).sum()
    shape = "asymétrique droite" if skew_val > 0.5 else ("asymétrique gauche" if skew_val < -0.5 else "symétrique")
    interp = "plus élevée pour les tumeurs malignes" if target_corr[var] > 0 else "plus élevée pour les bénignes"
    summary.append([var, shape, outliers, interp])

summary_df = pd.DataFrame(summary, columns=['Variable', 'Forme', 'Outliers (n)', 'Interprétation'])
print(summary_df)


# Question 4.4 – Analyse bivariée (scatter plots)

# In[82]:


# Cellule 4.4.1 : 4 scatter plots colorés par classe
# Choisir 4 paires de variables intéressantes (top corrélées avec la cible mais pas trop entre elles)

pairs = [('area_worst', 'perimeter_worst'),
         ('concavity_mean', 'concave points_mean'),   # ← espace ici
         ('radius_mean', 'texture_mean'),
         ('area_se', 'smoothness_se')]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (x, y) in enumerate(pairs):
    sns.scatterplot(data=df, x=x, y=y, hue='diagnosis', alpha=0.5, ax=axes[i], palette='Set1')
    axes[i].set_title(f'{x} vs {y}')

plt.tight_layout()
plt.savefig('graphs/scatter_plots_bivariate.png', dpi=300, bbox_inches='tight')
plt.show()


# In[83]:


# Cellule 4.4.2 (Bonus) : Pairplot des 5 variables les plus corrélées (attention, peut être long)
import warnings
warnings.filterwarnings('ignore')
top5 = target_corr.head(5).index.tolist()
sns.pairplot(df, vars=top5, hue='diagnosis', diag_kind='kde', plot_kws={'alpha':0.5})
plt.savefig('graphs/pairplot_top5.png', dpi=300, bbox_inches='tight')
plt.show()


# In[84]:


# Cellule finale : Exporter le dataframe avec les nouvelles features
df.to_csv('breast_cancer_featured.csv', index=False)
print("Dataset avec feature engineering sauvegardé sous 'breast_cancer_featured.csv'")


# IMPORTS

# In[85]:


# Scikit-learn pour preprocessing, sélection, modélisation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, chi2, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import warnings
warnings.filterwarnings('ignore')


# In[86]:


# Cellule C2 : Chargement du dataset enrichi (produit par membre B)
df = pd.read_csv('breast_cancer_featured.csv')
# Normalisation des noms de colonnes (remplacer espaces par underscores si nécessaire)
df.columns = df.columns.str.replace(' ', '_')
print("Dimensions :", df.shape)
print("Colonnes :", df.columns.tolist())
df.head()


# PARTIE 5 – PRÉPARATION POUR LE MACHINE LEARNING

# Question 5.1 – Sélection des features (8–15 features pertinentes)

# In[108]:


# Cellule C5.1 - Sélection des features avec barplot des corrélations
import matplotlib.pyplot as plt
import seaborn as sns

# Encodage de la cible (si pas déjà fait)
df['target'] = (df['diagnosis'] == 'M').astype(int)

# Calcul des corrélations absolues avec la target
corr_target = df.drop(columns=['diagnosis', 'target']).corrwith(df['target']).abs().sort_values(ascending=False)

# Affichage des 15 meilleures corrélations
print("Top 15 corrélations avec la cible :")
print(corr_target.head(15))

# Diagramme en barres (barplot) des 15 premières
plt.figure(figsize=(12, 6))
sns.barplot(x=corr_target.head(15).values, y=corr_target.head(15).index, palette='viridis')
plt.title('Top 15 corrélations absolues avec la variable cible (diagnosis)', fontsize=14)
plt.xlabel('Corrélation absolue')
plt.tight_layout()
plt.savefig('graphs/top_correlations.png', dpi=300, bbox_inches='tight')
plt.show()

# Sélection manuelle des features (8 à 15, avec |r|>0.1, sans multicolinéarité, ≥3 features créées)
selected_features = [
    'area_worst',           # corrélation très élevée
    'concave_points_worst',
    'perimeter_worst',
    'radius_worst',
    'concavity_mean',
    'area_trend',           # nouvelle feature (créée)
    'n_severe_features',    # nouvelle feature
    'ratio_perimeter_area', # nouvelle feature
    'area_mean_log',        # transformation log (si existe, sinon utiliser area_mean)
    'perimeter_mean',
    'texture_worst',
    'smoothness_worst',
    'concave_points_se'
]
# Filtrer les colonnes existantes
selected_features = [f for f in selected_features if f in df.columns]
print(f"\nNombre de features sélectionnées : {len(selected_features)}")
print("Features retenues :", selected_features)

# Justification (texte à mettre dans le rapport)
print("\n=== Justification ===")
print("- Forte corrélation avec la cible (>0.6) pour area_worst, concave_points_worst, etc.")
print("- Exclusion des features redondantes (area_mean très corrélé (0.99) à perimeter_mean).")
print("- Au moins 3 nouvelles features créées (area_trend, n_severe_features, ratio_perimeter_area) sont incluses.")
print("- Élimination des variables avec faible corrélation (<0.1) comme fractal_dimension_mean.")


# Question 5.2 – Split Train/Test stratifié

# In[88]:


# Cellule C5.2 : Split 80/20 avec stratification
X = df[selected_features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train : {X_train.shape[0]} lignes, Test : {X_test.shape[0]} lignes")
print(f"Distribution dans train : classe 1 = {y_train.mean()*100:.1f}%")
print(f"Distribution dans test  : classe 1 = {y_test.mean()*100:.1f}%")
# Sans stratification, le déséquilibre pourrait être plus marqué ou mal réparti


# Question 5.3 – Normalisation (StandardScaler)

# In[89]:


# Cellule C5.3 : Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Vérification
print(f"Moyenne (train) : {X_train_scaled.mean(axis=0).mean():.2e}")
print(f"Écart-type (train) : {X_train_scaled.std(axis=0).mean():.2f}")

# Boxplot avant/après pour une variable
fig, axes = plt.subplots(1,2, figsize=(12,5))
sns.boxplot(x=X_train['area_worst'], ax=axes[0])
axes[0].set_title('area_worst - original')
sns.boxplot(x=X_train_scaled[:, X_train.columns.get_loc('area_worst')], ax=axes[1])
axes[1].set_title('area_worst - normalisé')
plt.tight_layout()
plt.savefig('graphs/normalisation_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()


# Question 5.4 – Gestion du déséquilibre (SMOTE)

# In[90]:


# Cellule C5.4 : SMOTE sur le train uniquement
print(f"Avant SMOTE - Classe 1 : {y_train.sum()} ({y_train.mean()*100:.1f}%)")
smote = SMOTE(random_state=42, sampling_strategy=0.8)  # 0.8 = ratio minoritaire/majoritaire
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"Après SMOTE - Classe 1 : {y_train_resampled.sum()} ({y_train_resampled.mean()*100:.1f}%)")
print(f"Nouveau ratio : {y_train_resampled.mean():.2f}")
# Attention : SMOTE ne doit jamais être appliqué au test set, car on doit évaluer sur des données réelles.


# Question 5.5 (Bonus) – Pipeline Scikit-learn

# In[91]:


# Cellule C5.5 (Bonus) : Pipeline complet (scaler → SMOTE → classifieur)
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42, sampling_strategy=0.8)),
    ('classifier', LogisticRegression())
])
# Ce pipeline sera utilisé plus tard.


# PARTIE 6 – SÉLECTION AVANCÉE DES FEATURES

# Question 6.1 – Méthodes Filtre (ANOVA, Chi², corrélation)

# In[92]:


# Cellule C6.1 : Filtres statistiques
# ANOVA (f_classif) pour variables continues
f_scores, p_values = f_classif(X_train_scaled, y_train)
anova_res = pd.DataFrame({'feature': selected_features, 'F_score': f_scores, 'p_value': p_values})
anova_res = anova_res.sort_values('p_value')
print("ANOVA - top features (p-value < 0.05) :")
print(anova_res[anova_res.p_value < 0.05])

# Chi² nécessite des données non-négatives, on peut l'utiliser après normalisation positive
# On le saute ici car toutes nos variables sont continues.

# Barplot des scores d'importance (F-score)
plt.figure(figsize=(10,6))
sns.barplot(data=anova_res, x='F_score', y='feature', hue='feature', palette='viridis', legend=False)
plt.title('F-score (ANOVA) par feature')
plt.tight_layout()
plt.savefig('graphs/anova_f_scores.png', dpi=300, bbox_inches='tight')
plt.show()

# Élimination des features avec p-value > 0.05
features_filtre = anova_res[anova_res.p_value <= 0.05]['feature'].tolist()
print(f"Features conservées après filtre : {len(features_filtre)}")
print(features_filtre)


# Question 6.2 – Méthodes Wrapper – RFE Séquentielle

# In[93]:


# Cellule C6.2 : RFE avec RandomForest
estimator = RandomForestClassifier(n_estimators=50, random_state=42)
scores_rfe = []
n_features_list = [5, 8, 10, 12]

for n in n_features_list:
    rfe = RFE(estimator=estimator, n_features_to_select=n)
    rfe.fit(X_train_scaled, y_train)
    selected = X_train.columns[rfe.support_].tolist()
    # Évaluation rapide par validation croisée (F1)
    from sklearn.model_selection import cross_val_score
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    score = cross_val_score(model, X_train_scaled[:, rfe.support_], y_train, cv=5, scoring='f1').mean()
    scores_rfe.append(score)
    print(f"n={n} : F1 moyen = {score:.4f}, features = {selected}")

# Courbe du score en fonction du nombre de features
plt.figure(figsize=(8,5))
plt.plot(n_features_list, scores_rfe, marker='o')
plt.xlabel('Nombre de features')
plt.ylabel('F1-score (CV)')
plt.title('Performance RFE en fonction du nombre de features')
plt.grid(True)
plt.savefig('graphs/rfe_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# Nombre optimal (ici 10 donne le meilleur score)
best_n = n_features_list[np.argmax(scores_rfe)]
print(f"Nombre optimal de features : {best_n}")


# Question 6.3 – Méthodes Embedded – Importance par RandomForest

# In[94]:


# Cellule C6.3 : Importance des features avec RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
importances = pd.Series(rf.feature_importances_, index=selected_features).sort_values(ascending=False)

plt.figure(figsize=(10,6))
importances.plot(kind='bar')
plt.title('Feature importances - RandomForest')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('graphs/feature_importances_rf.png', dpi=300, bbox_inches='tight')
plt.show()

# SelectFromModel avec seuil 'mean'
selector = SelectFromModel(rf, threshold='mean')
X_train_sel = selector.fit_transform(X_train_scaled, y_train)
selected_embedded = [selected_features[i] for i in range(len(selected_features)) if selector.get_support()[i]]
print(f"Features au-dessus de la moyenne : {len(selected_embedded)}")
print(selected_embedded)

# Comparaison avec GradientBoosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train_scaled, y_train)
imp_gb = pd.Series(gb.feature_importances_, index=selected_features).sort_values(ascending=False)
print("Top 5 features (GradientBoosting) :")
print(imp_gb.head(5))


# Question 6.4 – Analyse par Permutation

# In[95]:


# Cellule C6.4 corrigée (version robuste avec barres d'erreur manuelles)
from sklearn.inspection import permutation_importance

result = permutation_importance(rf, X_test_scaled, y_test, n_repeats=10, random_state=42)
perm_imp = pd.DataFrame({
    'feature': selected_features,
    'mean': result.importances_mean,
    'std': result.importances_std
}).sort_values('mean', ascending=False)

plt.figure(figsize=(10, 6))
# Barplot sans xerr
ax = sns.barplot(data=perm_imp, x='mean', y='feature', hue='feature', 
                 palette='rocket', legend=False)
# Ajout manuel des barres d'erreur (écart-type)
for i, (mean, std) in enumerate(zip(perm_imp['mean'], perm_imp['std'])):
    ax.errorbar(mean, i, xerr=std, fmt='none', color='black', capsize=3, elinewidth=1)

plt.title('Permutation importance (variabilité)')
plt.xlabel('Importance moyenne')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('graphs/permutation_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("Features les plus robustes (mean élevé) :")
print(perm_imp.head(5))
print("\nFeatures instables (std élevé) :")
print(perm_imp.sort_values('std', ascending=False).head(3))


# Question 6.5 – Synthèse – Jeu de Features Final

# In[96]:


# Cellule C6.5 : Croisement des résultats
filtre_set = set(features_filtre)
wrapper_set = set(X_train.columns[rfe.support_])   # avec n optimal
embedded_set = set(selected_embedded)

# Intersection d'au moins deux méthodes
final_features_set = (filtre_set & wrapper_set) | (filtre_set & embedded_set) | (wrapper_set & embedded_set)
final_features = list(final_features_set)
print(f"Nombre de features finales : {len(final_features)}")
print("Features retenues :", final_features)

# Préparation des jeux finaux
X_final_train = X_train_scaled[:, [selected_features.index(f) for f in final_features]]
X_final_test = X_test_scaled[:, [selected_features.index(f) for f in final_features]]

# Justification : chaque feature est discriminante (filtre), importante pour le modèle (embedded),
# et/ou sélectionnée par RFE. On exclut les features redondantes ou peu informatives.


# PARTIE 7 – MODÉLISATION

# Question 7.1 – Choix des modèles

# In[103]:


# Cellule C7.1 - Entraînement des modèles de base (version corrigée)
import time
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# 1. Définition des modèles (hyperparamètres par défaut, sauf random_state)
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# 2. Application de SMOTE sur le jeu final (X_final_train, y_train)
#    (Ne pas utiliser y_train_resampled qui a 513 lignes)
smote = SMOTE(random_state=42)
X_final_train_resampled, y_final_train_resampled = smote.fit_resample(X_final_train, y_train)

print(f"X_final_train original shape : {X_final_train.shape}")
print(f"X_final_train_resampled shape : {X_final_train_resampled.shape}")
print(f"y_final_train_resampled shape : {y_final_train_resampled.shape}\n")

# 3. Entraînement avec mesure du temps et sauvegarde
training_times = {}

for name, model in models.items():
    start = time.time()
    model.fit(X_final_train_resampled, y_final_train_resampled)
    end = time.time()
    elapsed = end - start
    training_times[name] = elapsed

    # Sauvegarde du modèle avec joblib (dans le dossier courant)
    filename = f"model_{name.replace(' ', '_')}.pkl"
    joblib.dump(model, filename)
    print(f"{name:20} -> {elapsed:.3f} secondes (modèle sauvegardé dans {filename})")

# 4. Tableau récapitulatif des temps
print("\n=== Temps d'entraînement ===")
times_df = pd.DataFrame(list(training_times.items()), columns=['Modèle', 'Temps (secondes)'])
times_df = times_df.sort_values('Temps (secondes)')
print(times_df.to_string(index=False))

# 5. Résultats attendus (à reporter dans le rapport)
fastest = times_df.iloc[0]['Modèle']
slowest = times_df.iloc[-1]['Modèle']
total_time = times_df['Temps (secondes)'].sum()

print(f"\nModèle le plus rapide : {fastest}")
print(f"Modèle le plus lent : {slowest}")
print(f"Temps total d'entraînement : {total_time:.3f} secondes")


# Question 7.2 – Entraînement et évaluation (métriques, ROC)
# 

# In[104]:


# Cellule C7.2 - Évaluation et comparaison des modèles
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import pandas as pd
import matplotlib.pyplot as plt

# 1. Calcul des métriques pour chaque modèle
results = {}
for name, model in models.items():
    y_pred = model.predict(X_final_test)
    y_prob = model.predict_proba(X_final_test)[:, 1]
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_prob)
    }

# 2. Création du DataFrame et ajout du rang (selon AUC-ROC décroissant)
results_df = pd.DataFrame(results).T.round(4)
results_df['Rang'] = results_df['AUC-ROC'].rank(ascending=False, method='min').astype(int)

# Affichage du tableau
print("=== Tableau comparatif des performances ===")
print(results_df.to_string())

# 3. Graphique en barres groupées (toutes métriques)
ax = results_df.drop('Rang', axis=1).plot(kind='bar', figsize=(12,6), colormap='viridis')
plt.title('Comparaison des modèles', fontsize=14)
plt.ylabel('Score', fontsize=12)
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig('graphs/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Courbes ROC superposées
plt.figure(figsize=(8,6))
for name, model in models.items():
    y_prob = model.predict_proba(X_final_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = results[name]['AUC-ROC']
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
plt.xlabel('Taux de faux positifs (FPR)', fontsize=12)
plt.ylabel('Taux de vrais positifs (TPR)', fontsize=12)
plt.title('Courbes ROC', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('graphs/roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# Optionnel : sauvegarder le tableau en CSV pour le rapport
results_df.to_csv('model_performances.csv')


# Question 7.3 – Matrices de confusion et analyse des erreurs

# In[106]:


# Cellule C7.3 - Matrices de confusion et analyse des erreurs (version complète)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# 1. Afficher les matrices de confusion pour chaque modèle
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for ax, (name, model) in zip(axes, models.items()):
    ConfusionMatrixDisplay.from_estimator(model, X_final_test, y_test, ax=ax, cmap='Blues')
    ax.set_title(name)
# Cacher le sixième sous-graphique s'il y a 5 modèles
if len(models) < len(axes):
    axes[-1].axis('off')
plt.tight_layout()
plt.savefig('graphs/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Calculer les faux positifs (FP) et faux négatifs (FN) pour chaque modèle
results_errors = {}
for name, model in models.items():
    y_pred = model.predict(X_final_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    results_errors[name] = {'FP': fp, 'FN': fn}
    print(f"{name}: FP = {fp}, FN = {fn}")

# 3. Identifier les meilleurs modèles
best_recall_model = min(results_errors, key=lambda x: results_errors[x]['FN'])
best_precision_model = min(results_errors, key=lambda x: results_errors[x]['FP'])


print(f"Modèle avec le moins de faux négatifs : {best_recall_model}")
print(f"Modèle avec le moins de faux positifs : {best_precision_model}")

# 4. Compromis Precision / Recall (contexte médical)
print("\nCompromis retenu :")
print("Dans le diagnostic du cancer du sein, un faux négatif (prédire 'bénin' alors que la tumeur est maligne) "
      "est extrêmement dangereux car il retarde le traitement. En revanche, un faux positif (prédire 'malin' pour une tumeur bénigne) "
      "entraîne seulement des examens complémentaires. Par conséquent, nous privilégions le **Recall** (sensibilité) "
      "même si cela augmente les faux positifs. Le modèle avec le meilleur recall est celui qui minimise les FN.")

# 5. Identifier les échantillons systématiquement mal classés par tous les modèles
# Récupérer les prédictions de tous les modèles
y_preds_all = np.array([model.predict(X_final_test) for model in models.values()])
# Vérifier pour chaque échantillon si tous les modèles se trompent (aucun prédit la bonne classe)
misclassified_by_all = np.all(y_preds_all != y_test.values.reshape(1, -1), axis=0)
misclassified_indices = np.where(misclassified_by_all)[0]

if len(misclassified_indices) > 0:
    print(f"\n{len(misclassified_indices)} échantillon(s) mal classé(s) par TOUS les modèles : indices {misclassified_indices}")
    # Option : afficher les vraies valeurs et les prédictions de chaque modèle
    for idx in misclassified_indices:
        print(f"\nÉchantillon {idx}: vraie classe = {y_test.iloc[idx]}")
        for name, model in models.items():
            pred = model.predict(X_final_test[idx].reshape(1, -1))[0]
            print(f"  {name} a prédit : {pred}")
else:
    print("\nAucun échantillon n'est mal classé par tous les modèles.")


# Question 7.4 – Validation croisée et GridSearch

# In[107]:


# Cellule C7.4 - Validation croisée et GridSearchCV 
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 1. Validation croisée 5-fold sur les 2 meilleurs modèles (Random Forest et Gradient Boosting)
best_models_names = ['Random Forest', 'Gradient Boosting']
for name in best_models_names:
    model = models[name]   # modèles déjà entraînés (ou on peut les récupérer)
    cv_f1 = cross_val_score(model, X_final_train_resampled, y_final_train_resampled, cv=5, scoring='f1')
    print(f"{name} - CV F1 moyen : {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")

# 2. GridSearchCV sur Random Forest (meilleur modèle)
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf,
                       cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_rf.fit(X_final_train_resampled, y_final_train_resampled)
print("\nMeilleurs hyperparamètres RF :", grid_rf.best_params_)

# 3. Comparaison avant/après tuning (Accuracy, F1, AUC sur le test set)
# Modèle par défaut (déjà dans models, mais on le récupère)
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_final_train_resampled, y_final_train_resampled)
y_pred_default = rf_default.predict(X_final_test)
y_prob_default = rf_default.predict_proba(X_final_test)[:,1]

# Modèle optimisé
rf_tuned = grid_rf.best_estimator_
y_pred_tuned = rf_tuned.predict(X_final_test)
y_prob_tuned = rf_tuned.predict_proba(X_final_test)[:,1]

print("\n=== Comparaison Random Forest (avant/après tuning) ===")
print(f"Accuracy avant : {accuracy_score(y_test, y_pred_default):.4f}, après : {accuracy_score(y_test, y_pred_tuned):.4f}")
print(f"F1 avant : {f1_score(y_test, y_pred_default):.4f}, après : {f1_score(y_test, y_pred_tuned):.4f}")
print(f"AUC avant : {roc_auc_score(y_test, y_prob_default):.4f}, après : {roc_auc_score(y_test, y_prob_tuned):.4f}")

# (Optionnel) GridSearch sur Gradient Boosting
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1]
}
grid_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb,
                       cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_gb.fit(X_final_train_resampled, y_final_train_resampled)
print("\nMeilleurs hyperparamètres GB :", grid_gb.best_params_)

# Comparaison pour Gradient Boosting
gb_default = models['Gradient Boosting']
y_pred_gb_default = gb_default.predict(X_final_test)
y_prob_gb_default = gb_default.predict_proba(X_final_test)[:,1]
gb_tuned = grid_gb.best_estimator_
y_pred_gb_tuned = gb_tuned.predict(X_final_test)
y_prob_gb_tuned = gb_tuned.predict_proba(X_final_test)[:,1]

print("\n=== Comparaison Gradient Boosting (avant/après tuning) ===")
print(f"Accuracy avant : {accuracy_score(y_test, y_pred_gb_default):.4f}, après : {accuracy_score(y_test, y_pred_gb_tuned):.4f}")
print(f"F1 avant : {f1_score(y_test, y_pred_gb_default):.4f}, après : {f1_score(y_test, y_pred_gb_tuned):.4f}")
print(f"AUC avant : {roc_auc_score(y_test, y_prob_gb_default):.4f}, après : {roc_auc_score(y_test, y_prob_gb_tuned):.4f}")


# Question 7.5 – Prévisions sur nouveaux exemples

# In[ ]:


# Cellule C7.5 

# 1. S'assurer que final_features existe (le jeu de features final)
# Si tu as oublié de le définir, prends les colonnes utilisées pour l'entraînement final :
if 'final_features' not in dir():
    final_features = X_final_train.columns.tolist()  # ou list(X.columns)

# 2. Créer les nouveaux exemples (seulement les features finales)
new_samples = pd.DataFrame({
    'area_worst': [500, 1200, 800, 2000, 300, 1500, 600, 100],
    'concave_points_worst': [0.05, 0.20, 0.10, 0.25, 0.02, 0.18, 0.07, 0.01],
    'perimeter_worst': [80, 130, 95, 160, 60, 140, 85, 50],
    'radius_worst': [12, 18, 14, 22, 10, 20, 13, 9],
    'concavity_mean': [0.03, 0.15, 0.08, 0.20, 0.01, 0.12, 0.05, 0.02],
    'area_trend': [50, 300, 150, 500, 30, 400, 100, 10],
    'n_severe_features': [1, 6, 3, 8, 0, 5, 2, 0],
    'ratio_perimeter_area': [0.16, 0.11, 0.12, 0.08, 0.20, 0.10, 0.14, 0.22],
    'area_mean_log': [6.0, 7.2, 6.5, 7.8, 5.5, 7.4, 6.2, 5.0],
    'perimeter_mean': [70, 110, 85, 140, 55, 120, 75, 45],
    'texture_worst': [18, 25, 20, 30, 15, 28, 19, 14],
    'smoothness_worst': [0.08, 0.15, 0.10, 0.18, 0.06, 0.14, 0.09, 0.05],
    'concave_points_se': [0.01, 0.04, 0.02, 0.05, 0.005, 0.03, 0.015, 0.002]
})

# 3. Ne garder que les colonnes final_features (vérifier qu'elles existent)
missing_cols = set(final_features) - set(new_samples.columns)
if missing_cols:
    print(f"Colonnes manquantes dans new_samples : {missing_cols}")
    # Ajouter des valeurs par défaut ou ajuster la liste
new_samples = new_samples[final_features]

# 4. Normalisation : il faut un scaler entraîné sur les mêmes features
# Option A : recréer un scaler sur X_final_train (plus propre)
from sklearn.preprocessing import StandardScaler
scaler_final = StandardScaler()
scaler_final.fit(X_final_train_resampled)  # ou X_final_train
new_scaled = scaler_final.transform(new_samples)

# Option B : utiliser le scaler existant mais sur les colonnes d'origine (moins fiable)
# new_scaled = scaler.transform(new_samples)   # si scaler a été fit sur X_train_scaled

# 5. Prédiction avec le meilleur modèle (après tuning)
best_model = grid_rf.best_estimator_
pred_class = best_model.predict(new_scaled)
pred_proba = best_model.predict_proba(new_scaled)[:, 1]

# 6. Affichage
results_new = pd.DataFrame({
    'Exemple': range(1,9),
    'Classe prédite': ['Malin' if p else 'Bénin' for p in pred_class],
    'Probabilité (%)': (pred_proba*100).round(1),
    'Confiance': ['Haute' if (p>0.8 or p<0.2) else 'Moyenne' for p in pred_proba]
})
print(results_new)


# Sauvegarde des résultats

# In[ ]:


# Sauvegarde du modèle et du scaler 
import joblib
joblib.dump(best_model, 'best_model_rf.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Modèle et scaler sauvegardés.")

