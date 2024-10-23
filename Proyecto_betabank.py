
# %% [markdown]
# # Introduccion
# Los clientes de Beta Bank se están yendo, cada mes, poco a poco. Los banqueros descubrieron que es más barato salvar a los clientes existentes que atraer nuevos.
# 
# Necesitamos predecir si un cliente dejará el banco pronto. Tenemos los datos sobre el comportamiento pasado de los clientes y la terminación de contratos con el banco.
# 
# Crearemos un modelo con el máximo valor F1 posible. Necesitamos un valor F1 de al menos 0.59. Verificaremos F1 para el conjunto de prueba. 
# 
# Además, debemos medir la métrica AUC-ROC y compararla con el valor F1

# %% [markdown]
# ## Inicializacion y carga de datos

# %%
#importar librerias y funciones
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix ,recall_score, precision_score, f1_score, roc_curve,roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# %%
#leer el data frame
df = pd.read_csv('datasets/Churn.csv')

# %%
# visualizar la estructura del data frame
df.info()
df.head()


# %% [markdown]
# ## Estandarizacion de datos
# Primero es claro que la columna Tenure cuenta con valores ausentes los cuales deben de ser tratados antes de continuar

# %%
print(100*df.isna().sum()/df.shape[0])

# %%
df = df.dropna(subset=['Tenure'])
print(100*df.isna().sum()/df.shape[0])


# %% [markdown]
# Ya con los Nan eliminados pasamos a la codificacion one hot y al escaldado de caracteristicas

# %% [markdown]
# ### Codificacion OHE y escalado de caracteristicas 

# %%
df_ohe=pd.get_dummies(df,drop_first=True)
print(df_ohe.head(3))

# %%
target = df_ohe['Exited']
features = df_ohe.drop('Exited', axis=1)

features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.25, random_state=12345)

numeric = ['CustomerId', 'CreditScore', 'Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']

scaler = StandardScaler()
scaler.fit(features_train[numeric])
features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])



# %%
print(features_train.shape)
print(features_valid.shape)


# %% [markdown]
# ## Examinar el equilibrio de clases

# %%
#Entrena el modelo sin tener en cuenta el desequilibrio
model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid) #< escribe el código aquí >

accuracy_valid = accuracy_score(target_valid, predicted_valid)

print(accuracy_valid)

# %% [markdown]
# ### Prueba de consistencia

# %%
class_frequency = df_ohe['Exited'].value_counts(normalize=True)
print(class_frequency)
class_frequency.plot(kind='bar')

# %% [markdown]
# Las estadisticas nos dicen que hasta ahora el 2% de los clientes han abandonado el banco y cancelado su contrato.

# %%
#Analizar las frecuencias de clase de las predicciones del árbol de decisión
predicted_valid = pd.Series(model.predict(features_valid))
class_frequency = predicted_valid.value_counts(normalize=True)
print(class_frequency)
class_frequency.plot(kind='bar')


# %% [markdown]
# Nuestro modelo de árbol de decisión produce proporciones de clases que son similares a las que se observan simplemente eligiendo la clase más común en los datos, lo cual se aprecia en la primera parte de la prueba de consistencia, donde analizamos la frecuencia de las clases en la variable objetivo. Para verificar si el modelo está realmente aprendiendo patrones útiles, es crucial compararlo con un modelo constante en una prueba de cordura.

# %%
target_pred_constant = pd.Series(0, index=target.index)

print(accuracy_score(target, target_pred_constant))

# %% [markdown]
# Las predicciones no son mejores que las de un modelo constante que siempre elige la clase más frecuente. Este resultado indica que el modelo no está capturando patrones significativos en los datos, lo cual es crucial para un desempeño fiable y muestra que hay un desequilibrio de clases en nuestro problema, lo que afecta cómo se entrena el modelo.

# %% [markdown]
# ### Evaluacion del modelo

# %%
print(confusion_matrix(target_valid,predicted_valid))

# %% [markdown]
# El modelo es bastante pesimista. A menudo ve respuestas negativas donde no debería.

# %%
print(recall_score(target_valid,predicted_valid))

# %% [markdown]
# Llamando a recall podemos ver que el modelo tiene una capacidad moderada para detectar casos positivos,por lo que el modelo tiene margen de mejora en la identificación de todos los casos positivos reales en los datos.

# %%
print(precision_score(target_valid,predicted_valid))

# %%
print(f1_score(target_valid,predicted_valid))

# %% [markdown]
# Recordemos que necesitamos de un valor F1 de al menos 0.59 por lo que es necesario mejorar el desequilibrio. 


# %% [markdown]
# ## Mejora de la calidad del modelo

# %%
#Usamos el peso de clases para ver como mejora el modelo
model = DecisionTreeClassifier(random_state=12345,class_weight='balanced')
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)
print('F1:', f1_score(target_valid, predicted_valid))

# %% [markdown]
# El modelo ha mejorado un poco pero aun no es suficiente, pasemos al sobremuestreo

# %% [markdown]
# ### Sobremuestreo y submuestreo

# %%
def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345
    )

    return features_upsampled, target_upsampled


features_upsampled, target_upsampled = upsample(
    features_train, target_train, 8
)

model = DecisionTreeClassifier(random_state=12345,class_weight='balanced')
model.fit(features_upsampled,target_upsampled)
predicted_valid = model.predict(features_valid)

print('F1:', f1_score(target_valid, predicted_valid))

# %%
def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)]
        + [features_ones]
    )
    target_downsampled = pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)]
        + [target_ones]
    )

    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345
    )

    return features_downsampled, target_downsampled


features_downsampled, target_downsampled = downsample(
    features_train, target_train, .6
)
model = DecisionTreeClassifier(random_state=12345,class_weight='balanced')
model.fit(features_upsampled,target_upsampled)
predicted_valid = model.predict(features_valid)

print('F1:', f1_score(target_valid, predicted_valid))


# %% [markdown]
# Aunque el modelo ha mejorado ligeramente, aun estamos lejos de nuestro objetivo, intentemos con otros tipo de modelos
# 
# ### Regresion logistica

# %%
#veamos que pasa con una regresion logistica en sobremuestreo
model_lr = LogisticRegression(random_state=12345, solver ='liblinear',class_weight='balanced')
model_lr.fit(features_upsampled,target_upsampled)
predicted_valid_lr = model_lr.predict(features_valid)

print('F1:', f1_score(target_valid, predicted_valid_lr))

# %%
#veamos que pasa con una regresion logistica en submuestreo
model_lr = LogisticRegression(random_state=12345, solver ='liblinear',class_weight='balanced')
model_lr.fit(features_downsampled,target_downsampled)
predicted_valid_lr = model_lr.predict(features_valid)

print('F1:', f1_score(target_valid, predicted_valid_lr))

# %% [markdown]
# La regresion logistica no es nuestra mejor opcion, nos ha empeorado los valores, pasemos a un bosque aleatorio

# %% [markdown]
# ### Bosque aleatorio

# %%
#sobremuestreo
model_rf = RandomForestClassifier(n_estimators=150, random_state=12345, class_weight='balanced',min_samples_leaf=3)
model_rf.fit(features_upsampled,target_upsampled)
predicted_valid_rf = model_rf.predict(features_valid)

print('F1:', f1_score(target_valid, predicted_valid_rf))

# %%
#submuestreo
model_rf = RandomForestClassifier(n_estimators=150, random_state=12345, class_weight='balanced')
model_rf.fit(features_downsampled,target_downsampled)
predicted_valid_rf = model_rf.predict(features_valid)

print('F1:', f1_score(target_valid, predicted_valid_rf))


# %% [markdown]
# Perfecto, parece que nuestra mejor opcion es tener un bosque aleatorio con sobremuestreo, clases balanceadas y estimadores de 150, con un valor de .06063 hemos alcanzado nuestro objetivo .059. Por ultimo pasemos a nuestra curva ROC.

# %% [markdown]
# ## Curva Roc

# %%
#primero obtenemos las probabilidades de nuestro mejor modelo, el cual fue el randomforest
probabilities_valid = model_rf.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]
#obtenemos curva ROC y la trazamos
fpr, tpr, thresholds = roc_curve(target_valid,probabilities_one_valid)

plt.figure()
plt.plot(fpr,tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Tasa de falsos positivos')
plt.ylabel('Tasa de verdaderos positivos')
plt.title('Curva ROC')
plt.show()

# %%
#por ultimo obtenemos el valor auc_roc
auc_roc = roc_auc_score(target_valid,probabilities_one_valid)
print(auc_roc)

# %% [markdown]
# Podemos ver que el resultado es mejor que el modelo aleatorio, pero todavía está lejos de ser perfecto ya que le mejor escenario posible es que AUC-ROC sea igual a 1, sin embargo para este proyecto podemos concluir que hemos logrado nuestro objetivo.
# Hemos trabajado con diferentes modelos y es importante denotar que nunca hay que quedarse con una solo opcion y hay que buscar otras soluciones. No hay que olvidar el revisar los datos de antemano para asegurarnos que podemos proceder al modelado ya que muchas veces necesitan ser procesados de antemano. 




