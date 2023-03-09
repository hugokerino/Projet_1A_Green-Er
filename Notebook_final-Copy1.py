#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy as sp
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt


def loadData(fileName):
    return pd.read_csv(fileName, index_col=0, parse_dates=True, dtype='float32')

data_folder = Path('.')

csvFile = data_folder /'conso_globale.csv'
df1 = loadData(csvFile)

csvFile = data_folder /'weather.csv'
df2 = loadData(csvFile)

csvFile = data_folder /'calendar.csv'
df3 = loadData(csvFile)


# In[2]:


#Chargement et remplissage avec bfill des différentes données
temp = df1['Temperature'].fillna(method='bfill')
conso= df1['Global Consumption'].fillna(method='bfill')

ghi = df2['GHI'].fillna(method='bfill')
dni = df2['DNI'].fillna(method='bfill')
dhi = df2['DHI'].fillna(method='bfill')

presence = df3['Presence'].fillna(method='bfill')
day = df3['Day of week'].fillna(method='bfill')


conso = conso.asfreq('H',method='bfill')
temp = temp.asfreq('H',method ='bfill')
ghi = ghi.asfreq('H',method ='bfill')
dni = dni.asfreq('H',method ='bfill')
dhi = dhi.asfreq('H',method ='bfill')


# In[3]:


fig, ax = plt.subplots(figsize = (15,5))
ax.set_ylabel('Global Consumption (kWh)')
ax.plot(conso)


# In[4]:


presence_true = np.zeros(len(temp))

for i in range(presence_true.shape[0]):
    presence_true[i] = presence[i//24]
    
conso = conso.asfreq('H',method='bfill')
temp = temp.asfreq('H',method ='bfill')
ghi = ghi.asfreq('H',method ='bfill')
dni = dni.asfreq('H',method ='bfill')
dhi = dhi.asfreq('H',method ='bfill')


# In[5]:


#Option 2: Include ONLY the daily past values at the same hour than the target

nsample = len(conso)
history = 7
forecast = 1 # one day
offset = forecast*24+ (history-1)*24
yh_target = conso[offset:]
xh_temp = temp[(history-1)*24:nsample-forecast*24]


# Take past consumption values per day (same hour)
Xh = np.zeros( shape=(nsample-offset,history) ) # past values 
for h in range(nsample-offset):
    # Take past values
    Xh[h,:] = conso[h:h+history*24:24]
    
# Flip the predictor variables to get the current value as first column
Xh = np.fliplr(Xh)

# Add current temperature as a feature (cat at the last column of X)
Xh = np.concatenate( (Xh, xh_temp.to_numpy().reshape(-1, 1) ) ,axis = 1  )



Yh_target = np.zeros(len(yh_target)) #Transform panda data to np array
for i in range(len(yh_target)):
    Yh_target[i]= yh_target[i]


# In[6]:


degres = [1,2,3]
alpha_lasso = [0,0.01,0.05,0.1,0.2,0.3,0.5,1] 

inputs = Xh
targets = Yh_target

from sklearn.model_selection import KFold

num_fold = 6

kfold = KFold(n_splits = num_fold,shuffle= False)
fold_no = 1


mse_scores = np.zeros([len(degres),len(alpha_lasso),num_fold])
r2_scores = np.zeros([len(degres),len(alpha_lasso),num_fold])


# In[7]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


# In[8]:



k = 0

for train, test in kfold.split(inputs,targets):
    
    for i in range(len(degres)):

        for j in range(len(alpha_lasso)):

            poly = PolynomialFeatures(degree=degres[i],include_bias=False)


            polyf = poly.fit_transform(inputs[train])

            poly_reg_model = Lasso(alpha = alpha_lasso[j])
            poly_reg_model.fit(polyf,targets[train])

            polyf_test = poly.fit_transform(inputs[test])
            y_predicted = poly_reg_model.predict(polyf_test)

            mse = mean_squared_error(targets[test],y_predicted)
            r2 = r2_score(targets[test],y_predicted)

            mse_scores[i,j,k] = mse
            r2_scores[i,j,k] = r2

            print("Results fold",k," for degree = ",degres[i]," alpha = ",alpha_lasso[j],":")
            print(mse_scores[i,j,k])
            print(r2_scores[i,j,k])
            print("\n")
    k=k+1


# In[9]:


#determination des meilleurs paramètres

mse_scores_mean = np.zeros([len(degres),len(alpha_lasso)])
r2_scores_mean = np.zeros([len(degres),len(alpha_lasso)])

for i in range(len(degres)):
    for j in range(len(alpha_lasso)):
        Sum_r2 = 0
        Sum_mse = 0
        for k in range(num_fold):
            Sum_r2 = Sum_r2 + r2_scores[i,j,k]
            Sum_mse = Sum_mse + mse_scores[i,j,k]
            
        Moy_r2 = Sum_r2/num_fold
        r2_scores_mean[i,j] = Moy_r2
        
        Moy_mse = Sum_mse/num_fold
        mse_scores_mean[i,j] = Moy_mse


min_mse = np.min(mse_scores_mean)
max_r2 = np.max(r2_scores_mean)

ind_mse = np.where(mse_scores_mean == min_mse)
ind_r2 = np.where(r2_scores_mean == max_r2)


print("Meilleur parametre mse:")
print("degre = ",degres[ind_mse[0][0]]," alpha = ",alpha_lasso[ind_mse[1][0]])
print('\n')
print("Meilleur parametre r2:")
print("degre = ",degres[ind_r2[0][0]]," alpha = ",alpha_lasso[ind_r2[1][0]])


# In[10]:


#Algo with best parameters
split = int(8*Yh_target.shape[0]/10) #test sur 7 mois pour 8/10

X_train2,Y_train2 = Xh[(Yh_target.shape[0]-split):,:],Yh_target[(Yh_target.shape[0]-split):]
X_test2, Y_test2 = Xh[:(Yh_target.shape[0]-split),:],Yh_target[:(Yh_target.shape[0]-split)]

poly = PolynomialFeatures(degree=degres[ind_r2[0][0]],include_bias=False)

polyf = poly.fit_transform(X_train2)

poly_reg_model = Lasso(alpha = alpha_lasso[ind_r2[1][0]])
poly_reg_model.fit(polyf,Y_train2)

polyf_test = poly.fit_transform(X_test2)
y_predicted = poly_reg_model.predict(polyf_test)

mse = mean_squared_error(Y_test2,y_predicted)
r2 = r2_score(Y_test2,y_predicted)

print("Erreur de test :")
print("mse = ",mse)
print("r2 = ",r2)


# In[11]:


#Plot
fig, ax = plt.subplots(figsize = (15,5))
ax.plot(Y_test2,label='target')
ax.plot(y_predicted,label='predicted')
ax.set_ylabel('Global Consumption (kWh)')
ax.set_xlabel('Hours')
ax.legend()


# In[12]:


#zoom
fig, ax = plt.subplots(figsize = (15,5))
ax.plot(Y_test2[3000:3350],label='Target')
ax.plot(y_predicted[3000:3350],label='Predicted')
ax.set_ylabel('Global Consumption (kWh)')
ax.set_xlabel('Hours')
ax.legend()


# In[13]:


#plt.plot(alpha_lasso,r2_scores_mean[3,:],label='Degree = 4')
plt.plot(alpha_lasso,r2_scores_mean[2,:],label='Degree = 3')
plt.plot(alpha_lasso,r2_scores_mean[1,:],label='Degree = 2')
plt.plot(alpha_lasso,r2_scores_mean[0,:],label='Degree = 1')
plt.xlabel('Alpha')
plt.ylabel('R2 scores')
plt.title('R2 scores for polynomial regression lasso regularization')
plt.legend()


# In[14]:


plt.plot(degres,r2_scores_mean[:,1],label='alpha=0.01')
plt.plot(degres,r2_scores_mean[:,3],label='alpha=0.1')
plt.plot(degres,r2_scores_mean[:,7],label='alpha=1')
plt.xlabel('Degree')
plt.ylabel('R2 scores')
plt.title('R2 scores for polynomial regression lasso regularization')
plt.legend()


# In[15]:


coeff = poly_reg_model.coef_
coeff_nul = coeff.shape[0]-np.count_nonzero(coeff)
max_coeff =max(coeff)
indice_coef_max = np.where(coeff == max_coeff)

print(round(100*(coeff_nul/coeff.shape[0])),"% of coefficients deleted (",coeff_nul,"on ",coeff.shape[0],")")
print("Coefficient max =",max_coeff," correspond à l'indice ",indice_coef_max)


# Ajout des nouvelles entrées

# In[16]:



xh_ghi = ghi[(history-1)*24:nsample-forecast*24]
xh_dni = dni[(history-1)*24:nsample-forecast*24]
xh_dhi = dhi[(history-1)*24:nsample-forecast*24]
xh_presence = presence_true[(history-1)*24:nsample-forecast*24]

Xh = np.concatenate( (Xh, xh_ghi.to_numpy().reshape(-1, 1) ) ,axis = 1  )
Xh = np.concatenate( (Xh, xh_dni.to_numpy().reshape(-1, 1) ) ,axis = 1  )
Xh = np.concatenate( (Xh, xh_dhi.to_numpy().reshape(-1, 1) ) ,axis = 1  )
Xh = np.concatenate( (Xh, xh_presence.reshape(-1, 1) ) ,axis = 1  )


# In[17]:


inputs = Xh
targets = Yh_target

fold_no = 1


# In[18]:


k = 0

for train, test in kfold.split(inputs,targets):
    
    for i in range(len(degres)):

        for j in range(len(alpha_lasso)):

            poly = PolynomialFeatures(degree=degres[i],include_bias=False)


            polyf = poly.fit_transform(inputs[train])

            poly_reg_model = Lasso(alpha = alpha_lasso[j])
            poly_reg_model.fit(polyf,targets[train])

            polyf_test = poly.fit_transform(inputs[test])
            y_predicted = poly_reg_model.predict(polyf_test)

            mse = mean_squared_error(targets[test],y_predicted)
            r2 = r2_score(targets[test],y_predicted)

            mse_scores[i,j,k] = mse
            r2_scores[i,j,k] = r2

            print("Results fold",k," for degree = ",degres[i]," alpha = ",alpha_lasso[j],":")
            print(mse_scores[i,j,k])
            print(r2_scores[i,j,k])
            print("\n")
    k=k+1
  


# In[19]:


#determination des meilleurs paramètres

for i in range(len(degres)):
    for j in range(len(alpha_lasso)):
        Sum_r2 = 0
        Sum_mse = 0
        for k in range(num_fold):
            Sum_r2 = Sum_r2 + r2_scores[i,j,k]
            Sum_mse = Sum_mse + mse_scores[i,j,k]
            
        Moy_r2 = Sum_r2/num_fold
        r2_scores_mean[i,j] = Moy_r2
        
        Moy_mse = Sum_mse/num_fold
        mse_scores_mean[i,j] = Moy_mse
        
        
min_mse = np.min(mse_scores_mean)
max_r2 = np.max(r2_scores_mean)

ind_mse = np.where(mse_scores_mean == min_mse)
ind_r2 = np.where(r2_scores_mean == max_r2)


print("Meilleur parametre mse:")
print("degre = ",degres[ind_mse[0][0]]," alpha = ",alpha_lasso[ind_mse[1][0]])
print('\n')
print("Meilleur parametre r2:")
print("degre = ",degres[ind_r2[0][0]]," alpha = ",alpha_lasso[ind_r2[1][0]])


# In[34]:


split = int(8*Yh_target.shape[0]/10) #test sur 7 mois pour 8/10

X_train2,Y_train2 = Xh[(Yh_target.shape[0]-split):,:],Yh_target[(Yh_target.shape[0]-split):]
X_test2, Y_test2 = Xh[:(Yh_target.shape[0]-split),:],Yh_target[:(Yh_target.shape[0]-split)]

poly = PolynomialFeatures(degree=degres[ind_r2[0][0]],include_bias=False)

polyf = poly.fit_transform(X_train2)

poly_reg_model = Lasso(alpha = alpha_lasso[ind_r2[1][0]])
poly_reg_model.fit(polyf,Y_train2)

polyf_test = poly.fit_transform(X_test2)
y_predicted = poly_reg_model.predict(polyf_test)

mse = mean_squared_error(Y_test2,y_predicted)
r2 = r2_score(Y_test2,y_predicted)

print("mse = ",mse)
print("r2 = 0 ",r2)


# In[35]:


coeff = poly_reg_model.coef_
coeff_nul = coeff.shape[0]-np.count_nonzero(coeff)
max_coeff =max(coeff)
indice_coef_max = np.where(coeff == max_coeff)

print(round(100*(coeff_nul/coeff.shape[0])),"% of coefficients deleted (",coeff_nul,"on ",coeff.shape[0],")")
print("Coefficient max =",max_coeff," correspond à l'indice ",indice_coef_max)


# In[ ]:




