#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
import pandas as pd

df = pd.read_csv('E:\\my_file_revised - Copy.csv')


# In[51]:


df.fillna(0,inplace=True)


# In[52]:


# split data into X and y
X = df.iloc[:,0:104]
y = df.iloc[:,-1]
y = (y == 1.0).astype(int)


# In[53]:


X_encoded = pd.get_dummies(X, columns =['ICU_transferred_From','Organ_Support','fluid_infused_type','Gender','Infection _PrimaryReason_Hospital _Admission','Location_sepsis_diagnosis','Infection _PrimaryReason_ICU_Admission ','Suspected_origin_of_Infection','Risk_factor_infection','Systolic_BP_lesser_than_90','septic shock','Septic_Shock_Developd_later','Clinical_Feature_hypoperfusion','Albumin_infusion _3hr','Blood_Culture_Sent','Blood_Culture_KNOWN','Any_Other_Culture_Sent','Lactate_Level_1hr_presentation','vasopressor','inotropes','Hemodymic_monitoring','SourceofControl','Empirical_Antibiotic_given','Is_Empiric_choice_correct','Antibiotic_given as extented infusion','Loading_dose','Descalation_of_antibiotic','TDM_Performed','Rel_adjustment','Empirical_antifungal','Antiviral','Culture1','Inf_organism_cultured','Multiplex_PCR','Resistance gene detected','Galactomann_sent','Beta_D_glucan_sent','Corticosteroids','VitaminC','Thiamine','Extracorporeal_therapy','Other_immu2modulator_used','Bicarbote_therapy','NIV','HFNO','Succesful_Extubated','Reintubation','Recruit_ma2uever','Rel_Support','encephalopathy','coagulopathy','Liver_dysfunction','nosocomial_infection','any_incidencesec_sepsis','Nutrition_started','Blood_transfusion','Invasive_ventilation','Prone','Platelet_transfusion','FFP_transfusion','DVT_prophylaxis','Stress_Ulcer_Prophylaxis','Cumulative_fluid','CI_sedation','Was_sedation_scale','continuous_analgesia','Pain_scale_monitored','delirium_assessed','NMB_infusion','TOF_monitored','ECMO_used','Repeat_culture_sent','Repeat_culture_sent_value','Clinical_improvement','ICU_Discharge_Status','Discharge_medical_advice'],drop_first=True)
X_encoded.head()


# In[58]:


X_encoded.columns = X_encoded.columns.str.replace('[^a-zA-Z0-9_]', '_')


# In[59]:


import pandas as pd
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import train_test_split


sel = SelectKBest(f_classif, k=40).fit(X_encoded, y)

# Get the selected feature names
selected_feature_names = X_encoded.columns[sel.get_support()]
print("Selected feature names:", selected_feature_names)

# Get the p-values for all features
p_values = pd.Series(sel.pvalues_, index=X_encoded.columns)

# Filter p-values for only the selected features
selected_p_values = p_values[sel.get_support()]

# Sort the selected p-values
sorted_selected_p_values = selected_p_values.sort_values(ascending=True)

# Print sorted p-values for the top 40 selected features
print("Sorted p-values for the top 40 selected features:\n", sorted_selected_p_values)

# Transform X_encoded to include only the top 40 selected features
X_selected = sel.transform(X_encoded)
X = X_selected

# Split the dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X_selected, y, test_size=0.2, random_state=42)


# In[61]:


import random
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix



def initialize_search_agents(n, int_hyperparameter_ranges, float_hyperparameter_ranges):
    """
    Initialize search agents with random hyperparameters.

    Args:
    - n (int): Number of search agents.
    - int_hyperparameter_ranges (dict): Dictionary specifying the range for each integer hyperparameter.
    - float_hyperparameter_ranges (dict): Dictionary specifying the range for each float hyperparameter.

    Returns:
    - search_agents (list of dicts): List of search agents, where each agent is represented by a dictionary of hyperparameters.
    """
    X = []
    for _ in range(n):
        agent_hyperparameters = {}
        # Generate random values for integer hyperparameters
        for param, (min_val, max_val) in int_hyperparameter_ranges.items():
            agent_hyperparameters[param] = random.randint(min_val, max_val)
        # Generate random values for float hyperparameters
        for param, (min_val, max_val) in float_hyperparameter_ranges.items():
            agent_hyperparameters[param] = random.uniform(min_val, max_val)
        X.append(agent_hyperparameters)
    return X


def train_evaluate_lightgbm(hyperparameters, xtrain, ytrain, xtest, ytest):
    """
    Train a LightGBM model with given hyperparameters and evaluate its performance.

    Args:
    - hyperparameters (dict): Dictionary containing hyperparameters for LightGBM.
    - xtrain (numpy.ndarray): Training features.
    - ytrain (numpy.ndarray): Training labels.
    - xtest (numpy.ndarray): Testing features.
    - ytest (numpy.ndarray): Testing labels.

    Returns:
    - accuracy (float): Accuracy of the trained model on the testing dataset.
    """
    # Ensure each hyperparameter is a scalar and of the correct type
    hyperparameters['num_leaves'] = int(np.array(hyperparameters['num_leaves']).flatten()[0])
    hyperparameters['bagging_freq'] = int(np.array(hyperparameters['bagging_freq']).flatten()[0])
    hyperparameters['max_depth'] = int(np.array(hyperparameters['max_depth']).flatten()[0])
    hyperparameters['feature_fraction'] = float(np.array(hyperparameters['feature_fraction']).flatten()[0])
    hyperparameters['bagging_fraction'] = float(np.array(hyperparameters['bagging_fraction']).flatten()[0])
    hyperparameters['colsample_bytree'] = float(np.array(hyperparameters['colsample_bytree']).flatten()[0])
    hyperparameters['subsample'] = float(np.array(hyperparameters['subsample']).flatten()[0])
    hyperparameters['min_child_samples'] = int(np.array(hyperparameters['min_child_samples']).flatten()[0])

    
    
    # Extract hyperparameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',  
        'boosting_type': 'gbdt',
        'num_leaves': hyperparameters['num_leaves'],
        'bagging_freq': hyperparameters['bagging_freq'],
        'max_depth': hyperparameters['max_depth'],
        'feature_fraction': hyperparameters['feature_fraction'],
        'bagging_fraction': hyperparameters['bagging_fraction'],
        'colsample_bytree': hyperparameters['colsample_bytree'],
        'subsample': hyperparameters['subsample'],
        'min_child_samples' : hyperparameters['min_child_samples'],
        'verbose' : -1
    }
    
    
    # Train the LightGBM model
    train_data_lgb = lgb.Dataset(xtrain, label=ytrain)
    model = lgb.train(params, train_data_lgb, num_boost_round=100)

    # Make predictions on the testing dataset
    predictions = model.predict(xtest, num_iteration=model.best_iteration)

    # Convert probabilities to binary predictions (0 or 1)
    predicted_labels = (predictions >= 0.5).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(ytest, predicted_labels)  
    
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(ytest, predicted_labels).ravel()
    conf_matrix = confusion_matrix(ytest, predicted_labels)
    
    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return accuracy, sensitivity, specificity,conf_matrix

# Example hyperparameter ranges (you should customize these according to your specific problem):
int_hyperparameter_ranges = {
    'num_leaves': (40, 100),
    'bagging_freq': (1, 10),
    'max_depth' : (3,8),
    'min_child_samples' : (45,60)
}

float_hyperparameter_ranges = {
    'feature_fraction': (0.1, 1.0),
    'bagging_fraction': (0.1, 1.0),
    'colsample_bytree' : (0.1, 1.0),
    'subsample' : (0.1,1.0)
}



def update_incumbent(X, X_best, A, C, p, b, l, lb, ub, best_sensitivity, best_specificity):
    """
    

    Args:
    - X (dict): Current solution represented as a dictionary of hyperparameters.
    - X_best (dict): Best solution found so far represented as a dictionary of hyperparameters.
    - A (float): Coefficient controlling step size or movement.
    - C (float): Coefficient related to the exploration/exploitation balance.
    - p (float): Random number for decision making.
    - b (float): Scaling factor for the cosine term.
    - l (numpy.ndarray): Random vector for exploration.
    - lb (float): Lower bound of the search space.
    - ub (float): Upper bound of the search space.
    - best_sensitivity (float): Best sensitivity found so far.
    - best_specificity (float): Best specificity found so far.

    Returns:
    - X_new (dict): Updated solution represented as a dictionary of hyperparameters.
    """
    D = np.abs(C * np.array(list(X_best.values())) - np.array(list(X.values())))
    if p < 0.5:
        if np.abs(A) < 1:
            X_new = {param: value - (A * d) for (param, value), d in zip(X.items(), D)}
        else:
            rand_solution = {param: random.uniform(lb, ub) for param in X}
            X_new = {param: value - (A * d) for (param, value), d in zip(rand_solution.items(), D)}
    else:
        D_prime = np.abs(np.array(list(X_best.values())) - np.array(list(X.values())))
        X_new = {param: d * np.exp(b * l) * np.cos(2 * np.pi * l) + value for (param, value), d in zip(X_best.items(), D_prime)}
    
    # Clip values to boundary
    X_new_clipped = {param: np.clip(value, lb, ub) for param, value in X_new.items()}
    
    
    return X_new_clipped





def whale_optimization_algorithm(n, dimension, lb, ub, max_iter, int_hyperparameter_ranges, float_hyperparameter_ranges, xtrain, ytrain, xtest, ytest):
    X = initialize_search_agents(n, int_hyperparameter_ranges, float_hyperparameter_ranges)
    X_best = X[np.argmax([train_evaluate_lightgbm(agent, xtrain, ytrain, xtest, ytest)[0] for agent in X])]
    best_accuracy, best_sensitivity, best_specificity,best_conf_matrix = train_evaluate_lightgbm(X_best, xtrain, ytrain, xtest, ytest)
    best_solution = None
    best_accuracy = np.inf
    best_sensitivity = np.inf
    best_specificity = np.inf
    best_conf_matrix = None
    
    for iteration in range(max_iter):
        for i in range(n):
            a = 2 - 2 * iteration / max_iter  # a decreases linearly from 2 to 0
            A = 2 * a * np.random.rand() - a 
            C = 2 * np.random.rand()  
            p = np.random.rand()  
            b = 1  # Scaling factor b is kept constant
            l = np.random.uniform(-1, 1,dimension) 
            
            X[i] = update_incumbent(X[i], X_best, A, C, p, b, l, lb, ub, best_sensitivity, best_specificity)
            
        for i in range(len(X)):
            for param in X[i]:
                X[i][param] = np.clip(X[i][param], lb, ub)
            if 'bagging_fraction' in X[i]:
                X[i]['bagging_fraction'] = np.clip(X[i]['bagging_fraction'], 0.1, 1.0)
            if 'feature_fraction' in X[i]:
                X[i]['feature_fraction'] = np.clip(X[i]['feature_fraction'], 0.1, 1.0)
        
        
        fitness = [train_evaluate_lightgbm(agent, xtrain, ytrain, xtest, ytest) for agent in X]
          
        max_fitness_index = np.argmax([f[0] for f in fitness])
        max_fitness = fitness[max_fitness_index][0]
        
        if max_fitness < best_accuracy:
            best_accuracy = max_fitness
            best_solution = X_best
            best_sensitivity = fitness[max_fitness_index][1]
            best_specificity = fitness[max_fitness_index][2]
            best_conf_matrix = fitness[max_fitness_index][3]
        
    return best_solution, best_accuracy, best_sensitivity, best_specificity,best_conf_matrix



n = 24  # Number of search agents
dimension = 8  # Dimensionality of the search space
lb = 5  # Lower bound of the search space
ub = 10  # Upper bound of the search space
max_iter = 10  # Maximum number of iterations

best_solution, best_accuracy, best_sensitivity, best_specificity, best_conf_matrix = whale_optimization_algorithm(n, dimension, lb, ub, max_iter, int_hyperparameter_ranges, float_hyperparameter_ranges, xtrain, ytrain, xtest, ytest)
print("Best solution:", best_solution)
print("Best accuracy:", best_accuracy)
print("Best sensitivity:", best_sensitivity)
print("Best specificity:", best_specificity)
print("Best confusion matrix:",best_conf_matrix)






