import numpy as np
from scipy.stats import randint

from sklearn.model_selection import RandomizedSearchCV
import keras
from keras.layers import Dense # Dense layers are "fully connected" layers
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier


y_train_bin = to_categorical(y_train)
epochs = np.arange(10,151)
neurons = np.arange(20,1001)

n_output = 5
                   
parameters ={'nb_epoch':epochs,
            'learn': uniform(0.0001,0.2),
            'neurons':neurons}

#definición del modelo
def create_model(neurons,learn):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=100, activation='sigmoid'))
    model.add(Dense(n_output, activation='softmax'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=learn), metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model,verbose=0)
#hacemos uso de RandomizedSearchCV y pasamos los hiperparámetros.
grid= RandomizedSearchCV(estimator=model, param_distributions=parameters,n_iter=10,cv=4)


#entrenamos el modelo.
grid_result = grid.fit(X_train_pca,y_train_bin)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
