import keras
import tensorflow
from keras import Model
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras.regularizers import l1

MODEL_INPUT_SHAPE = 490
import os
import xgboost as xgb

print(keras.__version__)
print(tensorflow.__version__)
XGBS_PARAMS = {'max_depth': 4, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}


# XGBS_PARAMS = {
#     "objective": ["binary:hinge"],
#     "booster": ["gbtree"],
#     "eta": [0.1],
#     'gamma': [0.5],
#     'max_depth': range(2, 4, 2),
#     'min_child_weight': [1],
#     'subsample': [0.6],
#     'colsample_bytree': [0.6],
#     "lambda": [1],
# }


def api_model(shape):
    x = Input(shape=(shape,), name="input")
    ann_dense1 = Dense(100, activation='tanh', name='dense_100')(x)
    ann_dense2 = Dense(50, activation='tanh', kernel_constraint=maxnorm(3), activity_regularizer=l1(0.001),
                       kernel_regularizer=l1(0.001), name='dense_50')(ann_dense1)
    ann_dropout = Dropout(rate=0.5, name='dropout')(ann_dense2)
    ann_dense3 = Dense(20, activation='tanh', name='dense_20')(ann_dropout)
    ann_output = Dense(1, activation='sigmoid', name='output')(ann_dense3)
    model = Model(x, ann_output, name="ann_model")
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['acc'])
    return model


def load_trained_model(model_type, org_name, models_f):
    if model_type == 'base':
        model = api_model(MODEL_INPUT_SHAPE)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        load_weights_path = os.path.join(models_f, f"{org_name}/")
        model.load_weights(load_weights_path)
    else:
        model_name = os.path.join(models_f, f"{org_name}.dat")
        model = xgb.XGBClassifier(kwargs=XGBS_PARAMS)  # init model
        model.load_model(model_name)

    return model
