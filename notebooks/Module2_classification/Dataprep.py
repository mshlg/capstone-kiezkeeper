#dataprep.py
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

#def function to fit impute (+optional scale) on X_tr, apply to both - log is parameter-free
#KNN imputation is applied to treat missings in both models to establish comparability, knowing that XGBoost could handle NaNs independently
def preprocess_fit_transform(X_tr, X_ev, feature_cols, log_cols, scale=True):
    imputer = KNNImputer(n_neighbors=5)
    Xtr = pd.DataFrame(imputer.fit_transform(X_tr), columns=feature_cols, index=X_tr.index)
    Xev = pd.DataFrame(imputer.transform(X_ev), columns=feature_cols, index=X_ev.index)
    for col in log_cols:
        Xtr[col] = np.log1p(Xtr[col])
        Xev[col] = np.log1p(Xev[col])
    if scale:
        scaler = StandardScaler()
        Xtr = pd.DataFrame(scaler.fit_transform(Xtr), columns=feature_cols, index=Xtr.index)
        Xev = pd.DataFrame(scaler.transform(Xev), columns=feature_cols, index=Xev.index)
    return Xtr, Xev