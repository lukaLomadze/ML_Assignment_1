import mlflow
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np


def train_model(x,y,x_val,y_val, name,model, scoring):
    with mlflow.start_run(run_name=name) :
        cv_scores = cross_val_score(model, x, y, scoring=scoring, cv=5)
        cv_rmsle_mean = -cv_scores.mean()
        cv_rmsle_std = cv_scores.std()
        model.fit(x, y)
        t_rmsle= np.sqrt(mean_squared_error(y, model.predict(x)))
        v_rmsle = np.sqrt(mean_squared_error(y_val, model.predict(x_val)))
        params ={"name": type(model).__name__}
        params.update(model.get_params())
        mlflow.log_params(params)
        mlflow.log_metrics({
            "cv_rmsle_mean": cv_rmsle_mean,
            "cv_rmsle_std": cv_rmsle_std,
            "train_rmsle": t_rmsle,
            "validation_rmsle": v_rmsle,
            "overfit": v_rmsle-t_rmsle



        })

        mlflow.sklearn.log_model(model, name ="model")
        print("name : ", name)
        print("cv_rmsle_mean: ", cv_rmsle_mean)
        print("cv_rmsle_std: ", cv_rmsle_std)
        print("train_rmsle: ", t_rmsle)
        print("validation_rmsle: ", v_rmsle)

        return v_rmsle







