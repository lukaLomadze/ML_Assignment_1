import mlflow
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


def train_model(x,y,x_val,y_val, name,model, scoring):
    with mlflow.start_run(run_name=name) :
        cv_scores = cross_val_score(model, x, y, scoring=scoring, cv=5)
        cv_rmsle_mean = -cv_scores.mean()
        cv_rmsle_std = cv_scores.std()
        model.fit(x, y)

        t_pred= model.predict(x)
        v_pred= model.predict(x_val)


        t_rmsle= rmsle(y,t_pred)
        v_rmsle = rmsle(y_val,v_pred)

        params ={"name": type(model).__name__}
        params.update(model.get_params())
        mlflow.log_params(params)

        mlflow.log_metrics({
            "cv_rmsle_mean": cv_rmsle_mean,
            "cv_rmsle_std": cv_rmsle_std,
            "train_rmsle": t_rmsle,
            "val_rmsle": v_rmsle,
            "overfit": v_rmsle-t_rmsle



        })

        mlflow.sklearn.log_model(model, "model")
        print("name : ", name)
        print("cv_rmsle_mean: ", cv_rmsle_mean)
        print("cv_rmsle_std: ", cv_rmsle_std)
        print("train_rmsle: ", t_rmsle)
        print("validation_rmsle: ", v_rmsle)

        return v_rmsle







