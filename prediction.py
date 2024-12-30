import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df = df[["Age", "Attrition", "MonthlyIncome", "JobSatisfaction", "Department", "BusinessTravel", "JobRole","EducationField"]]
    return df

def one_hot_encoding(df, categorical_features):
    category = pd.get_dummies(df[categorical_features], dtype=int)
    encoding_df = pd.concat([df, category], axis=1)
    encoding_df = encoding_df.drop(columns=categorical_features)
    return encoding_df

def data_standardization(df, feature_to_scale):
    scaler = StandardScaler()

    df[feature_to_scale] = scaler.fit_transform(df[feature_to_scale])
    return df

def transformed_df(df):
    numerical_feature = ["Age", "Attrition", "MonthlyIncome", "JobSatisfaction"]
    categorical_feature = ["Department", "BusinessTravel", "JobRole", "EducationField"]

    transformed_df = one_hot_encoding(df, categorical_feature)
    transformed_df = data_standardization(transformed_df, numerical_feature)
    return transformed_df

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def model_predict(data_path, model_path):
    df = load_dataset(data_path)
    model = load_model(model_path)
    kmeans_df = transformed_df(df)
    clusters = model.predict(kmeans_df)
    df["Clusters_Segmentation"] = clusters.astype(int)
    df.to_csv("./clustering.csv", index=False)
    return df.sample(5)

if __name__ == "__main__":
    dataset_path = "./dataset/evaluate_dataset.csv"
    model_path = "./kmeans_clustering_model.joblib"
    
    result = model_predict(dataset_path, model_path)

    print(result)