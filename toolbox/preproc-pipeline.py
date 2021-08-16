import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


class Preprocessing:


    def load_data(self):

        data = pd.read_csv(f"https://wagon-public-datasets.s3.amazonaws.com/Machine%20Learning%20Datasets/ML_Houses_dataset.csv")

        data = data[['GrLivArea','BedroomAbvGr','KitchenAbvGr', 'OverallCond','RoofSurface','GarageFinish','CentralAir','ChimneyStyle','MoSold','SalePrice']].copy()

        return data


    def initial_cleaning(self):

        data = self.load_data()

        data = data.drop_duplicates()

        data.GarageFinish.replace(np.nan, "NoGarage", inplace=True)



        imputer = SimpleImputer(
            strategy="median"
        )  # Instanciate a SimpleImputer object with strategy of choice

        imputer.fit(data[['RoofSurface']])  # Call the "fit" method on the object

        data['RoofSurface'] = imputer.transform(
            data[['RoofSurface']])  # Call the "transform" method on the object




        scaler = MinMaxScaler()

        scaler.fit(data[['RoofSurface']])

        data['RoofSurface'] = scaler.transform(data[['RoofSurface']])



        r_scaler = RobustScaler()  # Instanciate Robust Scaler

        r_scaler.fit(data[['GrLivArea']])  # Fit scaler to feature

        data['GrLivArea'] = r_scaler.transform(data[['GrLivArea']])  #Scale



        scaler_BedroomAbvGr = MinMaxScaler()
        scaler_OverallCond = MinMaxScaler()
        scaler_KitchenAbvGr = MinMaxScaler()

        scaler_BedroomAbvGr.fit(data[['BedroomAbvGr']])

        scaler_OverallCond.fit(data[['OverallCond']])

        scaler_KitchenAbvGr.fit(data[['KitchenAbvGr']])

        data['BedroomAbvGr'] = scaler_BedroomAbvGr.transform(data[['BedroomAbvGr']])
        data['OverallCond'] = scaler_OverallCond.transform(data[['OverallCond']])
        data['KitchenAbvGr'] = scaler_KitchenAbvGr.transform(data[['KitchenAbvGr']])

        ohe = OneHotEncoder(sparse=False)  # Instanciate encode

        ohe.fit(data[['GarageFinish']])  # Fit encoder

        GarageFinish_encoded = ohe.transform(data[['GarageFinish']])  # Encode alley

        data["NoGarage"], data["Fin"], data["RFn"], data[
            "Unf"] = GarageFinish_encoded.T  # Transpose encoded Alley back into dataframe


        ohe = OneHotEncoder(drop='if_binary',
                            sparse=False)  # Instanciate binary encoder

        ohe.fit(data[['CentralAir']])  # Fit encoder

        CentralAir_encoded = ohe.transform(data[['CentralAir']])  # Encode alley

        data[
            "CentralAir"] = CentralAir_encoded  # Transpose encoded Alley back into dataframe


        data["MoSold_norm"] = 2 * np.pi * data["MoSold"] / data["MoSold"].max()

        data["sin_MoSold"] = np.sin(data["MoSold_norm"])
        data["cos_MoSold"] = np.cos(data["MoSold_norm"])

        data.drop(columns="MoSold", inplace=True)
        data.drop(columns="MoSold_norm", inplace=True)
        data.drop(columns='GarageFinish', inplace=True)

        data.to_csv("/data/clean_dataset.csv",
                    index=False)
