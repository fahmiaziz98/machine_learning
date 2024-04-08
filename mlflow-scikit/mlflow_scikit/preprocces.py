import logging
from scipy import stats
from scipy.stats import skew
from scipy.special import boxcox1p

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocces_data(
    path: Optional[Path[str]]
):
    """
    Preprocess data by handling missing values & feature engineering
    
    Args:
    path (str): Path to the data directory.
    """
    
    train = pd.read_csv(f"{path}/train.csv")
    test = pd.read_csv(f"{path}/test.csv")
    train = train.drop(["Id"], axis=1)
    test = test.drop(["Id"], axis=1)

    data, y_train = handle_outlier_missing_nan(train, test)
    train, test = feature_engineering(data)
    logger.info("Preproccesing Data is Done...")
    return train, test, y_train
    

def feature_engineering(
    data: pd.DataFrame,
):
    """
    Perform feature engineering on the input data.
    
    Args:
        data (pd.DataFrame): Preprocessed data.
        scaler (callable): Scaler object.
        encoder (callable): Encoder object.

    """
    logger.info("Feature engineering...")
    data['MSSubClass'] = data['MSSubClass'].apply(str)
    #Changing OverallCond into a categorical variable
    data['OverallCond'] = data['OverallCond'].astype(str)
    #Year and month sold are transformed into categorical features.
    data['YrSold'] = data['YrSold'].astype(str)
    data['MoSold'] = data['MoSold'].astype(str)
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
    # process columns, apply LabelEncoder to categorical features
    for c in cols:
        lbl = LabelEncoder() 
        lbl.fit(list(data[c].values)) 
        data[c] = lbl.transform(list(data[c].values))

    # Adding total sqfootage feature 
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    numeric_feats = data.dtypes[data.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness = skewness[abs(skewness) > 0.75]

    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        logger.info(f"Applying boxcox1p lambda={lam} to feature {feat}")
        #all_data[feat] += 1
        data[feat] = boxcox1p(data[feat], lam)
    
    #all_data[skewed_features] = np.log1p(all_data[skewed_features])
    data = pd.get_dummies(data)
    
    train = data[:ntrain]
    test = data[ntrain:]

    logger.info("Feature Engineering Done...")

    return train, test


def handle_outlier_missing_nan(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Preprocess the input data by handling outliers, missing values and
    data type conversion.

    Args:
        train (pd.DataFrame): The training data.
        test (pd.DataFrame): The test data.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: The preprocessed data and the target variable.
    """
    logger.info("Start preprocess data...")
    logger.info("Deleting Outlier...")

    # https://ww2.amstat.org/publications/jse/v19n3/Decock/DataDocumentation.txt
    train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

    # log transformation target variable
    logger.info("Transformation target variable...")
    train["SalePrice"] = np.log1p(train["SalePrice"])

    # concate all data
    logger.info("Concatenating data...")
    global ntrain
    ntrain = train.shape[0]
    ntest = test.shape[0]

    y_train = train.SalePrice.values
    all_data = pd.concat((train, test)).reset_index(drop=True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)

    # Input missing value
    logger.info("Preprocess NaN...")
    missing_cols_cat = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "MSSubClass",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond", "MasVnrType",
        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"
    ]
    for col in missing_cols_cat:
        all_data[col] = all_data[col].fillna("None")

    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )

    missing_cols_num = [
        'GarageYrBlt', 'GarageArea', 'GarageCars', "MasVnrArea",
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'
    ]
    for col in missing_cols_num:
        all_data[col] = all_data[col].fillna(0)

    missing_frequent = [
        "MSZoning", "Electrical", "KitchenQual",
        "SaleType", "Exterior1st", "Exterior2nd"
    ]
    for col in missing_frequent:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    all_data = all_data.drop(['Utilities'], axis=1)
    all_data["Functional"] = all_data["Functional"].fillna("Typ")
    logger.info("Preprocess NaN done...")

    return all_data, y_train

