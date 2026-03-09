def streamline_features(dir):
    #Import Pandas
    import pandas as pd
    import numpy as np

    #Load Data
    df = pd.read_csv(dir)

    #Clean Data
    df = df[df["price"] != 0]
    df.loc[df["yr_renovated"] == 0, "yr_renovated"] = df["yr_built"]
    df["sqft_vertical"] = df["sqft_above"] + df["sqft_basement"]
    df["sqft_horizontal"] = df["sqft_living"] + df["sqft_lot"]
    df["curr_year"] = np.full(len(df), 2014)
    df["apparent_age"] = (df["curr_year"] - df["yr_renovated"]) + ((df["yr_renovated"] - df["yr_built"])/2)
    df["quality"] = (0.75 * df["floors"]) + (3 * df["waterfront"]) + (2 * df["view"]) + df["condition"]
    df = df.drop(columns=["street", "country", "date"])
    columns = ["statezip", "city"]
    df[columns] = df[columns].astype("category")
    #Separate Data
    X = df[["apparent_age", "quality", "sqft_horizontal", "sqft_vertical", "statezip", "city", "bedrooms", "bathrooms"]]
    Y = df["price"]

    #Return Data
    return X, Y