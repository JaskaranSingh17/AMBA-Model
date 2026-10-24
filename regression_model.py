"""
For each outlet:  Sales = a + b * (MRP * item_factor)
item_factor = how well items with that ID number sell vs their store's average
              (FDW58 -> 58), smoothed over neighbouring numbers.
"""
import pandas as pd
from sklearn.linear_model import LinearRegression

train = pd.read_csv("train_v9rqX0R.csv")
test = pd.read_csv("test_AbJTz2l.csv")
for df in (train, test):
    df["Item_Num"] = df["Item_Identifier"].str[3:].astype(int)

# Item factor: average units sold relative to the outlet's average, per item number
units = train["Item_Outlet_Sales"] / train["Item_MRP"]
relative = units / units.groupby(train["Outlet_Identifier"]).transform("mean")
factor = relative.groupby(train["Item_Num"]).mean().rolling(3, center=True, min_periods=1).mean()
for df in (train, test):
    df["X"] = df["Item_MRP"] * df["Item_Num"].map(factor)

# One linear regression per outlet
test["Item_Outlet_Sales"] = 0.0
for outlet, group in train.groupby("Outlet_Identifier"):
    model = LinearRegression().fit(group[["X"]], group["Item_Outlet_Sales"])
    rows = test["Outlet_Identifier"] == outlet
    test.loc[rows, "Item_Outlet_Sales"] = model.predict(test.loc[rows, ["X"]])

test["Item_Outlet_Sales"] = test["Item_Outlet_Sales"].clip(lower=33)
test[["Item_Identifier", "Outlet_Identifier", "Item_Outlet_Sales"]].to_csv("regression_model.csv", index=False)
print("Saved regression_model.csv")
