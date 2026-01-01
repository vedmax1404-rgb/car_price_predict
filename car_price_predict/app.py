import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Car price prediction")

with open("car_price_predict/my.pkl", "rb") as f:
    pack = pickle.load(f)

model = pack["model"]
scaler = pack["scaler"]
ohe_cols = pack["ohe_cols"]
ohe_src_cols = pack["ohe_src_cols"]

st.header("EDA")

df_train = pd.read_csv(
    "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
)
df_test = pd.read_csv(
    "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv"
)

num = df_train.select_dtypes(include="number").columns
df_train[num] = df_train[num].fillna(df_train[num].median())
df_test[num] = df_test[num].fillna(df_train[num].median())

pri = [col for col in df_train.columns if col != "selling_price"]
df_train = df_train.drop_duplicates(subset=pri, keep="first")
df_train = df_train.reset_index(drop=True)

for col in ["mileage", "engine", "max_power"]:
    df_train[col] = df_train[col].str.split().str[0]
    df_train[col] = pd.to_numeric(df_train[col], errors="coerce")

    df_test[col] = df_test[col].str.split().str[0]
    df_test[col] = pd.to_numeric(df_test[col], errors="coerce")

df_train = df_train.drop(columns=["torque"])
df_test = df_test.drop(columns=["torque"])

for col in ["engine", "seats"]:
    df_train[col] = df_train[col].fillna(df_train[col].median())
    df_test[col] = df_test[col].fillna(df_train[col].median())

df_train["engine"] = df_train["engine"].astype(int)
df_train["seats"] = df_train["seats"].astype(int)

df_test["engine"] = df_test["engine"].astype(int)
df_test["seats"] = df_test["seats"].astype(int)

st.subheader("Pairplot (train)")

num = df_train.select_dtypes(include="number").columns
fig = sns.pairplot(
    df_train[num].sample(1000, random_state=42), corner=True, diag_kind="hist"
)
st.pyplot(fig)

st.subheader("Pairplot (test)")

num2 = df_test.select_dtypes(include="number").columns
num2 = [col for col in num2 if col != "selling_price"]

fig = sns.pairplot(
    df_test[num2].sample(1000, random_state=42), corner=True, diag_kind="hist"
)
st.pyplot(fig)

st.subheader("Correlation heatmap (train)")

corr = df_train[num].corr(method="pearson")

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)

st.pyplot(fig)

st.header("Prediction")

f = st.file_uploader("CSV file", type=["csv"])

if f is not None:
    d = pd.read_csv(f)

    for c in ["mileage", "engine", "max_power"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c].astype(str).str.split().str[0], errors="coerce")

    if "torque" in d.columns:
        d = d.drop(columns=["torque"])

    for c in ["engine", "seats"]:
        if c in d.columns:
            d[c] = d[c].fillna(d[c].median()).astype(int)

    n = d.select_dtypes(include="number").columns
    d[n] = d[n].fillna(d[n].median())

    d = pd.get_dummies(d, columns=ohe_src_cols, drop_first=True)
    d = d.reindex(columns=ohe_cols, fill_value=0)

    xs = scaler.transform(d)
    p = model.predict(xs)

    d["pred"] = p

    st.dataframe(d[["pred"]])

st.header("Model weights")

w = model.coef_
c = ohe_cols
t = pd.DataFrame({"feature": c, "weight": w})

t["abs_weight"] = t["weight"].abs()
t = t.sort_values("abs_weight", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(t["feature"], t["weight"])
ax.invert_yaxis()
ax.set_xlabel("Weight value")
ax.set_ylabel("Feature")

st.pyplot(fig)

