import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


@st.cache_data
def load_data():
    etf = pd.read_csv("ETFs.csv")
    mf = pd.read_csv("MutualFunds.csv")
    return etf, mf


@st.cache_resource
def train_model(data, target_col="fund_mean_annual_return_5years"):
    X = data.drop(columns=[target_col])
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def main():
    st.title("📈 Fund Return Prediction Dashboard")

    st.sidebar.header("Configuration")
    show_raw_data = st.sidebar.checkbox("Show Raw Data")
    selected_fund_type = st.sidebar.selectbox("Select Fund Type", ["ETF", "MutualFund"])

    etf_data, mf_data = load_data()
    data = etf_data if selected_fund_type == "ETF" else mf_data

    if show_raw_data:
        st.subheader("Raw Data Preview")
        st.dataframe(data.head())

    st.subheader("🔧 Feature Selection")
    features = st.multiselect(
        "Select Features for Prediction",
        data.columns.drop("fund_mean_annual_return_5years"),
        default=["fund_stdev_5years", "fund_sharpe_ratio_5years"]
    )

    if st.button("Train Model"):
        with st.spinner("Training the model..."):
            model, X_test, y_test = train_model(data[features + ["fund_mean_annual_return_5years"]])
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)

            st.success(f"Model training completed! MAE: {mae:.2f}")

            fig, ax = plt.subplots()
            ax.scatter(y_test, predictions)
            ax.set_xlabel("Actual Return")
            ax.set_ylabel("Predicted Return")
            ax.set_title("Predicted vs Actual Return")
            st.pyplot(fig)


if __name__ == "__main__":
    main()
