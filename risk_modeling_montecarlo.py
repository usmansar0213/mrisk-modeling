import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import openai
import os
from PIL import Image

def main():
    # Initialize session state if not already initialized
    if 'processed_data' not in st.session_state:
        st.session_state['processed_data'] = None

    # Function to display the title and logo
    def display_title():
       st.title("Risk Modeling")

    # Call the display_title function to show logo and title
    display_title()

    # Step 1: File Upload 
    st.subheader("Step 1: Untuk menjalankan aplikasi ini, silahkan upload Data variabel Anda")
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("Uploaded data:", df.head())

        if 'Year' not in df.columns:
            st.error("Kolom 'Year' tidak ditemukan.")
        else:
            variables = [col for col in df.columns if col != 'Year']
            dependent_variable = st.selectbox("Pilih dependent variable:", variables)
            independent_variables = st.multiselect("Pilih independent variables:", variables)

            # Step 2: Input Target Value
            st.subheader("Step 2: Input Target Value")
            target_value = st.slider("Masukkan Target Value:", min_value=-10000.0, max_value=10000.0, value=5.0, step=0.1)

            if independent_variables:
                last_year = df['Year'].max()
                years_to_predict = st.slider("Tahun untuk diprediksi:", 1, 20, 10)
                future_years = np.array(range(last_year + 1, last_year + 1 + years_to_predict)).reshape(-1, 1)

                predictions = {}
                for var in independent_variables:
                    model = LinearRegression()
                    model.fit(df[['Year']], df[var])
                    predictions[var] = model.predict(future_years)

                history_data = df[['Year'] + independent_variables].sort_values(by='Year', ascending=False)
                future_data = pd.DataFrame({'Year': future_years.flatten(), **predictions})
                combined_df = pd.concat([history_data, future_data], ignore_index=True).sort_values(by='Year', ascending=False)

                # Step 3: Display Combined Data
                    st.subheader("Step 3: Combined Data")
                    st.write(
                        "Berikut data gabungan historis dan prediksi ke depan. "
                        "Silakan periksa dan edit (misal di Excel) jika diperlukan secara terpisah."
                    )
                    st.dataframe(combined_df)
                    

                # Step 4: Process Data
                if st.button("Process Data"):
                    st.session_state['processed_data'] = combined_df

                    st.header("Step 4: Regression and Monte Carlo Simulation")

                    # Simulasi Regresi Linear
                    regression_model = LinearRegression()
                    regression_model.fit(df[independent_variables], df[dependent_variable])

                    num_simulations = st.number_input("Masukkan jumlah simulasi:", min_value=100, step=100, value=1000)

                    monte_carlo_results = []
                    for _ in range(num_simulations):
                        simulated_values = {}
                        for i, var in enumerate(independent_variables):
                            simulated_values[var] = np.random.normal(df[var].mean(), df[var].std(), 1)[0]
                        predicted_value = regression_model.intercept_ + sum(
                            regression_model.coef_[i] * simulated_values[var] for i, var in enumerate(independent_variables)
                        )
                        monte_carlo_results.append(predicted_value)

                    r_squared = regression_model.score(df[independent_variables], df[dependent_variable])
                    confidence_level = 5
                    var = np.percentile(monte_carlo_results, confidence_level)

                    monte_carlo_rounded = [round(value) for value in monte_carlo_results]
                    unique_values, counts = np.unique(monte_carlo_rounded, return_counts=True)
                    relative_probabilities = counts / num_simulations

                    # Visualization and Metrics
                    st.subheader("Visualization and Metrics")
                    plt.figure(figsize=(10, 6))
                    plt.hist(monte_carlo_results, bins=30, color='blue', alpha=0.7)
                    plt.title("Monte Carlo Simulation Results")
                    plt.xlabel(dependent_variable)
                    plt.ylabel("Frequency")
                    st.pyplot(plt)

                    st.write(f"Rata-rata hasil simulasi: {np.mean(monte_carlo_results):.2f}")
                    st.write(f"Standar deviasi hasil simulasi: {np.std(monte_carlo_results):.2f}")
                    st.write(f"R-Squared dari model regresi: {r_squared:.2f}")
                    st.write(f"Value at Risk (VaR) pada level {confidence_level}%: {var:.2f}")

                    st.subheader("Step 5: Predicted Values Over Time")

                    # Combine historical and future data for Random Forest
                    historical_years = df['Year'].values
                    predicted_years = future_years.flatten()[:len(monte_carlo_results)]  # Match length of Monte Carlo results
                    all_years = np.concatenate([historical_years, predicted_years])

                    y = np.concatenate([df[dependent_variable].values, monte_carlo_results[:len(predicted_years)]])  # Match lengths
                    X = all_years.reshape(-1, 1)  # Use years as features

                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Train Random Forest Regressor
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf_model.fit(X_train, y_train)

                    # Make predictions
                    predictions = rf_model.predict(X)

                    # Calculate MSE
                    mse = mean_squared_error(y_test, rf_model.predict(X_test))
                    st.write(f"Mean Squared Error: {mse:.2f}")

                    # Plot Historical and Predicted Values
                    plt.figure(figsize=(12, 6))
                    plt.plot(historical_years, df[dependent_variable].values, label="Historical Data", color="blue")
                    plt.plot(predicted_years, monte_carlo_results[:len(predicted_years)], label="Monte Carlo Predictions", color="orange", linestyle="--")
                    plt.plot(all_years, predictions, label="Random Forest Predictions", color="green")
                    plt.xlabel("Year")
                    plt.ylabel(dependent_variable)
                    plt.title("Predicted Values Over Time")
                    plt.legend()
                    st.pyplot(plt)

                    # Step 6: Risk Factors & Contributions
                    st.subheader("Step 6: Risk Factors & Contributions")
                    coefficients = regression_model.coef_
                    total_impact = np.sum(np.abs(coefficients))
                    impact_data = {
                        "Independent Variable": independent_variables,
                        "Impact Value": coefficients,
                        "Absolute Impact (%)": (np.abs(coefficients) / total_impact) * 100
                    }
                    impact_df = pd.DataFrame(impact_data)
                    impact_df = impact_df.sort_values(by="Absolute Impact (%)", ascending=False)
                    impact_df["Absolute Impact (%)"] = impact_df["Absolute Impact (%)"].map("{:.2f}%".format)
                    st.table(impact_df[["Independent Variable", "Impact Value", "Absolute Impact (%)"]])

                    # Step 8: Prepare Data for AI Analysis
                    st.header("Step 8: Prepare Data For AI Analysis")

                    # Prepare AI data
                    if "processed_data" in st.session_state:
                        processed_data = st.session_state["processed_data"]

                        # Summary statistics for simulation
                        mean_prediction = np.mean(predictions)
                        median_prediction = np.median(predictions)
                        std_dev_prediction = np.std(predictions)
                        avg_r2 = mse  # Using MSE as a placeholder for R2 in this example

                        # Bin data for probability calculations
                        bin_edges = np.histogram_bin_edges(predictions, bins=10)
                        bin_counts, _ = np.histogram(predictions, bins=bin_edges)
                        bin_data = [{"Range": f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}", "Probability": f"{bin_counts[i]/len(predictions)*100:.2f}%"} for i in range(len(bin_counts))]

                        # Prepare AI input
                        ai_data = {
                            "target_value": target_value,
                            "simulation_summary": {
                                "mean_prediction": mean_prediction,
                                "median_prediction": median_prediction,
                                "std_dev_prediction": std_dev_prediction,
                                "avg_r2": avg_r2,
                                "num_iterations": num_simulations
                            },
                            "probability_data": bin_data,
                            "risk_factors": impact_df[["Independent Variable", "Impact Value", "Absolute Impact (%)"]].to_dict(orient="records")
                        }

                        # Convert all ndarray objects in ai_data to lists (for serializability)
                        def convert_ndarray_to_list(data):
                            if isinstance(data, dict):
                                return {key: convert_ndarray_to_list(value) for key, value in data.items()}
                            elif isinstance(data, np.ndarray):
                                return data.tolist()
                            elif isinstance(data, list):
                                return [convert_ndarray_to_list(item) for item in data]
                            else:
                                return data

                        ai_data = convert_ndarray_to_list(ai_data)

                        # Request GPT analysis
                        def get_gpt_analysis(data):
                            openai.api_key = os.getenv("OPENAI_API_KEY")

                            try:
                                response = openai.ChatCompletion.create(
                                    model="gpt-4",
                                    messages=[
                                        {"role": "system", "content": "You are an AI assistant that provides analysis based on the provided data."},
                                        {"role": "user", "content": f"Please analyze the following data: {data}"}
                                    ]
                                )
                                return response["choices"][0]["message"]["content"]
                            except Exception as e:
                                return f"Error occurred: {e}"

                        gpt_response = get_gpt_analysis(ai_data)

                        # Display GPT response
                        st.subheader("Step 7: GPT Analysis Result:")
                        st.write(gpt_response)
                    else:
                        st.write("No processed data available in session state.")
if __name__ == "__main__":
    main()
