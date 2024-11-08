import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats


#----------------------------------------------------------#
#Author MLR Assumptions: Team4/MBAN
#Business Statistics---------------------------------------#
#Used ChatGPT to solve some code related problems----------#
#----------------------------------------------------------#

#Excel that has cleaned data
data = pd.read_excel('surveys_cleaned_data.xlsx')


# Define X variables for each model, excluding the other dependent variable
X_amount = data.drop(columns=['Amount', 'Confidence'])  # Exclude 'Confidence' for the 'Amount' model
X_confidence = data.drop(columns=['Amount', 'Confidence'])  # Exclude 'Amount' for the 'Confidence' model
X_amount = sm.add_constant(X_amount)  # Add constant for intercept in 'Amount' model
X_confidence = sm.add_constant(X_confidence)  # Add constant for intercept in 'Confidence' model

# Define dependent variables
y_amount = data['Amount']
y_confidence = data['Confidence']


# Function to perform diagnostics on a model
def perform_diagnostics(y, X, model_name):
    # Fit the model
    model = sm.OLS(y, X).fit()
    
    print(f"\n--- Diagnostics for {model_name} Model ---\n")

    # 1. Linearity and Homoscedasticity Check: Residuals vs Fitted Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(model.fittedvalues, model.resid, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title(f"Residuals vs Fitted Values ({model_name})")
    plt.show()

    # Breusch-Pagan Test for Homoscedasticity
    bp_test_stat, bp_test_p_value, _, _ = het_breuschpagan(model.resid, model.model.exog)
    print(f"{model_name} - Breusch-Pagan Test for Homoscedasticity:")
    print(f"  Statistic: {bp_test_stat}, p-value: {bp_test_p_value}")
    if bp_test_p_value > 0.05:
        print("  Interpretation: Homoscedasticity assumption is met (p > 0.05).")
    else:
        print("  Interpretation: Homoscedasticity assumption is violated (p <= 0.05).")

    # 2. Normality of Residuals: Q-Q Plot and Shapiro-Wilk Test
    plt.figure(figsize=(6, 6))
    sm.qqplot(model.resid, line='s', ax=plt.gca())
    plt.title(f"Q-Q Plot of Residuals ({model_name})")
    plt.show()

    # Shapiro-Wilk Test for Normality
    shapiro_test_stat, shapiro_test_p_value = stats.shapiro(model.resid)
    print(f"{model_name} - Shapiro-Wilk Test for Normality:")
    print(f"  Statistic: {shapiro_test_stat}, p-value: {shapiro_test_p_value}")
    if shapiro_test_p_value > 0.05:
        print("  Interpretation: Normality assumption is met (p > 0.05).")
    else:
        print("  Interpretation: Normality assumption is violated (p <= 0.05).")

    # 3. Independence of Observations: Durbin-Watson Test
    dw_stat = durbin_watson(model.resid)
    print(f"{model_name} - Durbin-Watson Test for Independence of Observations:")
    print(f"  Statistic: {dw_stat}")
    if 1.5 < dw_stat < 2.5:
        print("  Interpretation: Independence of observations assumption is likely met (statistic between 1.5 and 2.5).")
    else:
        print("  Interpretation: Independence of observations assumption may be violated (statistic outside 1.5 - 2.5 range).")

    # 4. Multicollinearity Check: Variance Inflation Factor (VIF)
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(f"{model_name} - Variance Inflation Factor (VIF) for Multicollinearity:")
    print(vif_data)
    if vif_data['VIF'].max() < 5:
        print("  Interpretation: Multicollinearity is low (all VIF < 5).")
    else:
        print("  Interpretation: Potential multicollinearity concern (some VIF >= 5).")

# Run diagnostics for the "Amount" model
perform_diagnostics(y_amount, X_amount, "Amount")

# Run diagnostics for the "Confidence" model
perform_diagnostics(y_confidence, X_confidence, "Confidence")

