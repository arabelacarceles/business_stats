import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#----------------------------------------------------------#
#Author1DataCleaning: Team 4/MBAN--------------------------#
#Author2DataCleaning: Team 3/MBAN--------------------------#
#Business Statistics---------------------------------------#
#Used ChatGPT to solve some code related problems----------#
#----------------------------------------------------------#



#The imported datased has the three surveys conducted combined
#into a single sheet. 

df1 = pd.read_excel(r'Surveys.xlsx', sheet_name='Surveys1')
df2 = pd.read_excel(r'Surveys.xlsx', sheet_name='Surveys2')
df3 = pd.read_excel(r'Surveys.xlsx', sheet_name='Surveys3')

#-------------------------DATASET FORMAT----------------------

def clean_company_columns(df):
    company_cols = df.filter(regex='^Company').columns
    df[company_cols] = df[company_cols].replace(["nan", ""], np.nan)
    df[company_cols] = df[company_cols].apply(pd.to_numeric, errors='coerce')
    df_cleaned = df.dropna(subset=company_cols, how='any')
    return df_cleaned

df1=clean_company_columns(df1)
df2=clean_company_columns(df2)
df3=clean_company_columns(df3)

#Delete Timestamp and Email columns
df1 = df1.drop(columns=['Timestamp', 'Email address'])
df2 = df2.drop(columns=['Timestamp', 'Email address'])
df3 = df3.drop(columns=['Timestamp', 'Email address'])
#Rename Gender column
df1 = df1.rename(columns={'¿Gender?': 'Gender'})
df2 = df2.rename(columns={'¿Gender?': 'Gender'})
df3 = df3.rename(columns={'¿Gender?': 'Gender'})
#Gender as a binary variable
df1['Gender'] = df1['Gender'].map({'Female': 1, 'Male': 0})
df2['Gender'] = df2['Gender'].map({'Female': 1, 'Male': 0})
df3['Gender'] = df3['Gender'].map({'Female': 1, 'Male': 0})

# This matrix is based on the combination for every company profile
matrix_data = {
    'a': [1, 1, 1],
    'b': [1, 1, 0],
    'c': [1, 0, 1],
    'd': [1, 0, 0],
    'e': [0, 1, 1],
    'f': [0, 1, 0],
    'g': [0, 0, 1],
    'h': [0, 0, 0]
}
matrix_df = pd.DataFrame(matrix_data, index=['Graphs/Tables', '5years/1year', 'Risks/NoRisks']).T

# Crear el DataFrame final


def process_and_combine_company_data(dataframes, matrix_df):
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame to store all results

    for df in dataframes:
        final_df = pd.DataFrame()

        # Iterate over each row in the dataframe
        for idx, row in df.iterrows():
            for i in range(1, 4):  # Assume there are 3 company columns per row
                # Find the columns for Company and Confidence based on the beginning of their names
                company_col = df.filter(regex=f'^Company {i}').columns[0]
                confidence_col = df.filter(regex=f'^Confidence Company {i}').columns[0]

                # Extract the letter from the column name
                if '(' in company_col and ')' in company_col:
                    letter = company_col.split('(')[-1][0]  # Get the first letter in parentheses

                    # Check if the letter exists in the matrix_df
                    if letter in matrix_df.index:
                        # Create a new row with only the relevant values
                        new_row = {
                            'Gender': row['Gender'],
                            'Experience': row['Experience'],
                            'Aggressiveness (Risk Tolerance)': row['Aggressiveness (Risk Tolerance)'],
                            'Market Knowledge': row['Market Knowledge'],
                            'Investment Horizon': row['Investment Horizon'],
                            'Size Company': row['Size Company'],
                            'Sector': row['Sector'],
                            'Company age': row['Company age'],
                            'Growth Potential': row['Growth Potential'],
                            'Graphs/Tables': matrix_df.loc[letter, 'Graphs/Tables'],
                            '5years/1year': matrix_df.loc[letter, '5years/1year'],
                            'Risks/NoRisks': matrix_df.loc[letter, 'Risks/NoRisks'],
                            'Amount': row[company_col],
                            'Confidence': row[confidence_col]
                        }

                        # Append the new row to the final DataFrame for this dataframe
                        final_df = pd.concat([final_df, pd.DataFrame([new_row])], ignore_index=True)
                    else:
                        print(f"Warning: Letter '{letter}' not found in matrix_df for column {company_col}")
                else:
                    print(f"Warning: Expected format with '(letter)' in column {company_col}")

        # Concatenate the processed DataFrame to the combined DataFrame
        combined_df = pd.concat([combined_df, final_df], ignore_index=True)

    return combined_df

df = process_and_combine_company_data([df1,df2,df3], matrix_df)

#Deeleting no significant Amount values
df['Amount'] = df['Amount'].apply(lambda x: x if x >= 10 else None)
df = df.dropna(subset=['Amount'])

#Check if there are null values on independent values
#Substitute them with average rounded to next int
df = df.fillna(np.ceil(df.mean(numeric_only=True)))
print("Null values:\n",df.isnull().sum())

#Create Excel to perform regression analysis
df.to_excel('surveys_cleaned_data.xlsx')


#-------------------------PRELIMINARY ANALYSIS----------------------

#Mean values of Amount and Confidence by each factor
mean_by_visualization = df.groupby('Graphs/Tables')[['Amount', 'Confidence']].mean()
print("\nMean by Graphs/Tables:\n", mean_by_visualization)
mean_by_time_frame = df.groupby('5years/1year')[['Amount', 'Confidence']].mean()
print("\nMean by 5years/1year:\n", mean_by_time_frame)
mean_by_risk = df.groupby('Risks/NoRisks')[['Amount', 'Confidence']].mean()
print("\nMean by Risks/NoRisks:\n", mean_by_risk)


#Box plots for Amount and Confidence by each factor
plt.figure(figsize=(10, 6))
sns.boxplot(x='Graphs/Tables', y='Amount', data=df)
plt.title('Amount by Graphs/Tables')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='5years/1year', y='Amount', data=df)
plt.title('Amount by 5years/1year')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Risks/NoRisks', y='Amount', data=df)
plt.title('Amount by Risks/NoRisks')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Graphs/Tables', y='Confidence', data=df)
plt.title('Confidence by Graphs/Tables')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='5years/1year', y='Confidence', data=df)
plt.title('Confidence by 5years/1year')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Risks/NoRisks', y='Confidence', data=df)
plt.title('Confidence by Risks/NoRisks')
plt.show()


#Historgrams for Amount and Confidence
plt.figure(figsize=(10, 6))
sns.histplot(df['Amount'], bins=20, kde=True)
plt.title('Distribution of Amount')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['Confidence'], bins=20, kde=True)
plt.title('Distribution of Confidence')
plt.show()

# Correlation Matrix of Confidence with all other factors (excluding Amount)
confidence_correlations = df.drop(columns=['Amount']).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(confidence_correlations, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix with Confidence and All Factors (excluding Amount)")
plt.show()

# Correlation Matrix of Amount with all other factors (excluding Confidence)
amount_correlations = df.drop(columns=['Confidence']).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(amount_correlations, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix with Amount and All Factors (excluding Confidence)")
plt.show()




