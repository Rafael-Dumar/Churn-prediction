#import
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

def generate_churn_report():
    # import data
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # preprocess data
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors ='coerce')
    # fill NaN values in TotalCharges with 0
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # convert categorical variables to numerical
    X_num = df.select_dtypes(include=['int64', 'float64'])
    X_cat = df.select_dtypes(include=['object']).drop(columns=['customerID', 'Churn'])
    # convert categorical variables to dummy variables
    X_cat = pd.get_dummies(X_cat, drop_first=True)
    # concatenate numerical and categorical variables
    X_encoded = pd.concat([X_num, X_cat], axis=1) 
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0) # convert churn to binary (1 for Yes, 0 for No)
    # save the preprocessed data
    model_columns = X_encoded.columns
    joblib.dump(model_columns, 'model_columns.pkl')

    # scaling the data
    Scaler = StandardScaler() # create a scaler object
    X_scaled = Scaler.fit_transform(X_encoded) # scale the features
    # save the scaler for future use
    joblib.dump(Scaler, 'scaler.pkl')

    # training the model
    model = LogisticRegression(random_state=42) # create a logistic regression model
    model.fit(X_scaled, y) # fit the model on the scaled data
    # save the model for future use
    joblib.dump(model, 'churn_model.pkl') 


    # make predictions
    churn_probabilities = model.predict_proba(X_scaled)[:, 1] # get probabilities of churn(Yes)

    # Creating a DataFrame for the report
    report_df = pd.DataFrame({
        'CustomerID': df['customerID'],
        'Churn Probability': churn_probabilities,
        'Churn Prediction': ['Yes' if prob > 0.5 else 'No' for prob in churn_probabilities]
    }).sort_values(by='Churn Probability', ascending=False) # Sort by churn probability in descending order

    # Save the report to a CSV file
    report_df.to_csv('churn_risk_report.csv', index=False) # Save the report to a CSV file

    print(f'Churn report generated! report saved as "churn_risk_report.csv" with {len(report_df)} records.')
    print('Showing the top 20 records of the report:')
    print(report_df.head(20)) # Display the top 20 records of the report
    print('Churn report generation completed successfully!')

# Main function to execute the churn report generation
if __name__ == "__main__":
    generate_churn_report()




