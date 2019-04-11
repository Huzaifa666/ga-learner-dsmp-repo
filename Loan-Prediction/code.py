# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 



# code starts here
bank = pd.read_csv(path)
categorical_var = bank.select_dtypes(include = 'object')

numerical_var = bank.select_dtypes(include = 'number')



# code ends here


# --------------
# code starts here
banks = bank.drop('Loan_ID', axis = 1)
print(banks.isnull().sum())

bank_mode=banks.mode()
#simply using a forloop with object 
for x in banks.columns.values:
    banks[x]=banks[x].fillna(value=bank_mode[x].iloc[0])
#code ends here


# --------------
# Code starts here

avg_loan_amount = pd.pivot_table(banks, values = 'LoanAmount', index = ['Gender', 'Married', 'Self_Employed'] , aggfunc = 'mean')


# code ends here



# --------------
# code starts here

loan_approved_se = len(banks[(banks['Self_Employed'] == 'Yes')& (banks['Loan_Status'] == 'Y')])
loan_approved_nse = len(banks[(banks['Self_Employed'] == 'No') & (banks['Loan_Status'] == 'Y')])

percentage_se = (loan_approved_se/614)*100
percentage_nse = (loan_approved_nse/614)*100
# code ends here


# --------------
# code starts here

loan_term = banks['Loan_Amount_Term'].apply(lambda x : x / 12)
big_loan_term = len(loan_term[lambda x : x >= 25])



# code ends here


# --------------
# code starts here

loan_groupby = banks.groupby('Loan_Status')
loan_groupby = loan_groupby['ApplicantIncome','Credit_History']
mean_values = loan_groupby.agg(np.mean)


# code ends here


