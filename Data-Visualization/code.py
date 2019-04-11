# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(path)

loan_status = data['Loan_Status'].value_counts()
loan_status.plot(kind = 'bar')
#Code starts here


# --------------
#Code starts here
property_and_loan = data.groupby(['Property_Area', 'Loan_Status']).size().unstack()

property_and_loan.plot(kind = 'bar', figsize = (10,8))
plt.xticks(rotation = 45)
plt.xlabel('Property Area')
plt.ylabel('Loan_Status')
plt.show()


# --------------
#Code starts here
education_and_loan = data.groupby(['Education', 'Loan_Status']).size().unstack()
education_and_loan.plot(kind='bar', stacked = True)
plt.xlabel('Education Status')
plt.ylabel('Loan Status')
plt.xticks(rotation = 45)
plt.show()


# --------------
#Code starts here

graduate = data.loc[data['Education'] == 'Graduate']
not_graduate = data.loc[data['Education'] == 'Not Graduate']
graduate['LoanAmount'].plot( kind = 'density', label = 'Graduate')


not_graduate['LoanAmount'].plot(kind = 'density', label = 'Not Graduate')








#Code ends here

#For automatic legend display
plt.legend()


# --------------
#Code starts here
fig = plt.figure(figsize = (20,10))
ax_1 = fig.add_subplot(131)
plt.title('Applicant Income')
ax_2 = fig.add_subplot(132)
plt.title('Coapplicant Income')
ax_3 = fig.add_subplot(133)
plt.title('Total Income')
data.plot.scatter(x = 'ApplicantIncome', y = 'LoanAmount', c = 'orange', ax = ax_1)
data.plot.scatter(x = 'CoapplicantIncome', y = 'LoanAmount', c = 'red', ax = ax_2)
data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
data.plot.scatter(x = 'TotalIncome', y = 'LoanAmount', c = 'blue', ax = ax_3)


