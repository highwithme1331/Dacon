#Linear Regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()



#Linear Regression(statsmodels)
import statsmodels.api as sm

X_sm = sm.add_constant(X)

model = sm.OLS(y, X_sm)