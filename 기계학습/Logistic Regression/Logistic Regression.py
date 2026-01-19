#Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()



#Logistic Regression(statsmodels)
import statsmodels.api as sm

X_con = sm.add_constant(X)

sm_model = sm.Logit(y, X_con)



#Parameter
model = LogisticRegression(
    penalty='l2', 		# L2 정규화 사용
    C=0.5,  		# 정규화 강도 (낮을수록 강한 정규화)
    fit_intercept=True, 	# 절편을 포함
    random_state=42,	# 결과 재현을 위한 난수 시드
    solver='lbfgs', 	# 최적화를 위한 알고리즘
    max_iter=100,	# 최대 반복 횟수
    multi_class='auto',	# 다중 클래스 처리 방식
    verbose=0, 		# 로그 출력 정도 (0은 출력하지 않음)
    n_jobs=1 		# 사용할 CPU 코어 수 (1은 하나의 코어 사용)
)