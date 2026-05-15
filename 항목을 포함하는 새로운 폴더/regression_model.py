import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
df = pd.read_csv('data/kbo_pitcher_processed.csv')
print(f"투수 수: {len(df)}명")

# 변수 설정
feature_cols = ['K%', 'BB%', 'K-BB%', 'HR%', 'FIP', 'WHIP', 'IP/G']
target = 'WAR▼'

df_reg = df.dropna(subset=feature_cols + [target])
X = df_reg[feature_cols]
y = df_reg[target]

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 랜덤포레스트 회귀 모델
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 예측 및 성능 평가
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n회귀 모델 성능")
print(f"MSE  : {mse:.4f}")
print(f"R²   : {r2:.4f}")

# 변수 중요도
importances = pd.Series(rf.feature_importances_, index=feature_cols)
print("\n변수 중요도:")
print(importances.sort_values(ascending=False).round(3))

# 전체 예측값 추가
df_reg = df_reg.copy()
df_reg['WAR_predicted'] = rf.predict(X)
print("\n예측 결과 샘플:")
print(df_reg[['Name', 'Team', 'WAR▼', 'WAR_predicted']].head(10).round(3))

# 모델 저장
joblib.dump(rf, 'data/war_model.pkl')
print("\n회귀 모델 저장 완료!")