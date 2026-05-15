import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
df = pd.read_csv('data/kbo_pitcher_processed.csv')
print(f"투수 수: {len(df)}명")

# 변수 설정
feature_cols = ['K%', 'BB%', 'K-BB%', 'HR%', 'FIP', 'WHIP', 'IP/G']

X = df[feature_cols].dropna()
df_clean = df.loc[X.index].copy()

# 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means 클러스터링으로 타겟 생성
kmeans = KMeans(n_clusters=3, random_state=42)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)

# 클러스터 이름
cluster_names = {0: '피홈런형', 1: '안정형', 2: '불안정형'}
df_clean['유형'] = df_clean['cluster'].map(cluster_names)

print("\n클러스터별 분포:")
print(df_clean['유형'].value_counts())

# 분류 모델
X_cls = df_clean[feature_cols]
y_cls = df_clean['cluster']

X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

# 랜덤포레스트 분류 모델
rf_cls = RandomForestClassifier(n_estimators=100, random_state=42)
rf_cls.fit(X_train, y_train)

# 예측 및 성능 평가
y_pred = rf_cls.predict(X_test)

print(f"\n분류 모델 성능")
print(f"정확도: {accuracy_score(y_test, y_pred):.4f}")
print("\n분류 리포트:")
print(classification_report(y_test, y_pred))
print("\n혼동행렬:")
print(confusion_matrix(y_test, y_pred))

# 변수 중요도
importances = pd.Series(rf_cls.feature_importances_, index=feature_cols)
print("\n변수 중요도:")
print(importances.sort_values(ascending=False).round(3))

# 예측 결과 샘플
df_clean['예측유형'] = rf_cls.predict(X_cls).tolist()
df_clean['예측유형'] = df_clean['예측유형'].map(cluster_names)
print("\n예측 결과 샘플:")
print(df_clean[['Name', 'Team', '유형', '예측유형']].head(10))

# 모델 저장
joblib.dump(rf_cls, 'data/type_model.pkl')
print("\n분류 모델 저장 완료!")