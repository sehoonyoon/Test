import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Mac)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_csv('data/kbo_pitcher_processed.csv')
print(f"투수 수: {len(df)}명")

# 클러스터링에 사용할 변수
features = ['K%', 'BB%', 'K-BB%', 'HR%', 'FIP', 'WHIP', 'IP/G']

X = df[features].dropna()
df_clean = df.loc[X.index].copy()

# 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means 클러스터링 (3개 유형)
kmeans = KMeans(n_clusters=3, random_state=42)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)

# 클러스터별 평균
cluster_summary = df_clean.groupby('cluster')[features].mean().round(3)
print("\n클러스터별 평균:")
print(cluster_summary)

# 투수별 클러스터 확인
print("\n투수 유형 분류:")
print(df_clean[['Name', 'Team', 'cluster']].to_string(index=False))

# 저장
df_clean.to_csv('data/kbo_pitcher_clustered.csv', index=False, encoding='utf-8-sig')
print("\n저장 완료!")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 회귀 모델 - WAR 예측
target = 'WAR▼'
feature_cols = ['K%', 'BB%', 'K-BB%', 'HR%', 'FIP', 'WHIP', 'IP/G']

df_reg = df_clean.dropna(subset=feature_cols + [target])

X_reg = df_reg[feature_cols]
y_reg = df_reg[target]

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# 랜덤포레스트 모델
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 예측
y_pred = rf.predict(X_test)

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n회귀 모델 성능")
print(f"MSE  : {mse:.4f}")
print(f"R²   : {r2:.4f}")

# 변수 중요도
importances = pd.Series(rf.feature_importances_, index=feature_cols)
print("\n변수 중요도:")
print(importances.sort_values(ascending=False).round(3))

# 전체 투수 WAR 예측값 추가
df_clean['WAR_predicted'] = rf.predict(X_reg)
df_clean[['Name', 'Team', 'cluster', 'K%', 'BB%', 'K-BB%', 'HR%', 'FIP', 'WHIP', 'IP/G', 'WAR▼', 'WAR_predicted']].round(3).to_csv(
    'data/kbo_pitcher_final.csv', index=False, encoding='utf-8-sig'
)

print("\n최종 저장 완료!")

# ✅ 타자 클러스터링
df_batter = pd.read_csv('data/kbo_batter_processed.csv')

# 타자 클러스터링 변수
b_features = ['OBP', 'SLG', 'OPS', 'BB%', 'K%', 'ISO']

X_b = df_batter[b_features].dropna()
df_batter_clean = df_batter.loc[X_b.index].copy()

# 정규화
scaler_b = StandardScaler()
X_b_scaled = scaler_b.fit_transform(X_b)

# K-Means (4개 유형)
kmeans_b = KMeans(n_clusters=4, random_state=42)
df_batter_clean['cluster'] = kmeans_b.fit_predict(X_b_scaled)

# 클러스터별 평균
b_cluster_summary = df_batter_clean.groupby('cluster')[b_features].mean().round(3)
print("\n타자 클러스터별 평균:")
print(b_cluster_summary)

# 클러스터 이름 자동 매핑
means = b_cluster_summary
batter_cluster_names = {}
batter_cluster_names[means['OPS'].idxmax()] = '출루형'
batter_cluster_names[means['K%'].idxmin()] = '컨택형'
batter_cluster_names[means['K%'].idxmax()] = '삼진형'

# ✅ 4개로 수정
for i in range(4):
    if i not in batter_cluster_names:
        batter_cluster_names[i] = '혼합형'

df_batter_clean['유형'] = df_batter_clean['cluster'].map(batter_cluster_names)

print("\n타자 유형 분류:")
print(df_batter_clean[['Name', 'Team', '유형']].to_string(index=False))

# 저장
df_batter_clean.to_csv('data/kbo_batter_clustered.csv', index=False, encoding='utf-8-sig')
print("\n타자 저장 완료!")

# ✅ 투수 vs 타자 상성 분석 함수
pitcher_cluster_names = {0: '안정형', 1: '불안정형', 2: '피홈런형'}  # ✅ 수정

def match_analysis(pitcher_name, batter_name):
    p = df_clean[df_clean['Name'] == pitcher_name]
    b = df_batter_clean[df_batter_clean['Name'] == batter_name]

    if p.empty:
        print(f"투수 {pitcher_name} 를 찾을 수 없어요.")
        return
    if b.empty:
        print(f"타자 {batter_name} 를 찾을 수 없어요.")
        return

    p = p.iloc[0]
    b = b.iloc[0]

    p_type = pitcher_cluster_names[p['cluster']]
    b_type = b['유형']

    print(f"\n{'='*40}")
    print(f"⚾ {pitcher_name} ({p_type}) vs {batter_name} ({b_type})")
    print(f"{'='*40}")

    advantage = []
    disadvantage = []

    if b['BB%'] > 0.12 and p['BB%'] > 0.08:
        advantage.append(f"✅ 타자 선구안 좋음 ({b['BB%']:.1%}) + 투수 볼넷율 높음 → 출루 유리")
    if b['K%'] > 0.20 and p['K%'] > 0.22:
        disadvantage.append(f"⚠️ 타자 삼진율 높음 ({b['K%']:.1%}) + 투수 삼진율 높음 → 삼진 위험")
    if b['ISO'] > 0.20 and p['HR%'] > 0.02:
        advantage.append(f"💣 타자 장타력 높음 (ISO {b['ISO']:.3f}) + 투수 홈런 허용 → 장타 기회")
    if p['WHIP'] > 1.3:
        advantage.append(f"📈 투수 WHIP 높음 ({p['WHIP']}) → 출루 가능성 높음")
    if p['FIP'] < 3.0 and b['OPS'] < 0.800:
        disadvantage.append(f"🔴 투수 FIP 낮음 ({p['FIP']}) + 타자 OPS 낮음 → 불리한 상황")

    if advantage:
        print("\n[ 타자 유리한 점 ]")
        for a in advantage:
            print(f"  {a}")
    if disadvantage:
        print("\n[ 타자 불리한 점 ]")
        for d in disadvantage:
            print(f"  {d}")
    if not advantage and not disadvantage:
        print("\n  → 특별한 유불리 없음, 기본기 싸움")

    print(f"{'='*40}\n")

# ✅ 분류분석 - 투수 유형 예측
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 분류 모델
target_cls = 'cluster'
X_cls = df_clean[feature_cols]
y_cls = df_clean[target_cls]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

rf_cls = RandomForestClassifier(n_estimators=100, random_state=42)
rf_cls.fit(X_train_c, y_train_c)

y_pred_c = rf_cls.predict(X_test_c)

print("\n분류분석 모델 성능")
print(f"정확도: {accuracy_score(y_test_c, y_pred_c):.4f}")
print("\n분류 리포트:")
print(classification_report(y_test_c, y_pred_c))
print("\n혼동행렬:")
print(confusion_matrix(y_test_c, y_pred_c))

# 분류 변수 중요도
cls_importances = pd.Series(rf_cls.feature_importances_, index=feature_cols)
print("\n분류 변수 중요도:")
print(cls_importances.sort_values(ascending=False).round(3))

# 모델 저장
import joblib
joblib.dump(rf, 'data/war_model.pkl')
joblib.dump(rf_cls, 'data/type_model.pkl')
print("\n모델 저장 완료!")