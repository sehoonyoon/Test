import pandas as pd

# 데이터 로드
df = pd.read_csv('data/kbo_pitcher_all.csv', header=[0, 1])  # ✅ 변경

# 컬럼 단순화
df.columns = [col[0] if col[0] == col[1] else col[1] for col in df.columns]

# 숫자 변환
cols = ['SO', 'BB', 'HR', 'TBF', 'IP', 'G', 'FIP', 'WHIP', 'ERA', 'WAR▼']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ✅ 파생변수 계산
df['K%']    = df['SO'] / df['TBF']
df['BB%']   = df['BB'] / df['TBF']
df['K-BB%'] = df['K%'] - df['BB%']
df['HR%']   = df['HR'] / df['TBF']
df['IP/G']  = df['IP'] / df['G']

# 필요한 변수만 추출
result = df[['Name', 'Team', 'K%', 'BB%', 'K-BB%', 'HR%', 'FIP', 'WHIP', 'IP/G', 'WAR▼']].copy()

# 소수점 정리
result = result.round(3)

print(result.head(10))

# 저장
result.to_csv('data/kbo_pitcher_processed.csv', index=False, encoding='utf-8-sig')
print("저장 완료!")

# ✅ 타자 전처리
df_b = pd.read_csv('data/kbo_batter_all.csv', header=[0, 1])  # ✅ 변경

# 컬럼 단순화
df_b.columns = [col[0] if col[0] == col[1] else col[1] for col in df_b.columns]

print("\n타자 원본 컬럼:", df_b.columns.tolist())
print(df_b.head())

# 숫자 변환
b_cols = ['AB', 'H', 'BB', 'SO', 'HR', 'PA', 'OBP', 'SLG', 'OPS', 'WAR▼', 'AVG']
for col in b_cols:
    if col in df_b.columns:
        df_b[col] = pd.to_numeric(df_b[col], errors='coerce')

# 파생변수 계산
df_b['K%']  = df_b['SO'] / df_b['PA']
df_b['BB%'] = df_b['BB'] / df_b['PA']
df_b['ISO'] = df_b['SLG'] - df_b['AVG']

# 필요한 변수 추출
b_result = df_b[['Name', 'Team', 'OBP', 'SLG', 'OPS', 'BB%', 'K%', 'ISO', 'WAR▼']].copy()
b_result = b_result.round(3)

print("\n타자 전처리 결과:")
print(b_result.head(10))

b_result.to_csv('data/kbo_batter_processed.csv', index=False, encoding='utf-8-sig')
print("타자 저장 완료!")