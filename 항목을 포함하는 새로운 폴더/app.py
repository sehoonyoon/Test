import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df_pitcher = pd.read_csv('data/kbo_pitcher_final.csv')
df_batter = pd.read_csv('data/kbo_batter_clustered.csv')

pitcher_cluster_names = {0: '피홈런형', 1: '안정형', 2: '불안정형'}
batter_type_colors = {'출루형': 'blue', '컨택형': 'green', '혼합형': 'orange', '삼진형': 'red'}
pitcher_colors = {0: 'green', 1: 'blue', 2: 'red'}

def radar_chart(ax, values, labels, title, color):
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += angles[:1]
    ax.plot(angles, values_plot, color=color, linewidth=2)
    ax.fill(angles, values_plot, color=color, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=15)
    ax.grid(True)

# 페이지 설정
st.set_page_config(page_title="KBO 투수-타자 분석", layout="wide")
st.title("⚾ KBO 투수-타자 분석")

# 사이드바 메뉴
menu = st.sidebar.selectbox(
    "분석 메뉴",
    ["투수 군집 분석", "투수 분석", "타자 군집 분석", "타자 분석", "회귀 분석", "분류 분석", "투수 vs 타자 매치업"]
)

# ===========================
# 투수 분석
# ===========================
if menu == "투수 분석":
    st.header("🔵 투수 분석 (23년~25년) ")
    pitcher_list = df_pitcher['Name'].tolist()
    selected = st.selectbox("투수 선택", pitcher_list)

    row = df_pitcher[df_pitcher['Name'] == selected].iloc[0]
    p_type = pitcher_cluster_names[row['cluster']]

    col1, col2 = st.columns(2)

    with col1:
        p_features = ['K%', 'BB%', 'K-BB%', 'HR%', 'FIP', 'WHIP', 'IP/G']
        labels = ['삼진율\nK%', '볼넷율\nBB%', '지배력\nK-BB%',
                  '홈런허용\nHR%', 'FIP', 'WHIP', '이닝소화\nIP/G']
        mins = df_pitcher[p_features].min()
        maxs = df_pitcher[p_features].max()
        norm = [(row[f] - mins[f]) / (maxs[f] - mins[f] + 1e-9) for f in p_features]
        for i, f in enumerate(p_features):
            if f in ['FIP', 'WHIP', 'BB%', 'HR%']:
                norm[i] = 1 - norm[i]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        radar_chart(ax, norm, labels, f"{selected} ({p_type})",
                    pitcher_colors[row['cluster']])
        st.pyplot(fig)

    with col2:
        st.subheader(f"📋 {selected} 리포트")
        st.markdown(f"**유형:** {p_type}")
        st.markdown(f"**실제 WAR:** {row['WAR▼']}  |  **예측 WAR:** {row['WAR_predicted']:.2f}")
        st.divider()
        st.markdown("**세이버메트릭스 지표**")
        st.markdown(f"- FIP `{row['FIP']}` — 수비 무관 방어율 (낮을수록 좋음)")
        st.markdown(f"- WHIP `{row['WHIP']}` — 이닝당 출루 허용 (낮을수록 좋음)")
        st.markdown(f"- K% `{row['K%']:.1%}` — 삼진 유도율 (높을수록 좋음)")
        st.markdown(f"- BB% `{row['BB%']:.1%}` — 볼넷 허용율 (낮을수록 좋음)")
        st.markdown(f"- HR% `{row['HR%']:.1%}` — 홈런 허용율 (낮을수록 좋음)")
        st.markdown(f"- IP/G `{row['IP/G']}` — 경기당 이닝 소화 (높을수록 좋음)")
        st.divider()
        st.markdown("**공략 포인트**")
        if row['cluster'] == 1:
            st.info("🎯 안정형 투수 — 초구부터 적극 공략 필요")
        elif row['cluster'] == 2:
            st.warning("👀 불안정형 투수 — 볼넷 기다리기 유리")
        else:
            st.error("⚡ 피홈런형 투수 — 장타 노려볼 만")
        if row['BB%'] > 0.09:
            st.success(f"✅ 볼넷율 높음 ({row['BB%']:.1%}) — 참을수록 유리")
        if row['K%'] > 0.22:
            st.warning(f"⚠️ 삼진율 높음 ({row['K%']:.1%}) — 컨택 집중")
        if row['WHIP'] > 1.4:
            st.success(f"📈 WHIP 높음 ({row['WHIP']}) — 출루 가능성 높음")

# ===========================
# 타자 분석
# ===========================
elif menu == "타자 분석":
    st.header("🔴 타자 분석(23년~25년)")
    batter_list = df_batter['Name'].tolist()
    selected = st.selectbox("타자 선택", batter_list)

    row = df_batter[df_batter['Name'] == selected].iloc[0]
    b_type = row['유형']
    color = batter_type_colors.get(b_type, 'gray')

    col1, col2 = st.columns(2)

    with col1:
        b_features = ['OBP', 'SLG', 'OPS', 'BB%', 'K%', 'ISO']
        b_labels = ['출루율\nOBP', '장타율\nSLG', 'OPS', '볼넷율\nBB%', '삼진율\nK%', '장타력\nISO']
        mins = df_batter[b_features].min()
        maxs = df_batter[b_features].max()
        norm = [(row[f] - mins[f]) / (maxs[f] - mins[f] + 1e-9) for f in b_features]
        norm[4] = 1 - norm[4]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        radar_chart(ax, norm, b_labels, f"{selected} ({b_type})", color)
        st.pyplot(fig)

    with col2:
        st.subheader(f"📋 {selected} 리포트")
        st.markdown(f"**유형:** {b_type}")
        st.markdown(f"**WAR:** {row['WAR▼']}")
        st.divider()
        st.markdown("**세이버메트릭스 지표**")
        st.markdown(f"- OBP `{row['OBP']}` — 출루율 (얼마나 자주 나가는가)")
        st.markdown(f"- SLG `{row['SLG']}` — 장타율 (타격의 파워)")
        st.markdown(f"- OPS `{row['OPS']}` — 출루율+장타율 (종합 타격 능력)")
        st.markdown(f"- ISO `{row['ISO']:.3f}` — 순수 장타력 (높을수록 파워형)")
        st.markdown(f"- BB% `{row['BB%']:.1%}` — 볼넷율 (선구안)")
        st.markdown(f"- K% `{row['K%']:.1%}` — 삼진율 (낮을수록 컨택 좋음)")
        st.divider()
        st.markdown("**투수 입장 공략 포인트**")
        if b_type == '출루형':
            st.warning("👀 출루형 타자 — 제구 흔들리면 볼넷 허용, 스트라이크 승부 필요")
        elif b_type == '컨택형':
            st.info("🎯 컨택형 타자 — 볼배합 중요, 변화구로 헛스윙 유도")
        elif b_type == '혼합형':
            st.info("⚖️ 혼합형 타자 — 컨택 좋고 균형잡힘, 다양한 공략 필요")
        elif b_type == '삼진형':
            st.error("⚡ 삼진형 타자 — 삼진 많음, 변화구로 적극 공략")
        if row['K%'] > 0.20:
            st.success(f"✅ 삼진율 높음 ({row['K%']:.1%}) — 헛스윙 유도 가능")
        if row['ISO'] > 0.20:
            st.warning(f"⚠️ 장타력 높음 (ISO {row['ISO']:.3f}) — 실투 조심")

# ===========================
# 투수 군집 분석
# ===========================
elif menu == "투수 군집 분석":
    st.header("🔵 투수 군집 분석(23년~25년)")

    st.subheader("📊 군집별 평균 지표")
    cluster_summary = df_pitcher.groupby('cluster')[
        ['K%', 'BB%', 'K-BB%', 'HR%', 'FIP', 'WHIP', 'IP/G']
    ].mean().round(3)
    cluster_summary.index = [pitcher_cluster_names[i] for i in cluster_summary.index]
    st.dataframe(cluster_summary)

    st.divider()

    st.subheader("📋 군집별 특성")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🟡 피홈런형 (클러스터 0)**")
        st.markdown("- HR% 높음 → 홈런 허용 많음")
        st.markdown("- FIP 높음 → 장타 허용 취약")
        st.markdown("- K-BB% 낮음 → 지배력 부족")
        st.markdown("- 대표: 알칸타라, 원태인, 윌커슨")
    with col2:
        st.markdown("**🟢 안정형 (클러스터 1)**")
        st.markdown("- K% 가장 높음 → 삼진 유도 강함")
        st.markdown("- FIP 낮음 → 실력 기반 성적")
        st.markdown("- BB% 낮음 → 제구 안정적")
        st.markdown("- 대표: 페디, 안우진, 올러")
    with col3:
        st.markdown("**🔴 불안정형 (클러스터 2)**")
        st.markdown("- BB% 높음 → 볼넷 허용 많음")
        st.markdown("- WHIP 높음 → 출루 허용 많음")
        st.markdown("- IP/G 낮음 → 이닝 소화 부족")
        st.markdown("- 대표: 고영표, 후라도, 뷰캐넌")

    st.divider()

    st.subheader("📈 투수 유형 분포 (PCA)")
    selected_pitcher = st.selectbox("투수 선택", df_pitcher['Name'].tolist())

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from matplotlib.lines import Line2D

    features = ['K%', 'BB%', 'K-BB%', 'HR%', 'FIP', 'WHIP', 'IP/G']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_pitcher[features])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_pitcher['PCA1'] = X_pca[:, 0]
    df_pitcher['PCA2'] = X_pca[:, 1]

    colors = {0: 'green', 1: 'blue', 2: 'red'}
    fig, ax = plt.subplots(figsize=(10, 6))

    for cluster, group in df_pitcher.groupby('cluster'):
        others = group[group['Name'] != selected_pitcher]
        ax.scatter(others['PCA1'], others['PCA2'], color=colors[cluster], s=60, alpha=0.2)

    selected_row = df_pitcher[df_pitcher['Name'] == selected_pitcher].iloc[0]
    ax.scatter(selected_row['PCA1'], selected_row['PCA2'],
               color=colors[selected_row['cluster']], s=300, zorder=5, edgecolors='black', linewidth=2)
    ax.annotate(selected_row['Name'], (selected_row['PCA1'], selected_row['PCA2']),
                fontsize=11, fontweight='bold', xytext=(8, 8), textcoords='offset points')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='피홈런형'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='안정형'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='불안정형'),
    ]
    ax.legend(handles=legend_elements)
    explained = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}% 설명)')
    ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}% 설명)')
    ax.set_title(f'{selected_pitcher} 위치 확인 - 투수 유형 분포')
    st.pyplot(fig)

    st.divider()
    st.subheader(f"📋 {selected_pitcher} 상세 지표")
    p_type = pitcher_cluster_names[selected_row['cluster']]
    st.markdown(f"**유형:** {p_type}")

    detail_df = pd.DataFrame({
        '지표': ['K%', 'BB%', 'K-BB%', 'HR%', 'FIP', 'WHIP', 'IP/G', 'WAR'],
        '설명': [
            '삼진 유도율 (높을수록 좋음)',
            '볼넷 허용율 (낮을수록 좋음)',
            '지배력 (높을수록 좋음)',
            '홈런 허용율 (낮을수록 좋음)',
            '수비 무관 방어율 (낮을수록 좋음)',
            '이닝당 출루 허용 (낮을수록 좋음)',
            '경기당 이닝 소화 (높을수록 좋음)',
            '대체선수 대비 기여도 (높을수록 좋음)'
        ],
        '값': [
            f"{selected_row['K%']:.1%}",
            f"{selected_row['BB%']:.1%}",
            f"{selected_row['K-BB%']:.1%}",
            f"{selected_row['HR%']:.1%}",
            f"{selected_row['FIP']}",
            f"{selected_row['WHIP']}",
            f"{selected_row['IP/G']}",
            f"{selected_row['WAR▼']}"
        ]
    })
    st.dataframe(detail_df, hide_index=True, use_container_width=True)

# ===========================
# 타자 군집 분석
# ===========================
elif menu == "타자 군집 분석":
    st.header("🔴 타자 군집 분석(23년~25년)")

    st.subheader("📊 군집별 평균 지표")
    cluster_summary_b = df_batter.groupby('cluster')[
        ['OBP', 'SLG', 'OPS', 'BB%', 'K%', 'ISO']
    ].mean().round(3)

    batter_cluster_names_local = {0: '출루형', 1: '컨택형', 2: '혼합형', 3: '삼진형'}
    cluster_summary_b.index = [batter_cluster_names_local[i] for i in cluster_summary_b.index]
    st.dataframe(cluster_summary_b)

    st.divider()

    st.subheader("📋 군집별 특성")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**🔵 출루형 (클러스터 0)**")
        st.markdown("- OPS/ISO 최고 → 장타+출루 모두 강함")
        st.markdown("- BB% 높음 → 선구안 좋음")
        st.markdown("- 대표: 노시환, 최정, 오스틴")
    with col2:
        st.markdown("**🟢 컨택형 (클러스터 1)**")
        st.markdown("- OPS 낮지만 K% 낮음 → 안타 중심")
        st.markdown("- ISO 낮음 → 장타보다 컨택")
        st.markdown("- 대표: 박민우, 김선빈, 홍창기")
    with col3:
        st.markdown("**🟡 혼합형 (클러스터 2)**")
        st.markdown("- K% 가장 낮음 → 컨택 좋음")
        st.markdown("- OPS 중간 → 균형잡힌 타자")
        st.markdown("- 대표: 박성한, 박찬호, 정수빈")
    with col4:
        st.markdown("**🔴 삼진형 (클러스터 3)**")
        st.markdown("- K% 가장 높음 → 삼진 많음")
        st.markdown("- ISO 중간 → 장타력 보통")
        st.markdown("- 대표: 강백호, 노시환(24), 박동원")

    st.divider()

    st.subheader("📈 타자 유형 분포 (PCA)")
    selected_batter = st.selectbox("타자 선택", df_batter['Name'].tolist())

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from matplotlib.lines import Line2D

    b_features = ['OBP', 'SLG', 'OPS', 'BB%', 'K%', 'ISO']
    scaler_b = StandardScaler()
    X_b_scaled = scaler_b.fit_transform(df_batter[b_features])
    pca_b = PCA(n_components=2)
    X_b_pca = pca_b.fit_transform(X_b_scaled)
    df_batter['PCA1'] = X_b_pca[:, 0]
    df_batter['PCA2'] = X_b_pca[:, 1]

    b_colors = {0: 'blue', 1: 'green', 2: 'orange', 3: 'red'}
    fig, ax = plt.subplots(figsize=(10, 6))

    for cluster, group in df_batter.groupby('cluster'):
        others = group[group['Name'] != selected_batter]
        ax.scatter(others['PCA1'], others['PCA2'], color=b_colors[cluster], s=60, alpha=0.2)

    selected_b_row = df_batter[df_batter['Name'] == selected_batter].iloc[0]
    ax.scatter(selected_b_row['PCA1'], selected_b_row['PCA2'],
               color=b_colors[selected_b_row['cluster']], s=300, zorder=5, edgecolors='black', linewidth=2)
    ax.annotate(selected_b_row['Name'], (selected_b_row['PCA1'], selected_b_row['PCA2']),
                fontsize=11, fontweight='bold', xytext=(8, 8), textcoords='offset points')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='출루형'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='컨택형'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='혼합형'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='삼진형'),
    ]
    ax.legend(handles=legend_elements)
    explained_b = pca_b.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({explained_b[0]*100:.1f}% 설명)')
    ax.set_ylabel(f'PC2 ({explained_b[1]*100:.1f}% 설명)')
    ax.set_title(f'{selected_batter} 위치 확인 - 타자 유형 분포')
    st.pyplot(fig)

    st.divider()
    st.subheader(f"📋 {selected_batter} 상세 지표")
    batter_cluster_names_fix = {0: '출루형', 1: '컨택형', 2: '혼합형', 3: '삼진형'}
    b_type = batter_cluster_names_fix[selected_b_row['cluster']]
    st.markdown(f"**유형:** {b_type}")

    detail_b_df = pd.DataFrame({
        '지표': ['OBP', 'SLG', 'OPS', 'ISO', 'BB%', 'K%', 'WAR'],
        '설명': [
            '출루율 (얼마나 자주 나가는가)',
            '장타율 (타격의 파워)',
            '출루율+장타율 (종합 타격 능력)',
            '순수 장타력 (높을수록 파워형)',
            '볼넷율 (선구안 좋을수록 높음)',
            '삼진율 (낮을수록 컨택 좋음)',
            '대체선수 대비 기여도 (높을수록 좋음)'
        ],
        '값': [
            f"{selected_b_row['OBP']}",
            f"{selected_b_row['SLG']}",
            f"{selected_b_row['OPS']}",
            f"{selected_b_row['ISO']:.3f}",
            f"{selected_b_row['BB%']:.1%}",
            f"{selected_b_row['K%']:.1%}",
            f"{selected_b_row['WAR▼']}"
        ]
    })
    st.dataframe(detail_b_df, hide_index=True, use_container_width=True)

# ===========================
# 회귀 분석
# ===========================
elif menu == "회귀 분석":
    st.header("📈 회귀 분석 (WAR 예측)")
    st.markdown("7개 변수로 투수 WAR를 예측하는 랜덤포레스트 회귀 모델 (2023~2025 시즌 67명)")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 모델 성능")
        st.metric("R² (설명력)", "0.6736")
        st.metric("MSE (평균제곱오차)", "0.8279")
        st.markdown("- **R² = 0.67** → 투수 WAR의 67%를 7개 변수로 설명 가능")
        st.markdown("- **독립변수** → K%, BB%, K-BB%, HR%, FIP, WHIP, IP/G")
        st.markdown("- **종속변수** → WAR (대체선수 대비 기여도)")

    with col2:
        st.subheader("🔑 변수 중요도")
        importances_reg = {
            'WHIP\n(이닝당출루)': 0.534,
            'IP/G\n(이닝소화력)': 0.309,
            'BB%\n(볼넷율)': 0.041,
            'K%\n(삼진율)': 0.034,
            'FIP\n(수비무관방어율)': 0.030,
            'HR%\n(홈런허용율)': 0.026,
            'K-BB%\n(지배력)': 0.025,
        }
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.barh(list(importances_reg.keys()), list(importances_reg.values()), color='cornflowerblue')
        ax1.set_title('회귀 변수 중요도')
        st.pyplot(fig1)

    st.divider()
    st.subheader("💡 해석")
    st.markdown("- **WHIP (53.4%)** → 이닝당 출루 허용이 WAR에 가장 큰 영향")
    st.markdown("- **IP/G (30.9%)** → 이닝 소화력이 두 번째로 중요")
    st.markdown("- WHIP + IP/G 합산 84.3% → 이 두 변수가 투수 가치를 결정")

    st.divider()
    st.subheader("📋 실제 vs 예측 WAR 비교")
    import joblib
    try:
        rf = joblib.load('data/war_model.pkl')
        feature_cols = ['K%', 'BB%', 'K-BB%', 'HR%', 'FIP', 'WHIP', 'IP/G']
        df_reg = df_pitcher.dropna(subset=feature_cols + ['WAR▼'])
        df_reg = df_reg.copy()
        df_reg['WAR_predicted'] = rf.predict(df_reg[feature_cols])
        df_reg['오차'] = (df_reg['WAR▼'] - df_reg['WAR_predicted']).round(3)
        st.dataframe(
            df_reg[['Name', 'Team', 'WAR▼', 'WAR_predicted', '오차']].round(3),
            hide_index=True, use_container_width=True
        )
    except:
        st.warning("model.py를 먼저 실행해주세요.")

# ===========================
# 분류 분석
# ===========================
elif menu == "분류 분석":
    st.header("🎯 분류 분석 (투수 유형 예측)")
    st.markdown("7개 변수로 투수 유형(안정형/불안정형/피홈런형)을 예측하는 랜덤포레스트 분류 모델")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 모델 성능")
        st.metric("정확도", "0.8333")
        st.success("✅ 2023~2025 시즌 67명 기반 (과적합 해결)")
        st.markdown("- **특징변수** → K%, BB%, K-BB%, HR%, FIP, WHIP, IP/G")
        st.markdown("- **타겟변수** → 투수 유형 (안정형 / 불안정형 / 피홈런형)")
        st.divider()
        st.markdown("**클러스터별 분포**")
        dist_df = pd.DataFrame({
            '유형': ['불안정형', '피홈런형', '안정형'],
            '인원': [31, 18, 10]
        })
        fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
        ax_pie.pie(dist_df['인원'], labels=dist_df['유형'],
                   autopct='%1.1f%%', colors=['red', 'green', 'blue'])
        ax_pie.set_title('투수 유형 분포')
        st.pyplot(fig_pie)

    with col2:
        st.subheader("🔑 변수 중요도")
        importances_cls = {
            'HR%\n(홈런허용율)': 0.253,
            'WHIP\n(이닝당출루)': 0.221,
            'K-BB%\n(지배력)': 0.136,
            'K%\n(삼진율)': 0.129,
            'FIP\n(수비무관방어율)': 0.125,
            'BB%\n(볼넷율)': 0.076,
            'IP/G\n(이닝소화력)': 0.059,
        }
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.barh(list(importances_cls.keys()), list(importances_cls.values()), color='lightcoral')
        ax2.set_title('분류 변수 중요도')
        st.pyplot(fig2)

    st.divider()
    st.subheader("💡 해석")
    st.markdown("- **HR% (25.3%)** → 홈런 허용율이 투수 유형을 가장 잘 구분")
    st.markdown("- **WHIP (22.1%)** → 이닝당 출루 허용이 두 번째로 중요")
    st.markdown("- 회귀와 달리 분류는 HR%, WHIP, K-BB% 중요 → 유형 구분에는 장타/지배력이 핵심")

# ===========================
# 투수 vs 타자 매치업
# ===========================
elif menu == "투수 vs 타자 매치업":
    st.header("⚾ 투수 vs 타자 매치업 분석")

    col_p, col_b = st.columns(2)
    with col_p:
        pitcher = st.selectbox("투수 선택", df_pitcher['Name'].tolist())
    with col_b:
        batter = st.selectbox("타자 선택", df_batter['Name'].tolist())

    p = df_pitcher[df_pitcher['Name'] == pitcher].iloc[0]
    b = df_batter[df_batter['Name'] == batter].iloc[0]
    p_type = pitcher_cluster_names[p['cluster']]
    b_type = b['유형']

    st.subheader(f"🆚 {pitcher} ({p_type}) vs {batter} ({b_type})")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        p_features = ['K%', 'BB%', 'K-BB%', 'HR%', 'FIP', 'WHIP', 'IP/G']
        p_labels = ['삼진율\nK%', '볼넷율\nBB%', '지배력\nK-BB%',
                    '홈런허용\nHR%', 'FIP', 'WHIP', '이닝소화\nIP/G']
        mins = df_pitcher[p_features].min()
        maxs = df_pitcher[p_features].max()
        p_norm = [(p[f] - mins[f]) / (maxs[f] - mins[f] + 1e-9) for f in p_features]
        for i, f in enumerate(p_features):
            if f in ['FIP', 'WHIP', 'BB%', 'HR%']:
                p_norm[i] = 1 - p_norm[i]
        fig1, ax1 = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        radar_chart(ax1, p_norm, p_labels, f"{pitcher}\n({p_type})", pitcher_colors[p['cluster']])
        st.pyplot(fig1)

    with col2:
        st.subheader("📊 상성 분석")
        st.markdown("**세이버메트릭스**")
        st.markdown(f"투수 FIP `{p['FIP']}` — 수비 무관 방어율 (낮을수록 좋음)")
        st.markdown(f"투수 WHIP `{p['WHIP']}` — 이닝당 출루 허용 (낮을수록 좋음)")
        st.markdown(f"투수 WAR `{p['WAR▼']}` — 대체선수 대비 기여도 (높을수록 좋음)")
        st.markdown(f"타자 OPS `{b['OPS']}` — 출루율+장타율 종합 타격 능력")
        st.markdown(f"타자 WAR `{b['WAR▼']}` — 대체선수 대비 기여도 (높을수록 좋음)")
        st.divider()

        advantage = []
        disadvantage = []
        if b['BB%'] > 0.12 and p['BB%'] > 0.08:
            advantage.append("✅ 선구안 좋음 + 투수 볼넷율 높음 → 출루 유리")
        if b['K%'] > 0.20 and p['K%'] > 0.22:
            disadvantage.append("⚠️ 삼진율 높음 + 투수 삼진율 높음 → 삼진 위험")
        if b['ISO'] > 0.20 and p['HR%'] > 0.02:
            advantage.append("💣 장타력 높음 + 투수 홈런 허용 → 장타 기회")
        if p['WHIP'] > 1.3:
            advantage.append(f"📈 투수 WHIP 높음 ({p['WHIP']}) → 출루 가능성")
        if p['FIP'] < 3.0 and b['OPS'] < 0.800:
            disadvantage.append("🔴 투수 FIP 낮음 + 타자 OPS 낮음 → 타자 불리")

        if advantage:
            st.markdown("**타자 유리한 점**")
            for a in advantage:
                st.success(a)
        if disadvantage:
            st.markdown("**타자 불리한 점**")
            for d in disadvantage:
                st.error(d)
        if not advantage and not disadvantage:
            st.info("→ 특별한 유불리 없음, 기본기 싸움")

    with col3:
        b_features = ['OBP', 'SLG', 'OPS', 'BB%', 'K%', 'ISO']
        b_labels = ['출루율\nOBP', '장타율\nSLG', 'OPS', '볼넷율\nBB%', '삼진율\nK%', '장타력\nISO']
        mins = df_batter[b_features].min()
        maxs = df_batter[b_features].max()
        b_norm = [(b[f] - mins[f]) / (maxs[f] - mins[f] + 1e-9) for f in b_features]
        b_norm[4] = 1 - b_norm[4]
        fig2, ax2 = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        radar_chart(ax2, b_norm, b_labels, f"{batter}\n({b_type})", batter_type_colors.get(b_type, 'gray'))
        st.pyplot(fig2)

    st.divider()
    st.subheader("📈 주요 지표 비교")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    x = range(3)
    width = 0.35
    ax3.bar([i - width/2 for i in x], [p['K%'], p['BB%'], p['WAR▼']],
            width, label=f'{pitcher} (투수)', color='cornflowerblue', alpha=0.8)
    ax3.bar([i + width/2 for i in x], [b['K%'], b['BB%'], b['WAR▼']],
            width, label=f'{batter} (타자)', color='lightcoral', alpha=0.8)
    ax3.set_xticks(list(x))
    ax3.set_xticklabels(['삼진율 K%', '볼넷율 BB%', 'WAR'])
    ax3.legend()
    st.pyplot(fig3)
