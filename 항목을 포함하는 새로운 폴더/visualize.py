import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('data/kbo_pitcher_final.csv')
df_batter_viz = pd.read_csv('data/kbo_batter_clustered.csv')

cluster_names = {0: '안정형', 1: '불안정형', 2: '피홈런형'}
df['유형'] = df['cluster'].map(cluster_names)

features = ['K%', 'BB%', 'K-BB%', 'HR%', 'FIP', 'WHIP', 'IP/G']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

colors = {0: 'blue', 1: 'red', 2: 'green'}
plt.figure(figsize=(12, 8))
for cluster, group in df.groupby('cluster'):
    plt.scatter(group['PCA1'], group['PCA2'], label=cluster_names[cluster], color=colors[cluster], s=120)
    for _, row in group.iterrows():
        plt.annotate(row['Name'], (row['PCA1'], row['PCA2']), fontsize=8, ha='left', va='bottom')

explained = pca.explained_variance_ratio_
plt.xlabel(f'PC1 ({explained[0]*100:.1f}% 설명)')
plt.ylabel(f'PC2 ({explained[1]*100:.1f}% 설명)')
plt.title('투수 유형 클러스터링\n(7개 변수: 삼진율·볼넷율·지배력·홈런허용율·FIP·WHIP·이닝소화력)')
plt.legend()
plt.tight_layout()
plt.savefig('data/cluster_plot.png', dpi=150)
plt.show()
print("그래프 1 저장완료!")

importances = pd.Series({
    'WHIP\n(이닝당출루)': 0.620,
    'IP/G\n(이닝소화력)': 0.216,
    'FIP\n(수비무관방어율)': 0.052,
    'BB%\n(볼넷율)': 0.034,
    'K%\n(삼진율)': 0.028,
    'HR%\n(홈런허용율)': 0.027,
    'K-BB%\n(지배력)': 0.023
}).sort_values(ascending=True)

plt.figure(figsize=(9, 6))
importances.plot(kind='barh', color='steelblue')
plt.xlabel('중요도')
plt.title('WAR 예측 변수 중요도')
plt.tight_layout()
plt.savefig('data/importance_plot.png', dpi=150)
plt.show()
print("그래프 2 저장완료!")

plt.figure(figsize=(10, 7))
df['error'] = abs(df['WAR▼'] - df['WAR_predicted'])
for _, row in df.iterrows():
    size = 80 + row['error'] * 500
    plt.scatter(row['WAR▼'], row['WAR_predicted'], color=colors[row['cluster']], s=size, alpha=0.8, edgecolors='white', linewidth=0.5)
    plt.annotate(row['Name'], (row['WAR▼'], row['WAR_predicted']), fontsize=8, ha='left', va='bottom', xytext=(4, 4), textcoords='offset points')

plt.plot([df['WAR▼'].min(), df['WAR▼'].max()], [df['WAR▼'].min(), df['WAR▼'].max()], 'r--', linewidth=1.5)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='안정형'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='불안정형'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='피홈런형'),
    Line2D([0], [0], linestyle='--', color='red', label='완벽한 예측선'),
]
plt.legend(handles=legend_elements, loc='upper left')
plt.xlabel('실제 WAR')
plt.ylabel('예측 WAR')
plt.title('실제 WAR vs 예측 WAR\n(점 크기 = 예측 오차, 색상 = 투수 유형)')
plt.tight_layout()
plt.savefig('data/war_plot.png', dpi=150)
plt.show()
print("그래프 3 저장완료!")


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


def get_attack_point(row):
    points = []
    if row['cluster'] == 0:
        points.append("🎯 안정형 투수 — 볼카운트 싸움이 불리, 스트라이크 적극 공략")
    elif row['cluster'] == 1:
        points.append("👀 불안정형 투수 — 볼넷 기다리기 유리")
    else:
        points.append("⚡ 피홈런형 투수 — 장타 노려볼 만")
    if row['BB%'] > 0.09:
        points.append(f"✅ 볼넷율 높음 ({row['BB%']:.1%}) — 참을수록 유리")
    if row['K%'] > 0.22:
        points.append(f"⚠️ 삼진율 높음 ({row['K%']:.1%}) — 삼진 조심, 컨택 집중")
    if row['HR%'] > 0.02:
        points.append(f"💣 홈런 허용율 있음 ({row['HR%']:.1%}) — 장타 노려볼 만")
    if row['WHIP'] > 1.4:
        points.append(f"📈 WHIP 높음 ({row['WHIP']:.2f}) — 출루 가능성 높음")
    if row['IP/G'] < 5.5:
        points.append(f"⏱️ 이닝 소화력 낮음 ({row['IP/G']:.1f}이닝) — 불펜 압박 전략 유효")
    return points


def analyze_pitcher(name):
    row = df[df['Name'] == name]
    if row.empty:
        print(f"{name} 선수를 찾을 수 없어요.")
        return
    row = row.iloc[0]

    cluster_names_local = {0: '안정형', 1: '불안정형', 2: '피홈런형'}
    colors_local = {0: 'blue', 1: 'red', 2: 'green'}
    color = colors_local[row['cluster']]

    p_features = ['K%', 'BB%', 'K-BB%', 'HR%', 'FIP', 'WHIP', 'IP/G']
    labels = ['삼진율\nK%', '볼넷율\nBB%', '지배력\nK-BB%', '홈런허용\nHR%', 'FIP', 'WHIP', '이닝소화\nIP/G']

    mins = df[p_features].min()
    maxs = df[p_features].max()
    normalized = [(row[f] - mins[f]) / (maxs[f] - mins[f] + 1e-9) for f in p_features]
    for i, f in enumerate(p_features):
        if f in ['FIP', 'WHIP', 'BB%', 'HR%']:
            normalized[i] = 1 - normalized[i]

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, polar=True)
    radar_chart(ax1, normalized, labels, f"{name} ({cluster_names_local[row['cluster']]})", color)

    ax2 = fig.add_subplot(122)
    ax2.axis('off')

    attack_points = get_attack_point(row)
    report_text = f"[ {name} 투수 공략 리포트 ]\n\n"
    report_text += f"유형: {cluster_names_local[row['cluster']]}\n"
    report_text += f"실제 WAR: {row['WAR▼']}  |  예측 WAR: {row['WAR_predicted']:.2f}\n"
    report_text += f"K%: {row['K%']:.1%}  BB%: {row['BB%']:.1%}  FIP: {row['FIP']}\n"
    report_text += f"WHIP: {row['WHIP']}  IP/G: {row['IP/G']}\n\n"
    report_text += "━━━━━━━━━━━━━━━━━━━━\n공략 포인트\n━━━━━━━━━━━━━━━━━━━━\n\n"
    for p in attack_points:
        report_text += f"{p}\n\n"

    ax2.text(0.05, 0.95, report_text, transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.suptitle(f"{name} 투수 분석", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'data/report_{name}.png', dpi=150)
    plt.show()
    print(f"{name} 리포트 저장완료!")


def get_batter_attack_point(row):
    points = []
    if row['유형'] == '파워형':
        points.append("💪 파워형 타자 — 장타 생산 능력 높음, 실투 놓치지 않음")
    elif row['유형'] == '컨택형':
        points.append("🎯 컨택형 타자 — 삼진 적고 안타 생산, 볼배합 중요")
    elif row['유형'] == '출루형':
        points.append("👀 출루형 타자 — 선구안 좋고 볼넷 많음, 제구 흔들리면 위험")
    else:
        points.append("⚖️ 혼합형 타자 — 특정 약점 없음, 다양한 공략 필요")
    if row['K%'] > 0.20:
        points.append(f"✅ 삼진율 높음 ({row['K%']:.1%}) — 변화구로 헛스윙 유도 가능")
    if row['BB%'] > 0.13:
        points.append(f"⚠️ 볼넷율 높음 ({row['BB%']:.1%}) — 제구 흔들리면 출루 허용")
    if row['ISO'] > 0.20:
        points.append(f"💣 장타력 높음 (ISO {row['ISO']:.3f}) — 실투 조심 필요")
    if row['OBP'] > 0.42:
        points.append(f"📈 출루율 높음 ({row['OBP']:.3f}) — 주자 누적 위험")
    if row['K%'] < 0.10:
        points.append(f"🔴 삼진율 낮음 ({row['K%']:.1%}) — 삼진으로 잡기 어려움")
    return points


def analyze_batter(name):
    row = df_batter_viz[df_batter_viz['Name'] == name]
    if row.empty:
        print(f"{name} 선수를 찾을 수 없어요.")
        return
    row = row.iloc[0]

    type_colors = {'파워형': 'red', '컨택형': 'green', '출루형': 'blue', '혼합형': 'orange'}
    color = type_colors.get(row['유형'], 'gray')

    b_features = ['OBP', 'SLG', 'OPS', 'BB%', 'K%', 'ISO']
    b_labels = ['출루율\nOBP', '장타율\nSLG', 'OPS', '볼넷율\nBB%', '삼진율\nK%', '장타력\nISO']

    mins = df_batter_viz[b_features].min()
    maxs = df_batter_viz[b_features].max()
    normalized = [(row[f] - mins[f]) / (maxs[f] - mins[f] + 1e-9) for f in b_features]
    normalized[4] = 1 - normalized[4]

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, polar=True)
    radar_chart(ax1, normalized, b_labels, f"{name} ({row['유형']})", color)

    ax2 = fig.add_subplot(122)
    ax2.axis('off')

    attack_points = get_batter_attack_point(row)
    report_text = f"[ {name} 타자 분석 리포트 ]\n\n"
    report_text += f"유형: {row['유형']}\n"
    report_text += f"WAR: {row['WAR▼']}\n"
    report_text += f"OBP: {row['OBP']}  SLG: {row['SLG']}  OPS: {row['OPS']}\n"
    report_text += f"BB%: {row['BB%']:.1%}  K%: {row['K%']:.1%}  ISO: {row['ISO']:.3f}\n\n"
    report_text += "━━━━━━━━━━━━━━━━━━━━\n투수 입장 공략 포인트\n━━━━━━━━━━━━━━━━━━━━\n\n"
    for p in attack_points:
        report_text += f"{p}\n\n"

    ax2.text(0.05, 0.95, report_text, transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    plt.suptitle(f"{name} 타자 분석", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'data/report_batter_{name}.png', dpi=150)
    plt.show()
    print(f"{name} 타자 리포트 저장완료!")


def match_report(pitcher_name, batter_name):
    p = df[df['Name'] == pitcher_name]
    b = df_batter_viz[df_batter_viz['Name'] == batter_name]

    if p.empty:
        print(f"투수 {pitcher_name} 를 찾을 수 없어요.")
        return
    if b.empty:
        print(f"타자 {batter_name} 를 찾을 수 없어요.")
        return

    p = p.iloc[0]
    b = b.iloc[0]

    pitcher_cluster_names = {0: '안정형', 1: '불안정형', 2: '피홈런형'}
    p_type = pitcher_cluster_names[p['cluster']]
    b_type = b['유형']

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"⚾ {pitcher_name} ({p_type}) vs {batter_name} ({b_type})", fontsize=16, fontweight='bold')

    ax1 = fig.add_subplot(231, polar=True)
    p_features = ['K%', 'BB%', 'K-BB%', 'HR%', 'FIP', 'WHIP', 'IP/G']
    p_labels = ['삼진율\nK%', '볼넷율\nBB%', '지배력\nK-BB%', '홈런허용\nHR%', 'FIP', 'WHIP', '이닝소화\nIP/G']
    p_mins = df[p_features].min()
    p_maxs = df[p_features].max()
    p_norm = [(p[f] - p_mins[f]) / (p_maxs[f] - p_mins[f] + 1e-9) for f in p_features]
    for i, f in enumerate(p_features):
        if f in ['FIP', 'WHIP', 'BB%', 'HR%']:
            p_norm[i] = 1 - p_norm[i]
    radar_chart(ax1, p_norm, p_labels, f"{pitcher_name}\n({p_type})", 'blue')

    ax2 = fig.add_subplot(232)
    ax2.axis('off')

    advantage = []
    disadvantage = []
    if b['BB%'] > 0.12 and p['BB%'] > 0.08:
        advantage.append("✅ 선구안 좋음 + 투수 볼넷율 높음\n   → 출루 유리")
    if b['K%'] > 0.20 and p['K%'] > 0.22:
        disadvantage.append("⚠️ 타자 삼진율 높음 + 투수 삼진율 높음\n   → 삼진 위험")
    if b['ISO'] > 0.20 and p['HR%'] > 0.02:
        advantage.append("💣 장타력 높음 + 투수 홈런 허용\n   → 장타 기회")
    if p['WHIP'] > 1.3:
        advantage.append(f"📈 투수 WHIP 높음 ({p['WHIP']})\n   → 출루 가능성 높음")
    if p['FIP'] < 3.0 and b['OPS'] < 0.800:
        disadvantage.append("🔴 투수 FIP 낮음 + 타자 OPS 낮음\n   → 타자 불리")

    report = "[ 세이버메트릭스 지표 ]\n\n"
    report += f"[ 투수 : {pitcher_name} ]\n"
    report += f"  FIP  {p['FIP']}  (수비 무관 방어율 / 낮을수록 좋음)\n"
    report += f"  WHIP {p['WHIP']}  (이닝당 출루 허용 / 낮을수록 좋음)\n"
    report += f"  K%   {p['K%']:.1%}  (삼진 유도율 / 높을수록 좋음)\n"
    report += f"  BB%  {p['BB%']:.1%}  (볼넷 허용율 / 낮을수록 좋음)\n"
    report += f"  HR%  {p['HR%']:.1%}  (홈런 허용율 / 낮을수록 좋음)\n"
    report += f"  IP/G {p['IP/G']}  (경기당 이닝 소화 / 높을수록 좋음)\n"
    report += f"  WAR  {p['WAR▼']}  (대체선수 대비 기여도)\n\n"
    report += f"[ 타자 : {batter_name} ]\n"
    report += f"  OBP  {b['OBP']}  (출루율 / 얼마나 자주 나가는가)\n"
    report += f"  SLG  {b['SLG']}  (장타율 / 타격의 파워)\n"
    report += f"  OPS  {b['OPS']}  (출루율+장타율 / 종합 타격 능력)\n"
    report += f"  ISO  {b['ISO']:.3f}  (순수 장타력 / 높을수록 파워형)\n"
    report += f"  BB%  {b['BB%']:.1%}  (볼넷율 / 선구안 좋을수록 높음)\n"
    report += f"  K%   {b['K%']:.1%}  (삼진율 / 낮을수록 컨택 좋음)\n"
    report += f"  WAR  {b['WAR▼']}  (대체선수 대비 기여도)\n\n"
    report += "━━━━━━━━━━━━━━━━━\n[ 상성 분석 ]\n\n"

    if advantage:
        report += "타자 유리한 점\n"
        for a in advantage:
            report += f"{a}\n\n"
    if disadvantage:
        report += "타자 불리한 점\n"
        for d in disadvantage:
            report += f"{d}\n\n"
    if not advantage and not disadvantage:
        report += "→ 특별한 유불리 없음\n   기본기 싸움\n"

    ax2.text(0.05, 0.95, report, transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax3 = fig.add_subplot(233, polar=True)
    b_features = ['OBP', 'SLG', 'OPS', 'BB%', 'K%', 'ISO']
    b_labels = ['출루율\nOBP', '장타율\nSLG', 'OPS', '볼넷율\nBB%', '삼진율\nK%', '장타력\nISO']
    b_mins = df_batter_viz[b_features].min()
    b_maxs = df_batter_viz[b_features].max()
    b_norm = [(b[f] - b_mins[f]) / (b_maxs[f] - b_mins[f] + 1e-9) for f in b_features]
    b_norm[4] = 1 - b_norm[4]
    radar_chart(ax3, b_norm, b_labels, f"{batter_name}\n({b_type})", 'red')

    ax4 = fig.add_subplot(212)
    x = range(3)
    width = 0.35
    p_vals = [p['K%'], p['BB%'], p['WAR▼']]
    b_vals = [b['K%'], b['BB%'], b['WAR▼']]
    ax4.bar([i - width/2 for i in x], p_vals, width, label=f'{pitcher_name} (투수)', color='cornflowerblue', alpha=0.7)
    ax4.bar([i + width/2 for i in x], b_vals, width, label=f'{batter_name} (타자)', color='lightcoral', alpha=0.7)
    ax4.set_xticks(list(x))
    ax4.set_xticklabels(['삼진율 K%', '볼넷율 BB%', 'WAR'])
    ax4.set_title('주요 지표 비교')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(f'data/matchup_{pitcher_name}_vs_{batter_name}.png', dpi=150)
    plt.show()
    print(f"{pitcher_name} vs {batter_name} 매치업 리포트 저장완료!")


# ✅ 테스트
analyze_pitcher("올러")
analyze_batter("최주환")
analyze_batter("김민석")
match_report("올러", "최주환")
match_report("최민석", "김민석")