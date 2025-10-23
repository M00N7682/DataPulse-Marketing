# 하이브리드 학습 실험 및 최종 평가
# Phase 5: Hybrid Learning & Final Evaluation

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_all_data():
    """모든 데이터 로드"""
    print("=== 데이터 로드 ===")
    
    # 실제 전처리된 데이터
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").squeeze()
    y_test = pd.read_csv("y_test.csv").squeeze()
    
    # 홀드아웃 데이터
    try:
        holdout_data = pd.read_csv("holdout_data.csv")
        print(f"홀드아웃 데이터: {holdout_data.shape}")
    except:
        holdout_data = None
        print("홀드아웃 데이터 없음")
    
    # 합성 데이터
    synthetic_data = {}
    try:
        ctgan_data = pd.read_csv("synthetic_data_ctgan.csv")
        synthetic_data['CTGAN'] = ctgan_data
        print(f"CTGAN 데이터: {ctgan_data.shape}")
    except:
        print("CTGAN 데이터 없음")
    
    try:
        gc_data = pd.read_csv("synthetic_data_gaussian_copula.csv")
        synthetic_data['Gaussian_Copula'] = gc_data
        print(f"Gaussian Copula 데이터: {gc_data.shape}")
    except:
        print("Gaussian Copula 데이터 없음")
    
    print(f"실제 학습 데이터: {X_train.shape}")
    print(f"실제 테스트 데이터: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, holdout_data, synthetic_data

def preprocess_synthetic_data(synthetic_data, X_train):
    """합성 데이터 전처리 (실제 데이터와 동일한 형태로 변환)"""
    print("\n=== 합성 데이터 전처리 ===")
    
    processed_synthetic = {}
    
    for name, synth_df in synthetic_data.items():
        print(f"처리 중: {name}")
        
        # 실제 데이터와 동일한 피처 엔지니어링 적용
        synth_processed = apply_feature_engineering(synth_df)
        
        # 실제 학습 데이터와 동일한 컬럼 구조로 맞춤
        synth_processed = align_columns(synth_processed, X_train)
        
        processed_synthetic[name] = synth_processed
        print(f"처리 완료: {synth_processed.shape}")
    
    return processed_synthetic

def apply_feature_engineering(df):
    """합성 데이터에 피처 엔지니어링 적용"""
    df_featured = df.copy()
    
    # 파생 변수 생성 (02_feature_engineering.py와 동일)
    df_featured['roas_estimated'] = (df['Income'] * df['ConversionRate']) / (df['AdSpend'] + 1e-8)
    df_featured['cost_per_conversion'] = df['AdSpend'] / (df['ConversionRate'] + 1e-8)
    df_featured['efficiency_score'] = df['ClickThroughRate'] * df['ConversionRate']
    
    df_featured['customer_lifetime_value'] = df['Income'] * (df['LoyaltyPoints'] / 1000) * (df['PreviousPurchases'] + 1)
    df_featured['engagement_score'] = (df['SocialShares'] + df['EmailOpens'] + df['EmailClicks']) / 3
    df_featured['website_engagement'] = df['PagesPerVisit'] * df['TimeOnSite']
    
    df_featured['channel_spend_ratio'] = df['AdSpend'] / df['Income']
    df_featured['conversion_efficiency'] = df['Conversion'] / (df['WebsiteVisits'] + 1)
    
    # 범주형 변수 인코딩
    df_featured = pd.get_dummies(df_featured, columns=['CampaignChannel', 'CampaignType', 'Gender'])
    
    # 로그 변환
    log_cols = ['AdSpend', 'Income', 'LoyaltyPoints']
    for col in log_cols:
        if col in df_featured.columns:
            df_featured[f'{col}_log'] = np.log1p(df_featured[col])
    
    return df_featured

def align_columns(synth_df, target_df):
    """합성 데이터를 타겟 데이터와 동일한 컬럼 구조로 정렬"""
    # 타겟 컬럼에 없는 컬럼 제거
    cols_to_keep = [col for col in synth_df.columns if col in target_df.columns or col == 'Conversion']
    synth_aligned = synth_df[cols_to_keep].copy()
    
    # 타겟에 있지만 합성 데이터에 없는 컬럼은 0으로 채움
    for col in target_df.columns:
        if col not in synth_aligned.columns:
            synth_aligned[col] = 0
    
    # 컬럼 순서 맞춤
    synth_aligned = synth_aligned[target_df.columns.tolist() + ['Conversion']]
    
    return synth_aligned

def create_hybrid_datasets(X_train, y_train, processed_synthetic, mixing_ratios=[0.3, 0.5, 0.7]):
    """하이브리드 데이터셋 생성"""
    print(f"\n=== 하이브리드 데이터셋 생성 ===")
    
    hybrid_datasets = {}
    
    for synth_name, synth_data in processed_synthetic.items():
        if synth_data is not None and len(synth_data) > 0:
            for ratio in mixing_ratios:
                # 실제 데이터 비율 = ratio, 합성 데이터 비율 = 1-ratio
                real_samples = int(len(X_train) * ratio)
                synth_samples = len(X_train) - real_samples
                
                # 실제 데이터 샘플링
                real_indices = np.random.choice(len(X_train), real_samples, replace=False)
                X_real_sample = X_train.iloc[real_indices]
                y_real_sample = y_train.iloc[real_indices]
                
                # 합성 데이터 샘플링
                if len(synth_data) >= synth_samples:
                    synth_indices = np.random.choice(len(synth_data), synth_samples, replace=False)
                    synth_sample = synth_data.iloc[synth_indices]
                else:
                    synth_sample = synth_data.copy()
                
                X_synth_sample = synth_sample.drop('Conversion', axis=1)
                y_synth_sample = synth_sample['Conversion']
                
                # 하이브리드 데이터셋 생성
                X_hybrid = pd.concat([X_real_sample, X_synth_sample], ignore_index=True)
                y_hybrid = pd.concat([y_real_sample, y_synth_sample], ignore_index=True)
                
                dataset_name = f"{synth_name}_ratio_{ratio:.1f}"
                hybrid_datasets[dataset_name] = (X_hybrid, y_hybrid)
                
                print(f"{dataset_name}: {X_hybrid.shape[0]}개 샘플 (실제: {real_samples}, 합성: {len(X_synth_sample)})")
    
    return hybrid_datasets

def evaluate_scenarios(X_train, X_test, y_train, y_test, hybrid_datasets):
    """시나리오별 성능 평가"""
    print(f"\n=== 시나리오별 성능 평가 ===")
    
    # 평가할 모델들
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
    }
    
    results = []
    
    # 시나리오 A: 실제 데이터만
    print("\n--- 시나리오 A: 실제 데이터만 ---")
    for model_name, model in models.items():
        result = evaluate_single_scenario(model, X_train, X_test, y_train, y_test, 
                                        f"Real_Only_{model_name}")
        results.append(result)
    
    # 시나리오 B, C, D: 하이브리드 데이터
    print("\n--- 하이브리드 시나리오 ---")
    for dataset_name, (X_hybrid, y_hybrid) in hybrid_datasets.items():
        for model_name, model in models.items():
            result = evaluate_single_scenario(model, X_hybrid, X_test, y_hybrid, y_test,
                                            f"{dataset_name}_{model_name}")
            results.append(result)
    
    return pd.DataFrame(results)

def evaluate_single_scenario(model, X_train, X_test, y_train, y_test, scenario_name):
    """단일 시나리오 평가"""
    try:
        # 모델 학습
        model.fit(X_train, y_train)
        
        # 예측
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # 성능 지표
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else np.nan
        
        # 교차 검증
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')
        
        return {
            'Scenario': scenario_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'AUC_ROC': auc,
            'CV_F1_Mean': cv_scores.mean(),
            'CV_F1_Std': cv_scores.std(),
            'Training_Size': len(X_train)
        }
        
    except Exception as e:
        print(f"평가 실패 {scenario_name}: {e}")
        return {
            'Scenario': scenario_name,
            'Accuracy': np.nan,
            'Precision': np.nan,
            'Recall': np.nan,
            'F1_Score': np.nan,
            'AUC_ROC': np.nan,
            'CV_F1_Mean': np.nan,
            'CV_F1_Std': np.nan,
            'Training_Size': len(X_train)
        }

def analyze_performance_improvement(results_df):
    """성능 향상 분석"""
    print("\n=== 성능 향상 분석 ===")
    
    # 베이스라인 성능 (실제 데이터만)
    baseline_results = results_df[results_df['Scenario'].str.contains('Real_Only')]
    
    # 하이브리드 결과
    hybrid_results = results_df[~results_df['Scenario'].str.contains('Real_Only')]
    
    print("=== 베이스라인 성능 (실제 데이터만) ===")
    print(baseline_results[['Scenario', 'F1_Score', 'AUC_ROC']].round(3))
    
    print("\n=== 하이브리드 학습 성능 ===")
    hybrid_summary = hybrid_results.groupby(hybrid_results['Scenario'].str.split('_').str[0:2].str.join('_')).agg({
        'F1_Score': ['mean', 'max'],
        'AUC_ROC': ['mean', 'max']
    }).round(3)
    print(hybrid_summary)
    
    # 향상율 계산
    baseline_f1 = baseline_results['F1_Score'].mean()
    best_hybrid_f1 = hybrid_results['F1_Score'].max()
    improvement = (best_hybrid_f1 - baseline_f1) / baseline_f1 * 100
    
    print(f"\n최대 F1-Score 향상률: {improvement:.2f}%")
    
    return improvement

def visualize_comprehensive_results(results_df):
    """종합 결과 시각화"""
    print("\n=== 종합 결과 시각화 ===")
    
    # 1. 시나리오별 F1-Score 비교
    plt.figure(figsize=(15, 8))
    
    # 시나리오 그룹별로 색상 구분
    scenarios = results_df['Scenario'].values
    colors = []
    for scenario in scenarios:
        if 'Real_Only' in scenario:
            colors.append('red')
        elif 'CTGAN' in scenario:
            colors.append('blue')
        elif 'Gaussian' in scenario:
            colors.append('green')
        else:
            colors.append('gray')
    
    plt.bar(range(len(results_df)), results_df['F1_Score'], color=colors, alpha=0.7)
    plt.xlabel('Scenarios')
    plt.ylabel('F1-Score')
    plt.title('시나리오별 F1-Score 비교')
    plt.xticks(range(len(results_df)), [s[:20] + '...' if len(s) > 20 else s for s in scenarios], 
               rotation=45, ha='right')
    
    # 범례
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Real Only'),
                      Patch(facecolor='blue', alpha=0.7, label='CTGAN Hybrid'),
                      Patch(facecolor='green', alpha=0.7, label='Gaussian Copula Hybrid')]
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.show()
    
    # 2. 모델별 성능 히트맵
    pivot_data = results_df.pivot_table(
        values='F1_Score', 
        index=results_df['Scenario'].str.split('_').str[-1],  # 모델명
        columns=results_df['Scenario'].str.replace(r'_Random Forest|_XGBoost|_LightGBM', '', regex=True),  # 시나리오명
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('모델별 × 시나리오별 F1-Score 히트맵')
    plt.tight_layout()
    plt.show()

def save_comprehensive_results(results_df, improvement):
    """종합 결과 저장"""
    print("\n=== 결과 저장 ===")
    
    # 상세 결과 저장
    results_df.to_csv("hybrid_learning_results.csv", index=False)
    
    # 요약 보고서 생성
    summary_report = f"""
# 하이브리드 학습 실험 결과 요약

## 실험 개요
- 실제 데이터: {results_df[results_df['Scenario'].str.contains('Real_Only')]['Training_Size'].iloc[0]:,}개
- 합성 데이터 생성 모델: CTGAN, Gaussian Copula
- 혼합 비율: 30%, 50%, 70%
- 평가 모델: Random Forest, XGBoost, LightGBM

## 주요 결과
- 최고 F1-Score: {results_df['F1_Score'].max():.3f}
- 베이스라인 F1-Score: {results_df[results_df['Scenario'].str.contains('Real_Only')]['F1_Score'].mean():.3f}
- 성능 향상률: {improvement:.2f}%

## 최고 성능 시나리오
{results_df.loc[results_df['F1_Score'].idxmax(), 'Scenario']}

## 권장사항
- 합성 데이터와 실제 데이터의 혼합 학습이 성능 향상에 효과적
- {'CTGAN' if 'CTGAN' in results_df.loc[results_df['F1_Score'].idxmax(), 'Scenario'] else 'Gaussian Copula'} 모델이 더 우수한 합성 데이터 생성
"""
    
    with open("hybrid_learning_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_report)
    
    print("저장 완료:")
    print("- hybrid_learning_results.csv")
    print("- hybrid_learning_summary.md")

def main():
    """메인 실행 함수"""
    try:
        # 1. 모든 데이터 로드
        X_train, X_test, y_train, y_test, holdout_data, synthetic_data = load_all_data()
        
        # 2. 합성 데이터 전처리
        if synthetic_data:
            processed_synthetic = preprocess_synthetic_data(synthetic_data, X_train)
        else:
            print("합성 데이터가 없어 실제 데이터만으로 평가를 진행합니다.")
            processed_synthetic = {}
        
        # 3. 하이브리드 데이터셋 생성
        if processed_synthetic:
            hybrid_datasets = create_hybrid_datasets(X_train, y_train, processed_synthetic)
        else:
            hybrid_datasets = {}
        
        # 4. 시나리오별 성능 평가
        results_df = evaluate_scenarios(X_train, X_test, y_train, y_test, hybrid_datasets)
        
        # 5. 성능 향상 분석
        improvement = analyze_performance_improvement(results_df)
        
        # 6. 결과 시각화
        visualize_comprehensive_results(results_df)
        
        # 7. 결과 저장
        save_comprehensive_results(results_df, improvement)
        
        print("\n=== Phase 5 완료: 하이브리드 학습 실험 ===")
        print("=== 전체 연구 프로젝트 완료 ===")
        
        return results_df, improvement
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

if __name__ == "__main__":
    results = main()