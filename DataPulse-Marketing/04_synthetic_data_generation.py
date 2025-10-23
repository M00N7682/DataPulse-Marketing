# 합성 데이터 생성 및 품질 평가
# Phase 4: Synthetic Data Generation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import wasserstein_distance
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# CTGAN 및 SDV 라이브러리
try:
    from ctgan import CTGAN
    from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    print("SDV 및 CTGAN 라이브러리 로드 완료")
except ImportError as e:
    print(f"라이브러리 설치 필요: {e}")

def load_original_data():
    """원본 데이터 로드"""
    print("=== 원본 데이터 로드 ===")
    
    df = pd.read_csv("digital_marketing_campaign_dataset.csv")
    print(f"데이터 형태: {df.shape}")
    
    # 합성 데이터 생성용 주요 변수 선택
    key_features = [
        'Age', 'Gender', 'Income', 'CampaignChannel', 'CampaignType',
        'AdSpend', 'ClickThroughRate', 'ConversionRate', 'WebsiteVisits',
        'PagesPerVisit', 'TimeOnSite', 'SocialShares', 'EmailOpens', 
        'EmailClicks', 'PreviousPurchases', 'LoyaltyPoints', 'Conversion'
    ]
    
    df_selected = df[key_features].copy()
    print(f"선택된 변수 수: {len(key_features)}")
    
    return df_selected

def create_metadata(df):
    """메타데이터 생성"""
    print("\n=== 메타데이터 생성 ===")
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    
    # 범주형 변수 명시적 지정
    categorical_cols = ['Gender', 'CampaignChannel', 'CampaignType']
    for col in categorical_cols:
        metadata.update_column(col, sdtype='categorical')
    
    # 이진 변수 지정
    metadata.update_column('Conversion', sdtype='categorical')
    
    print("메타데이터 생성 완료")
    return metadata

def generate_ctgan_data(df, num_samples=8000):
    """CTGAN으로 합성 데이터 생성"""
    print(f"\n=== CTGAN 합성 데이터 생성 ({num_samples}개) ===")
    
    try:
        # CTGAN 모델 초기화
        ctgan = CTGAN(epochs=300, batch_size=500, generator_lr=2e-4, discriminator_lr=2e-4)
        
        print("CTGAN 학습 시작...")
        ctgan.fit(df)
        
        print("합성 데이터 생성 중...")
        synthetic_data = ctgan.sample(num_samples)
        
        print(f"CTGAN 합성 데이터 생성 완료: {synthetic_data.shape}")
        return synthetic_data, ctgan
        
    except Exception as e:
        print(f"CTGAN 생성 실패: {e}")
        return None, None

def generate_gaussian_copula_data(df, metadata, num_samples=8000):
    """Gaussian Copula로 합성 데이터 생성"""
    print(f"\n=== Gaussian Copula 합성 데이터 생성 ({num_samples}개) ===")
    
    try:
        # Gaussian Copula 모델 초기화
        gc_synthesizer = GaussianCopulaSynthesizer(metadata)
        
        print("Gaussian Copula 학습 시작...")
        gc_synthesizer.fit(df)
        
        print("합성 데이터 생성 중...")
        synthetic_data = gc_synthesizer.sample(num_rows=num_samples)
        
        print(f"Gaussian Copula 합성 데이터 생성 완료: {synthetic_data.shape}")
        return synthetic_data, gc_synthesizer
        
    except Exception as e:
        print(f"Gaussian Copula 생성 실패: {e}")
        return None, None

def calculate_statistical_similarity(real_data, synthetic_data):
    """통계적 유사성 계산"""
    print("\n=== 통계적 유사성 평가 ===")
    
    results = {}
    
    # 수치형 변수에 대한 KS 테스트
    numeric_cols = real_data.select_dtypes(include=[np.number]).columns
    ks_distances = []
    
    for col in numeric_cols:
        ks_stat, ks_p = stats.ks_2samp(real_data[col], synthetic_data[col])
        ks_distances.append(ks_stat)
        
    results['KS_Distance_Mean'] = np.mean(ks_distances)
    results['KS_Distance_Std'] = np.std(ks_distances)
    
    # Wasserstein Distance
    wasserstein_distances = []
    for col in numeric_cols:
        wd = wasserstein_distance(real_data[col], synthetic_data[col])
        wasserstein_distances.append(wd)
        
    results['Wasserstein_Distance_Mean'] = np.mean(wasserstein_distances)
    results['Wasserstein_Distance_Std'] = np.std(wasserstein_distances)
    
    print(f"평균 KS Distance: {results['KS_Distance_Mean']:.4f}")
    print(f"평균 Wasserstein Distance: {results['Wasserstein_Distance_Mean']:.4f}")
    
    return results

def calculate_correlation_preservation(real_data, synthetic_data):
    """상관관계 보존 평가"""
    print("\n=== 상관관계 보존 평가 ===")
    
    # 수치형 변수만 선택
    numeric_cols = real_data.select_dtypes(include=[np.number]).columns
    
    real_corr = real_data[numeric_cols].corr()
    synthetic_corr = synthetic_data[numeric_cols].corr()
    
    # 상관관계 매트릭스 차이 계산
    corr_diff = np.abs(real_corr - synthetic_corr)
    correlation_preservation = 1 - corr_diff.mean().mean()
    
    print(f"상관관계 보존율: {correlation_preservation:.4f}")
    
    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 실제 데이터 상관관계
    sns.heatmap(real_corr, annot=False, cmap='RdBu_r', center=0, ax=axes[0])
    axes[0].set_title('Real Data Correlation')
    
    # 합성 데이터 상관관계
    sns.heatmap(synthetic_corr, annot=False, cmap='RdBu_r', center=0, ax=axes[1])
    axes[1].set_title('Synthetic Data Correlation')
    
    # 차이
    sns.heatmap(corr_diff, annot=False, cmap='Reds', ax=axes[2])
    axes[2].set_title('Correlation Difference')
    
    plt.tight_layout()
    plt.show()
    
    return correlation_preservation

def visualize_distributions(real_data, synthetic_data, model_name="Synthetic"):
    """분포 비교 시각화"""
    print(f"\n=== {model_name} 데이터 분포 비교 ===")
    
    # 주요 수치형 변수 선택
    numeric_cols = ['Age', 'Income', 'AdSpend', 'ClickThroughRate', 'ConversionRate', 'WebsiteVisits']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if col in real_data.columns and col in synthetic_data.columns:
            # KDE 플롯
            axes[i].hist(real_data[col], alpha=0.5, density=True, label='Real', bins=30)
            axes[i].hist(synthetic_data[col], alpha=0.5, density=True, label='Synthetic', bins=30)
            axes[i].set_title(f'{col} Distribution')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} vs Real Data Distribution Comparison')
    plt.tight_layout()
    plt.show()

def evaluate_categorical_similarity(real_data, synthetic_data):
    """범주형 변수 유사성 평가"""
    print("\n=== 범주형 변수 유사성 평가 ===")
    
    categorical_cols = ['Gender', 'CampaignChannel', 'CampaignType', 'Conversion']
    
    results = {}
    
    for col in categorical_cols:
        if col in real_data.columns and col in synthetic_data.columns:
            # 비율 비교
            real_dist = real_data[col].value_counts(normalize=True).sort_index()
            synthetic_dist = synthetic_data[col].value_counts(normalize=True).sort_index()
            
            # 공통 카테고리만 비교
            common_cats = set(real_dist.index) & set(synthetic_dist.index)
            
            if common_cats:
                real_common = real_dist[list(common_cats)]
                synthetic_common = synthetic_dist[list(common_cats)]
                
                # Jensen-Shannon Divergence 계산
                def js_divergence(p, q):
                    m = 0.5 * (p + q)
                    return 0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m)
                
                js_div = js_divergence(real_common.values, synthetic_common.values)
                results[f'{col}_JS_Divergence'] = js_div
                
                print(f"{col} JS Divergence: {js_div:.4f}")
    
    return results

def comprehensive_quality_evaluation(real_data, synthetic_models):
    """종합적인 품질 평가"""
    print("\n=== 종합 품질 평가 ===")
    
    evaluation_results = []
    
    for model_name, (synthetic_data, model) in synthetic_models.items():
        if synthetic_data is not None:
            print(f"\n--- {model_name} 평가 ---")
            
            # 통계적 유사성
            stat_similarity = calculate_statistical_similarity(real_data, synthetic_data)
            
            # 상관관계 보존
            corr_preservation = calculate_correlation_preservation(real_data, synthetic_data)
            
            # 범주형 변수 유사성
            cat_similarity = evaluate_categorical_similarity(real_data, synthetic_data)
            
            # 분포 시각화
            visualize_distributions(real_data, synthetic_data, model_name)
            
            # 결과 종합
            result = {
                'Model': model_name,
                'Correlation_Preservation': corr_preservation,
                **stat_similarity,
                **cat_similarity
            }
            
            evaluation_results.append(result)
    
    # 결과 DataFrame 생성
    results_df = pd.DataFrame(evaluation_results)
    print("\n=== 품질 평가 결과 요약 ===")
    print(results_df.round(4))
    
    return results_df

def save_synthetic_data(synthetic_models):
    """합성 데이터 저장"""
    print("\n=== 합성 데이터 저장 ===")
    
    for model_name, (synthetic_data, model) in synthetic_models.items():
        if synthetic_data is not None:
            filename = f"synthetic_data_{model_name.lower().replace(' ', '_')}.csv"
            synthetic_data.to_csv(filename, index=False)
            print(f"저장 완료: {filename}")

def main():
    """메인 실행 함수"""
    try:
        # 1. 원본 데이터 로드
        real_data = load_original_data()
        
        # 2. 메타데이터 생성
        metadata = create_metadata(real_data)
        
        # 3. 합성 데이터 생성
        print("\n" + "="*50)
        print("합성 데이터 생성 시작")
        print("="*50)
        
        # CTGAN
        ctgan_data, ctgan_model = generate_ctgan_data(real_data)
        
        # Gaussian Copula
        gc_data, gc_model = generate_gaussian_copula_data(real_data, metadata)
        
        # 4. 합성 모델 딕셔너리
        synthetic_models = {
            'CTGAN': (ctgan_data, ctgan_model),
            'Gaussian Copula': (gc_data, gc_model)
        }
        
        # 5. 품질 평가
        evaluation_results = comprehensive_quality_evaluation(real_data, synthetic_models)
        
        # 6. 합성 데이터 저장
        save_synthetic_data(synthetic_models)
        
        # 7. 평가 결과 저장
        evaluation_results.to_csv("synthetic_data_quality_evaluation.csv", index=False)
        
        print("\n=== Phase 4 완료: 합성 데이터 생성 및 평가 ===")
        print("다음 단계: 하이브리드 학습 실험")
        
        return real_data, synthetic_models, evaluation_results
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

if __name__ == "__main__":
    results = main()