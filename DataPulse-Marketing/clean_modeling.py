#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean Modeling - ConversionRate 제거 후 현실적인 모델링
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

def clean_and_prepare_data():
    """데이터 로드 및 전처리 (ConversionRate 제거)"""
    
    # 데이터 로드
    df = pd.read_csv('digital_marketing_campaign_dataset.csv')
    
    print("🧹 데이터 클리닝 시작")
    print("=" * 50)
    
    # CustomerID와 ConversionRate 제거 (데이터 누수 방지)
    print("❌ 제거할 컬럼들:")
    print("   - CustomerID (데이터 누수)")
    print("   - ConversionRate (타겟 변수와 동일 개념)")
    
    # 불필요한 컬럼들도 제거
    columns_to_remove = [
        'CustomerID', 
        'ConversionRate',  # 핵심: 이것이 데이터 누수의 주범
        'AdvertisingPlatform',  # 기밀 정보
        'AdvertisingTool'  # 기밀 정보
    ]
    
    df_clean = df.drop(columns=columns_to_remove)
    
    print(f"✅ 정리된 데이터 형태: {df_clean.shape}")
    print(f"📋 사용할 피처들: {list(df_clean.columns[:-1])}")
    
    return df_clean

def feature_engineering(df):
    """피처 엔지니어링"""
    
    print("\n🔧 피처 엔지니어링")
    print("=" * 30)
    
    # 기존 피처들 계산
    df['ROAS_estimate'] = (df['AdSpend'] * df['ClickThroughRate']).round(2)
    df['engagement_score'] = (df['WebsiteVisits'] * df['PagesPerVisit'] * df['TimeOnSite'] / 100).round(2)
    df['social_engagement'] = (df['SocialShares'] + 1).apply(np.log).round(2)
    df['email_effectiveness'] = ((df['EmailClicks'] + 1) / (df['EmailOpens'] + 1)).round(2)
    df['customer_value'] = (df['PreviousPurchases'] * df['LoyaltyPoints'] / 1000).round(2)
    df['visit_quality'] = (df['PagesPerVisit'] * df['TimeOnSite']).round(2)
    df['spend_efficiency'] = (df['ClickThroughRate'] / (df['AdSpend'] / 1000)).round(4)
    
    # 범주형 변수 인코딩
    le_gender = LabelEncoder()
    le_channel = LabelEncoder()
    le_campaign = LabelEncoder()
    
    df['Gender_encoded'] = le_gender.fit_transform(df['Gender'])
    df['CampaignChannel_encoded'] = le_channel.fit_transform(df['CampaignChannel'])
    df['CampaignType_encoded'] = le_campaign.fit_transform(df['CampaignType'])
    
    # 원본 범주형 변수 제거
    df_encoded = df.drop(['Gender', 'CampaignChannel', 'CampaignType'], axis=1)
    
    print(f"✅ 최종 피처 수: {df_encoded.shape[1] - 1}")
    
    return df_encoded

def train_clean_models(X, y):
    """깨끗한 데이터로 모델 훈련"""
    
    print("\n🤖 깨끗한 데이터로 모델 훈련")
    print("=" * 40)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 클래스 불균형 처리 (SMOTE)
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"📊 훈련 데이터 형태: {X_train_balanced.shape}")
    print(f"📊 클래스 분포: {np.bincount(y_train_balanced)}")
    
    # 모델들 정의
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
    }
    
    results = []
    
    # 교차 검증으로 각 모델 평가
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\n🔍 {name} 훈련 중...")
        
        # 교차 검증
        cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, 
                                   cv=cv, scoring='f1')
        
        # 전체 데이터로 훈련 후 테스트
        model.fit(X_train_balanced, y_train_balanced)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # 메트릭 계산
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results.append({
            'Model': name,
            'CV_F1_Mean': cv_scores.mean(),
            'CV_F1_Std': cv_scores.std(),
            'Test_F1': f1,
            'Test_AUC': auc
        })
        
        print(f"   CV F1: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"   Test F1: {f1:.4f}")
        print(f"   Test AUC: {auc:.4f}")
    
    return pd.DataFrame(results)

def main():
    """메인 실행 함수"""
    
    print("🧼 ConversionRate 제거 후 깨끗한 모델링")
    print("=" * 60)
    
    # 1. 데이터 클리닝
    df_clean = clean_and_prepare_data()
    
    # 2. 피처 엔지니어링
    df_featured = feature_engineering(df_clean.copy())
    
    # 3. X, y 분리
    X = df_featured.drop('Conversion', axis=1)
    y = df_featured['Conversion']
    
    print(f"\n📊 최종 데이터 요약:")
    print(f"   - 피처 수: {X.shape[1]}")
    print(f"   - 샘플 수: {X.shape[0]}")
    print(f"   - 전환율: {y.mean():.3f}")
    
    # 4. 모델 훈련 및 평가
    results = train_clean_models(X, y)
    
    # 5. 결과 저장 및 출력
    results.to_csv('clean_model_results.csv', index=False)
    
    print("\n🏆 최종 결과 (ConversionRate 제거 후):")
    print("=" * 50)
    print(results.round(4))
    
    best_model = results.loc[results['Test_F1'].idxmax()]
    print(f"\n🥇 최고 성능 모델: {best_model['Model']}")
    print(f"   - Test F1-Score: {best_model['Test_F1']:.4f}")
    print(f"   - Test AUC: {best_model['Test_AUC']:.4f}")
    
    print("\n✅ 이제 현실적인 성능이 나올 것입니다!")
    print("❌ ConversionRate 제거로 데이터 누수 해결!")

if __name__ == "__main__":
    main()