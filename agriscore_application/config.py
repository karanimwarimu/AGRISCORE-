
import os

VERSION = "0.1.0"

FEATURE_COLUMNS = [  
    'Unnamed: 0', 'hh_size', 'hh_dependency_ratio', 'age_manager',
       'hh_primary_education', 'hh_formal_education', 'farm_size', 'farm_work',
       'livestock', 'inorganic_fertilizer', 'fertilizer_per_ha', 'yield_kg_ha',
       'yield_stability', 'rainfall_mm', 'soil_quality_index', 'soil_npk',
       'drought_risk', 'climate_stress_index', 'crop_health_index',
       'tx_frequency', 'total_volume', 'avg_transaction',
       'input_purchase_ratio', 'cashflow_volatility', 'fraud_rate',
       'input_efficiency', 'hh_shock', 'drought_shock', 'dist_market','country'
       'main_crop_clean_MAIZE', 'main_crop_clean_MILLET',
       'main_crop_clean_NUTS', 'main_crop_clean_OTHER', 'main_crop_clean_RICE',
       'main_crop_clean_SORGHUM', 'main_crop_clean_TUBERS_ROOT',
       'main_crop_clean_WHEAT'
]

RISK_THRESHOLDS = {
    'low': 0.15,      # PD < 15% → Low risk
    'medium': 0.35,   # PD < 35% → Medium
    'high': 1.0
}

SCORE_SCALE = (300, 850)  # Min-Max credit score range