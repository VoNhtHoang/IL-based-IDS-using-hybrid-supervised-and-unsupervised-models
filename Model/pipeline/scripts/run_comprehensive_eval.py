import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg') # Fix lỗi hiển thị
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import AETrainer, IncrementalOCSVM, OpenSetXGBoost
from src.pipeline import SequentialHybridPipeline
from src.utils import (
    ScenarioDataLoader, ScenarioManager, 
    plot_cm, get_label_name, calculate_unknown_metrics,
    evaluate_final_pipeline
)

BASE_DATA_DIR = "merge1.4_3-4-5/case-from-3-incre-4class-incre-6class"
GLOBAL_SCALER_PATH = "Scenarios/global_scaler.joblib"
SAVE_ROOT = "results/comprehensive_eval"

PIPELINE_CONFIG = {
    'CONF_HIGH': 0.90,   
    'CONF_REJECT': 0.70, 
    'GRAY_LOGIC': 'HYBRID_SOFT'
}

class EvalPipeline(SequentialHybridPipeline):
    def predict(self, X, return_details=False):
        print(f"   -> Processing {len(X)} samples with Optimized Logic...")
        xgb_pred, xgb_conf = self.xgb.predict_with_confidence(X)
        
        ae_is_normal = self.ae.is_normal(X) if self.ae else None
        ocsvm_is_normal = (self.ocsvm.decision_function(X) > 0) if self.ocsvm else None
        
        final_preds = []
        for i in range(len(X)):
            p_val = int(xgb_pred[i])
            conf = xgb_conf[i]
            
            if conf < PIPELINE_CONFIG['CONF_REJECT']:
                final_preds.append("UNKNOWN")
                continue
            
            if p_val != 0:
                final_preds.append(self.label_map.get(p_val, "UNKNOWN"))
            else:
                if conf >= PIPELINE_CONFIG['CONF_HIGH']:
                    final_preds.append("BENIGN")
                else:
                    is_safe = False
                    if ae_is_normal[i]: is_safe = True
                    elif ocsvm_is_normal[i] and conf > (PIPELINE_CONFIG['CONF_REJECT'] + 0.15): is_safe = True
                    
                    final_preds.append("BENIGN" if is_safe else "UNKNOWN")
        
        return (final_preds, None) if return_details else final_preds

def load_models(Scenario_id, mgr):
    print(f"   -> Loading models from Scenario {Scenario_id}...")
    ae = AETrainer(81, 32)
    ocsvm = IncrementalOCSVM(nu=0.15)
    xgb = OpenSetXGBoost(0.7)
    mgr.load_models(Scenario_id, {'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb})
    return SequentialHybridPipeline(xgb=xgb, ae=ae, ocsvm=ocsvm)

def map_labels_for_pre_il(y_true_raw, unknown_target_labels):
    """
    Chuyển đổi các nhãn cụ thể (ví dụ 3 - Reconn) thành 'UNKNOWN'
    để vẽ biểu đồ Pre-IL đúng chuẩn Open Set.
    """
    y_mapped = []
    for val in y_true_raw:
        if val in unknown_target_labels:
            y_mapped.append("UNKNOWN")
        else:
            y_mapped.append(get_label_name(val))
    return y_mapped

def run_evaluation():
    print("🚀 STARTING COMPREHENSIVE EVALUATION (Logic Optimized + Correct Pre-IL CM)")
    os.makedirs(SAVE_ROOT, exist_ok=True)
    
    loader = ScenarioDataLoader()
    loader.load_scaler(GLOBAL_SCALER_PATH)
    mgr = ScenarioManager()

    # ==============================================================================
    # 1. Scenario 0: EVAL PHASE
    # ==============================================================================
    print(f"\n{'='*10} Scenario 0: EVALUATION {'='*10}")
    save_dir = os.path.join(SAVE_ROOT, "Scenario0_eval")
    os.makedirs(save_dir, exist_ok=True)
    
    X_test, y_test = loader.load_data_raw(os.path.join(BASE_DATA_DIR, "test_Scenario0.parquet"))
    X_test = loader.apply_scaling(X_test, fit=False)
    pipeline = load_models(0, mgr)
    
    preds = pipeline.predict(X_test)
    evaluate_final_pipeline(y_test, preds, "Scenario0_Final", save_dir)

    # ==============================================================================
    # 2. Scenario 1
    # ==============================================================================
    print(f"\n{'='*10} Scenario 1: RECONN {'='*10}")
    
    print(">>> Phase 1: Pre-IL (Target: Reconn -> UNKNOWN)")
    save_dir_pre = os.path.join(SAVE_ROOT, "Scenario1_phase1_pre_il")
    os.makedirs(save_dir_pre, exist_ok=True)
    
    X_train1, y_train1 = loader.load_data_raw(os.path.join(BASE_DATA_DIR, "train_Scenario1.parquet"))
    X_train1 = loader.apply_scaling(X_train1, fit=False)
    
    preds_pre = pipeline.predict(X_train1)
    
    y_true_mapped = map_labels_for_pre_il(y_train1, unknown_target_labels=[3])
    
    print("   Generating mapped CM for Pre-IL...")
    plot_cm(y_true_mapped, preds_pre, "CM Pipeline (Pre-IL) - Mapped", os.path.join(save_dir_pre, "cm_pre_il_mapped.png"))
    
    calculate_unknown_metrics(y_train1, preds_pre, unknown_label=3, save_dir=save_dir_pre, Scenario_name="Scenario1_PreIL")

    print("\n>>> Phase 3: Post-IL (Target: Reconn -> Reconn)")
    save_dir_post = os.path.join(SAVE_ROOT, "Scenario1_phase3_post_il")
    os.makedirs(save_dir_post, exist_ok=True)
    
    X_test1, y_test1 = loader.load_data_raw(os.path.join(BASE_DATA_DIR, "test_Scenario1.parquet"))
    X_test1 = loader.apply_scaling(X_test1, fit=False)
    
    pipeline = load_models(1, mgr)
    preds_post = pipeline.predict(X_test1)
    
    evaluate_final_pipeline(y_test1, preds_post, "Scenario1_PostIL", save_dir_post)

    # ==============================================================================
    # 3. Scenario 2
    # ==============================================================================
    print(f"\n{'='*10} Scenario 2: MITM & DNS {'='*10}")
    
    print(">>> Phase 1: Pre-IL (Target: MITM/DNS -> UNKNOWN)")
    save_dir_pre = os.path.join(SAVE_ROOT, "Scenario2_phase1_pre_il")
    os.makedirs(save_dir_pre, exist_ok=True)
    
    X_train2, y_train2 = loader.load_data_raw(os.path.join(BASE_DATA_DIR, "train_Scenario2.parquet"))
    X_train2 = loader.apply_scaling(X_train2, fit=False)
    
    preds_pre = pipeline.predict(X_train2)
    
    y_true_mapped = map_labels_for_pre_il(y_train2, unknown_target_labels=[4, 5])
    
    print("   Generating mapped CM for Pre-IL...")
    plot_cm(y_true_mapped, preds_pre, "CM Pipeline (Pre-IL) - Mapped", os.path.join(save_dir_pre, "cm_pre_il_mapped.png"))
    
    calculate_unknown_metrics(y_train2, preds_pre, unknown_label=[4, 5], save_dir=save_dir_pre, Scenario_name="Scenario2_PreIL")

    print("\n>>> Phase 3: Post-IL")
    save_dir_post = os.path.join(SAVE_ROOT, "Scenario2_phase3_post_il")
    os.makedirs(save_dir_post, exist_ok=True)
    
    X_test2, y_test2 = loader.load_data_raw(os.path.join(BASE_DATA_DIR, "test_Scenario2.parquet"))
    X_test2 = loader.apply_scaling(X_test2, fit=False)
    
    pipeline = load_models(2, mgr)
    preds_post = pipeline.predict(X_test2)
    
    evaluate_final_pipeline(y_test2, preds_post, "Scenario2_PostIL", save_dir_post)
    
    print(f"\n COMPLETED. Results at: {SAVE_ROOT}")

if __name__ == "__main__":
    run_evaluation()