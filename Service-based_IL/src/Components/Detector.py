# src/pipeline.py

# 1rd
import time

# 3rd libs
import numpy as np
from datetime import datetime

class Detectors_Pipeline:
    def __init__(self, ae, ocsvm, xgb):
        self.ae = ae
        self.ocsvm = ocsvm
        self.xgb = xgb
        # Chuyển label_map thành mảng để truy xuất nhanh bằng index
        self.labels = np.array(["Benign", "DDoS", "DoS", "Reconn", "MITM_ArpSoofing", "DNS_Spoofing", "Unknown"])
        self.CONF_REJECT = 0.7
        self.CONF_HIGH = 0.9

    def simple_predict(self, X):
        # Lấy dữ liệu thô từ model
        xgb_pred, xgb_conf = self.xgb.predict_with_confidence(X)
        
        n = len(X)
        # 1. Dùng empty để cấp phát cực nhanh, mặc định là Unknown
        out = np.empty(n, dtype=object)
        out[:] = "Unknown"

        # 2. Xử lý Benign tin cậy trước (Logic: XGB=0 và Conf >= 0.7)
        # Chúng ta lọc trước các ứng viên có thể là Benign
        is_xgb_benign = (xgb_pred == 0)
        conf_ok = (xgb_conf >= self.CONF_REJECT)
        
        # Những ai thỏa mãn Conf cực cao (>0.9) thì là Benign luôn
        out[is_xgb_benign & (xgb_conf >= self.CONF_HIGH)] = "Benign"

        # 3. Những ai 0.7 <= Conf < 0.9 thì mới gọi đến AE và OCSVM (Lazy Evaluation)
        # Đây là chỗ quan trọng nhất để tăng tốc: Chỉ chạy AE/OCSVM trên các dòng cần thiết
        check_idx = np.where(is_xgb_benign & conf_ok & (xgb_conf < self.CONF_HIGH))[0]
        
        if len(check_idx) > 0:
            # Chỉ lấy dữ liệu tại các index cần check để đưa vào AE/OCSVM
            X_sub = X[check_idx] 
            ae_ok = self.ae.is_normal(X_sub)
            oc_ok = self.ocsvm.decision_function(X_sub) > 0
            
            # Chỉ những mẫu pass cả 2 mới ghi đè thành Benign
            passed = ae_ok & oc_ok
            out[check_idx[passed]] = "Benign"

        # # 4. Cuối cùng ghi đè Malicious (Vì nó có độ ưu tiên cao nhất trong phân loại)
        # is_malicious = (~is_xgb_benign) & conf_ok
        # if np.any(is_malicious):
        #     # Ép kiểu int một lần cho toàn bộ mảng Malicious
        #     mal_labels = xgb_pred[is_malicious].astype(int)
        #     out[is_malicious] = self.labels[mal_labels]
        
        mal_idx = np.where((~is_xgb_benign) & conf_ok)[0]
        
        if len(mal_idx) > 0:
            # Lấy nhãn số, ép kiểu int để làm index cho mảng self.labels
            mal_label_indices = xgb_pred[mal_idx].astype(np.int8)
            # Fancy Indexing: self.labels[array_of_ints] trả về một mảng nhãn tương ứng
            out[mal_idx] = self.labels[mal_label_indices]
            
        return {"predictions": out.tolist()}

# class Detectors_Pipeline:
#     def __init__(self, ae, ocsvm, xgb):
#         self.ae = ae; self.ocsvm = ocsvm; self.xgb = xgb
#         self.label_map = {0:"BENIGN", 1:"DDoS", 2:"DoS", 3:"Reconn", 4:"MITM_ArpSoofing", 5:"DNS_Spoofing"}
    
#     def simple_predict(self, X):
#         # start_time = datetime.strftime(datetime.now(), "%d/%m/%Y %H:%M:%S.%f")
        
#         xgb_pred, xgb_conf = self.xgb.predict_with_confidence(X)

#         CONF_REJECT = 0.7
#         CONF_HIGH = 0.9

#         ae_ok = self.ae.is_normal(X)
#         oc_ok = self.ocsvm.decision_function(X) > 0

#         out = []

#         for i in range(len(X)):
#             p_val = int(xgb_pred[i])
#             conf = xgb_conf[i]
            
#             if conf < CONF_REJECT:
#                 out.append("Unknown")
#             elif p_val != 0:
#                 out.append(self.label_map.get(p_val, "Unknown"))
#             elif conf >= CONF_HIGH:
#                 out.append("Benign")
#             else:
#                 out.append("Benign" if ae_ok[i] and oc_ok[i] else "Unknown")

#         # total_time = time.perf_counter()*1000000 - start_time
#         return {
#             "predictions": out
#             # "start_predtime": start_time
#             # "avg_predtime_per_flow": round(total_time / len(X)) if len(X) > 0 else 0
#         }

class SequentialHybridPipeline:
    def __init__(self, xgb, ae=None, ocsvm=None):
        self.xgb = xgb; self.ae = ae; self.ocsvm = ocsvm
        self.label_map = {0:"Benign", 1:"DDoS", 2:"DoS", 3:"Reconnaisance", 4:"MITM_ArpSpoofing", 5:"DNS_Spoofing"}
        
    def predict(self, X, return_details=False):
        modes = ["XGB"]
        if self.ae: modes.append("AE")
        if self.ocsvm: modes.append("OCSVM")
        print(f"Pipeline [{' + '.join(modes)}] processing {len(X)} samples...")
        
        xgb_pred, xgb_conf = self.xgb.predict_with_confidence(X)
        
        # [OPTIMIZATION 1] Giảm ngưỡng để tin tưởng XGBoost hơn
        CONF_REJECT = 0.7  # Giảm nhẹ từ 0.7
        CONF_HIGH = 0.9    # Giảm từ 0.9 -> Giảm lượng mẫu rơi vào Gray Zone
        
        ae_is_normal = self.ae.is_normal(X) if self.ae else None
        ocsvm_is_normal = (self.ocsvm.decision_function(X) > 0) if self.ocsvm else None
        
        final_preds = []
        stats = {"low_unk":0, "atk_acc":0, "high_ben":0, "gray_pass":0, "gray_fail":0}
        
        for i in range(len(X)):
            p_val = int(xgb_pred[i])
            conf = xgb_conf[i]
            
            # 1. Low Conf -> Unknown (Vẫn giữ để loại bỏ tấn công lạ thực sự)
            if conf < CONF_REJECT:
                final_preds.append("Unknown")
                stats["low_unk"] += 1
                continue
            
            # 2. Attack -> Chấp nhận ngay
            if p_val != 0:
                final_preds.append(self.label_map.get(p_val, "Unknown")) # self.label_map.get(p_val, "Unknown")
                stats["atk_acc"] += 1
            else:
                # 3. Benign High Conf -> Chấp nhận
                if conf >= CONF_HIGH:
                    final_preds.append("Benign")
                    stats["high_ben"] += 1
                else:
                    # 4. Gray Zone (0.65 <= conf < 0.85)
                    # [OPTIMIZATION 2] Logic mềm dẻo hơn (OR logic hoặc Voting)
                    is_safe = True 
                    
                    if self.ae and self.ocsvm:
                        # Logic cũ: ae_is_normal[i] and ocsvm_is_normal[i] (Quá chặt)
                        # Logic MỚI: Chỉ cần 1 trong 2 bảo là Normal thì tạm tin là Normal
                        # (Vì XGB đã bảo là Benign rồi, ta cần bằng chứng mạnh để bác bỏ nó)
                        if ae_is_normal[i] and ocsvm_is_normal[i]: 
                            is_safe = True
                        else:
                            # Cả 2 đều bảo là Abnormal -> Chắc chắn là Unknown
                            is_safe = False
                    elif self.ae:
                        is_safe = ae_is_normal[i]
                    elif self.ocsvm:
                        is_safe = ocsvm_is_normal[i]

                    if is_safe:
                        final_preds.append("Benign")
                        stats["gray_pass"] += 1
                    else:
                        final_preds.append("Unknown")
                        stats["gray_fail"] += 1
        
        print(f"   Stats: HighBen: {stats['high_ben']} | GrayPass: {stats['gray_pass']} | GrayFail(Unk): {stats['gray_fail']}")
        
        details = {
            'ae_pred': ae_is_normal.astype(int) if ae_is_normal is not None else None,
            'ocsvm_pred': ocsvm_is_normal.astype(int) if ocsvm_is_normal is not None else None
        }
        return (final_preds, details) if return_details else final_preds

    def incremental_learning_old(self, X_new, y_new, X_old, y_old):
        print("=== INCREMENTAL LEARNING ===")
        X_benign = X_new[y_new == 0]
        if len(X_benign) > 0:
            if self.ae: 
                # [OPTIMIZATION 3] Tăng epoch để AE học kỹ Benign mới hơn
                print(f"   -> Fine-tuning AE on {len(X_benign)} samples (50 epochs)...")
                self.ae.train_on_known_data(X_benign, epochs=50, verbose=False)
            if self.ocsvm: 
                self.ocsvm.partial_fit(X_benign)
        self.xgb.safe_incremental_retrain(X_old, y_old, X_new, y_new)
        
    def incremental_learning(self, X, y):
        train_benign_X = X[y==0]
        if len(train_benign_X) >0:
            if self.ocsvm: 
                print(f"   -> Partialfit on {len(train_benign_X)} ...")
                self.ocsvm.partial_fit(train_benign_X)
            if self.ae:
                print(f"   -> Fine-tuning AE on {len(train_benign_X)} samples (50 epochs)...")
                self.ae.train_on_known_data(train_benign_X, epochs=50, verbose=False)
            if self.xgb:
                print(f"   -> Updating xgb ...")
                self.xgb.train(X, y, True)