# Standard libs
import os

# 3rd libs
import joblib
import torch
import xgboost as xgb
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split

from collections import Counter


# ==================== 1. DEEP AUTOENCODER (Requested Version) ====================
class Model:
    def __init__(self):
        self.model_name = ""
        
    def get_model_name(self):
        return self.model_name

class AnomalyAE(nn.Module):
    def __init__(self, input_dim=81, latent_dim=32):
        super(AnomalyAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.1), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.1), 
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.1),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1), nn.Dropout(0.1),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Dropout(0.1),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.1), nn.Dropout(0.1), 
            nn.Linear(512, input_dim)
        )
    def forward(self, x): return self.decoder(self.encoder(x))

class AETrainer(Model):
    def __init__(self, input_dim=81, encoding_dim=32, lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AnomalyAE(input_dim, encoding_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.known_threshold = None
        
        self.model_name = "ae.pt"
        self.loaded = False

    def train_on_known_data(self, X_benign, epochs=200, batch_size=512, verbose=True):
        if verbose: print(f"Training Deep AE on {len(X_benign)} samples...")
        self.model.train()
        tensor = torch.FloatTensor(X_benign.values).to(self.device)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor), batch_size=batch_size, shuffle=True, drop_last=True)
        
        for epoch in range(epochs):
            for batch in loader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(batch[0]), batch[0])
                loss.backward()
                self.optimizer.step()
        
        self.model.eval()
        errors = self.get_reconstruction_errors(X_benign)
        # Threshold: Mean + 1.0 * Std (Khá chặt chẽ)
        self.known_threshold = np.mean(errors) + 1.0 * np.std(errors)
        if verbose: print(f"AE Threshold set: {self.known_threshold:.6f}")

    def get_reconstruction_errors(self, data):
        self.model.eval()
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                tensor = torch.FloatTensor(data).to(self.device)
            elif isinstance(data, pd.DataFrame):
                tensor = torch.FloatTensor(data.values).to(self.device)  # .values → numpy array
            errors = []
            for i in range(0, len(tensor), 2048):
                batch = tensor[i:i+2048]
                errors.append(torch.mean((batch - self.model(batch))**2, dim=1).cpu().numpy())
            return np.concatenate(errors)

    # [Compatibility] Thêm hàm này để Pipeline gọi không bị lỗi
    def is_normal(self, X):
        return self.get_reconstruction_errors(X) <= self.known_threshold

    def save_model(self, p, version = 0, date_modified = "2025-12-21 00:00:00"): 
        os.makedirs(os.path.dirname(p), exist_ok=True)
        torch.save({'st': self.model.state_dict(), 'th': self.known_threshold, 'ver': version, 'date_m': date_modified}, p)
    
    def load_model(self, p):
        ckpt = torch.load(p, map_location=self.device, weights_only=False); 
        self.model.load_state_dict(ckpt['st']); self.known_threshold = ckpt['th'] ; 
        self.loaded = True
    
    def load_model_info(self, p):
        d = torch.load(p, map_location=self.device, weights_only=False); 
        return getattr(d, 'ver', 'N/A'), getattr(d, 'date_m', 'N/A')
        # return 'N/A', 'N/A'
# ==================== 2. INCREMENTAL OCSVM (Requested Version) ====================
class IncrementalOCSVM(Model):
    def __init__(self, nu=0.15, random_state=42):
        # Nystroem 800 components như yêu cầu
        self.feature_map = Nystroem(gamma=0.1, random_state=random_state, n_components=1000)
        self.model = SGDOneClassSVM(nu=nu, random_state=random_state, shuffle=True)
        # self.is_fitted = False
        
        self.model_name = "ocsvm.pkl"
        self.loaded = False

    def train(self, X): 
        self.model.fit(self.feature_map.fit_transform(X))
        # self.is_fitted = True

    def partial_fit(self, X): 
        # if self.is_fitted:
        self.model.partial_fit(self.feature_map.transform(X.astype(np.float32)))
        # else:
            # self.train(X)

    def decision_function(self, X): return self.model.decision_function(self.feature_map.transform(X))
    
    def save_model(self, p, version = 0, date_modified = "2025-12-21 00:00:00"): 
        os.makedirs(os.path.dirname(p), exist_ok=True)
        joblib.dump({'model': self.model, 'map': self.feature_map, 'fitted': self.is_fitted, 'ver': version, 'date_m': date_modified}, p)
    
    def load_model(self, p): 
        d = joblib.load(p);
        self.model, self.feature_map, self.is_fitted = d['model'], d['map'], d['fitted']; 
        self.loaded= True
    
    def load_model_info(self, p):
        d=joblib.load(p);
        return getattr(d, 'ver', 'N/A'), getattr(d, 'date_m', 'N/A')
        # return 'N/A', 'N/A'
# ==================== 3. XGBOOST (Giữ nguyên để Pipeline hoạt động) ====================
class OpenSetXGBoost(Model):
    def __init__(self, confidence_threshold=0.75, max_classes_buffer=20):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.label_encoder = {}
        self.reverse_encoder = {}
        self.max_classes = max_classes_buffer
        
        self.model_name = "xgb.pkl"
        self.loaded = False
        
    def _update_encoder(self, y):
        cur = set(self.label_encoder.keys()); new = set(np.unique(y)) - cur
        if not new and self.label_encoder: return
        idx = len(self.label_encoder)
        for l in sorted(new): self.label_encoder[l] = idx; self.reverse_encoder[idx] = l; idx += 1

    def train(self, X, y, is_incremental=False):
        self._update_encoder(y); y_enc = np.array([self.label_encoder[l] for l in y])
        w = dict(Counter(y_enc)); sample_weights = np.array([max(w.values())/w[c] for c in y_enc])
        
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X, y_enc, sample_weights, test_size=0.1, random_state=42, stratify=y_enc
        )

        if not is_incremental:
            print(f"XGBoost Initial Train ({len(X)} samples)...")
            self.model = xgb.XGBClassifier(
                n_estimators=800, max_depth=6, learning_rate=0.01, 
                objective='multi:softprob', num_class=self.max_classes, 
                n_jobs=-1, random_state=42, early_stopping_rounds=150
            )
            self.model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            print(f"XGBoost Incremental Update ({len(X)} samples)...")
            cur_est = self.model.get_booster().num_boosted_rounds()
            self.model.set_params(n_estimators=cur_est + 200)
            self.model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)], verbose=False, xgb_model=self.model.get_booster())

    def predict_with_confidence(self, X):
        proba = self.model.predict_proba(X)
        valid = list(self.reverse_encoder.keys()); valid_proba = proba[:, valid]
        idx = np.argmax(valid_proba, axis=1)
        return np.array([self.reverse_encoder[valid[i]] for i in idx]), np.max(valid_proba, axis=1)
    
    def safe_incremental_retrain(self, X_old, y_old, X_new, y_new):
        self.train(np.vstack([X_old, X_new]), np.hstack([y_old, y_new]), is_incremental=True)

    def save_model(self, p, version = 0, date_modified = "2025-12-21 00:00:00"): 
        os.makedirs(os.path.dirname(p), exist_ok=True); 
        joblib.dump({'m': self.model, 'le': self.label_encoder, 're': self.reverse_encoder, 'c': self.confidence_threshold, 'ver': version, 'date_m': date_modified}, p)
    
    def load_model(self, p):
        d=joblib.load(p); 
        self.model, self.label_encoder, self.reverse_encoder, self.confidence_threshold = d['m'], d['le'], d['re'], d['c']; 
        self.loaded=True
    
    def load_model_info(self, p):
        d=joblib.load(p);
        return getattr(d, 'ver', 'N/A'), getattr(d, 'date_m', 'N/A')
