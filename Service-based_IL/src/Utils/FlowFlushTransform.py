import gc
import pandas as pd
from pathlib import Path
from datetime import datetime

# ===== ALGO LIBS =====
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

# ===== HELPER
import joblib



# ======== VARIABLES / FUNC IMPORT=========
# from FlowZmqServer import CURR_DIR
from src.config.settings import settings
from src.Utils.func_convert import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ======== CONSTANCE =========
PKL_PATH = settings.PKL_DIR
STANDARD_SCALER_PATH = PKL_PATH/"standardScaler.pkl"
MINMAX_SCALER_PATH = PKL_PATH/"minmaxScaler.pkl"
MINMAX_COLS = ['Flow Duration', 'Fwd IAT Total', 'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg', 'Fwd Seg Size Min']

STANDARD_COLS = ['Total Fwd Packet', 'Total Bwd packets', 'Total Length of Fwd Packet', 'Total Length of Bwd Packet', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Fwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min', 'Packet Length Max', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Fwd Segment Size Avg', 'Bwd Segment Size Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'FWD Init Win Bytes', 'Bwd Init Win Bytes', 'Fwd Act Data Pkts', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']

COLS_TO_DROP = ["Flow ID", "Timestamp"]

SAMPLE_COLS_TO_REMOVE = ['Flow Duration', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Max', 'Bwd IAT Min', 'Bwd Bulk Rate Avg']
DO_NOT_TOUCH_COLS = ['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol' ]

DECIMAL_BIN = 6

class FlowFlushTransformer:
    def __init__(
        self,
        minmax_scaler_path, standard_scaler_path, minmax_cols, standard_cols, 
        header =None, label_mapping=None, 
        decimal_bin=6, out_dir="flows_parquet", file_prefix="batch"
    ):
        # Load old scaler
        self.minmaxScaler = MinMaxScaler()
        self.minmaxScaler = joblib.load(minmax_scaler_path)
        self.standardScaler = StandardScaler()
        self.standardScaler = joblib.load(standard_scaler_path)

        # 
        self.minmax_cols = minmax_cols
        self.standard_cols = standard_cols
                
        self.header = header or []
        self.label_mapping = label_mapping or {}
        self.numeric_cols = self.loadNumericCols()
        
        
        self.all_features = [col for col in self.header if col not in ["Flow ID", "Timestamp", "Label"] ]
        self.std_idx = [self.all_features.index(c) for c in self.standard_cols if c in self.all_features]
        self.mmax_idx = [self.all_features.index(c) for c in self.minmax_cols if c in self.all_features]
        self.sample_remove_idx = [self.all_features.index(c) for c in SAMPLE_COLS_TO_REMOVE if c in self.all_features]
        
        self.decimal_bin = decimal_bin
        
        # SAVE DIR        
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # state batch
        self.file_prefix = file_prefix
        self.curr_folder = f"{datetime.now().date()}"
        self.curr_index = 0
        
        # 
        with open('src/features.json') as json_file:
            self.feature_config = json.load(json_file)
        
        
        # Tạo sẵn dict dtypes để dùng cho Pandas .astype()
        self.target_dtypes = {}
        for key, t_name in self.feature_config.items():
            if t_name == "int8":
                self.target_dtypes[key] = np.int8
            elif t_name == "float64":
                self.target_dtypes[key] = np.float64
            elif t_name == "float32":
                self.target_dtypes[key] = np.float32
            else:
                self.target_dtypes[key] = str
                
        # self.dtypes = {}
        # for key, t_name in self.feature_config.items():
        #     if t_name == "int8":
        #         self.dtypes[key] = np.int8
        #     elif t_name == "float64":
        #         self.dtypes[key] = np.float64
        #     elif t_name == "float32":
        #         self.dtypes[key] = np.float32
        
        self.scaler  = StandardScaler()
        self.scaler = joblib.load("src/pkl/scaler.joblib")
        
        if (not Path.exists(self.out_dir/self.curr_folder)):
            Path.mkdir(self.out_dir/self.curr_folder, parents=True, exist_ok=True)
            
    def flush(self, df: list | pd.DataFrame, header = None):
        # ---- Build DataFrame ----
        if isinstance(df, list):
            if self.header.len() <1:
                print("[ERROR] There is no header!")
                return None
            
            df = pd.DataFrame(df, columns= self.header)
            
        # XỬ LÍ DATA
        df, indexdf = self.preProcessingData_flush(df)
        df = df.astype(self.target_dtypes, errors='ignore')
        df = pd.concat([indexdf, df], axis= 1)
                
        # LƯU
        temp = f"{datetime.now().date()}"
        
        if temp != self.curr_folder:
            self.curr_folder = temp
            self.curr_index = 0
            temp = self.out_dir / self.curr_folder
            Path.mkdir(temp, parents=True, exist_ok=True)
            
        fname = (
            self.out_dir / self.curr_folder / f"{self.file_prefix}_{self.curr_index}.parquet"
            #  self.out_dir / self.curr_folder / f"275kpps-20s.parquet"
        )
        
        self.curr_index +=1
        try:
            df.to_parquet(fname)
        
        except Exception as e:
            print("[FlowFlushTransform] Cannot flush!")
            
        del df, indexdf        
        return fname
    
    def detect(self, df: list | pd.DataFrame, header = None):
        # ---- Build DataFrame ----
        if isinstance(df, list):
            df = pd.DataFrame(df, columns=self.header)

        # XỬ LÍ DATA
        retX, indexdf = self.preProcessingData(df)
        # df = df.astype(self.target_dtypes)  
        retX = self.scaler.transform(retX)
   
        return indexdf, retX, None
    
    # =====================
    # MAX SPEED
    # =====================
    def preProcessingData(self, df):
        indexdf = df[COLS_TO_DROP].copy()
        
        # 1. Chuyển đổi dữ liệu thô sang số (Vectorized)
        df['Src IP'] = ip_to_float_vectorized(df['Src IP'])
        df['Dst IP'] = ip_to_float_vectorized(df['Dst IP'])
        df['Src Port'] = bucket_port_vectorized(df['Src Port'])
        df['Dst Port'] = bucket_port_vectorized(df['Dst Port'])
        
        # df['Protocol'] = pd.to_numeric(df['Protocol'], errors='coerce').fillna(0)
        
        X = df[self.all_features].to_numpy(dtype=np.float64, copy=True)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        mask = (X[:, self.sample_remove_idx] >= 0).all(axis=1)
        X = X[mask]
        
        if self.std_idx:
            # X[:, self.std_idx] chọn đúng các cột cần Standard Scale
            X[:, self.std_idx] = self.standardScaler.transform(np.log1p(X[:, self.std_idx]))
            X[:, self.std_idx] = np.round(X[:, self.std_idx], self.decimal_bin)
        
        # Apply MinMax Scaler dựa trên index mask
        if self.mmax_idx:
            X[:, self.mmax_idx] = self.minmaxScaler.transform(X[:, self.mmax_idx])
            X[:, self.mmax_idx] = np.round(X[:, self.mmax_idx], self.decimal_bin)

        # 6. Làm tròn toàn bộ
        # X = np.round(X, self.decimal_bin)
        return X, indexdf
    
    def loadNumericCols(self):
        return  [col for col in self.header if col not in ["Flow ID", "Timestamp", "Label"]]      
    
    def loadNumericCols_old(self):
        # Dùng set để tự động loại bỏ trùng lặp, tốc độ cực nhanh
        cols = set(self.minmax_cols) | set(self.standard_cols) | set(SAMPLE_COLS_TO_REMOVE)
        cols.add("Src Port")
        cols.add("Dst Port")
        return list(cols)
    
    
    def preProcessingData_flush(self, df):
        df = df.replace([np.inf, -np.inf, "inf", "-inf", "Infinity", "-Infinity", r'[N|n][a|A][N|n]', "(empty)"], 0)
        df = df.dropna()
        df = df.drop_duplicates()
        df = df.fillna(0)
        
        indexdf = df[COLS_TO_DROP]
        df = df.drop(columns=COLS_TO_DROP)

        # Bỏ các giá trị âm
        numeric_cols = self.loadNumericCols()
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        for col in SAMPLE_COLS_TO_REMOVE:
            df = df[df[col]>=0]
            
        
        df["Src Port"] = df["Src Port"].apply(bucket_port)
        df["Dst Port"] = df["Dst Port"].apply(bucket_port)
        
        df['Src IP'] = df['Src IP'].apply(ip_to_float)
        df['Dst IP'] = df['Dst IP'].apply(ip_to_float)
        
        df[self.standard_cols] = np.log1p(df[self.standard_cols])

        # scale
        df[self.standard_cols] = self.standardScaler.transform(df[self.standard_cols])
        df[self.minmax_cols] = self.minmaxScaler.transform(df[self.minmax_cols])

        # round
        df[self.minmax_cols] = df[self.minmax_cols].round(self.decimal_bin)
        df[self.standard_cols] = df[self.standard_cols].round(self.decimal_bin)
        
        gc.collect()
        return df, indexdf