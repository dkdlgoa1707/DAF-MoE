import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer
from scipy.stats import skew

class DAFPreprocessor:
    def __init__(self, num_cols, cat_cols, config=None):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        
        # Config에서 설정 가져오기 (없으면 기본값)
        n_quantiles = config.n_quantiles if config else 1000
        subsample = config.subsample if config else 100000
        
        # 1. Imputers
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        
        # 2. Processors
        self.scaler = StandardScaler()
        self.quantile_transformer = QuantileTransformer(
            output_distribution='uniform', 
            n_quantiles=n_quantiles, 
            random_state=42,
            subsample=subsample
        )
        self.label_encoders = {col: LabelEncoder() for col in cat_cols}
        
        # 3. Stats Cache
        self.num_stats = {}
        self.cat_stats = {}
        self.max_rows = 0

    def fit(self, df):
        self.max_rows = len(df)
        
        # Impute
        self.num_imputer.fit(df[self.num_cols])
        self.cat_imputer.fit(df[self.cat_cols])
        
        num_filled = pd.DataFrame(self.num_imputer.transform(df[self.num_cols]), columns=self.num_cols)
        cat_filled = pd.DataFrame(self.cat_imputer.transform(df[self.cat_cols]), columns=self.cat_cols)
        
        # Numerical Fit
        self.scaler.fit(num_filled)
        self.quantile_transformer.fit(num_filled)
        
        for col in self.num_cols:
            self.num_stats[col] = {'skew': skew(num_filled[col])}
            
        # Categorical Fit
        for col in self.cat_cols:
            series = cat_filled[col].astype(str)
            le = self.label_encoders[col]
            le.fit(series)
            
            # Vectorized Map Init
            classes = le.classes_
            id_map = dict(zip(classes, le.transform(classes)))
            default_id = id_map.get('Unknown', 0)
            
            counts = series.value_counts(normalize=True)
            log_card = np.log(len(classes) + 1e-9) / np.log(self.max_rows + 1e-9)
            
            self.cat_stats[col] = {
                'freq_map': counts.to_dict(), 
                'log_card': log_card,
                'id_map': id_map,
                'default_id': default_id
            }
        return self

    def transform(self, df):
        n_samples = len(df)
        
        # Impute
        x_num_filled_np = self.num_imputer.transform(df[self.num_cols])
        x_num_filled = pd.DataFrame(x_num_filled_np, columns=self.num_cols)
        x_cat_filled_np = self.cat_imputer.transform(df[self.cat_cols])
        x_cat_filled = pd.DataFrame(x_cat_filled_np, columns=self.cat_cols)

        # Numerical
        x_norm = self.scaler.transform(x_num_filled)
        x_p = self.quantile_transformer.transform(x_num_filled)
        
        x_gamma_list = [np.full((n_samples,), self.num_stats[col]['skew']) for col in self.num_cols]
        x_gamma = np.stack(x_gamma_list, axis=1)
        
        X_num_final = np.stack([x_norm, x_p, x_gamma], axis=-1).astype(np.float32)

        # Categorical
        X_cat_ids, X_cat_freqs, X_cat_cards = [], [], []
        
        for col in self.cat_cols:
            stats = self.cat_stats[col]
            series = x_cat_filled[col].astype(str)
            
            ids = series.map(stats['id_map']).fillna(stats['default_id']).values
            freqs = series.map(stats['freq_map']).fillna(0.0).values
            cards = np.full((n_samples,), stats['log_card'])
            
            X_cat_ids.append(ids)
            X_cat_freqs.append(freqs)
            X_cat_cards.append(cards)
            
        X_cat_ids_final = np.stack(X_cat_ids, axis=1).astype(np.int64)
        X_cat_meta_final = np.stack([np.stack(X_cat_freqs, axis=1), np.stack(X_cat_cards, axis=1)], axis=-1).astype(np.float32)
        
        return X_num_final, X_cat_ids_final, X_cat_meta_final