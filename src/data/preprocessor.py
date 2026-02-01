"""
DAF-MoE Data Preprocessor
=========================
Implements the distribution-aware preprocessing logic including:
1. Percentile ranking (Phi) and Skewness (Gamma) for numerical features.
2. Symmetric Rareness (Tilde F) and Log-cardinality (C) for categorical features.
(Section 3.1 & 3.3.1 in the paper)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer
from scipy.stats import skew

class DAFPreprocessor:
    """
    Handles distribution-embedded feature engineering.
    """
    def __init__(self, num_cols, cat_cols, config=None):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        
        n_quantiles = getattr(config, 'n_quantiles', 1000)
        subsample = getattr(config, 'subsample', 100000)
        
        # Imputers
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        
        # Processors
        self.scaler = StandardScaler()
        self.quantile_transformer = QuantileTransformer(
            output_distribution='uniform', 
            n_quantiles=n_quantiles, 
            random_state=42,
            subsample=subsample
        )
        self.label_encoders = {col: LabelEncoder() for col in cat_cols}
        
        # Stats storage
        self.num_stats = {}
        self.cat_stats = {}
        self.cat_offsets = {}  
        self.max_rows = 0

    def fit(self, df):
        self.max_rows = len(df)
        
        # 1. Numerical Fit: Calculate Global Statistics (Gamma)
        if len(self.num_cols) > 0:
            self.num_imputer.fit(df[self.num_cols])
            num_filled = pd.DataFrame(self.num_imputer.transform(df[self.num_cols]), columns=self.num_cols)
            
            self.scaler.fit(num_filled)
            self.quantile_transformer.fit(num_filled)
            
            for col in self.num_cols:
                # Gamma_j (Equation 1)
                self.num_stats[col] = {'skew': skew(num_filled[col])}
        
        # 2. Categorical Fit: Calculate Frequencies and Cardinalities (C_j)
        if len(self.cat_cols) > 0:
            cat_data = df[self.cat_cols].astype(str)
            self.cat_imputer.fit(cat_data)
            cat_filled = pd.DataFrame(self.cat_imputer.transform(cat_data), columns=self.cat_cols)
            
            current_offset = 0  # To prevent ID overlap in the unified embedding table
            
            for col in self.cat_cols:
                series = cat_filled[col].astype(str)
                le = self.label_encoders[col]
                le.fit(series)
                
                classes = le.classes_
                id_map = dict(zip(classes, le.transform(classes)))
                default_id = id_map.get('Unknown', 0)
                
                # Frequency F_j and Log-cardinality C_j (Equation 2)
                counts = series.value_counts(normalize=True)
                log_card = np.log(len(classes) + 1e-9) / np.log(self.max_rows + 1e-9)
                
                self.cat_stats[col] = {
                    'freq_map': counts.to_dict(), 
                    'log_card': log_card,
                    'id_map': id_map,
                    'default_id': default_id
                }
                
                self.cat_offsets[col] = current_offset
                current_offset += len(classes)
                
        return self

    def transform(self, df):
        """
        Transforms raw data into 3-channel (Numerical) and ID+Meta (Categorical) tensors.
        """
        n_samples = len(df)
        
        # 1. Numerical Transform: [Value, Phi(x), Gamma] (Equation 1)
        if len(self.num_cols) > 0:
            x_num_filled = pd.DataFrame(self.num_imputer.transform(df[self.num_cols]), columns=self.num_cols)
            
            x_norm = self.scaler.transform(x_num_filled)
            x_phi = self.quantile_transformer.transform(x_num_filled) # Phi(x_j)
            x_gamma = np.stack([np.full((n_samples,), self.num_stats[col]['skew']) for col in self.num_cols], axis=1)
            
            X_num_final = np.stack([x_norm, x_phi, x_gamma], axis=-1).astype(np.float32)
        else:
            X_num_final = np.zeros((n_samples, 0, 3), dtype=np.float32)

        # 2. Categorical Transform: [ID, Meta(F_j, C_j)] (Equation 2)
        if len(self.cat_cols) > 0:
            cat_data = df[self.cat_cols].astype(str)
            x_cat_filled = pd.DataFrame(self.cat_imputer.transform(cat_data), columns=self.cat_cols)
            
            X_cat_ids, X_cat_freqs, X_cat_cards = [], [], []
            
            for col in self.cat_cols:
                stats = self.cat_stats[col]
                offset = self.cat_offsets[col] 
                series = x_cat_filled[col].astype(str)
                
                ids = (series.map(stats['id_map']).fillna(stats['default_id']).values + offset)
                freqs = series.map(stats['freq_map']).fillna(0.0).values
                cards = np.full((n_samples,), stats['log_card'])
                
                X_cat_ids.append(ids)
                X_cat_freqs.append(freqs)
                X_cat_cards.append(cards)
            
            X_cat_ids_final = np.stack(X_cat_ids, axis=1).astype(np.int64)
            # Meta channels used for Symmetric Rareness Expansion in the model
            X_cat_meta_final = np.stack([np.stack(X_cat_freqs, axis=1), 
                                         np.stack(X_cat_cards, axis=1)], axis=-1).astype(np.float32)
        else:
            X_cat_ids_final = np.zeros((n_samples, 0), dtype=np.int64)
            X_cat_meta_final = np.zeros((n_samples, 0, 2), dtype=np.float32)
        
        return X_num_final, X_cat_ids_final, X_cat_meta_final