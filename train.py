import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.configs.default_config import DAFConfig
from src.data.preprocessor import DAFPreprocessor
from src.data.dataset import DAFDataset
from src.models.daf_moe_transformer import DAFMoETransformer
from src.losses.daf_moe_loss import DAFLoss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Config 생성 (기본값 로드)
    config = DAFConfig()
    
    # [Dummy Data Gen]
    print("⚠️ Generating Mock Data...")
    N = 1000
    df = pd.DataFrame({
        'age': np.random.normal(50, 10, N),
        'bp': np.random.normal(120, 20, N),
        'gender': np.random.choice(['M', 'F'], N),
        'type': np.random.choice(['A', 'B', 'C'], N),
        'target': np.random.randint(0, 2, N)
    })
    num_cols = ['age', 'bp']
    cat_cols = ['gender', 'type']

    # 2. Preprocessing with Config
    preprocessor = DAFPreprocessor(num_cols, cat_cols, config=config)
    preprocessor.fit(df)
    X_num, X_cat_idx, X_cat_meta = preprocessor.transform(df)
    
    # 3. Dynamic Config Update
    config.n_numerical = len(num_cols)
    config.n_categorical = len(cat_cols)
    config.n_features = config.n_numerical + config.n_categorical
    config.total_cats = sum([len(le.classes_) for le in preprocessor.label_encoders.values()]) + 10
    
    print(f"📊 Config Updated: {config.n_features} features, {config.total_cats} vocab")

    # 4. Loader & Model
    dataset = DAFDataset(X_num, X_cat_idx, X_cat_meta, df['target'].values, config.task_type)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    model = DAFMoETransformer(config).to(device)
    criterion = DAFLoss(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    # 5. Training Loop
    print("🚀 Start Training...")
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for inputs, targets in pbar:
            for k in inputs: inputs[k] = inputs[k].to(device)
            targets = targets.to(device)
            
            # Forward
            logits, hist, meta = model(**inputs)
            
            # Loss
            losses = criterion(logits, targets, hist, meta)
            
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
            
            epoch_loss += losses['total'].item()
            pbar.set_postfix({'Loss': f"{losses['total'].item():.4f}"})
            
    print("✅ Done!")

if __name__ == "__main__":
    main()