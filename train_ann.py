import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
from sklearn.metrics import classification_report , roc_auc_score


class Tabular_dataset(Dataset):
  def __init__(self , x , y ):
     # Convert features to float32 and targets to float32 for binary classification
     self.x = torch.tensor(x, dtype=torch.float32)
     self.y = torch.tensor(y, dtype=torch.float32)


  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]



class AnnChurn(nn.Module):
  def __init__(self , input_features , dropout_rate = .3):
    super().__init__()

    self.network = nn.Sequential(

        ## Layer 1
        nn.Linear(input_features , 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(dropout_rate),


        ## Layer 2

        nn.Linear(64 , 32 ),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(dropout_rate),

        ## Layer 3

        nn.Linear(32 , 1)

    )

  def forward(self , x ):
    return self.network(x)


def train_ann_pipeline(
    epochs = 25,
    lr=.01,
    batch_size = 64,
    data_dir = "Data/Processed"
    ):

  # 1) preparing data loaders
  x_train = pd.read_csv(os.path.join(data_dir , "x_train.csv")).values
  y_train = pd.read_csv(os.path.join(data_dir , "y_train.csv")).values
  x_test  = pd.read_csv(os.path.join(data_dir , "x_test.csv")).values
  y_test  = pd.read_csv(os.path.join(data_dir , "y_test.csv")).values

  train_dataset = Tabular_dataset( x_train , y_train)
  test_dataset = Tabular_dataset(x_test , y_test )

  train_loader = DataLoader( train_dataset , batch_size = batch_size , shuffle = True )
  test_loader = DataLoader( test_dataset , batch_size = batch_size , shuffle = False )


  # 2 handling Loss setup with class imbalance weighting ( 3:1 non-churn to churn )

  neg_count = len(y_train) - sum(y_train)
  pos_count = sum(y_train)
  pos_weight = torch.tensor(neg_count / pos_count, dtype=torch.float32)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # Pass x_train.shape[1] (features) instead of x_train.shape[0] (samples)
  model = AnnChurn(x_train.shape[1]).to(device)
  criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) )
  optimizer = torch.optim.AdamW(model.parameters() , lr = lr )

  ## 3 train loop

  val_losses = []
  train_losses = []
  print(f"Training PyTorch ANN on {device} for {epochs} epochs...\n" + "=" * 50)

  for epoch in range(epochs):
    model.train()
    batch_loss = 0.0
    for x_b , y_b in train_loader:
      x_b, y_b = x_b.to(device), y_b.to(device)
      optimizer.zero_grad()
      y_pred = model(x_b)
      loss = criterion( y_pred , y_b )
      loss.backward()
      optimizer.step()
      batch_loss += loss.item() * x_b.size(0)

    epoch_loss = batch_loss / len(train_dataset)
    train_losses.append(epoch_loss)



    ## validation phase
    model.eval()
    batch_val_loss = 0.0
    with torch.no_grad():
      for x_b , y_b in test_loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        y_pred = model(x_b)
        loss = criterion(y_pred , y_b )
        batch_val_loss += loss.item() * x_b.size(0)

      epoch_val_loss = batch_val_loss / len(test_dataset)
      val_losses.append(epoch_val_loss)

    if epoch % 5 == 0 or epoch == 1:
      print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

  
  # 4. Generate, Save, and Display Loss Curve
  plt.figure(figsize=(8, 5))
  plt.plot(range(1, epochs + 1), train_losses, label="Training Loss", color="#2563eb", linewidth=2)
  plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss", color="#dc2626", linewidth=2, linestyle="--")
  plt.title("PyTorch ANN Training vs. Validation Loss (Convergence Analysis)")
  plt.xlabel("Epoch")
  plt.ylabel("BCE Loss")
  plt.legend()
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig("loss_curve.png", dpi=200)
  plt.show()  # Corrected: Display the plot inline inside the notebook outputs
  plt.close()
  print("Saved convergence graph to loss_curve.png")


  # 5. Final Evaluation on Test Set
  model.eval()
  all_preds = []
  all_probs = []
  with torch.no_grad():
      for batch_X, _ in test_loader:
          batch_X = batch_X.to(device)
          logits = model(batch_X)
          probs = torch.sigmoid(logits)
          preds = (probs >= 0.5).int()
          all_probs.extend(probs.cpu().numpy().flatten())
          all_preds.extend(preds.cpu().numpy().flatten())

  print("\n" + "=" * 50)
  print("PyTorch ANN Classification Report:")
  print(classification_report(y_test, all_preds, target_names=["Retained", "Churned"]))
  print(f"Test ROC-AUC: {roc_auc_score(y_test, all_probs):.4f}")


if __name__ == "__main__":
    train_ann_pipeline()