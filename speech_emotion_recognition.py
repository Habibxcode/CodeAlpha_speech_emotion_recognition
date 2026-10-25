import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

# 1. Map RAVDESS numerical codes to actual emotions
# RAVDESS names files like: 03-01-05-01-01-01-01.wav (The 3rd number '05' is the emotion)
emotion_dictionary = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# 2. Extract Features from Real Audio
def load_data(data_path):
    X, y = [], []
    print("Extracting features from audio files... (This may take a few minutes)")
    
    # Loop through all actor folders
    for actor_folder in os.listdir(data_path):
        actor_path = os.path.join(data_path, actor_folder)
        if not os.path.isdir(actor_path):
            continue
            
        # Loop through audio files in each folder
        for file in os.listdir(actor_path):
            if file.endswith('.wav'):
                file_path = os.path.join(actor_path, file)
                
                # Get the emotion code from the filename
                emotion_code = file.split('-')[2]
                emotion_label = emotion_dictionary[emotion_code]
                
                # Load audio and extract MFCCs
                audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
                mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
                mfccs_scaled = np.mean(mfccs.T, axis=0)
                
                X.append(mfccs_scaled)
                y.append(int(emotion_code) - 1) # 0-indexed for PyTorch

    return np.array(X), np.array(y)

# 3. Load the dataset (Make sure your extracted folders are inside 'ravdess_data')
X, y = load_data("ravdess_data")

# 4. Create the Training Data and Test Data splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 5. Define Deep Learning Architecture (LSTM)
class EmotionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EmotionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# 8 classes for the 8 emotions in RAVDESS
model = EmotionLSTM(input_size=40, hidden_size=128, num_classes=8)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Train the Model on Real Human Voices
print("\nTraining LSTM Model on RAVDESS Dataset...")
epochs = 50
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

# 7. Evaluate on the Test Data (Unseen Voices)
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_classes = torch.argmax(predictions, dim=1)
    acc = accuracy_score(y_test_tensor.numpy(), predicted_classes.numpy())
    
    print(f"\nTest Data Accuracy: {acc*100:.2f}%")
    print("\nClassification Report (Real Data):")
    target_names = list(emotion_dictionary.values())
    print(classification_report(y_test_tensor.numpy(), predicted_classes.numpy(), 
                                target_names=target_names, zero_division=0))

# Save the real trained model
torch.save(model.state_dict(), 'real_speech_emotion_lstm.pth')
print("Model saved to 'real_speech_emotion_lstm.pth'")