import sys
import os

# 1. Force Python to see the main folder so Phase_3 can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 2. Standard Machine Learning Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 3. Agent Import (Cross-folder)
from Phase_3.copali_db import diagnose_anomaly

# 4. Local PyTorch Imports (Since dataset.py and train.py are in the same folder as test.py)
from dataset import prepare_cmapss_data
from train import AeroSense

def run_aerosense_stream(model, test_loader, scaler, trigger_threshold=0.80):
    """
    Simulates live telemetry streaming. 
    The PyTorch model evaluates every batch. If an anomaly is detected, it triggers the LLM.
    """
    print("\n[SYSTEM] Starting AeroSense Watchdog Stream...")
    
    # Ensure model is in evaluation mode (no training)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(test_loader):
            features = features.to(device)
            
            # 1. PyTorch Watchdog predicts failure probability
            logits = model(features)
            probabilities = torch.sigmoid(logits)
            
            # 2. Check each engine's telemetry in the current batch
            for i in range(len(probabilities)):
                prob = probabilities[i].item()
                
                # 3. IF Anomaly is detected!
                if prob >= trigger_threshold:
                    print(f"\n==================================================")
                    print(f"CRITICAL: Watchdog triggered! Probability: {prob:.4f}")
                    
                    # 4. Reverse the normalization to get REAL sensor values for the LLM
                    # (The LLM doesn't understand scaled numbers like 0.05, it needs the real RPMs/Temps)
                    scaled_features_np = features[i].cpu().numpy().reshape(1, -1)
                    raw_sensor_values = scaler.inverse_transform(scaled_features_np)[0]
                    
                    # For simulation, we'll pretend this is Engine 101 at cycle 150
                    mock_engine_id = 101
                    mock_cycle = 150
                    
                    # 5. Synthesize dynamic telemetry query for ColPali
                    # Extract key thermodynamic & aerodynamic indicators:
                    # Sensor s_3: HPC outlet temp (index 5)
                    # Sensor s_7: HPC outlet pressure (index 9)
                    # Sensor s_8: Physical fan speed (index 10)
                    # Sensor s_9: Physical core speed (index 11)
                    t30_val = raw_sensor_values[5]
                    p30_val = raw_sensor_values[9]
                    fan_speed = raw_sensor_values[10]
                    core_speed = raw_sensor_values[11]
                    
                    dynamic_query = (
                        f"Troubleshooting turbofan engine degradation: elevated HPC outlet temperature T30 ({t30_val:.1f} R), "
                        f"HPC outlet pressure P30 ({p30_val:.1f} psia), core speed Nc ({core_speed:.1f} rpm). "
                        f"Fault isolation procedure for High Pressure Compressor (HPC) and Fan Degradation."
                    )
                    
                    # 6. Hand off to the RAG + Vision LLM system
                    diagnose_anomaly(
                        sensor_readings=raw_sensor_values,
                        engine_unit_nr=mock_engine_id,
                        current_cycle=mock_cycle,
                        query=dynamic_query
                    )
                    
                    print("==================================================")
                    
                    # Return immediately after the first detection just to test the pipeline
                    # In production, you would 'continue' to keep monitoring other engines
                    return 

# ==========================================
# 5. FINAL FULL EXECUTION
# ==========================================

if __name__ == "__main__":
    print("=== Initializing AeroSense Pipeline ===")
    
    # 1. Resolve Data Paths
    data_dir = os.path.join(parent_dir, "NASA_Turbofan_Engine_Degradation")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(current_dir, "NASA_Turbofan_Engine_Degradation")
        
    train_file = os.path.join(data_dir, "train_FD003.txt")
    test_file = os.path.join(data_dir, "test_FD003.txt")
    truth_file = os.path.join(data_dir, "RUL_FD003.txt")
    
    if not os.path.exists(test_file):
        print(f"\n[ERROR] NASA C-MAPSS dataset not found at '{data_dir}'.")
        print("Please place 'train_FD003.txt', 'test_FD003.txt', and 'RUL_FD003.txt' in 'NASA_Turbofan_Engine_Degradation/'.")
        sys.exit(1)

    print("Loading CMAPSS Data...")
    train_dataset, test_dataset, scaler = prepare_cmapss_data(
        train_file=train_file, 
        test_file=test_file,
        truth_file=truth_file
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 2. Resolve Model Weights Path
    model_path = os.path.join(current_dir, "aerosense.pth")
    if not os.path.exists(model_path):
        alt_path = os.path.join(parent_dir, "aerosense.pth")
        if os.path.exists(alt_path):
            model_path = alt_path
            
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model weights not found at '{model_path}'.")
        print("Please run 'python pytorch/train.py' first to train and save 'aerosense.pth'.")
        sys.exit(1)
        
    print(f"Loading PyTorch Watchdog weights from '{model_path}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    watchdog_model = AeroSense()
    watchdog_model.load_state_dict(torch.load(model_path, weights_only=True))
    
    # 3. Run the live simulation!
    run_aerosense_stream(watchdog_model, test_loader, scaler, trigger_threshold=0.80)