import numpy as np

from PIL import Image


def anomaly_map_to_img(anomaly_map: np.ndarray):
    if len(anomaly_map.shape) == 4:
        anomaly_map = np.squeeze(anomaly_map, axis=0)
    
    if len(anomaly_map.shape) == 3:
        anomaly_map = np.squeeze(anomaly_map, axis=-1)
    # Normalize values between 0 and 1
    normalized_anomaly_map = (anomaly_map - np.min(anomaly_map)) / (np.max(anomaly_map) - np.min(anomaly_map))
    normalized_anomaly_map = normalized_anomaly_map * 255
    anomaly_image = Image.fromarray(anomaly_map.astype('uint8'))
    
    return anomaly_image


