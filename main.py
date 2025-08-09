import base64
import io
import librosa
import modal
import numpy as np
import requests
import torch.nn as nn
import torchaudio.transforms as T
import torch
from model import AudioCNN
from pydantic import BaseModel
import soundfile as sf


app = modal.App("audio-cnn-inference")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["libsndfile1"])
         .add_local_python_source("model"))

model_volume = modal.Volume.from_name("esc-model")


class AudioProcessor:
    def __init__(self):
        self.transform = nn.Sequential( 
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB()
    )
        
    def process_audio_chuck(self, audio_data):
        waveform = torch.from_numpy(audio_data).float()
        waveform = waveform.unsqueeze(0)  # Add batch dimension
        spectrogram = self.transform(waveform)
        return spectrogram.unsqueeze(0)  # Remove batch dimension

class InferenceRequest(BaseModel):
    audio_data: str

@app.cls(image=image, gpu="A10G", volumes={"/models": model_volume}, scaledown_window=15)
class AudioClassifier:
    @modal.enter()
    def load_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load('/models/best_model.pth', map_location=self.device)
        self.classes = checkpoint['classes']

        self.model = AudioCNN(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.processor = AudioProcessor()
        self.audio_processor = AudioProcessor()
        print("Model loaded successfully on enter.")

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: InferenceRequest):
        audio_bytes = base64.b64decode(request.audio_data)

        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype='float32')

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)  # Convert to mono if stereo

        if sample_rate != 22050:
            audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=22050)

        spectrogram = self.audio_processor.process_audio_chuck(audio_data)
        spectrogram = spectrogram.to(self.device)

        with torch.no_grad():
            output, feature_maps = self.model(spectrogram, return_feature_maps=True)
            output = torch.nan_to_num(output)
            probabilities = torch.softmax(output, dim=1)
            top3_probs, top3_indices = torch.topk(probabilities[0], 3)

            predictions = [{"class": self.classes[idx.item()], "confidence": prob.item()}
                           for prob, idx in zip(top3_probs, top3_indices)
                           ]
            viz_data = {}
            for name, tensor in feature_maps.items():
                if tensor.dim() == 4: # [batch_size, channels, height, width]
                    aggregated_tensor = torch.mean(tensor, dim=1) # Aggregate over batch and channels
                    squeezed_tensor = aggregated_tensor.squeeze(0)# Remove batch dimension
                    numpy_array = squeezed_tensor.cpu().numpy() # Convert to numpy array
                    clean_array = np.nan_to_num(numpy_array) # Clean NaN values
                    viz_data[name] = {
                        "shape": list(clean_array.shape),
                        "values": clean_array.tolist()
                    }  # Convert to list for JSON serialization

           

            spectrogram_np = spectrogram.squeeze(0).squeeze(0).cpu().numpy()  # Remove batch dimension and channels
            clean_spectrogram = np.nan_to_num(spectrogram_np)  # Clean NaN values
            
            max_samples = 8000
            waveform_sample_rate = 22050
            if len(audio_data) > max_samples:
                step = len(audio_data) // max_samples
                waveform_data = audio_data[::step]
            else:
                waveform_data = audio_data

        response = {
            "predictions": predictions,
            "visualization": viz_data,
            "input_spectrogram": {
                "shape": list(clean_spectrogram.shape),
                "values": clean_spectrogram.tolist()
            },
            "waveform": {
                "values": waveform_data.tolist(),
                "sample_rate": waveform_sample_rate,
                "duration": len(audio_data) / waveform_sample_rate
            }
        }

        return response
    
@app.local_entrypoint()
def main():
    audio_data, sample_rate = sf.read("knocking.wav")

    buffer = io.BytesIO()
    sf.write(buffer, audio_data, 22050, format='WAV')
    audio_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    payload = {"audio_data": audio_b64}

    server = AudioClassifier()
    url = server.inference.get_web_url()
    response = requests.post(url, json=payload)
    response.raise_for_status()

    result = response.json()

    waveform_info = result.get("waveform", {})
    if waveform_info:
        values = waveform_info.get("values", {})
        print(f"First 10 values: {[round(v, 4) for v in values[:10]]}...")
        print(f"Duration: {waveform_info.get("duration", 0)}")

    print("Top Predictions")

    for pred in result.get("predictions", []):
        print(f"Class: {pred['class']}, Confidence: {pred['confidence']:0.2%}")



