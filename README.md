# Deepfake Detection Platform

A web-based platform for testing videos against multiple state-of-the-art deepfake detection models.

## Features

- **Multiple Model Support**: Test videos against 6 different deepfake detection models
- **Upload & Analyze**: Drag-and-drop video upload with instant analysis
- **Batch Analysis**: Analyze with selected models or all models at once
- **Ensemble Results**: Get majority-vote results when analyzing with all models
- **Modern UI**: Clean, responsive interface with real-time progress tracking

## Supported Models

1. **DeepfakeBench (Effort)** - ICML 2025 Spotlight, 99.2% accuracy
2. **CViT2** - Transformer-based, trained on 5 datasets, 95.8% accuracy
3. **GenConViT** - Hybrid architecture with Autoencoder/VAE, 95.8% accuracy
4. **Xception + SE** - CNN with attention, 82.6% accuracy
5. **ResNet50** - Deep residual network, 70.3% accuracy
6. **Swin Transformer + SE** - Vision Transformer, 80.0% accuracy

## Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

### Single Model Analysis
1. Upload a video file (MP4, AVI, MOV, MKV, or WEBM)
2. Select one or more detection models from the grid
3. Click "Analyze Selected Models"
4. View results for each model

### All Models Analysis
1. Upload a video file
2. Click "Analyze All Models"
3. View ensemble result (majority vote) and individual model results

## Current Limitations

This is a **demo/simulation version** where model inference is simulated with random predictions. To use real deepfake detection models:

1. Install the actual model libraries (DeepfakeBench, CViT, etc.)
2. Download pretrained weights
3. Replace the `simulate_model_analysis()` function in `app.py` with actual model inference code

## Integration with Real Models

### For DeepfakeBench:
```python
# Install DeepfakeBench
git clone https://github.com/SCLBD/DeepfakeBench.git
cd DeepfakeBench
conda create -n DeepfakeBench python=3.7.2
conda activate DeepfakeBench
sh install.sh
```

### For CViT2:
```python
# Download pretrained weights
wget https://huggingface.co/datasets/Deressa/cvit/blob/main/cvit2_deepfake_detection_ep_50.pth
```

### Example Real Integration:
```python
def real_model_analysis(model_id, video_path):
    if model_id == 'cvit2':
        from cvit_prediction import predict_deepfake
        result = predict_deepfake(video_path, model='cvit2')
        return {
            'is_fake': result['prediction'] > 0.5,
            'confidence': result['confidence'],
            'frames_analyzed': result['num_frames']
        }
    # Add other models...
```

## File Structure

```
train video/
├── app.py                 # Flask backend
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Frontend UI
├── uploads/              # Uploaded videos (auto-created)
├── results/              # Analysis results (auto-created)
└── static/               # Static files (auto-created)
```

## API Endpoints

- `GET /` - Main page
- `GET /api/models` - Get available models
- `POST /api/upload` - Upload video file
- `POST /api/analyze` - Analyze with specific model
- `POST /api/analyze-all` - Analyze with all models

## License

This project is for educational and research purposes. The underlying models have their own licenses:
- DeepfakeBench: CC BY-NC 4.0
- CViT/GenConViT: GPL-3.0
- Others: Refer to respective repositories

## Acknowledgments

- [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) - Comprehensive benchmark
- [CViT](https://github.com/erprogs/CViT) - Convolutional Vision Transformer
- [GenConViT](https://github.com/truemediaorg/GenConViT) - Generative Convolutional Vision Transformer
- [ResNet/Xception/Swin](https://github.com/Vaioskn/deepfake-detection-resnet-xception-swin) - Model comparison

## Support

For issues or questions about the underlying models, please refer to their respective GitHub repositories.
