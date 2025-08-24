# 🚗 Car Damage Detection System


![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Features

- **Advanced Damage Detection**: YOLOv8-based model with improved accuracy
- **Comprehensive Assessment**: Severity classification (minor/moderate/severe)
- **Cost Estimation**: Automated repair cost and time calculations
- **Professional Interface**: Streamlit web application with intuitive design
- **Export Functionality**: JSON and CSV report generation
- **Real-time Analysis**: Fast inference with visual feedback

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 4GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/IITM-ML-Collective/car_damage_assessment.git
   cd car-damage-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run fixed_app.py
   ```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Detection Rate | 90-100% |
| Average Confidence | 0.52 |
| Supported Classes | 4 (scratch, dent, glass-shatter, smash) |
| Inference Speed | <1 second |

## 🔧 Usage

### Web Application
1. Start the app: `streamlit run fixed_app.py`
2. The model loads automatically on startup
3. Upload a car image
4. Click "Analyze Damage" for instant results
5. View comprehensive damage assessment and cost estimates

### Training Custom Model
```bash
python train_improved_model.py
```

### Testing System
```bash
python test_improved_system.py
```

## 📁 Project Structure

```
car_damage_assessment/
├── fixed_app.py              # Main Streamlit application
├── train_improved_model.py   # Model training script
├── test_improved_system.py   # System testing utilities
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .gitignore               # Git ignore rules
```

## 🎯 Damage Classes

The system detects four types of vehicle damage:

1. **Scratch** - Surface damage requiring paint work
2. **Dent** - Body panel deformation
3. **Glass-shatter** - Window/windshield damage
4. **Smash** - Major structural damage

## 💰 Cost Estimation

Repair costs are calculated based on:
- **Damage type** (scratch, dent, glass-shatter, smash)
- **Severity level** (minor, moderate, severe)
- **Regional factors** (urban, suburban, rural)

## 🔍 Technical Details

### Model Architecture
- **Base Model**: YOLOv8n
- **Input Size**: 640x640 pixels
- **Confidence Threshold**: 0.1 (adjustable)
- **IoU Threshold**: 0.4

### Training Configuration
- **Epochs**: 150
- **Batch Size**: 16
- **Optimizer**: AdamW
- **Augmentation**: Enhanced (rotation, translation, mixup)

## 📈 Performance Improvements

This system addresses common issues in car damage detection:

- **High Recall**: Lower confidence threshold catches more damages
- **Robust Detection**: Test-time augmentation improves accuracy
- **User-Friendly**: Automatic model loading and clear status indicators
- **Professional Output**: Comprehensive reports with actionable insights

## 🛠️ Development

### Adding New Damage Types
1. Update class names in `train_improved_model.py`
2. Add cost/time estimates in `fixed_app.py`
3. Retrain model with new labeled data

### Customizing Cost Estimates
Edit the cost dictionaries in `fixed_app.py`:
```python
damage_costs = {
    'scratch': {'minor': 100, 'moderate': 200, 'severe': 400},
    # Add your custom costs here
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section in the documentation

## 🙏 Acknowledgments

- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/) for the web framework
- The computer vision community for datasets and inspiration


