#!/usr/bin/env python3
"""
Car Damage Detection App

Completely fixed version with proper model persistence and session state management.
"""

import streamlit as st
import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image
import pandas as pd
import plotly.express as px

try:
    from ultralytics import YOLO
    import torch
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    st.error("Please install ultralytics: pip install ultralytics")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🚗 Car Damage Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;es the model loading issue and ensures th
        margin-bottom: 1rem;
    }
    .status-loaded {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .status-not-loaded {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .damage-severe { color: #d62728; font-weight: bold; }
    .damage-moderate { color: #ff7f0e; font-weight: bold; }
    .damage-minor { color: #2ca02c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_yolo_model(model_path):
    """Load and cache the YOLO model"""
    try:
        model = YOLO(model_path)
        return model, True, None
    except Exception as e:
        return None, False, str(e)

def find_available_models():
    """Find available trained models"""
    model_paths = [
        "runs/train/yolov8_damage_detection_improved/weights/best.pt",
        "runs/train/yolov8_damage_detection11/weights/best.pt",
        "runs/train/yolov8_damage_detection11/weights/last.pt",
        "yolov8n.pt"
    ]
    
    available = []
    for path in model_paths:
        if os.path.exists(path):
            available.append(path)
    
    return available

def initialize_session_state():
    """Initialize session state variables"""
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'current_model_path' not in st.session_state:
        st.session_state.current_model_path = None
    if 'model_load_attempted' not in st.session_state:
        st.session_state.model_load_attempted = False

def assess_severity(class_name, confidence, area, image_shape):
    """Assess damage severity"""
    image_area = image_shape[0] * image_shape[1]
    relative_area = area / image_area
    
    if class_name == 'glass-shatter':
        if relative_area > 0.05 or confidence > 0.8:
            return 'severe'
        elif relative_area > 0.02 or confidence > 0.6:
            return 'moderate'
        else:
            return 'minor'
    elif class_name == 'smash':
        if relative_area > 0.03 or confidence > 0.7:
            return 'severe'
        elif relative_area > 0.015 or confidence > 0.5:
            return 'moderate'
        else:
            return 'minor'
    elif class_name == 'dent':
        if relative_area > 0.04 or confidence > 0.8:
            return 'severe'
        elif relative_area > 0.02 or confidence > 0.6:
            return 'moderate'
        else:
            return 'minor'
    elif class_name == 'scratch':
        if confidence > 0.8 or relative_area > 0.03:
            return 'severe'
        elif confidence > 0.6 or relative_area > 0.015:
            return 'moderate'
        else:
            return 'minor'
    return 'moderate'

def detect_damages(model, image, conf_threshold=0.1, iou_threshold=0.4):
    """Detect damages using the loaded model"""
    try:
        results = model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=100,
            save=False,
            verbose=False,
            augment=True
        )
        
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    class_name = result.names[class_id] if hasattr(result, 'names') else f"class_{class_id}"
                    area = (x2 - x1) * (y2 - y1)
                    severity = assess_severity(class_name, confidence, area, image.shape)
                    
                    detections.append({
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'area': area,
                        'severity': severity
                    })
        
        return detections
    except Exception as e:
        st.error(f"Detection failed: {e}")
        return []

def visualize_detections(image, detections):
    """Visualize detections on image"""
    image_copy = image.copy()
    
    severity_colors = {
        'minor': (0, 255, 0),
        'moderate': (0, 165, 255),
        'severe': (0, 0, 255)
    }
    
    for detection in detections:
        x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]
        color = severity_colors[detection['severity']]
        
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        
        label = f"{detection['class_name']} ({detection['severity']})"
        conf_label = f"{detection['confidence']:.2f}"
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image_copy, (x1, y1-35), (x1 + label_size[0], y1), color, -1)
        
        cv2.putText(image_copy, label, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image_copy, conf_label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return image_copy

def calculate_costs(detections):
    """Calculate repair costs and time"""
    damage_costs = {
        'scratch': {'minor': 100, 'moderate': 200, 'severe': 400},
        'dent': {'minor': 200, 'moderate': 400, 'severe': 800},
        'glass-shatter': {'minor': 300, 'moderate': 500, 'severe': 800},
        'smash': {'minor': 500, 'moderate': 1000, 'severe': 2000}
    }
    
    repair_times = {
        'scratch': {'minor': 2, 'moderate': 4, 'severe': 8},
        'dent': {'minor': 3, 'moderate': 6, 'severe': 12},
        'glass-shatter': {'minor': 2, 'moderate': 3, 'severe': 4},
        'smash': {'minor': 8, 'moderate': 16, 'severe': 24}
    }
    
    total_cost = 0
    total_time = 0
    breakdown = {'minor': 0, 'moderate': 0, 'severe': 0}
    damage_types = {}
    
    for detection in detections:
        class_name = detection['class_name']
        severity = detection['severity']
        
        cost = damage_costs.get(class_name, {}).get(severity, 200)
        time = repair_times.get(class_name, {}).get(severity, 4)
        
        total_cost += cost
        total_time += time
        breakdown[severity] += 1
        damage_types[class_name] = damage_types.get(class_name, 0) + 1
    
    return total_cost, total_time, breakdown, damage_types

def main():
    """Main application"""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🚗 Car Damage Detection</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Enhanced AI-powered car damage detection with proper model persistence**
    
    """)
    
    # Find available models
    available_models = find_available_models()
    
    # Sidebar
    st.sidebar.markdown("## 🔧 Model Management")
    
    if not available_models:
        st.sidebar.error("❌ No trained models found!")
        st.sidebar.info("Please train a model first:\n`python train_improved_model.py`")
        st.error("No models available. Please train a model first.")
        return
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Model:",
        available_models,
        format_func=lambda x: os.path.basename(x),
        key="model_selector"
    )
    
    # Model loading
    if st.sidebar.button("🔄 Load Selected Model", key="load_btn"):
        with st.sidebar:
            with st.spinner("Loading model..."):
                model, success, error = load_yolo_model(selected_model)
                if success:
                    st.session_state.model_loaded = True
                    st.session_state.current_model_path = selected_model
                    st.session_state.model_load_attempted = True
                    st.success("✅ Model loaded successfully!")
                else:
                    st.session_state.model_loaded = False
                    st.session_state.current_model_path = None
                    st.error(f"❌ Failed to load model: {error}")
    
    # Auto-load first model if none loaded
    if not st.session_state.model_load_attempted and available_models:
        with st.sidebar:
            with st.spinner("Auto-loading first available model..."):
                model, success, error = load_yolo_model(available_models[0])
                if success:
                    st.session_state.model_loaded = True
                    st.session_state.current_model_path = available_models[0]
                    st.session_state.model_load_attempted = True
                    st.success(f"✅ Auto-loaded: {os.path.basename(available_models[0])}")
                else:
                    st.session_state.model_loaded = False
                    st.error(f"❌ Auto-load failed: {error}")
                st.session_state.model_load_attempted = True
    
    # Display model status
    if st.session_state.model_loaded and st.session_state.current_model_path:
        model_name = os.path.basename(st.session_state.current_model_path)
        st.sidebar.markdown(f"""
        <div class="status-loaded">
            ✅ <strong>Model Loaded:</strong><br>{model_name}
        </div>
        """, unsafe_allow_html=True)
        
        # Get the cached model
        model, _, _ = load_yolo_model(st.session_state.current_model_path)
        
    else:
        st.sidebar.markdown("""
        <div class="status-not-loaded">
            ❌ <strong>No Model Loaded</strong><br>Please select and load a model above.
        </div>
        """, unsafe_allow_html=True)
        st.warning("⚠️ Please load a model from the sidebar to continue.")
        return
    
    # Detection parameters
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Detection Settings")
    
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.01, 
        max_value=1.0, 
        value=0.1,
        step=0.01,
        help="Lower = more sensitive (detects more damages)"
    )
    
    iou_threshold = st.sidebar.slider(
        "IoU Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.4,
        step=0.05,
        help="Higher = fewer duplicate detections"
    )
    
    # File upload
    st.markdown("## 📁 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a car image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image showing potential vehicle damage"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        
        # Display images
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📷 Original Image")
            st.image(image, use_container_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze Damage", type="primary"):
            with st.spinner("Analyzing image for damage..."):
                # Detect damages
                detections = detect_damages(model, image_np, conf_threshold, iou_threshold)
                
                if detections:
                    # Visualize results
                    result_image = visualize_detections(image_np, detections)
                    
                    with col2:
                        st.markdown("### 🎯 Detection Results")
                        st.image(result_image, use_container_width=True)
                    
                    # Calculate metrics
                    total_cost, total_time, severity_breakdown, damage_types = calculate_costs(detections)
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## 📊 Damage Assessment Report")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Damages", len(detections))
                    with col2:
                        st.metric("Estimated Cost", f"${total_cost:,.2f}")
                    with col3:
                        st.metric("Repair Time", f"{total_time:.1f} hrs")
                    with col4:
                        avg_confidence = np.mean([d['confidence'] for d in detections])
                        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                    
                    # Charts
                    if len(detections) > 0:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if damage_types:
                                st.markdown("### 🔧 Damage Types")
                                damage_df = pd.DataFrame(
                                    list(damage_types.items()),
                                    columns=['Type', 'Count']
                                )
                                fig = px.pie(damage_df, values='Count', names='Type')
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            if any(severity_breakdown.values()):
                                st.markdown("### ⚠️ Severity Levels")
                                severity_df = pd.DataFrame(
                                    list(severity_breakdown.items()),
                                    columns=['Severity', 'Count']
                                )
                                colors = {'minor': '#2ca02c', 'moderate': '#ff7f0e', 'severe': '#d62728'}
                                fig = px.bar(severity_df, x='Severity', y='Count',
                                           color='Severity', color_discrete_map=colors)
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed list
                    st.markdown("### 📋 Detailed Damage List")
                    damage_data = []
                    for i, detection in enumerate(detections, 1):
                        damage_data.append({
                            'ID': i,
                            'Type': detection['class_name'].replace('-', ' ').title(),
                            'Severity': detection['severity'].title(),
                            'Confidence': f"{detection['confidence']:.2f}",
                            'Bbox': f"({detection['bbox'][0]:.0f}, {detection['bbox'][1]:.0f}, {detection['bbox'][2]:.0f}, {detection['bbox'][3]:.0f})"
                        })
                    
                    st.dataframe(pd.DataFrame(damage_data), use_container_width=True)
                    
                    # Recommendations
                    st.markdown("### 💡 Recommendations")
                    if severity_breakdown['severe'] > 0:
                        st.error("🚨 URGENT: Severe damage detected - immediate professional assessment recommended")
                    if 'glass-shatter' in damage_types:
                        st.warning("⚠️ Glass damage detected - safety hazard, repair immediately")
                    if total_cost > 2000:
                        st.info("💰 High repair cost - consider comprehensive insurance claim")
                
                else:
                    with col2:
                        st.markdown("### 🎯 Detection Results")
                        st.image(image_np, use_container_width=True)
                    
                    st.info("✅ No damage detected in this image")
                    st.markdown(f"""
                    **Try adjusting settings:**
                    - Lower confidence threshold (current: {conf_threshold:.2f})
                    - Check image quality and lighting
                    - Ensure damage is clearly visible
                    """)

if __name__ == "__main__":
    main()
