# Medical Text Classification Project

A comprehensive system for medical text classification, specifically focused on cancer type prediction and abstract classification. The project consists of a FastAPI backend service and a machine learning pipeline for model finetuning.

## Project Overview

This project provides:
- Cancer type prediction from medical abstracts
- Abstract classification (Cancer/Non-Cancer)
- Model finetuning pipeline for custom training
- RESTful API for easy integration
- Asynchronous processing with Redis

## Project Structure

```
.
├── backend/                 # FastAPI backend service
│   ├── api/                # API endpoints
│   ├── core/              # Core configuration
│   ├── models/            # Pydantic models
│   ├── services/          # Business logic
│   ├── tests/             # Test cases
│   └── Dockerfile         # Backend container config
│
├── ml/                     # Machine learning pipeline
│   ├── data/              # Data processing scripts
│   ├── engine/            # Training engine
│   ├── models/            # Model definitions
│   ├── utils/             # Utility functions
│   └── config/            # Training configurations
│
└── docs/                   # Documentation
```

## Model Architecture

### Base Model
- **Model**: Microsoft Phi-3 (3.8B parameters)
- **Finetuning Method**: Parameter-Efficient Fine-Tuning (PEFT) using LoRA
- **Training Data**: Medical abstracts with cancer-related annotations

### Performance Metrics

| Metric | Before Finetuning | After Finetuning | Improvement |
|--------|------------------|------------------|-------------|
| Accuracy | 84.67% | 89.67% | +5.00% |
| F1 Score (Cancer) | 85.63% | 89.12% | +3.49% |
| F1 Score (Non-Cancer) | 83.57% | 90.16% | +6.59% |

The finetuning process shows significant improvements across all metrics, with the most notable enhancement in the Non-Cancer class F1 score.

## Prerequisites

### System Requirements
- Python 3.10+
- CUDA-capable GPU (for model training and inference)
- Redis server
- Docker (optional)

