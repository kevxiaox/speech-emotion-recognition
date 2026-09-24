# Speech Emotion Recognition

Speech emotion recognition using CNN models and the RAVDESS dataset.

## Models

1. CNN trained from scratch
2. ResNet18 transfer-learning model

The data were separated by actor so that test speakers were unseen during training.

## Test results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Scratch CNN | 50.83% | 0.4667 | 0.4671 |
| ResNet18 transfer learning | 56.67% | 0.5457 | 0.5413 |

## Supported emotions

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## Web application

The Streamlit application can record or upload speech, display its Log-Mel spectrogram, and predict the emotion.

## Run locally

    python -m streamlit run app/app.py

## Dataset

The original RAVDESS audio files are not included because of their size.

## Team members and contributions

This project was completed by a team of two students.

- **Junjie Xiao**: Dataset preparation, speaker-independent splitting, feature extraction, implementation and training of the scratch CNN and ResNet18 transfer-learning model, and model evaluation.
- **Pengkun Xu**: Web-application testing, verification of the inference pipeline, review of result visualisations, documentation, and final project validation.
