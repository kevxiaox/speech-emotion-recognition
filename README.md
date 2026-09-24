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
