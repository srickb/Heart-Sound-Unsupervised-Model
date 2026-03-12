"""

Create a standalone script named 02_train_autoencoder.py.

Goal:
Train an unsupervised autoencoder on the cycle-level feature matrix produced by 01_preprocess.py. The purpose is to learn a compact latent representation of heart sound cycles that preserves important structural and temporal information for later clustering.

Important context:
- Use the feature-based cycle input created in the preprocessing stage.
- Do not use raw waveform input in this first training script.
- This is unsupervised learning. Do not use labels.
- The encoder output will later be used as the latent embedding for HDBSCAN.

Inputs to load:
outputs/{RUN_NAME}/preprocess/cycle_features.npy
outputs/{RUN_NAME}/preprocess/cycle_metadata.csv
outputs/{RUN_NAME}/preprocess/feature_names.json

Required behavior:
- Load only valid cycles from preprocessing outputs.
- Preserve sample ordering and sample_id.
- Split train/validation at the subject level if subject_id exists; otherwise use recording_id level splitting.
- Do not leak normalization statistics from validation into training.

Model requirements:
- Use TensorFlow/Keras.
- Build a simple and readable dense autoencoder appropriate for tabular feature vectors.
- Make hidden layer sizes and latent dimension configurable at the top of the script.
- Include feature normalization in a reproducible way.
- Prefer a Keras Normalization layer adapted on training data only, so the preprocessing behavior is part of the saved model.
- Use reconstruction loss only.
- Default to MSE loss unless clearly configurable.
- Keep the initial architecture conservative and easy to inspect.

Training requirements:
- Set random seed in a reproducible way.
- Use callbacks:
  - ModelCheckpoint
  - EarlyStopping with restore_best_weights=True
  - CSVLogger
  - TensorBoard
- Save both the full autoencoder and the encoder only.
- Save a text summary of the model architecture.
- Save training history and a compact JSON summary.

Output files to save:
outputs/{RUN_NAME}/training/autoencoder.keras
outputs/{RUN_NAME}/training/encoder.keras
outputs/{RUN_NAME}/training/model_summary.txt
outputs/{RUN_NAME}/training/training_history.csv
outputs/{RUN_NAME}/training/training_summary.json
outputs/{RUN_NAME}/training/tensorboard/
outputs/{RUN_NAME}/training/reconstruction_error_summary.csv

The training summary should include:
- input dimension
- number of training and validation samples
- latent dimension
- best validation loss
- final training loss
- final validation loss
- key hyperparameters

Implementation guidance:
- Add comments explaining tensor shapes.
- Keep code clear for a new lab member.
- Fail loudly if feature matrix shape is invalid.
- Do not perform clustering in this script.
- Do not add unnecessary abstraction.

Done when:
- the script trains end-to-end on the preprocessed feature matrix
- encoder and autoencoder are saved
- training history is saved
- reconstruction metrics are saved
- outputs are ready for embedding extraction in the next stage

"""