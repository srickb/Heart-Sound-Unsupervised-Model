"""

Create a standalone script named 02_train_autoencoder.py.

Goal:
Train an unsupervised autoencoder on the cycle-level numeric feature matrix produced by 01_preprocess.py. The purpose is to learn a compact latent representation of each heart sound cycle in feature space.

Important context:
- This is a tabular feature-learning task, not a raw time-series reconstruction task.
- The model input is a fixed-length numeric feature vector for each cycle.
- Do not use raw waveform tensors as model input.
- Do not build sequence models such as 1D CNN, RNN, LSTM, or Transformer for this stage.
- Use a simple dense autoencoder for tabular numeric data.
- The encoder output will later be used for HDBSCAN clustering.

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
- Build a simple dense autoencoder for tabular features.
- Make hidden layer sizes, activation, dropout, and latent dimension configurable.
- Normalize features in a reproducible way using training data only.
- Prefer a Keras Normalization layer adapted on training data only.
- Use reconstruction loss only.
- Default to MSE loss unless configured otherwise.
- Keep the model small, readable, and easy to inspect.

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
- Add comments explaining feature matrix shape.
- Keep code clear for a new lab member.
- Fail loudly if feature matrix shape is invalid.
- Do not perform clustering in this script.
- Do not add unnecessary abstraction.

Done when:
- the script trains end-to-end on the feature matrix
- encoder and autoencoder are saved
- training history is saved
- reconstruction metrics are saved
- outputs are ready for embedding extraction in the next stage

"""