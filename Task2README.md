# Conditional GAN for Audio Spectrograms — README

> This repository trains a conditional GAN (cGAN) to generate mel-spectrograms for audio categories and converts them back to audio using an inverse mel + Griffin‑Lim pipeline.

---

## Project summary

The goal of this project is to train a conditional GAN that generates log‑mel spectrograms conditioned on class labels. After training, the generator synthesizes spectrograms for a given class; these are inverted to audio using an inverse mel transform followed by Griffin‑Lim. This is a audio synthesis pipeline that is used here to get audio from the generated spectrograms

---

## File-by-file and code-block explanation

Below is a block‑level explanation of the main notebook code

### 0. Imports & Setup

* Standard imports: `os`, `random`, `math`, `json`, `Path`.
* PyTorch & torchaudio imports.
* Set seeds (`torch.manual_seed`, `random.seed`) for reproducibility, to reduce the randomness.
* Mount Google Drive in Colab: `drive.mount('/content/drive')`.

Purpose: prepare environment, deterministic behavior, and access to google drive.

---

### 1. `TrainAudioSpectrogramDataset` (custom `Dataset`)

* Loads `.wav` files from per-class subfolders.
* Resamples to a fixed `sample_rate` if necessary.
* Converts waveform -> mel spectrogram using `torchaudio.transforms.MelSpectrogram`.
* Applies `log1p` (log(1+x)) to compress magnitude range.
* Pads or crops spectrograms to fixed `max_frames`.
* Normalizes per-sample to `[-1, 1]` (simple heuristic). Optionally, compute dataset-level stats for a more principled normalization.
* Returns: `log_spec` shaped `(1, n_mels, max_frames)` and one-hot `label_vec`.

Reason we do this is because converting audio to a 2D image-like tensor makes it suitable for convolutional GANs.

---

### 2. Generator — `CGAN_Generator`

* Input: concatenated noise vector `z` and class one-hot `y`.
* `fc` projects the combined vector to a low-resolution feature map `(256, 8, 32)`.
* Several `ConvTranspose2d` (deconvolution) layers progressively upsample to `(1, 128, 512)` (n_mels × frames).
* Final activation: `tanh` to match dataset normalization `[-1, 1]`.

this makes the model learn to synthesize spectrogram images conditioned on label.

---

### 3. Discriminator — `CGAN_Discriminator`

* Input: real or fake spectrogram + label map.
* Label embedding: a linear layer expands the one-hot into a label map `(1, H, W)` and concatenates as an extra channel.
* Several `Conv2d` layers downsample the input; final conv produces a single logit per sample.
* Optionally use `spectral_norm` for stability.

To judge whether spectrograms are real or fake conditioned on label.

---

### 4. Utility functions

* `generate_audio_gan(generator, category_idx, num_samples, device, sample_rate=22050)`:

  * Produces a batch of fake spectrograms from the generator (with `no_grad()`).
  * Un-normalizes from `[-1,1]` → `[0,1]`, approximates `expm1` inversion (since dataset used `log1p`).
  * Converts mel → linear (`InverseMelScale`) and runs `GriffinLim` for waveform reconstruction.
  * Returns waveform tensor.

* `save_and_play(wav, sample_rate, filename)` — saves `.wav` and uses IPython `Audio` to play in Colab.

---

### 5. Training function — `train_gan(...)`

Steps taking place in each epoch: 

1. Iterate over `dataloader`.
2. Create `real_labels` and `fake_labels` tensors.
3. Train Discriminator:

   * `optimizer_D.zero_grad()`
   * Evaluate `discriminator(real_specs, labels)` and compute `loss_D_real`
   * Generate `fake_specs = generator(z, labels)`
   * Evaluate `discriminator(fake_specs.detach(), labels)` and compute `loss_D_fake`
   * `loss_D = loss_D_real + loss_D_fake`; `loss_D.backward()`; `optimizer_D.step()`
4. Train Generator:

   * `optimizer_G.zero_grad()`
   * `output = discriminator(fake_specs, labels)`
   * `loss_G = criterion(output, real_labels)` (generator tries to fool Discriminator)
   * `loss_G.backward()`; `optimizer_G.step()`

Extras:

* Periodically (every N epochs) generate and save spectrogram plots + audio for monitoring. This is wrapped in `no_grad()` so it does not affect gradients.
* Checkpoint saving (model + optimizer states) at epoch end.

---

### 6. Main execution block

* Set `DEVICE` (cuda if available), hyperparameters (latent dim, batch size, epochs, lr).
* Build `TrainAudioSpectrogramDataset` and `DataLoader` 
* Initialize models and move to device.
* Call `train_gan(...)`.

---

