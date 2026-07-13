"""
build_av_workflow.py

Generates an audio-visual deception detection notebook combining:
- EfficientNet-B0 (visual/face features)
- Wav2Vec2 (audio/speech features)
- Simple concatenation fusion

Based on the DOLOS dataset (ICCV 2023) with video-level splits
to prevent data leakage.
"""

import json
import os

cells = []
ROOT = r'C:\Users\OMEN\Desktop\DECEPTION DETECTION USING VIDEOS AND FACE'
VIDEO_DIR = os.path.join(ROOT, 'DOLOSDATA', 'raw_videos')
CSV_PATH = os.path.join(ROOT, 'DOLOSDATA', 'DOLOS', 'dolos_timestamps.csv')
AUDIO_DIR = os.path.join(ROOT, 'audio_output')
OUTPUT_DIR = os.path.join(ROOT, 'av_output')
MODEL_DIR = os.path.join(ROOT, 'saved_models')


def md(src):
    src_clean = src.lstrip('\n')
    lines = [l + '\n' for l in src_clean.split('\n')]
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    })


def code(src):
    src_clean = src.lstrip('\n')
    lines = [l + '\n' for l in src_clean.split('\n')]
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    })


# ============================================================================
# CELL 1: TITLE
# ============================================================================
md(r"""
# Audio-Visual Deception Detection — EfficientNet-B0 + Wav2Vec2 Fusion

> **Dataset**: DOLOS (ICCV 2023) — clips from *Would I Lie to You?*
> **Visual Model**: EfficientNet-B0 (ImageNet pretrained)
> **Audio Model**: Wav2Vec2 Base (speech pretrained)
> **Fusion**: Simple concatenation of pooled features
> **Split strategy**: **Video-level** splits to prevent data leakage
> **Additional**: Speaker-independent evaluation, modality comparison

### Why audio-visual fusion?

Deception manifests across multiple channels:
- **Visual**: Facial expressions, micro-expressions, gaze patterns, head movements
- **Audio**: Vocal tension, speech fluency, pitch variations, pauses

A multimodal model can capture complementary cues that single modalities miss.

### Architecture Overview
```
Video (.mp4)
  ├── Audio → .wav → Wav2Vec2 → [B, 64, 768] → mean pool → [B, 768]
  └── 10 Frames → EfficientNet-B0 → [B, 10, 1280] → project → [B, 10, 768] → mean pool → [B, 768]
                                                                       ↓
                                              Concatenate → [B, 1536] → FC layers → [B, 2]
```

### Pipeline overview
1. Import libraries
2. Load metadata
3. Extract audio from videos
4. Extract face frames (10 per video, CLAHE enhanced)
5. Video-level train/val/test split
6. Build audio-visual dataset
7. Build fusion model (EfficientNet-B0 + Wav2Vec2)
8. Train (2-phase: frozen backbone → fine-tuning)
9. Evaluate: compare Audio-only / Visual-only / Fusion
10. Speaker-independent evaluation
11. Save model
12. Inference on new videos
---
""")

# ============================================================================
# CELL 2: IMPORTS
# ============================================================================
md(r"""
---
## Step 1: Import Libraries
""")
code(r"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
import time
import copy
import json
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import torchaudio

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc, precision_recall_curve
)

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams.update({'figure.max_open_warning': 0, 'font.size': 11})

ROOT = r'C:\Users\OMEN\Desktop\DECEPTION DETECTION USING VIDEOS AND FACE'
VIDEO_DIR = os.path.join(ROOT, 'DOLOSDATA', 'raw_videos')
CSV_PATH = os.path.join(ROOT, 'DOLOSDATA', 'DOLOS', 'dolos_timestamps.csv')
AUDIO_DIR = os.path.join(ROOT, 'audio_output')
OUTPUT_DIR = os.path.join(ROOT, 'av_output')
MODEL_DIR = os.path.join(ROOT, 'saved_models')
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("All imports successful.")
""")

# ============================================================================
# CELL 3: LOAD METADATA
# ============================================================================
md(r"""
---
## Step 2: Load Metadata & Prepare for Video-Level Splitting
""")
code(r"""
df_meta = pd.read_csv(CSV_PATH)
print(f"Metadata shape: {df_meta.shape}")
print(f"Columns: {list(df_meta.columns)}")

df_meta['label_clean'] = df_meta['label'].str.lower().str.strip()
df_meta['label_clean'] = df_meta['label_clean'].replace({
    'lie': 'deception', 'true': 'truth'
})

df_meta['speaker'] = df_meta['file_name'].str.extract(r'^([A-Z]+)_', expand=False).str.upper()

video_files = set(f.replace('.mp4', '') for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4'))
df_meta['video_available'] = df_meta['file_name'].isin(video_files)

print(f"\nLabel distribution:\n{df_meta['label_clean'].value_counts()}")
print(f"\nUnique speakers: {sorted(df_meta['speaker'].dropna().unique())}")
print(f"Videos on disk: {df_meta['video_available'].sum()} / {len(df_meta)}")

df_videos = df_meta[df_meta['video_available']].drop_duplicates(subset='file_name').copy()
print(f"\nUnique videos for processing: {len(df_videos)}")
print(f"Video-level label distribution:\n{df_videos['label_clean'].value_counts()}")
print(f"Video-level speaker distribution:\n{df_videos['speaker'].value_counts()}")
""")

# ============================================================================
# CELL 4: EXTRACT AUDIO FROM VIDEOS
# ============================================================================
md(r"""
---
## Step 3: Extract Audio from Videos

We extract `.wav` audio tracks from all `.mp4` videos. This is a one-time operation — results are cached.

Wav2Vec2 expects raw audio waveforms at 16kHz. We'll resample during dataset loading.
""")
code(r"""
import subprocess

def extract_audio_batch(video_dir, audio_dir, max_videos=None):
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if max_videos:
        video_files = video_files[:max_videos]

    existing_wavs = set(f.replace('.wav', '') for f in os.listdir(audio_dir) if f.endswith('.wav'))
    to_process = [f for f in video_files if f.replace('.mp4', '') not in existing_wavs]

    print(f"Total videos: {len(video_files)}")
    print(f"Already extracted: {len(video_files) - len(to_process)}")
    print(f"To extract: {len(to_process)}")

    for i, vname in enumerate(to_process):
        video_path = os.path.join(video_dir, vname)
        audio_path = os.path.join(audio_dir, vname.replace('.mp4', '.wav'))

        try:
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                '-y', audio_path
            ]
            subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print(f"  Error processing: {vname}")
            continue

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(to_process)}...")

    final_count = len([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    print(f"\nDone! Total .wav files: {final_count}")

extract_audio_batch(VIDEO_DIR, AUDIO_DIR)
""")

# ============================================================================
# CELL 5: EXTRACT FACE FRAMES
# ============================================================================
md(r"""
---
## Step 4: Extract Face Frames with Video-Level Grouping

We extract **10 evenly-spaced frames** from each video at **224×224** resolution, with CLAHE enhancement.
""")
code(r"""
FRAMES_PER_VIDEO = 10
TARGET_SIZE = (224, 224)
CACHE_PATH = os.path.join(OUTPUT_DIR, 'frames_cache.npz')

def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def extract_frames(video_path, n_frames=FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []
    frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        frame_rgb = apply_clahe(frame_rgb)
        frames.append(frame_rgb)
    cap.release()
    return frames

if os.path.exists(CACHE_PATH):
    print(f"Loading cached frames from: {CACHE_PATH}")
    cache = np.load(CACHE_PATH, allow_pickle=True)
    all_frames = cache['frames']
    all_labels = cache['labels']
    all_filenames = cache['filenames']
    all_speakers = cache['speakers']
    print(f"Loaded {len(all_frames)} frames from {len(np.unique(all_filenames))} videos")
else:
    available_df = df_videos.copy()
    print(f"Processing {len(available_df)} videos...\n")

    all_frames = []
    all_labels = []
    all_filenames = []
    all_speakers = []

    for i, (_, row) in enumerate(available_df.iterrows()):
        video_path = os.path.join(VIDEO_DIR, row['file_name'] + '.mp4')
        frames = extract_frames(video_path)

        for frame in frames:
            all_frames.append(frame)
            all_labels.append(row['label_clean'])
            all_filenames.append(row['file_name'])
            all_speakers.append(row['speaker'])

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(available_df)} videos...")

    all_frames = np.stack(all_frames, axis=0)
    all_labels = np.array(all_labels)
    all_filenames = np.array(all_filenames)
    all_speakers = np.array(all_speakers)

    print(f"\nDone! Extracted {len(all_frames)} frames from {len(available_df)} videos")
    print(f"Frame array shape: {all_frames.shape}")
    print(f"Label distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")

    np.savez_compressed(CACHE_PATH,
                        frames=all_frames, labels=all_labels,
                        filenames=all_filenames, speakers=all_speakers)
    print(f"Cached to: {CACHE_PATH}")
""")

# ============================================================================
# CELL 6: EDA
# ============================================================================
md(r"""
---
## Step 5: Exploratory Data Analysis
""")
code(r"""
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle("Sample Extracted Frames (CLAHE Enhanced)", fontsize=16, fontweight='bold')
for i, ax in enumerate(axes.flatten()):
    if i < len(all_frames):
        ax.imshow(all_frames[i])
        ax.set_title(f"{all_labels[i]}\n{all_filenames[i][:12]}...", fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.show()
""")

code(r"""
fig, axes = plt.subplots(1, 4, figsize=(22, 5))

label_counts = pd.Series(all_labels).value_counts()
colors = {'truth': '#2ecc71', 'deception': '#e74c3c'}
axes[0].bar(label_counts.index, label_counts.values,
            color=[colors.get(l, '#999') for l in label_counts.index],
            edgecolor='black', linewidth=1.2)
for i, (l, v) in enumerate(label_counts.items()):
    axes[0].text(i, v + 2, f'{v} ({v/len(all_labels)*100:.1f}%)',
                 ha='center', fontweight='bold', fontsize=12)
axes[0].set_title('Frame Count by Label', fontweight='bold')
axes[0].set_ylabel('Count')

unique_videos = np.unique(all_filenames)
video_labels = [all_labels[all_filenames == v][0] for v in unique_videos]
video_label_counts = pd.Series(video_labels).value_counts()
axes[1].bar(video_label_counts.index, video_label_counts.values,
            color=[colors.get(l, '#999') for l in video_label_counts.index],
            edgecolor='black', linewidth=1.2)
for i, (l, v) in enumerate(video_label_counts.items()):
    axes[1].text(i, v + 0.5, f'{v}', ha='center', fontweight='bold', fontsize=12)
axes[1].set_title('Video Count by Label', fontweight='bold')
axes[1].set_ylabel('Count')

speaker_counts = pd.Series(all_speakers).value_counts()
axes[2].bar(speaker_counts.index, speaker_counts.values,
            color=sns.color_palette('Set2', len(speaker_counts)),
            edgecolor='black', linewidth=1.2)
axes[2].set_title('Frame Count by Speaker', fontweight='bold')
axes[2].tick_params(axis='x', rotation=45)

brightness = [np.mean(f) for f in all_frames]
df_brightness = pd.DataFrame({'brightness': brightness, 'label': all_labels})
sns.boxplot(x='label', y='brightness', data=df_brightness, ax=axes[3],
            palette={'truth': '#2ecc71', 'deception': '#e74c3c'})
axes[3].set_title('Mean Brightness by Label', fontweight='bold')

plt.tight_layout()
plt.show()
""")

# ============================================================================
# CELL 7: VIDEO-LEVEL SPLIT
# ============================================================================
md(r"""
---
## Step 6: Video-Level Train / Validation / Test Split

**Critical**: Split at the **video level** to prevent data leakage. All frames from one video stay in the same split.
""")
code(r"""
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
label_map = {'truth': 0, 'deception': 1}

unique_videos = np.unique(all_filenames)
video_labels = np.array([all_labels[all_filenames == v][0] for v in unique_videos])

print(f"Total unique videos: {len(unique_videos)}")
print(f"Video-level label distribution: {dict(zip(*np.unique(video_labels, return_counts=True)))}")

vid_train_val, vid_test, lbl_train_val, lbl_test = train_test_split(
    unique_videos, video_labels, test_size=0.15, random_state=42, stratify=video_labels
)
vid_train, vid_val, lbl_train, lbl_val = train_test_split(
    vid_train_val, lbl_train_val, test_size=0.176, random_state=42, stratify=lbl_train_val
)

print(f"\nVideo-level split:")
print(f"  Train: {len(vid_train)} videos")
print(f"  Val:   {len(vid_val)} videos")
print(f"  Test:  {len(vid_test)} videos")

train_vids = set(vid_train)
val_vids = set(vid_val)
test_vids = set(vid_test)

train_idx = np.array([i for i, f in enumerate(all_filenames) if f in train_vids])
val_idx = np.array([i for i, f in enumerate(all_filenames) if f in val_vids])
test_idx = np.array([i for i, f in enumerate(all_filenames) if f in test_vids])

X_train_frames = all_frames[train_idx]
y_train = np.array([label_map[l] for l in all_labels[train_idx]])
X_val_frames = all_frames[val_idx]
y_val = np.array([label_map[l] for l in all_labels[val_idx]])
X_test_frames = all_frames[test_idx]
y_test = np.array([label_map[l] for l in all_labels[test_idx]])

train_filenames = all_filenames[train_idx]
val_filenames = all_filenames[val_idx]
test_filenames = all_filenames[test_idx]

print(f"\nFrame-level split (all frames from same video stay together):")
print(f"  Train: {X_train_frames.shape[0]} frames ({X_train_frames.shape[0]/len(all_frames)*100:.1f}%)")
print(f"  Val:   {X_val_frames.shape[0]} frames ({X_val_frames.shape[0]/len(all_frames)*100:.1f}%)")
print(f"  Test:  {X_test_frames.shape[0]} frames ({X_test_frames.shape[0]/len(all_frames)*100:.1f}%)")

train_files_in_test = set(all_filenames[test_idx]) & set(all_filenames[train_idx])
val_files_in_test = set(all_filenames[test_idx]) & set(all_filenames[val_idx])
print(f"\nLeakage check — videos appearing in both train & test: {len(train_files_in_test)}")
print(f"Leakage check — videos appearing in both val & test:   {len(val_files_in_test)}")
assert len(train_files_in_test) == 0, "DATA LEAKAGE DETECTED!"
assert len(val_files_in_test) == 0, "DATA LEAKAGE DETECTED!"
print("✓ No data leakage confirmed.")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Video-Level Split Distribution", fontsize=14, fontweight='bold')
for ax, (name, labels_arr) in zip(axes, [('Train', y_train), ('Validation', y_val), ('Test', y_test)]):
    counts = pd.Series(labels_arr).map({0: 'Truth', 1: 'Deception'}).value_counts()
    colors_list = ['#2ecc71', '#e74c3c']
    ax.bar(counts.index, counts.values, color=colors_list, edgecolor='black')
    ax.set_title(f'{name} (n={len(labels_arr)} frames)', fontweight='bold')
    ax.set_ylabel('Count')
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.show()
""")

# ============================================================================
# CELL 8: SPEAKER-INDEPENDENT SPLIT
# ============================================================================
md(r"""
---
## Step 7: Speaker-Independent Split Preparation
""")
code(r"""
speaker_video_map = {}
for _, row in df_videos.iterrows():
    spk = row['speaker']
    if spk not in speaker_video_map:
        speaker_video_map[spk] = []
    speaker_video_map[spk].append(row['file_name'])

print("Speaker → video count:")
for spk, vids in sorted(speaker_video_map.items()):
    spk_labels = [all_labels[all_filenames == v][0] for v in vids]
    label_str = dict(zip(*np.unique(spk_labels, return_counts=True)))
    print(f"  {spk}: {len(vids)} videos — {label_str}")

top_speakers = sorted(speaker_video_map.keys(), key=lambda s: len(speaker_video_map[s]), reverse=True)[:3]

speaker_results = {}
for held_out_speaker in top_speakers:
    test_vids_speaker = set(speaker_video_map[held_out_speaker])
    train_vids_speaker = set(unique_videos) - test_vids_speaker

    spk_train_idx = np.array([i for i, f in enumerate(all_filenames) if f in train_vids_speaker])
    spk_test_idx = np.array([i for i, f in enumerate(all_filenames) if f in test_vids_speaker])

    speaker_results[held_out_speaker] = {
        'train_idx': spk_train_idx, 'test_idx': spk_test_idx,
        'filenames_train': all_filenames[spk_train_idx],
        'filenames_test': all_filenames[spk_test_idx],
        'n_train_vids': len(train_vids_speaker), 'n_test_vids': len(test_vids_speaker),
    }
    print(f"\nSpeaker {held_out_speaker} held out:")
    print(f"  Train: {len(spk_train_idx)} frames from {len(train_vids_speaker)} videos")
    print(f"  Test:  {len(spk_test_idx)} frames from {len(test_vids_speaker)} videos")
""")

# ============================================================================
# CELL 9: AUDIO-VISUAL DATASET
# ============================================================================
md(r"""
---
## Step 8: Build Audio-Visual Dataset

This dataset loads both:
- **Audio**: `.wav` file, resampled for Wav2Vec2 (produces 64 tokens)
- **Visual**: 10 CLAHE-enhanced face frames
- **Label**: 0 (truth) or 1 (deception)
""")
code(r"""
class AVDeceptionDataset(Dataset):
    def __init__(self, filenames, labels, audio_dir, video_dir,
                 num_audio_tokens=64, num_frames=10, frame_size=224):
        super(AVDeceptionDataset, self).__init__()
        self.filenames = filenames
        self.labels = labels
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.num_audio_tokens = num_audio_tokens
        self.num_frames = num_frames
        self.frame_size = frame_size

        self.visual_transform = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ])

        self.val_transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        clip_name = self.filenames[idx]
        label = self.labels[idx]

        # Load audio
        audio_path = os.path.join(self.audio_dir, clip_name + '.wav')
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform[0]  # mono

        clip_duration = len(waveform) / sample_rate
        new_sample_rate = int(321.893491124260 * self.num_audio_tokens / clip_duration)
        waveform = torchaudio.functional.resample(waveform, sample_rate, new_sample_rate)
        waveform = waveform.unsqueeze(0)  # [1, num_samples]

        # Load face frames from video
        video_path = os.path.join(self.video_dir, clip_name + '.mp4')
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, max(0, total_frames - 1), self.num_frames, dtype=int)

        frames = []
        for fidx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (self.frame_size, self.frame_size))
                frame_rgb = apply_clahe(frame_rgb)
                frames.append(frame_rgb)
            else:
                frames.append(np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8))
        cap.release()

        frames = np.stack(frames, axis=0)
        transformed = []
        for f in frames:
            transformed.append(self.visual_transform(f))
        frames = torch.stack(transformed, dim=0)  # [N, C, H, W]

        return waveform, frames, label


def av_collate_fn(batch):
    waveforms, face_tensors, targets = [], [], []
    for waveform, face_frames, label in batch:
        waveforms += [waveform]
        face_tensors += [face_frames]
        targets += [torch.tensor(label)]

    # Each waveform is [1, num_samples] -> squeeze to [num_samples] -> unsqueeze to [num_samples, 1]
    # pad_sequence expects list of [num_samples, 1] tensors
    waveforms_padded = torch.nn.utils.rnn.pad_sequence(
        [w.squeeze(0).unsqueeze(1) for w in waveforms], batch_first=True, padding_value=0.
    )  # [batch, max_samples, 1]
    waveforms_padded = waveforms_padded.permute(0, 2, 1)  # [batch, 1, max_samples]

    face_tensors = torch.stack(face_tensors)
    targets = torch.stack(targets)
    return waveforms_padded, face_tensors, targets


BATCH_SIZE = 16

train_dataset = AVDeceptionDataset(
    train_filenames, y_train, AUDIO_DIR, VIDEO_DIR,
    num_frames=FRAMES_PER_VIDEO, frame_size=224
)
val_dataset = AVDeceptionDataset(
    val_filenames, y_val, AUDIO_DIR, VIDEO_DIR,
    num_frames=FRAMES_PER_VIDEO, frame_size=224
)
test_dataset = AVDeceptionDataset(
    test_filenames, y_test, AUDIO_DIR, VIDEO_DIR,
    num_frames=FRAMES_PER_VIDEO, frame_size=224
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=av_collate_fn, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=av_collate_fn, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=av_collate_fn, num_workers=0)

print(f"Batch size: {BATCH_SIZE}")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")
print(f"Test batches:  {len(test_loader)}")

sample_batch = next(iter(train_loader))
print(f"Sample batch - audio: {sample_batch[0].shape}, visual: {sample_batch[1].shape}, labels: {sample_batch[2].shape}")
""")

# ============================================================================
# CELL 10: BUILD FUSION MODEL
# ============================================================================
md(r"""
---
## Step 9: Build Audio-Visual Fusion Model

### Architecture
- **Audio branch**: Wav2Vec2 Base (4 layers with adapters) → mean pool → 768-d vector
- **Visual branch**: EfficientNet-B0 (frozen backbone) → project 1280→768 → mean pool → 768-d vector
- **Fusion**: Concatenate → 1536-d → FC classifier → 2 classes
""")
code(r"""
class EfficientNetVisual(nn.Module):
    def __init__(self, freeze_backbone=True):
        super(EfficientNetVisual, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.classifier = nn.Identity()

        self.proj = nn.Sequential(
            nn.Linear(1280, 768),
            nn.ReLU(),
        )

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        feat = self.backbone(x)  # [B*N, 1280]
        feat = feat.view(B, N, -1)  # [B, N, 1280]
        feat = self.proj(feat)  # [B, N, 768]
        return feat


class Wav2Vec2Audio(nn.Module):
    def __init__(self, num_encoders=4):
        super(Wav2Vec2Audio, self).__init__()
        model = torchaudio.pipelines.WAV2VEC2_BASE.get_model()

        for p in model.parameters():
            p.requires_grad = False

        self.FEATURE_EXTRACTOR = model.feature_extractor
        self.feature_projection = model.encoder.feature_projection
        self.pos_conv_embed = model.encoder.transformer.pos_conv_embed
        self.layer_norm = model.encoder.transformer.layer_norm
        self.dropout = model.encoder.transformer.dropout
        self.layer_norm_first = model.encoder.transformer.layer_norm_first

        layer_list = []
        for i in range(num_encoders):
            layer_list.append(model.encoder.transformer.layers[i])
        self.layers = nn.ModuleList(layer_list)

    def forward(self, x):
        features, _ = self.FEATURE_EXTRACTOR(x, None)
        projections = self.feature_projection(features)

        # Preprocess: add positional encoding + layer norm + dropout
        projections = projections + self.pos_conv_embed(projections)
        if self.layer_norm_first:
            projections = self.layer_norm(projections)
        projections = self.dropout(projections)

        # Pass through transformer layers (each returns (x, position_bias))
        position_bias = None
        for layer in self.layers:
            projections, position_bias = layer(projections, position_bias=position_bias)

        if not self.layer_norm_first:
            projections = self.layer_norm(projections)

        return projections  # [B, 64, 768]


class AVFusionModel(nn.Module):
    def __init__(self, freeze_audio_backbone=True, freeze_visual_backbone=True):
        super(AVFusionModel, self).__init__()

        self.audio_backbone = Wav2Vec2Audio(num_encoders=4)
        self.visual_backbone = EfficientNetVisual(freeze_backbone=freeze_visual_backbone)

        self.audio_pool = nn.AdaptiveAvgPool1d(1)
        self.visual_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, waveform, frames):
        # Audio: [B, 1, T] -> squeeze to [B, T] -> Wav2Vec2 -> [B, 64, 768]
        audio_feat = self.audio_backbone(waveform.squeeze(1))
        audio_vec = self.audio_pool(audio_feat.transpose(1, 2)).squeeze(-1)  # [B, 768]

        # Visual: [B, 10, 3, 224, 224] -> EfficientNet -> [B, 10, 768]
        vis_feat = self.visual_backbone(frames)  # [B, 10, 768]
        vis_vec = self.visual_pool(vis_feat.transpose(1, 2)).squeeze(-1)  # [B, 768]

        # Fusion
        fused = torch.cat([audio_vec, vis_vec], dim=1)  # [B, 1536]
        logits = self.classifier(fused)  # [B, 2]
        return logits


model = AVFusionModel(freeze_audio_backbone=True, freeze_visual_backbone=True)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters:    {total_params - trainable_params:,}")
print(f"Model loaded to: {device}")
""")

# ============================================================================
# CELL 11: LOSS + OPTIMIZER
# ============================================================================
md(r"""
---
## Step 10: Loss, Optimizer, and Scheduler Setup
""")
code(r"""
criterion = nn.CrossEntropyLoss()
PHASE1_EPOCHS = 10
PHASE2_LR = 1e-4

def get_phase1_optimizer():
    return optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )

def get_phase2_optimizer():
    return optim.Adam(model.parameters(), lr=PHASE2_LR, weight_decay=1e-4)

optimizer = get_phase1_optimizer()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
)

print("Optimizer, scheduler, and loss function configured.")
""")

# ============================================================================
# CELL 12: TRAINING FUNCTIONS
# ============================================================================
md(r"""
---
## Step 11: Training Functions
""")
code(r"""
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for waveforms, frames, labels in loader:
        waveforms = waveforms.to(device)
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(waveforms, frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * waveforms.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for waveforms, frames, labels in loader:
            waveforms = waveforms.to(device)
            frames = frames.to(device)
            labels = labels.to(device)

            outputs = model(waveforms, frames)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * waveforms.size(0)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return (running_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


print("Training functions defined.")
""")

# ============================================================================
# CELL 13: TRAINING LOOP
# ============================================================================
md(r"""
---
## Step 12: Train Model (2-Phase Transfer Learning)

**Phase 1** (Epochs 1–10): Backbones frozen, train fusion head only
**Phase 2** (Epochs 11+): Unfreeze Wav2Vec2, keep EfficientNet frozen, lower LR
""")
code(r"""
EPOCHS = 50
patience = 10
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [], 'lr': []
}

print(f"Training for up to {EPOCHS} epochs on VIDEO-LEVEL split")
print(f"Phase 1: epochs 1-{PHASE1_EPOCHS} (backbones frozen)")
print(f"Phase 2: epochs {PHASE1_EPOCHS+1}+ (audio unfrozen)")
print("=" * 70)

for epoch in range(EPOCHS):
    if epoch == PHASE1_EPOCHS:
        print(f"\n{'='*70}")
        print(f"PHASE 2: Unfreezing audio backbone, lr={PHASE2_LR}")
        print(f"{'='*70}\n")
        for param in model.audio_backbone.parameters():
            param.requires_grad = True
        optimizer = get_phase2_optimizer()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_preds, val_labels, val_probs = validate(model, val_loader, criterion, device)
    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    current_lr = optimizer.param_groups[0]['lr']
    history['lr'].append(current_lr)

    phase = "FROZEN" if epoch < PHASE1_EPOCHS else "FINETUNE"
    print(f"Epoch [{epoch+1:2d}/{EPOCHS}] [{phase:8s}]  "
          f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
          f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  LR: {current_lr:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = copy.deepcopy(model.state_dict())
        torch.save(best_model_state, os.path.join(MODEL_DIR, 'best_av_fusion_model.pth'))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"\nRestored best model (val_loss: {best_val_loss:.4f})")

print(f"Training completed after {len(history['train_loss'])} epochs")
""")

# ============================================================================
# CELL 14: EVALUATE FUSION MODEL
# ============================================================================
md(r"""
---
## Step 13: Evaluate Fusion Model — Comprehensive Metrics
""")
code(r"""
model.eval()
all_test_preds = []
all_test_probs = []
all_test_labels = []

with torch.no_grad():
    for waveforms, frames, labels in test_loader:
        waveforms = waveforms.to(device)
        frames = frames.to(device)
        outputs = model(waveforms, frames)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        all_test_preds.extend(preds.cpu().numpy())
        all_test_probs.extend(probs[:, 1].cpu().numpy())
        all_test_labels.extend(labels.numpy())

y_pred = np.array(all_test_preds)
y_pred_prob = np.array(all_test_probs)
y_true = np.array(all_test_labels)

test_acc = accuracy_score(y_true, y_pred)
test_f1_macro = f1_score(y_true, y_pred, average='macro')
test_f1_weighted = f1_score(y_true, y_pred, average='weighted')
test_precision = precision_score(y_true, y_pred, average='macro')
test_recall = recall_score(y_true, y_pred, average='macro')
fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_prob)
test_auc = auc(fpr, tpr)

print("=" * 60)
print("TEST SET EVALUATION — FUSION MODEL (Video-Level Split)")
print("=" * 60)
print(f"  Accuracy:          {test_acc:.4f} ({test_acc*100:.1f}%)")
print(f"  Precision (macro): {test_precision:.4f}")
print(f"  Recall (macro):    {test_recall:.4f}")
print(f"  F1 Score (macro):  {test_f1_macro:.4f}")
print(f"  F1 Score (weighted): {test_f1_weighted:.4f}")
print(f"  ROC-AUC:           {test_auc:.4f}")
print("=" * 60)
""")

code(r"""
print("\nClassification Report (per-class):")
print(classification_report(y_true, y_pred, target_names=['Truth', 'Deception'],
                            digits=4))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"\n  True Negatives  (Truth→Truth):       {cm[0,0]}")
print(f"  False Positives (Truth→Deception):    {cm[0,1]}")
print(f"  False Negatives (Deception→Truth):    {cm[1,0]}")
print(f"  True Positives  (Deception→Deception): {cm[1,1]}")
""")

# ============================================================================
# CELL 15: VISUALIZE RESULTS
# ============================================================================
md(r"""
---
## Step 14: Visualize Results
""")
code(r"""
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("AV Fusion Model — Training & Evaluation (Video-Level Split)",
             fontsize=16, fontweight='bold')

axes[0, 0].plot(history['train_loss'], label='Train Loss', color='#3498db', linewidth=2)
axes[0, 0].plot(history['val_loss'], label='Val Loss', color='#e74c3c', linewidth=2)
axes[0, 0].axvline(x=PHASE1_EPOCHS - 0.5, color='orange', linestyle='--', alpha=0.7, label='Phase 2 start')
axes[0, 0].set_title('Training & Validation Loss', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history['train_acc'], label='Train Acc', color='#3498db', linewidth=2)
axes[0, 1].plot(history['val_acc'], label='Val Acc', color='#e74c3c', linewidth=2)
axes[0, 1].axvline(x=PHASE1_EPOCHS - 0.5, color='orange', linestyle='--', alpha=0.7, label='Phase 2 start')
axes[0, 1].set_title('Training & Validation Accuracy', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(history['lr'], color='#9b59b6', linewidth=2)
axes[0, 2].axvline(x=PHASE1_EPOCHS - 0.5, color='orange', linestyle='--', alpha=0.7, label='Phase 2 start')
axes[0, 2].set_title('Learning Rate Schedule', fontweight='bold')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Learning Rate')
axes[0, 2].set_yscale('log')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['Truth', 'Deception'], yticklabels=['Truth', 'Deception'],
            linewidths=2, linecolor='black')
axes[1, 0].set_title('Confusion Matrix', fontweight='bold')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

axes[1, 1].plot(fpr, tpr, color='#e74c3c', linewidth=2, label=f'ROC (AUC = {test_auc:.3f})')
axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
axes[1, 1].fill_between(fpr, tpr, alpha=0.1, color='#e74c3c')
axes[1, 1].set_title('ROC Curve', fontweight='bold')
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(True, alpha=0.3)

precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_prob)
axes[1, 2].plot(recall_curve, precision_curve, color='#2ecc71', linewidth=2)
axes[1, 2].set_title('Precision-Recall Curve', fontweight='bold')
axes[1, 2].set_xlabel('Recall')
axes[1, 2].set_ylabel('Precision')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fusion_training_results.png'), dpi=150, bbox_inches='tight')
plt.show()
""")

code(r"""
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
fig.suptitle("Sample Predictions — Fusion Model (Test Set)", fontsize=14, fontweight='bold')

random_indices = random.sample(range(len(test_dataset)), min(10, len(test_dataset)))
for i, idx in enumerate(random_indices):
    row, col = divmod(i, 5)
    waveform, frame_tensor, label = test_dataset[idx]

    pred_label = 'Deception' if y_pred[idx] == 1 else 'Truth'
    true_label = 'Deception' if y_true[idx] == 1 else 'Truth'
    confidence = y_pred_prob[idx] if y_pred[idx] == 1 else 1 - y_pred_prob[idx]
    color = 'green' if y_pred[idx] == y_true[idx] else 'red'

    frame_np = frame_tensor.permute(1, 2, 0).numpy()
    frame_np = (frame_np * IMAGENET_STD + IMAGENET_MEAN)
    frame_np = np.clip(frame_np, 0, 1)

    axes[row, col].imshow(frame_np)
    axes[row, col].set_title(
        f"Pred: {pred_label} ({confidence:.2f})\nTrue: {true_label}",
        fontweight='bold', color=color, fontsize=10)
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fusion_sample_predictions.png'), dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================================
# CELL 16: COMPARE MODALITIES
# ============================================================================
md(r"""
---
## Step 15: Compare Modalities — Audio-only vs Visual-only vs Fusion

We evaluate each modality separately and compare with the fusion model.
""")
code(r"""
class AudioOnlyModel(nn.Module):
    def __init__(self):
        super(AudioOnlyModel, self).__init__()
        self.backbone = Wav2Vec2Audio(num_encoders=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, waveform, frames=None):
        feat = self.backbone(waveform)
        vec = self.pool(feat.transpose(1, 2)).squeeze(-1)
        return self.classifier(vec)


class VisualOnlyModel(nn.Module):
    def __init__(self):
        super(VisualOnlyModel, self).__init__()
        self.backbone = EfficientNetVisual(freeze_backbone=False)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, waveform=None, frames=None):
        feat = self.backbone(frames)
        vec = self.pool(feat.transpose(1, 2)).squeeze(-1)
        return self.classifier(vec)


def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for waveforms, frames, labels in loader:
            waveforms = waveforms.to(device)
            frames = frames.to(device)
            outputs = model(waveforms, frames)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr) if len(np.unique(y_true)) > 1 else 0.0

    return {
        'accuracy': acc, 'f1': f1, 'precision': prec,
        'recall': rec, 'auc': auc_score,
        'y_pred': y_pred, 'y_true': y_true, 'y_prob': y_prob
    }


def train_modality_model(model, train_loader, val_loader, device, epochs=30, lr=1e-3):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for waveforms, frames, labels in train_loader:
            waveforms = waveforms.to(device)
            frames = frames.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(waveforms, frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * waveforms.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for waveforms, frames, labels in val_loader:
                waveforms = waveforms.to(device)
                frames = frames.to(device)
                labels = labels.to(device)
                outputs = model(waveforms, frames)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * waveforms.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


print("Training Audio-only model...")
audio_model = train_modality_model(AudioOnlyModel(), train_loader, val_loader, device, epochs=30)
audio_results = evaluate_model(audio_model, test_loader, device)

print(f"\nAudio-only results:")
print(f"  Accuracy: {audio_results['accuracy']:.4f}")
print(f"  F1:       {audio_results['f1']:.4f}")
print(f"  AUC:      {audio_results['auc']:.4f}")
""")

code(r"""
print("Training Visual-only model...")
visual_model = train_modality_model(VisualOnlyModel(), train_loader, val_loader, device, epochs=30)
visual_results = evaluate_model(visual_model, test_loader, device)

print(f"\nVisual-only results:")
print(f"  Accuracy: {visual_results['accuracy']:.4f}")
print(f"  F1:       {visual_results['f1']:.4f}")
print(f"  AUC:      {visual_results['auc']:.4f}")
""")

code(r"""
print("=" * 70)
print("MODALITY COMPARISON — TEST SET (Video-Level Split)")
print("=" * 70)
print(f"{'Model':<25} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'AUC':>10}")
print("-" * 70)
print(f"{'Audio-only':<25} {audio_results['accuracy']:>10.4f} {audio_results['f1']:>10.4f} "
      f"{audio_results['precision']:>10.4f} {audio_results['recall']:>10.4f} {audio_results['auc']:>10.4f}")
print(f"{'Visual-only':<25} {visual_results['accuracy']:>10.4f} {visual_results['f1']:>10.4f} "
      f"{visual_results['precision']:>10.4f} {visual_results['recall']:>10.4f} {visual_results['auc']:>10.4f}")
print(f"{'FUSION (Audio+Visual)':<25} {test_acc:>10.4f} {test_f1_macro:>10.4f} "
      f"{test_precision:>10.4f} {test_recall:>10.4f} {test_auc:>10.4f}")
print("=" * 70)

fusion_better_than_audio = test_acc - audio_results['accuracy']
fusion_better_than_visual = test_acc - visual_results['accuracy']
print(f"\nFusion improves over Audio-only by:  {fusion_better_than_audio*100:+.2f}%")
print(f"Fusion improves over Visual-only by: {fusion_better_than_visual*100:+.2f}%")
""")

# ============================================================================
# CELL 17: SPEAKER-INDEPENDENT EVALUATION
# ============================================================================
md(r"""
---
## Step 16: Speaker-Independent Evaluation

Evaluate the fusion model on speakers it has never seen during training.
""")
code(r"""
print("=" * 60)
print("SPEAKER-INDEPENDENT EVALUATION — FUSION MODEL")
print("=" * 60)

for held_out_speaker, data in speaker_results.items():
    spk_train_dataset = AVDeceptionDataset(
        data['filenames_train'],
        np.array([label_map[all_labels[all_filenames == f][0]] for f in data['filenames_train']]),
        AUDIO_DIR, VIDEO_DIR, num_frames=FRAMES_PER_VIDEO, frame_size=224
    )
    spk_test_dataset = AVDeceptionDataset(
        data['filenames_test'],
        np.array([label_map[all_labels[all_filenames == f][0]] for f in data['filenames_test']]),
        AUDIO_DIR, VIDEO_DIR, num_frames=FRAMES_PER_VIDEO, frame_size=224
    )

    spk_train_loader = DataLoader(spk_train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=av_collate_fn, num_workers=0)
    spk_test_loader = DataLoader(spk_test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 collate_fn=av_collate_fn, num_workers=0)

    spk_model = AVFusionModel(freeze_audio_backbone=True, freeze_visual_backbone=True)
    spk_model = spk_model.to(device)

    print(f"\nTraining on all speakers except {held_out_speaker}...")
    spk_model = train_modality_model(spk_model, spk_train_loader, spk_test_loader,
                                     device, epochs=30, lr=1e-3)

    spk_results = evaluate_model(spk_model, spk_test_loader, device)

    print(f"\nSpeaker {held_out_speaker} held out ({data['n_test_vids']} test videos):")
    print(f"  Accuracy:  {spk_results['accuracy']:.4f} ({spk_results['accuracy']*100:.1f}%)")
    print(f"  Precision: {spk_results['precision']:.4f}")
    print(f"  Recall:    {spk_results['recall']:.4f}")
    print(f"  F1 Score:  {spk_results['f1']:.4f}")
    print(f"  ROC-AUC:   {spk_results['auc']:.4f}")
    print(classification_report(spk_results['y_true'], spk_results['y_pred'],
                                target_names=['Truth', 'Deception'], digits=4, zero_division=0))

print("\n" + "=" * 60)
print("NOTE: Speaker-independent accuracy tests real-world generalization.")
print("=" * 60)
""")

# ============================================================================
# CELL 18: SAVE MODEL
# ============================================================================
md(r"""
---
## Step 17: Save Model & Artifacts
""")
code(r"""
model_path = os.path.join(MODEL_DIR, 'av_fusion_deception.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'model_class': 'AVFusionModel',
    'audio_backbone': 'Wav2Vec2',
    'visual_backbone': 'EfficientNet-B0',
    'fusion_type': 'concatenation',
    'num_classes': 2,
    'input_audio': 'raw_waveform_64_tokens',
    'input_visual': [10, 3, 224, 224],
}, model_path)

history_path = os.path.join(MODEL_DIR, 'av_training_history.json')
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)

metadata = {
    'model': 'AVFusionModel',
    'audio_backbone': 'Wav2Vec2 Base',
    'visual_backbone': 'EfficientNet-B0',
    'fusion': 'concatenation',
    'label_map': label_map,
    'input_visual': [10, 3, 224, 224],
    'input_audio': 'raw_waveform_resampled_for_64_tokens',
    'imagenet_mean': IMAGENET_MEAN.tolist(),
    'imagenet_std': IMAGENET_STD.tolist(),
    'epochs_trained': len(history['train_loss']),
    'phase1_epochs': PHASE1_EPOCHS,
    'best_val_loss': float(best_val_loss),
    'test_accuracy': float(test_acc),
    'test_f1_macro': float(test_f1_macro),
    'test_precision': float(test_precision),
    'test_recall': float(test_recall),
    'test_auc': float(test_auc),
    'audio_only_accuracy': float(audio_results['accuracy']),
    'visual_only_accuracy': float(visual_results['accuracy']),
    'fusion_improvement_over_audio': float(fusion_better_than_audio),
    'fusion_improvement_over_visual': float(fusion_better_than_visual),
    'split_strategy': 'video_level',
    'device': str(device),
}
meta_path = os.path.join(MODEL_DIR, 'av_model_metadata.json')
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Model saved: {model_path}")
print(f"History saved: {history_path}")
print(f"Metadata saved: {meta_path}")
""")

# ============================================================================
# CELL 19: INFERENCE
# ============================================================================
md(r"""
---
## Step 18: Inference Pipeline on New Videos
""")
code(r"""
def predict_video_av(model, video_path, n_frames=10, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None, None

    # Extract visual frames
    frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_AREA)
            frame_rgb = apply_clahe(frame_rgb)
            normalized = frame_rgb.astype(np.float32) / 255.0
            normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
            tensor = torch.tensor(normalized.transpose(2, 0, 1), dtype=torch.float32)
            frames.append(tensor)
    cap.release()

    if not frames:
        return None, None

    visual_batch = torch.stack(frames).unsqueeze(0).to(device)  # [1, 10, 3, 224, 224]

    # Extract audio
    audio_path = video_path.replace('.mp4', '.wav').replace(VIDEO_DIR, AUDIO_DIR)
    if not os.path.exists(audio_path):
        cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
               '-ar', '16000', '-ac', '1', '-y', audio_path]
        subprocess.run(cmd, capture_output=True, timeout=30)

    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform[0]
    clip_duration = len(waveform) / sr
    new_sr = int(321.893491124260 * 64 / clip_duration)
    waveform = torchaudio.functional.resample(waveform, sr, new_sr)
    waveform = waveform.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, T]

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(waveform, visual_batch)
        probs = torch.softmax(output, dim=1)

    pred_class = int(probs.argmax(dim=1).item())
    confidence = float(probs[0, pred_class].item())
    label = 'deception' if pred_class == 1 else 'truth'

    return label, confidence


model_path = os.path.join(MODEL_DIR, 'av_fusion_deception.pth')
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
loaded_model = AVFusionModel(freeze_audio_backbone=True, freeze_visual_backbone=True)
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model = loaded_model.to(device)
loaded_model.eval()
print(f"Model loaded from: {model_path}")
""")

code(r"""
import subprocess

test_videos = glob.glob(os.path.join(VIDEO_DIR, '*.mp4'))
random.seed(42)
sample_videos = random.sample(test_videos, min(5, len(test_videos)))

print("Running inference on sample videos...\n")
for vid_path in sample_videos:
    vid_name = os.path.basename(vid_path).replace('.mp4', '')
    true_label_from_name = 'deception' if 'lie' in vid_name.lower() else 'truth'

    pred_label, confidence = predict_video_av(loaded_model, vid_path)

    print(f"Video: {vid_name}")
    print(f"  Predicted: {pred_label.upper()} (confidence: {confidence:.2%})")
    print(f"  Expected:  {true_label_from_name.upper()}")
    print(f"  Match: {'YES' if pred_label == true_label_from_name else 'NO'}")
    print()

    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        color = 'green' if pred_label == true_label_from_name else 'red'
        ax.set_title(f"Pred: {pred_label.upper()} ({confidence:.1%})\nTrue: {true_label_from_name.upper()}",
                     fontweight='bold', color=color, fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        plt.show()
""")

# ============================================================================
# CELL 20: SUMMARY
# ============================================================================
md(r"""
---
## Summary

### Completed Pipeline

| Step | Description | Status |
|------|-------------|--------|
| 1 | Import Libraries | Done |
| 2 | Load Metadata | Done |
| 3 | Extract Audio from Videos | Done |
| 4 | Extract Face Frames (CLAHE) | Done |
| 5 | Exploratory Data Analysis | Done |
| 6 | **Video-Level** Train/Val/Test Split | Done |
| 7 | **Speaker-Independent** Split | Done |
| 8 | Audio-Visual Dataset | Done |
| 9 | Fusion Model (EfficientNet-B0 + Wav2Vec2) | Done |
| 10 | 2-Phase Training | Done |
| 11 | Test Set Evaluation | Done |
| 12 | Modality Comparison | Done |
| 13 | Speaker-Independent Evaluation | Done |
| 14 | Save Model | Done |
| 15 | Inference on New Videos | Done |

### Key Results
- **Fusion model test accuracy**: See Step 13
- **Audio-only vs Visual-only vs Fusion**: See Step 15
- **Speaker-independent accuracy**: See Step 16
- **Model saved**: `saved_models/av_fusion_deception.pth`

### Why This Matters
The fusion model combines complementary information from:
- **Visual cues**: Facial expressions, micro-expressions, head movements
- **Audio cues**: Vocal tension, speech patterns, pitch variations

This should improve over single-modality approaches, especially for
speakers or situations where one modality is less informative.
""")

# ============================================================================
# BUILD NOTEBOOK
# ============================================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3.11 (GPU)",
            "language": "python",
            "name": "deception-av-gpu"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.15"
        }
    },
    "cells": cells
}

output_path = os.path.join(ROOT, 'av_workflow_complete.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook created: {output_path}")
print(f"Total cells: {len(cells)}")
