import json
import os

cells = []
ROOT = r'C:\Users\OMEN\Desktop\DECEPTION DETECTION USING VIDEOS AND FACE'
OUT_DIR = os.path.join(ROOT, 'eda_output')
TSV = os.path.join(ROOT, 'DOLOSDATA', 'DOLOS', 'dolos_timestamps.csv')
PROTO_DIR = os.path.join(ROOT, 'DOLOSDATA', 'DOLOS', 'Training_Protocols')
VIDEO_DIR = os.path.join(ROOT, 'DOLOSDATA', 'raw_videos')

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
# Audio-Visual Deception Detection: Exploratory Data Analysis

**DOLOS Dataset & Parameter-Efficient Crossmodal Learning (PECL)**  
*ICCV 2023*

This notebook performs a comprehensive exploratory data analysis (EDA) of the **DOLOS** deception detection dataset. DOLOS is curated from the British TV gameshow *"Would I Lie to You?" (WILTY)* and contains audio-visual clips labelled as **truth** or **deception**.

### Dataset Overview
- **Source**: YouTube clips from the WILTY gameshow
- **Modalities**: Audio (speech) + Visual (face frames)
- **Labels**: Truth (0) / Deception (1)
- **Features**: MUMIN behavioural annotations, speaker metadata, gender
- **Paper**: [Audio-Visual Deception Detection: DOLOS Dataset and Parameter-Efficient Crossmodal Learning](https://arxiv.org/abs/2303.12745)

### Notebook Structure
Each visualisation is presented one at a time with a detailed explanation of:
- What the chart displays
- What each axis represents
- Key insights and observations
- Relevance to deception detection

---

""")

# ============================================================================
# CELL 2: IMPORTS AND SETUP
# ============================================================================
md(r"""
## Setup and Imports

This cell imports all necessary libraries for data manipulation, audio processing, computer vision, and visualisation.
""")

code(r"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import librosa
import cv2

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams.update({'figure.max_open_warning': 0, 'font.size': 10})
%matplotlib inline

ROOT = r'C:\Users\OMEN\Desktop\DECEPTION DETECTION USING VIDEOS AND FACE'
TSV = os.path.join(ROOT, 'DOLOSDATA', 'DOLOS', 'dolos_timestamps.csv')
PROTO_DIR = os.path.join(ROOT, 'DOLOSDATA', 'DOLOS', 'Training_Protocols')
VIDEO_DIR = os.path.join(ROOT, 'DOLOSDATA', 'raw_videos')

print("All imports successful.")
""")

# ============================================================================
# CELL 3: DATA LOADING - METADATA
# ============================================================================
md(r"""
## Data Loading: Metadata

We load the **`dolos_timestamps.csv`** file which contains:
- **file_name**: Unique clip identifier
- **label**: Ground truth (truth / deception / lie / true)
- **start_time / end_time**: Timestamps within the full episode
- **YT_Video_ID**: Source YouTube video
- **subject_name / subject_gender**: Speaker metadata

We perform initial cleaning: stripping whitespace, standardising labels to `truth` / `deception`, and computing clip duration in seconds.
""")

code(r"""
tsv = pd.read_csv(TSV)
tsv['file_name'] = tsv['file_name'].str.strip()

# Standardise labels
tsv['label_clean'] = tsv['label'].str.lower().str.strip()
tsv['label_clean'] = tsv['label_clean'].replace({'lie': 'deception', 'true': 'truth'})

# Convert timestamps to seconds
def mmss_to_seconds(t):
    if pd.isna(t):
        return np.nan
    t = str(t).strip()
    parts = list(map(float, t.split(':')))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 1:
        return parts[0] * 60
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return np.nan

tsv['start_sec'] = tsv['start_time'].apply(mmss_to_seconds)
tsv['end_sec'] = tsv['end_time'].apply(mmss_to_seconds)
tsv['duration_sec'] = tsv['end_sec'] - tsv['start_sec']

# Remove invalid durations
tsv = tsv[tsv['duration_sec'] > 0].copy()

# Extract speaker code from filename
tsv['speaker'] = tsv['file_name'].str.extract(r'^([A-Z]+)_', expand=False)
tsv['speaker'] = tsv['speaker'].str.upper()

print(f"Loaded {len(tsv)} clips from timestamps CSV.")
print(f"Label distribution (clean):\n{tsv['label_clean'].value_counts()}")
print(f"\nUnique speakers: {sorted(tsv['speaker'].dropna().unique())}")
""")

# ============================================================================
# CELL 4: GENDER LOADING
# ============================================================================
md(r"""
## Loading Gender Annotations

Gender information is not present in the main CSV. We extract it from the **Training Protocol CSV files** which contain triples: `(clip_name, label, gender)`. This gives us gender annotations for a subset of clips.
""")

code(r"""
gender_map = {}
for fname in os.listdir(PROTO_DIR):
    fpath = os.path.join(PROTO_DIR, fname)
    try:
        dfp = pd.read_csv(fpath, header=None)
    except Exception:
        continue
    if dfp.shape[1] >= 3:
        for _, row in dfp.iterrows():
            clip = str(row[0]).strip()
            gen = str(row[2]).strip().lower()
            if gen in ('male', 'female'):
                gender_map[clip] = gen

tsv['gender'] = tsv['file_name'].map(gender_map)
print(f"Gender annotations collected: {len(gender_map)} clips")
print(f"Coverage: {tsv['gender'].notna().sum()} / {len(tsv)} clips")
""")

# ============================================================================
# CELL 5: VIDEO FILE CHECK
# ============================================================================
md(r"""
## Video File Availability Check

We check which of the 1671 clips in the CSV have corresponding `.mp4` video files available on disk. This helps us understand the coverage of our local dataset.
""")

code(r"""
video_files = set(os.listdir(VIDEO_DIR))
tsv['video_exists'] = tsv['file_name'].apply(lambda x: f"{x}.mp4" in video_files)
print(f"Videos found on disk: {sum(tsv['video_exists'])} / {len(tsv)}")
""")

# ============================================================================
# CELL 6: LABEL DISTRIBUTION
# ============================================================================
md(r"""
---
## Visualisation 1: Label Distribution

**What it shows**: A bar chart of the clean label distribution — how many clips are labelled as **truth** vs **deception**.

**Axes**:
- **X-axis**: Label category (truth / deception)
- **Y-axis**: Count (number of clips)

**Key insights**:
- A balanced dataset is critical for unbiased model training.
- If one class dominates, the model may learn a trivial majority-class predictor.
- The percentages tell us the exact class balance.

""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 5))
label_counts = tsv['label_clean'].value_counts()
colors_label = ['#2ecc71', '#e74c3c']
bars = ax.bar(label_counts.index, label_counts.values, color=colors_label, edgecolor='black', linewidth=1.2)
for bar, val in zip(bars, label_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{val} ({val/len(tsv)*100:.1f}%)', ha='center', fontweight='bold', fontsize=12)
ax.set_title('Label Distribution (Clean)', fontweight='bold', fontsize=14)
ax.set_ylabel('Number of Clips', fontweight='bold')
ax.set_xlabel('Label', fontweight='bold')
sns.despine()
plt.tight_layout()
plt.show()
""")

# ============================================================================
# CELL 7: GENDER DISTRIBUTION
# ============================================================================
md(r"""
---
## Visualisation 2: Gender Distribution (Known)

**What it shows**: Distribution of speaker gender among clips where gender information is available.

**Axes**:
- **X-axis**: Gender (male / female)
- **Y-axis**: Count (number of clips)

**Key insights**:
- Gender imbalance can affect deception detection performance.
- Prior work has shown that deception cues may differ between genders.
- This informs whether we need gender-specific or gender-balanced evaluation protocols.

""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 5))
gender_valid = tsv['gender'].dropna()
gender_counts = gender_valid.value_counts()
colors_gen = ['#3498db', '#e91e63']
bars = ax.bar(gender_counts.index, gender_counts.values, color=colors_gen, edgecolor='black', linewidth=1.2)
for bar, val in zip(bars, gender_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{val} ({val/len(gender_valid)*100:.1f}%)', ha='center', fontweight='bold', fontsize=12)
ax.set_title('Gender Distribution (Known)', fontweight='bold', fontsize=14)
ax.set_ylabel('Number of Clips', fontweight='bold')
ax.set_xlabel('Gender', fontweight='bold')
sns.despine()
plt.tight_layout()
plt.show()
""")

# ============================================================================
# CELL 8: SPEAKER DISTRIBUTION
# ============================================================================
md(r"""
---
## Visualisation 3: Speaker Distribution

**What it shows**: Number of clips per unique speaker in the dataset.

**Axes**:
- **X-axis**: Speaker code (e.g., AN, BRI, LS, SB, YW)
- **Y-axis**: Count (number of clips for that speaker)

**Key insights**:
- Large variation in speaker representation can introduce speaker bias.
- Models may perform well on frequent speakers but poorly on rare ones.
- Speaker-based cross-validation may be necessary for fair evaluation.

""")

code(r"""
fig, ax = plt.subplots(figsize=(10, 5.5))
speaker_counts = tsv['speaker'].value_counts()
colors_sp = sns.color_palette('Set2', len(speaker_counts))
bars = ax.bar(speaker_counts.index, speaker_counts.values, color=colors_sp, edgecolor='black', linewidth=1.2)
for bar, val in zip(bars, speaker_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(val), ha='center', fontweight='bold', fontsize=9)
ax.set_title('Speaker Distribution', fontweight='bold', fontsize=14)
ax.set_ylabel('Number of Clips', fontweight='bold')
ax.set_xlabel('Speaker Code', fontweight='bold')
ax.tick_params(axis='x', rotation=45)
sns.despine()
plt.tight_layout()
plt.show()
""")

# ============================================================================
# CELL 9: CLIP DURATION DISTRIBUTION
# ============================================================================
md(r"""
---
## Visualisation 4: Clip Duration Distribution

**What it shows**: Histogram of clip durations (in seconds) across all samples.

**Axes**:
- **X-axis**: Duration (seconds)
- **Y-axis**: Count (number of clips)
- **Dashed lines**: Median (red) and Mean (green) duration

**Key insights**:
- Understanding duration helps set model input length constraints.
- Very short clips may lack sufficient content for reliable prediction.
- The distribution's skew informs data augmentation or truncation strategies.

""")

code(r"""
fig, ax = plt.subplots(figsize=(10, 5.5))
durations = tsv['duration_sec'].dropna()
ax.hist(durations, bins=40, color='#9b59b6', edgecolor='black', alpha=0.7, linewidth=1.2)
ax.axvline(durations.median(), color='red', ls='--', linewidth=2, label=f"Median: {durations.median():.1f}s")
ax.axvline(durations.mean(), color='green', ls='--', linewidth=2, label=f"Mean: {durations.mean():.1f}s")
ax.set_title('Clip Duration Distribution', fontweight='bold', fontsize=14)
ax.set_xlabel('Duration (seconds)', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.legend(fontsize=11)
sns.despine()
plt.tight_layout()
plt.show()
""")

# ============================================================================
# CELL 10: DURATION BY LABEL
# ============================================================================
md(r"""
---
## Visualisation 5: Duration by Label

**What it shows**: Box plot comparing the distribution of clip durations for truth vs deception clips.

**Axes**:
- **X-axis**: Label (truth / deception)
- **Y-axis**: Duration (seconds)

**Key insights**:
- If truth and deception clips have systematically different durations, the model could exploit duration as a confound.
- The box plot shows median, quartiles, and outliers for each class.

""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 5.5))
df_dur = tsv.dropna(subset=['duration_sec', 'label_clean'])
sns.boxplot(x='label_clean', y='duration_sec', data=df_dur, ax=ax,
            palette={'truth': '#2ecc71', 'deception': '#e74c3c'}, linewidth=1.5)
ax.set_title('Clip Duration by Label', fontweight='bold', fontsize=14)
ax.set_xlabel('Label', fontweight='bold')
ax.set_ylabel('Duration (seconds)', fontweight='bold')
sns.despine()
plt.tight_layout()
plt.show()
""")

# ============================================================================
# CELL 11: LABEL x GENDER
# ============================================================================
md(r"""
---
## Visualisation 6: Label x Gender Interaction

**What it shows**: Grouped bar chart showing the number of truth vs deception clips broken down by gender.

**Axes**:
- **X-axis**: Label (truth / deception)
- **Y-axis**: Count
- **Bars**: Coloured by gender (blue = male, pink = female)

**Key insights**:
- Reveals whether the label distribution is balanced within each gender group.
- Helps identify gender-specific biases in the dataset annotations.

""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 5.5))
gender_known = tsv['gender'].notna()
crosstab = pd.crosstab(tsv.loc[gender_known, 'label_clean'], tsv.loc[gender_known, 'gender'])
crosstab.plot(kind='bar', ax=ax, color=['#3498db', '#e91e63'], edgecolor='black', linewidth=1.2)
ax.set_title('Label Distribution by Gender', fontweight='bold', fontsize=14)
ax.set_xlabel('Label', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.legend(title='Gender', fontsize=11)
ax.tick_params(axis='x', rotation=0)
sns.despine()
plt.tight_layout()
plt.show()
""")

# ============================================================================
# CELL 12: RAW LABEL VALUES
# ============================================================================
md(r"""
---
## Visualisation 7: Raw Label Values in CSV

**What it shows**: Bar chart of the original (uncleaned) label values found in the CSV file.

**Axes**:
- **X-axis**: Raw label text (e.g., "lie", "deception", "true", "truth")
- **Y-axis**: Count of occurrences

**Key insights**:
- This reveals inconsistencies in the original annotation.
- Multiple synonymous labels (e.g., "lie" vs "deception", "true" vs "truth") need to be standardised.
- The count of unique raw labels tells us how messy the original data is.

""")

code(r"""
fig, ax = plt.subplots(figsize=(10, 5.5))
raw_label_counts = tsv['label'].value_counts()
colors_raw = sns.color_palette('husl', len(raw_label_counts))
bars = ax.bar(range(len(raw_label_counts)), raw_label_counts.values,
              color=colors_raw, edgecolor='black', linewidth=1.2)
ax.set_xticks(range(len(raw_label_counts)))
ax.set_xticklabels(raw_label_counts.index, rotation=45, ha='right', fontsize=10)
for bar, val in zip(bars, raw_label_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            str(val), ha='center', fontsize=10, fontweight='bold')
ax.set_title('Raw Label Values in CSV', fontweight='bold', fontsize=14)
ax.set_ylabel('Count', fontweight='bold')
sns.despine()
plt.tight_layout()
plt.show()
""")

# ============================================================================
# CELL 13: DUPLICATE FILE NAMES
# ============================================================================
md(r"""
---
## Visualisation 8: Duplicate file_name Entries

**What it shows**: Horizontal bar chart of `file_name` values that appear more than once in the dataset.

**Axes**:
- **Y-axis**: File name of the duplicate clip
- **X-axis**: Number of occurrences

**Key insights**:
- Duplicate entries can cause data leakage between train and test splits.
- They may indicate clips that were annotated multiple times or overlapping timestamps.
- These duplicates should be handled (deduplicated) before model training.

""")

code(r"""
fig, ax = plt.subplots(figsize=(10, max(4, len(tsv['file_name'].value_counts()[tsv['file_name'].value_counts() > 1]) * 0.3 + 2)))
dup_counts = tsv['file_name'].value_counts()
dups = dup_counts[dup_counts > 1]
ax.barh(range(len(dups)), dups.values, color='#e67e22', edgecolor='black', linewidth=1.2)
ax.set_yticks(range(len(dups)))
ax.set_yticklabels(dups.index, fontsize=8)
ax.set_title(f'Duplicate file_name Entries (n={len(dups)})', fontweight='bold', fontsize=14)
ax.set_xlabel('Count', fontweight='bold')
for i, v in enumerate(dups.values):
    ax.text(v + 0.1, i, str(v), va='center', fontsize=9, fontweight='bold')
sns.despine()
plt.tight_layout()
plt.show()
""")

# ============================================================================
# CELL 14: AUDIO FEATURES - SAMPLES
# ============================================================================
md(r"""
---
## Audio Feature Samples

In the next cells, we examine raw audio features extracted from sample video clips. For each clip, we generate:

1. **Waveform**: The raw audio amplitude over time
2. **Mel-Spectrogram**: Time-frequency representation using mel-scale filterbanks
3. **Pitch Contour**: Fundamental frequency (F0) tracked using PYIN algorithm

We compare one **truth** clip and one **deception** clip to look for acoustic differences.
""")

code(r"""
sample_files = tsv[tsv['video_exists']].drop_duplicates(subset='file_name')['file_name'].head(2).tolist()
sample_labels = []
for fname in sample_files:
    lbl = tsv.loc[tsv['file_name'] == fname, 'label_clean'].iloc[0]
    sample_labels.append(lbl)
    print(f"Sample: {fname}  |  Label: {lbl}")
""")

# ============================================================================
# CELL 15: WAVEFORM SAMPLE 1
# ============================================================================
md(r"""
---
### Visualisation 9: Waveform — Sample Clip 1

**What it shows**: The raw audio waveform (amplitude vs time) for a sample clip.

**Axes**:
- **X-axis**: Time (seconds)
- **Y-axis**: Amplitude (normalised audio signal)

**Interpretation**:
- The waveform shows the raw audio signal's amplitude variations over time.
- Speech waveforms typically show alternating loud (voiced) and quiet (unvoiced/ pause) segments.
- Differences in energy patterns between truth and deception may indicate vocal stress or arousal.
""")

code(r"""
fname = sample_files[0]
fpath = os.path.join(VIDEO_DIR, f"{fname}.mp4")
label = sample_labels[0]
y, sr = librosa.load(fpath, sr=16000, mono=True)
t = np.linspace(0, len(y)/sr, len(y))

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(t, y, color='#3498db', linewidth=0.5)
ax.set_title(f'Waveform — {fname} [{label}]', fontweight='bold', fontsize=13)
ax.set_xlabel('Time (s)', fontweight='bold')
ax.set_ylabel('Amplitude', fontweight='bold')
sns.despine()
plt.tight_layout()
plt.show()
""")

# ============================================================================
# CELL 16: MEL-SPECTROGRAM SAMPLE 1
# ============================================================================
md(r"""
---
### Visualisation 10: Mel-Spectrogram — Sample Clip 1

**What it shows**: A mel-scaled spectrogram showing how frequency content evolves over time.

**Axes**:
- **X-axis**: Time (seconds)
- **Y-axis**: Mel frequency bins (Hz, mel scale)
- **Colour**: Energy level in dB (brighter = higher energy)

**Interpretation**:
- The mel-spectrogram captures speech characteristics like formant structure and pitch harmonics.
- Darker regions indicate silence/pauses; brighter regions indicate voiced speech.
- Differences in spectral patterns between truth and deception may relate to vocal tension or articulation changes.
""")

code(r"""
fig, ax = plt.subplots(figsize=(14, 5))
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel',
                               fmax=8000, ax=ax)
ax.set_title(f'Mel-Spectrogram — {fname} [{label}]', fontweight='bold', fontsize=13)
ax.set_xlabel('Time (s)', fontweight='bold')
ax.set_ylabel('Mel Frequency (Hz)', fontweight='bold')
plt.colorbar(img, ax=ax, format='%+2.0f dB')
plt.tight_layout()
plt.show()
""")

# ============================================================================
# CELL 17: PITCH CONTOUR SAMPLE 1
# ============================================================================
md(r"""
---
### Visualisation 11: Pitch Contour — Sample Clip 1

**What it shows**: The fundamental frequency (F0) contour tracked over time using the PYIN algorithm.

**Axes**:
- **X-axis**: Time (seconds)
- **Y-axis**: Frequency (Hz)

**Interpretation**:
- Pitch (F0) is perceived as the "tone" of voice.
- Deception has been linked to changes in pitch range, mean pitch, and pitch variability.
- Gaps in the contour indicate unvoiced segments (e.g., consonants, silence).
""")

code(r"""
fig, ax = plt.subplots(figsize=(14, 4))
f0, voiced_flag, _ = librosa.pyin(y, fmin=80, fmax=400, sr=sr)
times = librosa.times_like(f0, sr=sr)
ax.plot(times, f0, color='#e67e22', linewidth=1.5)
ax.set_title(f'Pitch Contour — {fname} [{label}]', fontweight='bold', fontsize=13)
ax.set_xlabel('Time (s)', fontweight='bold')
ax.set_ylabel('Frequency (Hz)', fontweight='bold')
ax.set_ylim(50, 450)
sns.despine()
plt.tight_layout()
plt.show()
""")

# ============================================================================
# CELL 18-20: SAMPLE 2
# ============================================================================
md(r"""
---
### Visualisation 12: Waveform — Sample Clip 2
""")

code(r"""
fname2 = sample_files[1]
fpath2 = os.path.join(VIDEO_DIR, f"{fname2}.mp4")
label2 = sample_labels[1]
y2, sr2 = librosa.load(fpath2, sr=16000, mono=True)
t2 = np.linspace(0, len(y2)/sr2, len(y2))

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(t2, y2, color='#3498db', linewidth=0.5)
ax.set_title(f'Waveform — {fname2} [{label2}]', fontweight='bold', fontsize=13)
ax.set_xlabel('Time (s)', fontweight='bold')
ax.set_ylabel('Amplitude', fontweight='bold')
sns.despine()
plt.tight_layout()
plt.show()
""")

md(r"""
---
### Visualisation 13: Mel-Spectrogram — Sample Clip 2
""")

code(r"""
fig, ax = plt.subplots(figsize=(14, 5))
S2 = librosa.feature.melspectrogram(y=y2, sr=sr2, n_mels=128, fmax=8000)
S2_dB = librosa.power_to_db(S2, ref=np.max)
img2 = librosa.display.specshow(S2_dB, sr=sr2, x_axis='time', y_axis='mel',
                                fmax=8000, ax=ax)
ax.set_title(f'Mel-Spectrogram — {fname2} [{label2}]', fontweight='bold', fontsize=13)
ax.set_xlabel('Time (s)', fontweight='bold')
ax.set_ylabel('Mel Frequency (Hz)', fontweight='bold')
plt.colorbar(img2, ax=ax, format='%+2.0f dB')
plt.tight_layout()
plt.show()
""")

md(r"""
---
### Visualisation 14: Pitch Contour — Sample Clip 2
""")

code(r"""
fig, ax = plt.subplots(figsize=(14, 4))
f0_2, _, _ = librosa.pyin(y2, fmin=80, fmax=400, sr=sr2)
times2 = librosa.times_like(f0_2, sr=sr2)
ax.plot(times2, f0_2, color='#e67e22', linewidth=1.5)
ax.set_title(f'Pitch Contour — {fname2} [{label2}]', fontweight='bold', fontsize=13)
ax.set_xlabel('Time (s)', fontweight='bold')
ax.set_ylabel('Frequency (Hz)', fontweight='bold')
ax.set_ylim(50, 450)
sns.despine()
plt.tight_layout()
plt.show()
""")

# ============================================================================
# CELL 21-22: FACE DETECTION
# ============================================================================
md(r"""
---
## Face Detection Sample

We apply a Haar cascade classifier to detect faces in a sample video frame. This demonstrates the visual pre-processing pipeline used in the paper: face detection with MTCNN (here approximated with OpenCV's Haar cascade for illustration).
""")

code(r"""
avail = tsv[tsv['video_exists']].drop_duplicates(subset='file_name')
sample_vid = avail['file_name'].iloc[0] if len(avail) > 0 else None
print(f"Sample video for face detection: {sample_vid}")
""")

md(r"""
---
### Visualisation 15: Raw Video Frame

**What it shows**: A randomly selected frame from a sample video clip.

**Interpretation**:
- The raw frame contains the full scene, including background, other people, and the speaker.
- For deception detection, we focus on facial cues — hence the need for face detection and cropping.
""")

code(r"""
fpath_vid = os.path.join(VIDEO_DIR, f"{sample_vid}.mp4")
cap = cv2.VideoCapture(fpath_vid)
ret, frame = cap.read()
cap.release()

fig, ax = plt.subplots(figsize=(8, 6))
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
ax.imshow(frame_rgb)
ax.set_title(f'Raw Frame — {sample_vid}', fontweight='bold', fontsize=13)
ax.axis('off')
plt.tight_layout()
plt.show()
""")

md(r"""
---
### Visualisation 16: Face Detection Result

**What it shows**: The same frame with bounding boxes drawn around detected faces using OpenCV's Haar cascade classifier.

**Interpretation**:
- Successful face detection is the first step in the visual pipeline.
- The paper uses MTCNN for more accurate detection, but Haar cascade provides a quick visual check.
- Multiple face detections may indicate the presence of other panelists — the model must focus on the speaking subject.
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 6))
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)
frame_disp = frame_rgb.copy()
if len(faces) > 0:
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_disp, (x, y), (x + w, y + h), (0, 255, 0), 3)
    ax.set_title(f'Face Detected (Haar: {len(faces)} faces)', fontweight='bold', fontsize=13)
else:
    ax.set_title('No Face Detected (Haar)', fontweight='bold', fontsize=13)
ax.imshow(frame_disp)
ax.axis('off')
plt.tight_layout()
plt.show()
""")

# ============================================================================
# CELL 23-24: CORRELATION
# ============================================================================
md(r"""
---
## Feature Correlation & Outlier Analysis

We analyse relationships between metadata features:
- **Duration** (seconds)
- **Label** (truth=0, deception=1)
- **Gender** (male=0, female=1)
- **Speaker** (encoded as integers)

A correlation heatmap reveals linear relationships between these variables.
""")

code(r"""
df_feat = tsv.drop_duplicates(subset='file_name')[['file_name', 'duration_sec', 'label_clean', 'speaker', 'gender']].copy()
df_feat['label_num'] = (df_feat['label_clean'] == 'deception').astype(int)
df_feat['gender_num'] = (df_feat['gender'] == 'male').astype(int)
spk_map = {s: i for i, s in enumerate(df_feat['speaker'].unique())}
df_feat['speaker_num'] = df_feat['speaker'].map(spk_map)
""")

md(r"""
---
### Visualisation 17: Correlation Heatmap

**What it shows**: A heatmap of Pearson correlation coefficients between metadata features.

**Axes**:
- Both axes: Feature names (duration, label, gender, speaker)
- Cell values: Correlation coefficient (-1 to 1)
- Colour: Red = positive correlation, Blue = negative correlation

**Interpretation**:
- Values near +1 indicate strong positive linear relationship.
- Values near -1 indicate strong negative linear relationship.
- Values near 0 indicate no linear relationship.
- If any feature strongly correlates with the label, the model might learn a spurious shortcut.
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 6.5))
corr_cols = ['duration_sec', 'label_num', 'gender_num', 'speaker_num']
corr_data = df_feat[corr_cols].dropna()
corr = corr_data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, ax=ax, mask=mask, linewidths=1,
            vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Heatmap — Metadata Features', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()
""")

md(r"""
---
### Visualisation 18: Duration Outliers by Speaker

**What it shows**: Box plot of clip durations grouped by speaker, highlighting outliers.

**Axes**:
- **X-axis**: Speaker code
- **Y-axis**: Duration (seconds)
- Points beyond whiskers: Statistical outliers

**Interpretation**:
- Some speakers may have unusually long or short clips.
- Outliers can skew training — they may need to be clipped or removed.
- Speaker-specific duration patterns may reflect editing differences in the TV show.
""")

code(r"""
fig, ax = plt.subplots(figsize=(10, 5.5))
df_spk = df_feat.dropna(subset=['duration_sec'])
sns.boxplot(x='speaker', y='duration_sec', data=df_spk, ax=ax, palette='Set2', linewidth=1.5)
ax.set_title('Duration Outliers by Speaker', fontweight='bold', fontsize=14)
ax.set_xlabel('Speaker', fontweight='bold')
ax.set_ylabel('Duration (seconds)', fontweight='bold')
ax.tick_params(axis='x', rotation=45)
sns.despine()
plt.tight_layout()
plt.show()
""")

# ============================================================================
# CELL 25: LABEL x SPEAKER
# ============================================================================
md(r"""
---
## Visualisation 19: Label Proportion by Speaker

**What it shows**: Stacked horizontal bar chart showing the proportion of truth vs deception clips for each speaker.

**Axes**:
- **Y-axis**: Speaker code
- **X-axis**: Proportion (0 to 1)
- **Annotation**: `n=` shows total clips per speaker

**Interpretation**:
- Reveals whether certain speakers are more likely to be labelled as deceptive.
- If some speakers appear predominantly in one class, the model might learn speaker identity rather than deception cues.
- This is critical for designing speaker-independent evaluation protocols.
""")

code(r"""
fig, ax = plt.subplots(figsize=(10, 6))
crosstab_spk = pd.crosstab(tsv['speaker'], tsv['label_clean'])
crosstab_spk_pct = crosstab_spk.div(crosstab_spk.sum(1), axis=0)
crosstab_spk_pct.plot(kind='barh', stacked=True, ax=ax,
                       color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=1.2)
for i, spk in enumerate(crosstab_spk.index):
    total = crosstab_spk.sum(1)[spk]
    ax.text(0.5, i, f' n={total}', va='center', fontweight='bold', fontsize=10)
ax.set_title('Label Proportion by Speaker', fontweight='bold', fontsize=14)
ax.set_xlabel('Proportion', fontweight='bold')
ax.set_ylabel('Speaker', fontweight='bold')
ax.legend(title='Label', bbox_to_anchor=(1.05, 1), fontsize=10)
plt.tight_layout()
plt.show()
""")

# ============================================================================
# CELL 26: ACOUSTIC PROFILE TABLE
# ============================================================================
md(r"""
---
## Visualisation 20: Acoustic Profile of Sample Clips

**What it shows**: A table of acoustic features extracted from a sample of 6 video clips, comparing truth vs deception.

**Features extracted**:
- **pitch_mean_hz**: Average fundamental frequency (perceived pitch)
- **rms_mean**: Root-mean-square energy (loudness)
- **spectral_centroid_hz**: "Brightness" of the sound (frequency balance)
- **zcr**: Zero-crossing rate (noisiness / fricative content)

**Interpretation**:
- These acoustic parameters are commonly used in deception detection literature.
- Changes in pitch (higher or more variable) may indicate cognitive load or stress.
- Spectral centroid shifts may reflect changes in vocal tract tension.
- This table gives a quick glimpse of potential acoustic differences between classes.
""")

code(r"""
sample_set = tsv[tsv['video_exists']].drop_duplicates(subset='file_name').head(6)
ac_rows = []
for _, row in sample_set.iterrows():
    fpath = os.path.join(VIDEO_DIR, f"{row['file_name']}.mp4")
    try:
        y_a, sr_a = librosa.load(fpath, sr=16000, mono=True, duration=5.0)
        if len(y_a) < sr_a:
            continue
        rms = librosa.feature.rms(y=y_a)[0]
        cent = librosa.feature.spectral_centroid(y=y_a, sr=sr_a)[0]
        zcr = librosa.feature.zero_crossing_rate(y_a)[0]
        f0_a, voiced_a, _ = librosa.pyin(y_a, fmin=80, fmax=400, sr=sr_a)
        f0_clean = f0_a[~np.isnan(f0_a)]
        ac_rows.append({
            'file_name': row['file_name'],
            'label': row['label_clean'],
            'duration_s': f"{row['duration_sec']:.1f}",
            'pitch_mean_hz': f"{np.mean(f0_clean):.0f}" if len(f0_clean) > 0 else 'NA',
            'rms_mean': f"{np.mean(rms):.4f}",
            'spectral_centroid_hz': f"{np.mean(cent):.0f}",
            'zcr': f"{np.mean(zcr):.4f}",
        })
    except Exception:
        continue

ac_df_table = pd.DataFrame(ac_rows)
print("Acoustic Profile of Sample Clips:")
display(ac_df_table.style.hide(axis='index'))
""")

# ============================================================================
# CELL 27: OUTLIER REMOVAL
# ============================================================================
md(r"""
---
## Visualisation 21: Duration Outlier Removal (IQR Method)

**What it shows**: A before/after comparison of clip duration histograms with statistical outliers removed using the **Interquartile Range (IQR)** method.

**How IQR works**:
- Q1 = 25th percentile, Q3 = 75th percentile
- IQR = Q3 - Q1
- Lower bound = Q1 - 1.5 × IQR
- Upper bound = Q3 + 1.5 × IQR
- Values outside these bounds are considered outliers

**Axes**:
- Left plot (Before): All data with IQR bounds shown as vertical lines
- Right plot (After): Cleaned data with new mean and median

**Interpretation**:
- Outliers can distort statistical summaries and model training.
- Removing them reveals the underlying distribution more clearly.
- We report how many clips are removed and what the new bounds are.
""")

code(r"""
dur = tsv['duration_sec'].dropna()
Q1 = dur.quantile(0.25)
Q3 = dur.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
dur_clean = dur[(dur >= lower) & (dur <= upper)]

print(f"Duration IQR: {IQR:.2f}s")
print(f"Lower bound: {lower:.2f}s, Upper bound: {upper:.2f}s")
print(f"Outliers removed: {len(dur) - len(dur_clean)} / {len(dur)} ({(len(dur)-len(dur_clean))/len(dur)*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Duration Outlier Removal (IQR Method)', fontsize=14, fontweight='bold')

axes[0].hist(dur, bins=40, color='#e74c3c', edgecolor='black', alpha=0.7, linewidth=1.2)
axes[0].axvline(lower, color='blue', ls='--', linewidth=2, label=f'Lower: {lower:.1f}s')
axes[0].axvline(upper, color='blue', ls='--', linewidth=2, label=f'Upper: {upper:.1f}s')
axes[0].set_title(f'Before (n={len(dur)})', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Duration (seconds)', fontweight='bold')
axes[0].set_ylabel('Count', fontweight='bold')
axes[0].legend(fontsize=10)

axes[1].hist(dur_clean, bins=40, color='#2ecc71', edgecolor='black', alpha=0.7, linewidth=1.2)
axes[1].axvline(dur_clean.median(), color='red', ls='--', linewidth=2, label=f"Median: {dur_clean.median():.1f}s")
axes[1].axvline(dur_clean.mean(), color='green', ls='--', linewidth=2, label=f"Mean: {dur_clean.mean():.1f}s")
axes[1].set_title(f'After IQR Removal (n={len(dur_clean)})', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Duration (seconds)', fontweight='bold')
axes[1].set_ylabel('Count', fontweight='bold')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.show()
""")

# ============================================================================
# CELL 28: COMPUTE EXTENDED ACOUSTIC FEATURES
# ============================================================================
md(r"""
---
## Extended Acoustic Feature Analysis

We now compute acoustic features for a larger sample of **100 clips** to enable statistical comparison between truth and deception classes. The features computed are:

1. **pitch_mean_hz**: Average fundamental frequency — indicator of vocal pitch
2. **rms_mean**: Root-mean-square energy — correlates with loudness
3. **spectral_centroid_hz**: Centre of mass of the spectrum — indicates "brightness"
4. **zcr**: Zero-crossing rate — correlates with noisiness / frication

These features are used in the following visualisations: box plots, scatter plots, pairplots, and regression analysis.
""")

code(r"""
sample_set = tsv[tsv['video_exists']].drop_duplicates(subset='file_name').head(100).copy()
ac_rows = []
for _, row in sample_set.iterrows():
    fpath = os.path.join(VIDEO_DIR, f"{row['file_name']}.mp4")
    try:
        y_e, sr_e = librosa.load(fpath, sr=16000, mono=True, duration=5.0)
        if len(y_e) < sr_e:
            continue
        rms = librosa.feature.rms(y=y_e)[0]
        cent = librosa.feature.spectral_centroid(y=y_e, sr=sr_e)[0]
        zcr = librosa.feature.zero_crossing_rate(y_e)[0]
        f0_e, _, _ = librosa.pyin(y_e, fmin=80, fmax=400, sr=sr_e)
        f0_clean = f0_e[~np.isnan(f0_e)]
        ac_rows.append({
            'file_name': row['file_name'],
            'label': row['label_clean'],
            'duration_s': row['duration_sec'],
            'pitch_mean_hz': np.mean(f0_clean) if len(f0_clean) > 0 else np.nan,
            'rms_mean': np.mean(rms),
            'spectral_centroid_hz': np.mean(cent),
            'zcr': np.mean(zcr),
        })
    except Exception:
        continue

ac_df = pd.DataFrame(ac_rows).dropna()
print(f"Acoustic features computed for {len(ac_df)} clips")
print(f"\nLabel distribution in sample:\n{ac_df['label'].value_counts()}")
""")

# ============================================================================
# CELL 29-32: BOX PLOTS
# ============================================================================
md(r"""
---
### Visualisation 22: Pitch Distribution by Label (Box Plot)

**What it shows**: Box plot comparing the distribution of average pitch (fundamental frequency) between truth and deception clips.

**Axes**:
- **X-axis**: Label (truth / deception)
- **Y-axis**: Pitch (Hz)

**Interpretation**:
- Pitch is one of the most studied acoustic correlates of deception.
- Higher pitch may indicate stress, cognitive load, or emotional arousal.
- The box plot shows median (center line), IQR (box), and outliers (whiskers + points).
- Overlap between the two distributions suggests pitch alone is not a perfect discriminator.
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 5.5))
sns.boxplot(x='label', y='pitch_mean_hz', data=ac_df, ax=ax,
            palette={'truth': '#2ecc71', 'deception': '#e74c3c'}, linewidth=1.5)
ax.set_title('Pitch Distribution by Label', fontweight='bold', fontsize=14)
ax.set_xlabel('Label', fontweight='bold')
ax.set_ylabel('Pitch (Hz)', fontweight='bold')
sns.despine()
plt.tight_layout()
plt.show()
""")

md(r"""
---
### Visualisation 23: RMS Energy Distribution by Label (Box Plot)

**What it shows**: Box plot comparing RMS energy (loudness) between truth and deception clips.

**Axes**:
- **X-axis**: Label (truth / deception)
- **Y-axis**: RMS Energy (amplitude envelope)

**Interpretation**:
- RMS energy captures overall loudness/energy of the speech signal.
- Changes in vocal energy may reflect confidence, nervousness, or emphasis.
- Deceptive speech may exhibit different energy patterns than truthful speech.
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 5.5))
sns.boxplot(x='label', y='rms_mean', data=ac_df, ax=ax,
            palette={'truth': '#2ecc71', 'deception': '#e74c3c'}, linewidth=1.5)
ax.set_title('RMS Energy Distribution by Label', fontweight='bold', fontsize=14)
ax.set_xlabel('Label', fontweight='bold')
ax.set_ylabel('RMS Energy', fontweight='bold')
sns.despine()
plt.tight_layout()
plt.show()
""")

md(r"""
---
### Visualisation 24: Spectral Centroid Distribution by Label (Box Plot)

**What it shows**: Box plot comparing spectral centroid (frequency "brightness") between truth and deception clips.

**Axes**:
- **X-axis**: Label (truth / deception)
- **Y-axis**: Spectral Centroid (Hz)

**Interpretation**:
- Spectral centroid indicates where the "centre of mass" of the spectrum lies.
- Higher centroid = brighter, more high-frequency content.
- Changes may reflect vocal tension or articulatory precision under cognitive load.
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 5.5))
sns.boxplot(x='label', y='spectral_centroid_hz', data=ac_df, ax=ax,
            palette={'truth': '#2ecc71', 'deception': '#e74c3c'}, linewidth=1.5)
ax.set_title('Spectral Centroid Distribution by Label', fontweight='bold', fontsize=14)
ax.set_xlabel('Label', fontweight='bold')
ax.set_ylabel('Spectral Centroid (Hz)', fontweight='bold')
sns.despine()
plt.tight_layout()
plt.show()
""")

md(r"""
---
### Visualisation 25: Zero-Crossing Rate Distribution by Label (Box Plot)

**What it shows**: Box plot comparing zero-crossing rate (signal noisiness) between truth and deception clips.

**Axes**:
- **X-axis**: Label (truth / deception)
- **Y-axis**: Zero-Crossing Rate

**Interpretation**:
- ZCR measures how often the audio signal crosses zero amplitude.
- Higher ZCR indicates more high-frequency noise or fricative content.
- May correlate with vocal tension, articulation changes, or background noise during deception.
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 5.5))
sns.boxplot(x='label', y='zcr', data=ac_df, ax=ax,
            palette={'truth': '#2ecc71', 'deception': '#e74c3c'}, linewidth=1.5)
ax.set_title('Zero-Crossing Rate Distribution by Label', fontweight='bold', fontsize=14)
ax.set_xlabel('Label', fontweight='bold')
ax.set_ylabel('Zero-Crossing Rate', fontweight='bold')
sns.despine()
plt.tight_layout()
plt.show()
""")

# ============================================================================
# CELL 33-36: SCATTER PLOTS
# ============================================================================
md(r"""
---
## Scatter Plots: Acoustic Feature Relationships

The following scatter plots visualise pairwise relationships between acoustic features, with points coloured by label (green = truth, red = deception). These reveal:
- Whether acoustic features are correlated with each other
- Whether truth and deception samples cluster separately in feature space
- Potential interactions between features that may aid classification
""")

md(r"""
---
### Visualisation 26: Pitch vs RMS Energy

**What it shows**: Scatter plot of pitch (Hz) against RMS energy, coloured by label.

**Axes**:
- **X-axis**: Pitch (Hz)
- **Y-axis**: RMS Energy

**Interpretation**:
- This reveals the relationship between vocal pitch and loudness.
- If truth and deception occupy different regions of this 2D space, these features together may be discriminative.
- Outliers may indicate unusual vocal patterns worth investigating.
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 6))
for label in ['truth', 'deception']:
    sub = ac_df[ac_df['label'] == label]
    c = '#2ecc71' if label == 'truth' else '#e74c3c'
    ax.scatter(sub['pitch_mean_hz'], sub['rms_mean'],
               alpha=0.6, c=c, label=label, edgecolors='black', linewidth=0.5, s=50)
ax.set_xlabel('Pitch (Hz)', fontweight='bold')
ax.set_ylabel('RMS Energy', fontweight='bold')
ax.set_title('Pitch vs RMS Energy by Label', fontweight='bold', fontsize=13)
ax.legend(fontsize=11)
sns.despine()
plt.tight_layout()
plt.show()
""")

md(r"""
---
### Visualisation 27: Spectral Centroid vs Zero-Crossing Rate

**What it shows**: Scatter plot of spectral centroid against zero-crossing rate, coloured by label.

**Axes**:
- **X-axis**: Spectral Centroid (Hz)
- **Y-axis**: Zero-Crossing Rate

**Interpretation**:
- Both features capture aspects of high-frequency content.
- Strong correlation is expected (brighter sounds have more zero crossings).
- Differences between classes may reflect articulation or vocal tension changes.
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 6))
for label in ['truth', 'deception']:
    sub = ac_df[ac_df['label'] == label]
    c = '#2ecc71' if label == 'truth' else '#e74c3c'
    ax.scatter(sub['spectral_centroid_hz'], sub['zcr'],
               alpha=0.6, c=c, label=label, edgecolors='black', linewidth=0.5, s=50)
ax.set_xlabel('Spectral Centroid (Hz)', fontweight='bold')
ax.set_ylabel('Zero-Crossing Rate', fontweight='bold')
ax.set_title('Spectral Centroid vs ZCR by Label', fontweight='bold', fontsize=13)
ax.legend(fontsize=11)
sns.despine()
plt.tight_layout()
plt.show()
""")

md(r"""
---
### Visualisation 28: Duration vs Pitch

**What it shows**: Scatter plot of clip duration against pitch, coloured by label.

**Axes**:
- **X-axis**: Duration (seconds)
- **Y-axis**: Pitch (Hz)

**Interpretation**:
- Examines whether clip duration correlates with pitch features.
- Longer clips may have more pitch variation, potentially affecting average pitch estimates.
- This helps determine if duration should be controlled for in acoustic analysis.
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 6))
for label in ['truth', 'deception']:
    sub = ac_df[ac_df['label'] == label]
    c = '#2ecc71' if label == 'truth' else '#e74c3c'
    ax.scatter(sub['duration_s'], sub['pitch_mean_hz'],
               alpha=0.6, c=c, label=label, edgecolors='black', linewidth=0.5, s=50)
ax.set_xlabel('Duration (s)', fontweight='bold')
ax.set_ylabel('Pitch (Hz)', fontweight='bold')
ax.set_title('Duration vs Pitch by Label', fontweight='bold', fontsize=13)
ax.legend(fontsize=11)
sns.despine()
plt.tight_layout()
plt.show()
""")

md(r"""
---
### Visualisation 29: Duration vs RMS Energy

**What it shows**: Scatter plot of clip duration against RMS energy, coloured by label.

**Axes**:
- **X-axis**: Duration (seconds)
- **Y-axis**: RMS Energy

**Interpretation**:
- Examines whether longer clips systematically differ in energy from shorter ones.
- Short clips with high energy may represent emphatic or emotional statements — potentially relevant for deception detection.
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 6))
for label in ['truth', 'deception']:
    sub = ac_df[ac_df['label'] == label]
    c = '#2ecc71' if label == 'truth' else '#e74c3c'
    ax.scatter(sub['duration_s'], sub['rms_mean'],
               alpha=0.6, c=c, label=label, edgecolors='black', linewidth=0.5, s=50)
ax.set_xlabel('Duration (s)', fontweight='bold')
ax.set_ylabel('RMS Energy', fontweight='bold')
ax.set_title('Duration vs RMS Energy by Label', fontweight='bold', fontsize=13)
ax.legend(fontsize=11)
sns.despine()
plt.tight_layout()
plt.show()
""")

# ============================================================================
# CELL 37: PAIRPLOT ALL DATA
# ============================================================================
md(r"""
---
### Visualisation 30: Acoustic Feature Pairplot (All Data)

**What it shows**: A seaborn pairplot (scatter matrix) showing all pairwise relationships between 5 acoustic features: pitch, RMS energy, spectral centroid, ZCR, and duration. Points are coloured by label.

**Axes**:
- Off-diagonal: Scatter plots of pairwise feature combinations
- Diagonal: Kernel density estimate (KDE) of each feature's distribution

**Interpretation**:
- The pairplot provides a comprehensive view of the feature space.
- Look for feature pairs where truth (green) and deception (red) separate clearly — these are promising for classification.
- The diagonal shows the univariate distribution of each feature split by class.
- Correlations between features are also visible (e.g., centroid and ZCR should correlate).
""")

code(r"""
pair_cols = ['pitch_mean_hz', 'rms_mean', 'spectral_centroid_hz', 'zcr', 'duration_s', 'label']
pp = sns.pairplot(ac_df[pair_cols], hue='label',
                   palette={'truth': '#2ecc71', 'deception': '#e74c3c'},
                   diag_kind='kde',
                   plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'black', 'linewidth': 0.5})
pp.fig.suptitle('Acoustic Feature Pairplot (All Data)', fontsize=14, fontweight='bold', y=1.02)
plt.show()
""")

# ============================================================================
# CELL 38: PAIRPLOT CLEAN
# ============================================================================
md(r"""
---
### Visualisation 31: Acoustic Feature Pairplot (Outliers Removed)

**What it shows**: The same pairplot but with per-feature IQR outliers removed. Each feature's outliers are identified independently using the 1.5×IQR rule, and samples outside any feature's bounds are excluded.

**Purpose**:
- Outliers can dominate visualisations and distort correlations.
- This cleaner version reveals the core structure of the data without extreme values.
- Compare with the previous pairplot to see which patterns are robust and which are driven by outliers.
""")

code(r"""
ac_clean = ac_df.copy()
for col in ['pitch_mean_hz', 'rms_mean', 'spectral_centroid_hz', 'zcr', 'duration_s']:
    cQ1 = ac_clean[col].quantile(0.25)
    cQ3 = ac_clean[col].quantile(0.75)
    cIQR = cQ3 - cQ1
    c_low = cQ1 - 1.5 * cIQR
    c_high = cQ3 + 1.5 * cIQR
    ac_clean = ac_clean[(ac_clean[col] >= c_low) & (ac_clean[col] <= c_high)]

print(f"Clips after per-feature IQR removal: {len(ac_clean)} (from {len(ac_df)})")

pp2 = sns.pairplot(ac_clean[pair_cols], hue='label',
                    palette={'truth': '#2ecc71', 'deception': '#e74c3c'},
                    diag_kind='kde',
                    plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'black', 'linewidth': 0.5})
pp2.fig.suptitle('Acoustic Feature Pairplot (Outliers Removed)', fontsize=14, fontweight='bold', y=1.02)
plt.show()
""")

# ============================================================================
# CELL 39-42: REGRESSION LINES
# ============================================================================
md(r"""
---
## Regression Analysis

The following scatter plots include **linear regression lines** (with 95% confidence intervals) to reveal trends between feature pairs. The regression line shows the best-fit linear relationship, and the shaded region represents the uncertainty of the fit.

A non-horizontal line with a narrow confidence band indicates a statistically meaningful relationship between the two features.
""")

md(r"""
---
### Visualisation 32: Duration vs Pitch with Regression Line

**What it shows**: Scatter plot of duration vs pitch with an overlaid linear regression line and confidence interval.

**Axes**:
- **X-axis**: Duration (seconds)
- **Y-axis**: Pitch (Hz)
- **Red line**: Linear regression fit

**Interpretation**:
- The slope of the regression line indicates how pitch changes with clip duration.
- If the confidence band is narrow, the trend is statistically reliable.
- A flat line means no linear relationship between the two variables.
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 6))
sns.regplot(x='duration_s', y='pitch_mean_hz', data=ac_df, ax=ax,
            scatter_kws={'alpha': 0.5, 'edgecolor': 'black', 'linewidths': 0.5},
            line_kws={'color': 'red', 'linewidth': 2})
ax.set_xlabel('Duration (s)', fontweight='bold')
ax.set_ylabel('Pitch (Hz)', fontweight='bold')
ax.set_title('Duration vs Pitch with Linear Fit', fontweight='bold', fontsize=13)
sns.despine()
plt.tight_layout()
plt.show()
""")

md(r"""
---
### Visualisation 33: Duration vs RMS Energy with Regression Line

**What it shows**: Scatter plot of duration vs RMS energy with linear regression.

**Axes**:
- **X-axis**: Duration (seconds)
- **Y-axis**: RMS Energy

**Interpretation**:
- Does vocal energy change systematically with clip duration?
- Longer statements may show different energy patterns (e.g., trailing off, or more emphatic delivery).
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 6))
sns.regplot(x='duration_s', y='rms_mean', data=ac_df, ax=ax,
            scatter_kws={'alpha': 0.5, 'edgecolor': 'black', 'linewidths': 0.5},
            line_kws={'color': 'red', 'linewidth': 2})
ax.set_xlabel('Duration (s)', fontweight='bold')
ax.set_ylabel('RMS Energy', fontweight='bold')
ax.set_title('Duration vs RMS Energy with Linear Fit', fontweight='bold', fontsize=13)
sns.despine()
plt.tight_layout()
plt.show()
""")

md(r"""
---
### Visualisation 34: Pitch vs Spectral Centroid with Regression Line

**What it shows**: Scatter plot of pitch vs spectral centroid with linear regression.

**Axes**:
- **X-axis**: Pitch (Hz)
- **Y-axis**: Spectral Centroid (Hz)

**Interpretation**:
- Pitch and spectral centroid both involve frequency but capture different aspects.
- Pitch = fundamental frequency (perceived tone). Centroid = "centre of mass" of the full spectrum.
- A strong positive correlation would indicate that higher-pitched voices also have brighter timbre.
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 6))
sns.regplot(x='pitch_mean_hz', y='spectral_centroid_hz', data=ac_df, ax=ax,
            scatter_kws={'alpha': 0.5, 'edgecolor': 'black', 'linewidths': 0.5},
            line_kws={'color': 'red', 'linewidth': 2})
ax.set_xlabel('Pitch (Hz)', fontweight='bold')
ax.set_ylabel('Spectral Centroid (Hz)', fontweight='bold')
ax.set_title('Pitch vs Spectral Centroid with Linear Fit', fontweight='bold', fontsize=13)
sns.despine()
plt.tight_layout()
plt.show()
""")

md(r"""
---
### Visualisation 35: RMS Energy vs ZCR with Regression Line

**What it shows**: Scatter plot of RMS energy vs zero-crossing rate with linear regression.

**Axes**:
- **X-axis**: RMS Energy
- **Y-axis**: Zero-Crossing Rate

**Interpretation**:
- Both features capture aspects of the signal's character: energy level and noisiness.
- Louder speech may have different ZCR characteristics.
- The regression trend quantifies this relationship.
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 6))
sns.regplot(x='rms_mean', y='zcr', data=ac_df, ax=ax,
            scatter_kws={'alpha': 0.5, 'edgecolor': 'black', 'linewidths': 0.5},
            line_kws={'color': 'red', 'linewidth': 2})
ax.set_xlabel('RMS Energy', fontweight='bold')
ax.set_ylabel('Zero-Crossing Rate', fontweight='bold')
ax.set_title('RMS Energy vs ZCR with Linear Fit', fontweight='bold', fontsize=13)
sns.despine()
plt.tight_layout()
plt.show()
""")

# ============================================================================
# CELL 43: UNIVARIATE STATISTICS
# ============================================================================
md(r"""
---
## Visualisation 36: Complete Univariate Statistics

**What it shows**: A comprehensive statistical summary of all 5 acoustic features across the entire sample.

**Metrics reported**:
- **Count**: Number of samples
- **Mean**: Arithmetic average
- **Std**: Standard deviation (spread)
- **Min / Max**: Range
- **25%, 50%, 75%**: Quartiles (25th percentile, median, 75th percentile)

**Interpretation**:
- The mean and median together indicate skew — if they differ substantially, the distribution is asymmetric.
- Standard deviation tells us how much variation exists within each feature.
- The range (min-max) and IQR (Q3-Q1) help identify potential outliers and data quality issues.
- These statistics inform feature normalisation strategies for model training.
""")

code(r"""
uni_cols = ['duration_s', 'pitch_mean_hz', 'rms_mean', 'spectral_centroid_hz', 'zcr']
uni_rows = []
for col in uni_cols:
    s = ac_df[col].dropna()
    uni_rows.append({
        'Feature': col,
        'Count': len(s),
        'Mean': f"{s.mean():.4f}",
        'Std': f"{s.std():.4f}",
        'Min': f"{s.min():.4f}",
        '25%': f"{s.quantile(0.25):.4f}",
        '50%': f"{s.quantile(0.50):.4f}",
        '75%': f"{s.quantile(0.75):.4f}",
        'Max': f"{s.max():.4f}",
    })
uni_df = pd.DataFrame(uni_rows)
display(uni_df.style.hide(axis='index'))
""")

# ============================================================================
# CELL 44: SUMMARY
# ============================================================================
md(r"""
---
## EDA Summary

### Key Findings

1. **Dataset Balance**: The DOLOS dataset has a reasonably balanced label distribution, with slightly more deception (53.6%) than truth (46.4%) samples. This is favourable for training unbiased models.

2. **Gender Imbalance**: There are significantly more male (78.9%) than female (21.1%) speakers. Gender-specific evaluation protocols are necessary to ensure the model doesn't develop gender bias.

3. **Speaker Variation**: Coverage varies widely across speakers (from 49 to 398 clips per speaker). Speaker-independent evaluation is critical to measure generalisation.

4. **Duration**: Most clips are short (median ~5s, mean ~5.2s), with some outliers extending to 48s. The IQR method removes ~3.8% of extreme values.

5. **Acoustic Features**: 
   - Pitch, RMS, spectral centroid, and ZCR show considerable overlap between truth and deception classes, suggesting that no single acoustic feature is sufficient for reliable deception detection.
   - Pairwise interactions (visualised in scatter plots and pairplots) may reveal more discriminative combinations.
   - Regression analysis shows the linear trends between feature pairs.

6. **Data Quality**: Six distinct raw label values were found (requiring standardisation), and a small number of duplicate file names were identified.

### Implications for Model Building

- A **multimodal approach** (audio + visual) is likely necessary, as the unimodal acoustic features show limited separability.
- **Parameter-efficient fine-tuning** (adapters) helps avoid overfitting given the moderate dataset size.
- **Speaker-independent cross-validation** and **gender-balanced evaluation** are essential for fair assessment.
- **Duration normalisation** or consistent input length should be applied to avoid duration-related confounds.

---

*End of Exploratory Data Analysis*
""")


# ============================================================================
# WRITE NOTEBOOK
# ============================================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.14.0"
        }
    },
    "cells": cells
}

out_path = os.path.join(ROOT, 'eda_dolos_notebook.ipynb')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to: {out_path}")
print(f"Total cells: {len(cells)}")
