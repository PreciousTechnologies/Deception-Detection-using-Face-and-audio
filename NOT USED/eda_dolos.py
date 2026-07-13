import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams.update({'figure.max_open_warning': 0, 'font.size': 10})

ROOT = r'C:\Users\OMEN\Desktop\DECEPTION DETECTION USING VIDEOS AND FACE'
OUT_DIR = os.path.join(ROOT, 'eda_output')
os.makedirs(OUT_DIR, exist_ok=True)

TSV = os.path.join(ROOT, 'DOLOSDATA', 'DOLOS', 'dolos_timestamps.csv')
PROTO_DIR = os.path.join(ROOT, 'DOLOSDATA', 'DOLOS', 'Training_Protocols')
VIDEO_DIR = os.path.join(ROOT, 'DOLOSDATA', 'raw_videos')

print("=" * 70)
print("DOLOS DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

###############################################################################
# 1. LOAD AND CLEAN METADATA
###############################################################################
print("\n[1] Loading and cleaning metadata...")

tsv = pd.read_csv(TSV)
print(f"  dolos_timestamps.csv: {tsv.shape[0]} rows, {tsv.shape[1]} cols")
tsv['file_name'] = tsv['file_name'].str.strip()

tsv['label_clean'] = tsv['label'].str.lower().str.strip()
tsv['label_clean'] = tsv['label_clean'].replace({'lie': 'deception', 'true': 'truth'})
print(f"  Label values (raw): {tsv['label'].unique()}")
print(f"  Label values (clean): {tsv['label_clean'].unique()}")

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

bad_dur = tsv['duration_sec'] <= 0
if bad_dur.any():
    print(f"  Removing {bad_dur.sum()} rows with non-positive duration")
    tsv = tsv[~bad_dur].copy()

tsv['speaker'] = tsv['file_name'].str.extract(r'^([A-Z]+)_', expand=False)
tsv['speaker'] = tsv['speaker'].str.upper()
print(f"  Unique speakers: {sorted(tsv['speaker'].dropna().unique())}")

###############################################################################
# 2. LOAD GENDER
###############################################################################
print("\n[2] Loading gender annotations from protocol CSVs...")
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
print(f"  Gender annotations collected: {len(gender_map)} clips")
tsv['gender'] = tsv['file_name'].map(gender_map)
print(f"  Gender coverage: {tsv['gender'].notna().sum()} / {len(tsv)}")

###############################################################################
# 3. VIDEO FILE CHECK
###############################################################################
print("\n[3] Checking video file availability...")
video_files = set(os.listdir(VIDEO_DIR))
tsv['video_exists'] = tsv['file_name'].apply(lambda x: f"{x}.mp4" in video_files)
print(f"  Videos found on disk: {sum(tsv['video_exists'])} / {len(tsv)}")

###############################################################################
# 4. DATASET COMPOSITION PLOTS
###############################################################################
print("\n[4] Generating dataset composition plots...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("DOLOS Dataset Composition", fontsize=16, fontweight='bold')

ax = axes[0, 0]
label_counts = tsv['label_clean'].value_counts()
colors_label = ['#2ecc71', '#e74c3c']
bars = ax.bar(label_counts.index, label_counts.values, color=colors_label, edgecolor='black')
for bar, val in zip(bars, label_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{val} ({val/len(tsv)*100:.1f}%)', ha='center', fontweight='bold')
ax.set_title('Label Distribution (Clean)', fontweight='bold')
ax.set_ylabel('Count')

ax = axes[0, 1]
gender_valid = tsv['gender'].dropna()
gender_counts = gender_valid.value_counts()
colors_gen = ['#3498db', '#e91e63']
bars = ax.bar(gender_counts.index, gender_counts.values, color=colors_gen, edgecolor='black')
for bar, val in zip(bars, gender_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{val} ({val/len(gender_valid)*100:.1f}%)', ha='center', fontweight='bold')
ax.set_title('Gender Distribution (Known)', fontweight='bold')
ax.set_ylabel('Count')

ax = axes[0, 2]
speaker_counts = tsv['speaker'].value_counts()
colors_sp = sns.color_palette('Set2', len(speaker_counts))
bars = ax.bar(speaker_counts.index, speaker_counts.values, color=colors_sp, edgecolor='black')
for bar, val in zip(bars, speaker_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(val), ha='center', fontweight='bold', fontsize=8)
ax.set_title('Speaker Distribution', fontweight='bold')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=45)

ax = axes[1, 0]
durations = tsv['duration_sec'].dropna()
ax.hist(durations, bins=40, color='#9b59b6', edgecolor='black', alpha=0.7)
ax.axvline(durations.median(), color='red', ls='--', label=f"Median: {durations.median():.1f}s")
ax.axvline(durations.mean(), color='green', ls='--', label=f"Mean: {durations.mean():.1f}s")
ax.set_title('Clip Duration Distribution', fontweight='bold')
ax.set_xlabel('Duration (seconds)')
ax.set_ylabel('Count')
ax.legend(fontsize=8)

ax = axes[1, 1]
df_dur = tsv.dropna(subset=['duration_sec', 'label_clean'])
sns.boxplot(x='label_clean', y='duration_sec', data=df_dur, ax=ax,
            palette={'truth': '#2ecc71', 'deception': '#e74c3c'})
ax.set_title('Duration by Label', fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('Duration (seconds)')

ax = axes[1, 2]
gender_known = tsv['gender'].notna()
if gender_known.any():
    crosstab = pd.crosstab(tsv.loc[gender_known, 'label_clean'], tsv.loc[gender_known, 'gender'])
    crosstab.plot(kind='bar', ax=ax, color=['#3498db', '#e91e63'], edgecolor='black')
    ax.set_title('Label x Gender', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Count')
    ax.legend(title='Gender')
    ax.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '01_dataset_composition.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> 01_dataset_composition.png")

###############################################################################
# 5. DATA QUALITY
###############################################################################
print("\n[5] Data quality assessment...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Data Quality Assessment", fontsize=14, fontweight='bold')

ax = axes[0]
raw_label_counts = tsv['label'].value_counts()
colors_raw = sns.color_palette('husl', len(raw_label_counts))
bars = ax.bar(range(len(raw_label_counts)), raw_label_counts.values,
              color=colors_raw, edgecolor='black')
ax.set_xticks(range(len(raw_label_counts)))
ax.set_xticklabels(raw_label_counts.index, rotation=45, ha='right')
for bar, val in zip(bars, raw_label_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            str(val), ha='center', fontsize=9)
ax.set_title('Raw Label Values in CSV', fontweight='bold')
ax.set_ylabel('Count')

ax = axes[1]
dup_counts = tsv['file_name'].value_counts()
dups = dup_counts[dup_counts > 1]
ax.barh(range(len(dups)), dups.values, color='#e67e22', edgecolor='black')
ax.set_yticks(range(len(dups)))
ax.set_yticklabels(dups.index, fontsize=7)
ax.set_title(f'Duplicate file_name Entries (n={len(dups)})', fontweight='bold')
ax.set_xlabel('Count')
for i, v in enumerate(dups.values):
    ax.text(v + 0.1, i, str(v), va='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_data_quality.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> 02_data_quality.png")

###############################################################################
# 6. AUDIO FEATURE SAMPLES (2 clips only)
###############################################################################
print("\n[6] Extracting audio features from sample videos...")
import librosa

try:
    sample_files = tsv[tsv['video_exists']].drop_duplicates(subset='file_name')['file_name'].head(2).tolist()
    fig, axes = plt.subplots(len(sample_files), 3, figsize=(16, 4 * len(sample_files)))
    fig.suptitle("Audio Feature Samples by Label", fontsize=14, fontweight='bold')
    if len(sample_files) == 1:
        axes = axes.reshape(1, -1)

    for idx, fname in enumerate(sample_files):
        fpath = os.path.join(VIDEO_DIR, f"{fname}.mp4")
        label = tsv.loc[tsv['file_name'] == fname, 'label_clean'].iloc[0]
        try:
            y, sr = librosa.load(fpath, sr=16000, mono=True)
        except Exception as e:
            print(f"    Skipping {fname}: {e}")
            continue

        t = np.linspace(0, len(y)/sr, len(y))

        ax = axes[idx, 0]
        ax.plot(t, y, color='#3498db', linewidth=0.5)
        ax.set_title(f'Waveform - {fname} [{label}]', fontsize=9, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')

        ax = axes[idx, 1]
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel',
                                       fmax=8000, ax=ax)
        ax.set_title(f'Mel-Spectrogram - {fname} [{label}]', fontsize=9, fontweight='bold')
        fig.colorbar(img, ax=ax, format='%+2.0f dB', shrink=0.8)

        ax = axes[idx, 2]
        f0, voiced_flag, _ = librosa.pyin(y, fmin=80, fmax=400, sr=sr)
        times = librosa.times_like(f0, sr=sr)
        ax.plot(times, f0, color='#e67e22', linewidth=1)
        ax.set_title(f'Pitch Contour - {fname} [{label}]', fontsize=9, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim(50, 450)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '03_audio_features.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> 03_audio_features.png")
except Exception as e:
    print(f"  Audio extraction failed (non-critical): {e}")

###############################################################################
# 7. FACE DETECTION SAMPLE
###############################################################################
print("\n[7] Visual preprocessing: face detection sample...")
import cv2

try:
    avail = tsv[tsv['video_exists']].drop_duplicates(subset='file_name')
    sample_vid = avail['file_name'].iloc[0] if len(avail) > 0 else None
    if sample_vid:
        fpath = os.path.join(VIDEO_DIR, f"{sample_vid}.mp4")
        cap = cv2.VideoCapture(fpath)
        ret, frame = cap.read()
        cap.release()
        if ret:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(f"Face Detection Sample - {sample_vid}", fontsize=13, fontweight='bold')

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[0].imshow(frame_rgb)
            axes[0].set_title('Raw Frame', fontweight='bold')
            axes[0].axis('off')

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            frame_disp = frame_rgb.copy()
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame_disp, (x, y), (x + w, y + h), (0, 255, 0), 3)
                axes[1].set_title(f'Face Detected (Haar: {len(faces)})', fontweight='bold')
            else:
                axes[1].set_title('No Face Detected (Haar)', fontweight='bold')
            axes[1].imshow(frame_disp)
            axes[1].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, '04_face_detection.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("  -> 04_face_detection.png")
except Exception as e:
    print(f"  Face detection failed (non-critical): {e}")

###############################################################################
# 8. CORRELATION & OUTLIER ANALYSIS (metadata only)
###############################################################################
print("\n[8] Feature correlation and outlier analysis...")

df_feat = tsv.drop_duplicates(subset='file_name')[['file_name', 'duration_sec', 'label_clean', 'speaker', 'gender']].copy()
df_feat['label_num'] = (df_feat['label_clean'] == 'deception').astype(int)
df_feat['gender_num'] = (df_feat['gender'] == 'male').astype(int)
spk_map = {s: i for i, s in enumerate(df_feat['speaker'].unique())}
df_feat['speaker_num'] = df_feat['speaker'].map(spk_map)

# Stats table
print("\n  Univariate Statistics (duration_sec):")
s = df_feat['duration_sec'].dropna()
stats_df = pd.DataFrame([{
    'Feature': 'duration_sec', 'Count': len(s), 'Mean': f"{s.mean():.3f}",
    'Std': f"{s.std():.3f}", 'Min': f"{s.min():.3f}",
    '25%': f"{s.quantile(0.25):.3f}", '50%': f"{s.quantile(0.50):.3f}",
    '75%': f"{s.quantile(0.75):.3f}", 'Max': f"{s.max():.3f}"
}])
print(stats_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 2))
ax.axis('off')
table = ax.table(cellText=stats_df.values, colLabels=stats_df.columns,
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)
ax.set_title("Univariate Statistics (duration_sec)", fontweight='bold', fontsize=14, pad=20)
plt.savefig(os.path.join(OUT_DIR, '05_univariate_stats.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> 05_univariate_stats.png")

# 8a. Correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Correlation & Feature Analysis", fontsize=14, fontweight='bold')

ax = axes[0]
corr_cols = ['duration_sec', 'label_num', 'gender_num', 'speaker_num']
corr_data = df_feat[corr_cols].dropna()
corr = corr_data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, ax=ax, mask=mask, linewidths=1,
            vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Heatmap', fontweight='bold')

# 8b. Duration box plot by speaker
ax = axes[1]
df_spk = df_feat.dropna(subset=['duration_sec'])
sns.boxplot(x='speaker', y='duration_sec', data=df_spk, ax=ax, palette='Set2')
ax.set_title('Duration Outliers by Speaker', fontweight='bold')
ax.set_xlabel('Speaker')
ax.set_ylabel('Duration (seconds)')
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '06_correlation_outliers.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> 06_correlation_outliers.png")

###############################################################################
# 9. LABEL v SPEAKER BREAKDOWN
###############################################################################
print("\n[9] Label x Speaker breakdown...")
fig, ax = plt.subplots(figsize=(10, 6))
crosstab_spk = pd.crosstab(tsv['speaker'], tsv['label_clean'])
crosstab_spk_pct = crosstab_spk.div(crosstab_spk.sum(1), axis=0)
crosstab_spk_pct.plot(kind='barh', stacked=True, ax=ax,
                       color=['#2ecc71', '#e74c3c'], edgecolor='black')
for i, spk in enumerate(crosstab_spk.index):
    total = crosstab_spk.sum(1)[spk]
    ax.text(0.5, i, f' n={total}', va='center', fontweight='bold', fontsize=9)
ax.set_title('Label Proportion by Speaker', fontweight='bold')
ax.set_xlabel('Proportion')
ax.set_ylabel('Speaker')
ax.legend(title='Label', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '07_label_by_speaker.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> 07_label_by_speaker.png")

###############################################################################
# 10. ACOUSTIC PROFILE OF SAMPLE CLIPS (table)
###############################################################################
print("\n[10] Acoustic profile of sample clips...")
try:
    sample_set = tsv[tsv['video_exists']].drop_duplicates(subset='file_name').head(6)
    ac_rows = []
    for _, row in sample_set.iterrows():
        fpath = os.path.join(VIDEO_DIR, f"{row['file_name']}.mp4")
        try:
            y, sr = librosa.load(fpath, sr=16000, mono=True, duration=5.0)
            if len(y) < sr:
                continue
            rms = librosa.feature.rms(y=y)[0]
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            f0, voiced, _ = librosa.pyin(y, fmin=80, fmax=400, sr=sr)
            f0_clean = f0[~np.isnan(f0)]
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

    if ac_rows:
        ac_df = pd.DataFrame(ac_rows)
        fig, ax = plt.subplots(figsize=(14, 2 + 0.4 * len(ac_rows)))
        ax.axis('off')
        table = ax.table(cellText=ac_df.values, colLabels=ac_df.columns,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax.set_title("Acoustic Profile of Sample Clips", fontweight='bold', fontsize=14, pad=20)
        plt.savefig(os.path.join(OUT_DIR, '08_acoustic_profile.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  -> 08_acoustic_profile.png")
        print(ac_df.to_string(index=False))
except Exception as e:
    print(f"  Acoustic profile failed: {e}")

###############################################################################
# 12. OUTLIER REMOVAL VISUALIZATION
###############################################################################
print("\n[12] Outlier removal visualization...")

dur = tsv['duration_sec'].dropna()
Q1 = dur.quantile(0.25)
Q3 = dur.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
dur_clean = dur[(dur >= lower) & (dur <= upper)]

print(f"  Duration IQR: {IQR:.2f}s")
print(f"  Lower bound: {lower:.2f}s, Upper bound: {upper:.2f}s")
print(f"  Outliers removed: {len(dur) - len(dur_clean)} / {len(dur)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Duration Outlier Removal (IQR Method)", fontsize=14, fontweight='bold')

axes[0].hist(dur, bins=40, color='#e74c3c', edgecolor='black', alpha=0.7)
axes[0].axvline(lower, color='blue', ls='--', label=f'Lower: {lower:.1f}s')
axes[0].axvline(upper, color='blue', ls='--', label=f'Upper: {upper:.1f}s')
axes[0].set_title(f'Before (n={len(dur)})', fontweight='bold')
axes[0].set_xlabel('Duration (seconds)')
axes[0].set_ylabel('Count')
axes[0].legend(fontsize=8)

axes[1].hist(dur_clean, bins=40, color='#2ecc71', edgecolor='black', alpha=0.7)
axes[1].axvline(dur_clean.median(), color='red', ls='--', label=f"Median: {dur_clean.median():.1f}s")
axes[1].axvline(dur_clean.mean(), color='green', ls='--', label=f"Mean: {dur_clean.mean():.1f}s")
axes[1].set_title(f'After IQR Removal (n={len(dur_clean)})', fontweight='bold')
axes[1].set_xlabel('Duration (seconds)')
axes[1].set_ylabel('Count')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '12_outlier_removal.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> 12_outlier_removal.png")

###############################################################################
# 13. COMPUTE ACOUSTIC FEATURES FOR EXTENDED SAMPLE
###############################################################################
print("\n[13] Computing acoustic features for extended sample...")
try:
    sample_set = tsv[tsv['video_exists']].drop_duplicates(subset='file_name').head(100).copy()
    ac_rows = []
    for _, row in sample_set.iterrows():
        fpath = os.path.join(VIDEO_DIR, f"{row['file_name']}.mp4")
        try:
            y, sr = librosa.load(fpath, sr=16000, mono=True, duration=5.0)
            if len(y) < sr:
                continue
            rms = librosa.feature.rms(y=y)[0]
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            f0, _, _ = librosa.pyin(y, fmin=80, fmax=400, sr=sr)
            f0_clean = f0[~np.isnan(f0)]
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
    print(f"  Acoustic features computed for {len(ac_df)} clips")

    ###############################################################################
    # 13a. DETAILED BOX PLOTS PER PARAMETER
    ###############################################################################
    print("\n[13a] Box plots per acoustic parameter by label...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Acoustic Feature Distributions by Label", fontsize=14, fontweight='bold')

    feat_cfg = [
        ('pitch_mean_hz', 'Pitch (Hz)'),
        ('rms_mean', 'RMS Energy'),
        ('spectral_centroid_hz', 'Spectral Centroid (Hz)'),
        ('zcr', 'Zero Crossing Rate'),
    ]
    for ax, (col, title) in zip(axes.flatten(), feat_cfg):
        sns.boxplot(x='label', y=col, data=ac_df, ax=ax,
                    palette={'truth': '#2ecc71', 'deception': '#e74c3c'})
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '13a_boxplots_acoustic.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> 13a_boxplots_acoustic.png")

    ###############################################################################
    # 13b. SCATTER PLOTS
    ###############################################################################
    print("\n[13b] Scatter plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Acoustic Feature Relationships by Label", fontsize=14, fontweight='bold')

    scatter_cfgs = [
        ('pitch_mean_hz', 'rms_mean', 'Pitch (Hz)', 'RMS Energy'),
        ('spectral_centroid_hz', 'zcr', 'Spectral Centroid (Hz)', 'ZCR'),
        ('duration_s', 'pitch_mean_hz', 'Duration (s)', 'Pitch (Hz)'),
        ('duration_s', 'rms_mean', 'Duration (s)', 'RMS Energy'),
    ]
    for ax, (xcol, ycol, xlab, ylab) in zip(axes.flatten(), scatter_cfgs):
        for label in ['truth', 'deception']:
            sub = ac_df[ac_df['label'] == label]
            c = '#2ecc71' if label == 'truth' else '#e74c3c'
            ax.scatter(sub[xcol], sub[ycol], alpha=0.6, c=c, label=label, edgecolors='black', linewidth=0.3)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)
        ax.set_title(f'{xlab} vs {ylab}', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '13b_scatter_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> 13b_scatter_plots.png")

    ###############################################################################
    # 13c. SCATTER MATRIX / PAIRPLOT (all data)
    ###############################################################################
    print("\n[13c] Scatter matrix - all data...")
    pair_cols = ['pitch_mean_hz', 'rms_mean', 'spectral_centroid_hz', 'zcr', 'duration_s', 'label']
    pp = sns.pairplot(ac_df[pair_cols], hue='label',
                       palette={'truth': '#2ecc71', 'deception': '#e74c3c'},
                       diag_kind='kde',
                       plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'black', 'linewidth': 0.3})
    pp.fig.suptitle("Acoustic Feature Pairplot (All Data)", fontsize=14, fontweight='bold', y=1.02)
    pp.savefig(os.path.join(OUT_DIR, '13c_pairplot_all.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> 13c_pairplot_all.png")

    ###############################################################################
    # 13c2. PAIRPLOT WITH OUTLIERS REMOVED (IQR per feature)
    ###############################################################################
    print("\n[13c2] Scatter matrix - outliers removed...")
    ac_clean = ac_df.copy()
    for col in ['pitch_mean_hz', 'rms_mean', 'spectral_centroid_hz', 'zcr', 'duration_s']:
        cQ1 = ac_clean[col].quantile(0.25)
        cQ3 = ac_clean[col].quantile(0.75)
        cIQR = cQ3 - cQ1
        c_low = cQ1 - 1.5 * cIQR
        c_high = cQ3 + 1.5 * cIQR
        ac_clean = ac_clean[(ac_clean[col] >= c_low) & (ac_clean[col] <= c_high)]
    print(f"  Clips after per-feature IQR removal: {len(ac_clean)}")

    pp2 = sns.pairplot(ac_clean[pair_cols], hue='label',
                        palette={'truth': '#2ecc71', 'deception': '#e74c3c'},
                        diag_kind='kde',
                        plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'black', 'linewidth': 0.3})
    pp2.fig.suptitle("Acoustic Feature Pairplot (Outliers Removed)", fontsize=14, fontweight='bold', y=1.02)
    pp2.savefig(os.path.join(OUT_DIR, '13c2_pairplot_clean.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> 13c2_pairplot_clean.png")

    ###############################################################################
    # 13d. REGRESSION LINES
    ###############################################################################
    print("\n[13d] Scatter plots with regression lines...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Feature Relationships with Regression Lines", fontsize=14, fontweight='bold')

    reg_cfgs = [
        ('duration_s', 'pitch_mean_hz', 'Duration (s)', 'Pitch (Hz)'),
        ('duration_s', 'rms_mean', 'Duration (s)', 'RMS Energy'),
        ('pitch_mean_hz', 'spectral_centroid_hz', 'Pitch (Hz)', 'Spectral Centroid (Hz)'),
        ('rms_mean', 'zcr', 'RMS Energy', 'ZCR'),
    ]
    for ax, (xcol, ycol, xlab, ylab) in zip(axes.flatten(), reg_cfgs):
        sns.regplot(x=xcol, y=ycol, data=ac_df, ax=ax,
                    scatter_kws={'alpha': 0.5, 'edgecolor': 'black', 'linewidths': 0.3},
                    line_kws={'color': 'red'})
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(f'{xlab} vs {ylab} (linear fit)', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '13d_regression_lines.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> 13d_regression_lines.png")

    ###############################################################################
    # 13e. COMPLETE UNIVARIATE ANALYSIS FOR ALL PARAMETERS
    ###############################################################################
    print("\n[13e] Complete univariate analysis for all parameters...")
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
    print(uni_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(16, 3 + 0.4 * len(uni_df)))
    ax.axis('off')
    table = ax.table(cellText=uni_df.values, colLabels=uni_df.columns,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    ax.set_title("Complete Univariate Statistics (All Acoustic Features)", fontweight='bold', fontsize=14, pad=20)
    plt.savefig(os.path.join(OUT_DIR, '13e_univariate_stats.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> 13e_univariate_stats.png")

except Exception as e:
    print(f"  Extended acoustic analysis failed (non-critical): {e}")
    import traceback
    traceback.print_exc()

###############################################################################
# 11. SUMMARY REPORT
###############################################################################
print("\n" + "=" * 70)
print("EDA SUMMARY REPORT")
print("=" * 70)
print(f"  Total clips in timestamps CSV: {len(tsv)}")
print(f"  Unique clips: {tsv['file_name'].nunique()}")
print(f"  Unique YouTube videos: {tsv['YT_Video_ID'].nunique()}")
print(f"  Label distribution (clean):")
for label, count in label_counts.items():
    print(f"    {label}: {count} ({count/len(tsv)*100:.1f}%)")
if len(gender_valid) > 0:
    print(f"  Gender distribution (known):")
    for g, c in gender_counts.items():
        print(f"    {g}: {c} ({c/len(gender_valid)*100:.1f}%)")
    print(f"    Unknown: {tsv['gender'].isna().sum()} clips")
print(f"  Speakers: {sorted(tsv['speaker'].dropna().unique())}")
print(f"  Videos on disk: {sum(tsv['video_exists'])} / {len(tsv)}")
durations = tsv['duration_sec'].dropna()
print(f"  Duration range: {durations.min():.1f}s - {durations.max():.1f}s")
print(f"  Duration mean: {durations.mean():.2f}s, median: {durations.median():.2f}s")
print(f"  Speaker counts:")
for spk, cnt in tsv['speaker'].value_counts().items():
    print(f"    {spk}: {cnt}")
if len(dups) > 0:
    print(f"  Duplicate file_name entries: {len(dups)}")
else:
    print(f"  Duplicate file_name entries: None")
print(f"  Label inconsistencies: {len(raw_label_counts)} unique raw labels")
print(f"\n  Output directory: {OUT_DIR}")
print("=" * 70)
print("EDA complete!")
