"""
互补特征融合集成方案 - CBU5201 Mini Project
- 专家 A: LightCRNN (160维) + 专家 B: MobileNetV3-Small (576维)
- 融合: 特征拼接 (736维) + MLP 分类器
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import glob
from tqdm import tqdm
import soundfile as sf
import pyworld as pw
from joblib import Parallel, delayed
from dataclasses import dataclass, field
from typing import List, Optional

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# PyTorch for deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# timm for pretrained models
import timm

# torchaudio for audio processing
import torchaudio
import torchaudio.transforms as T


# Device detection
def _detect_device_once():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.environ.get("DEVICE_INFO_PRINTED") != "1":
        gpu_info = f"GPU: {torch.cuda.get_device_name(0)}" if device == 'cuda' else "CPU only"
        print(f"[Device] {gpu_info}")
        os.environ["DEVICE_INFO_PRINTED"] = "1"
    return device

DEVICE = _detect_device_once()


# ==================== 配置系统 ====================

@dataclass
class DataConfig:
    data_dir: str = r"C:\Users\guson\Desktop\ml_project\Data\MLEndHW\MLEndHW_sample_800"
    pretrained_weights_dir: str = r"C:\Users\guson\Desktop\ml_project\pretrained_weights"
    valid_songs: List[str] = field(default_factory=lambda: [
        'TryEverything', 'RememberMe', 'NewYork', 'Friend',
        'Necessities', 'Married', 'Happy', 'Feeling'
    ])
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 45
    n_jobs: Optional[int] = None

@dataclass
class AudioConfig:
    sample_rate: int = 22050
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    duration_seconds: float = 12.0
    max_length: int = 512

@dataclass
class ModelConfig:
    crnn_rnn_hidden_size: int = 80
    crnn_rnn_num_layers: int = 2
    crnn_dropout_rate: float = 0.7
    crnn_use_attention: bool = True
    crnn_attention_hidden_size: int = 64
    crnn_cnn_channels: List[int] = field(default_factory=lambda: [24, 48, 96, 192])
    crnn_cnn_dropout: float = 0.1
    mobilenet_pretrained: bool = True
    mobilenet_feature_dim: int = 576
    fusion_hidden_dim: int = 128
    fusion_dropout_rate: float = 0.5
    num_classes: int = 8
    freeze_experts: bool = True

@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs_experts: int = 60
    num_epochs_fusion: int = 15
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    scheduler_mode: str = 'max'
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    num_workers: int = 4
    pin_memory: bool = True
    early_stopping_patience: Optional[int] = None
    save_best_only: bool = True
    verbose: bool = True

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        self.data.song_to_label = {song: idx for idx, song in enumerate(self.data.valid_songs)}
        self.data.label_to_song = {idx: song for idx, song in enumerate(self.data.valid_songs)}

    def print_config(self):
        print(f"[Config] seed={self.data.random_seed}, batch={self.training.batch_size}, "
              f"lr={self.training.learning_rate}, epochs={self.training.num_epochs_experts}/{self.training.num_epochs_fusion}")


# 创建全局配置实例
CONFIG = Config()

# 兼容性：保留原有的全局变量（避免破坏现有代码）
RANDOM_SEED = CONFIG.data.random_seed
DEFAULT_DATA_DIR = CONFIG.data.data_dir
PRETRAINED_WEIGHTS_DIR = CONFIG.data.pretrained_weights_dir
VALID_SONGS = CONFIG.data.valid_songs
SONG_TO_LABEL = CONFIG.data.song_to_label
LABEL_TO_SONG = CONFIG.data.label_to_song


# 随机种子设置
def set_random_seed(seed=None):
    seed = seed or RANDOM_SEED
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if os.environ.get("SEED_INFO_PRINTED") != "1":
        print(f"[Seed] {seed}")
        os.environ["SEED_INFO_PRINTED"] = "1"
    return seed

set_random_seed(RANDOM_SEED)


def log_section(title):
    """打印分节日志"""
    print(f"\n{'='*60}\n{title}\n{'='*60}")

def get_or_download_pretrained_mobilenetv3():
    """获取或下载 MobileNetV3-Small 预训练权重"""
    os.makedirs(PRETRAINED_WEIGHTS_DIR, exist_ok=True)
    cache_path = os.path.join(PRETRAINED_WEIGHTS_DIR, 'mobilenetv3_small_timm.pth')

    if os.path.exists(cache_path):
        try:
            model = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=0, in_chans=1)
            state_dict = torch.load(cache_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            return model, True
        except Exception:
            pass

    try:
        model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=0, in_chans=1)
        torch.save(model.state_dict(), cache_path)
        return model, False
    except Exception:
        model = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=0, in_chans=1)
        return model, False


# ==================== 数据处理 ====================

def parse_filenames(files, return_mappings=False):
    """解析文件名生成 DataFrame"""
    data = []
    for file_path in files:
        filename = os.path.basename(file_path)
        try:
            parts = filename.replace('.wav', '').split('_')
            if len(parts) >= 4:
                data.append({
                    'file_id': filename, 'file_path': file_path,
                    'participant': parts[0], 'type': parts[1],
                    'interpretation': parts[2], 'song': parts[3],
                    'label': SONG_TO_LABEL.get(parts[3], -1)
                })
        except Exception as e:
            print(f"Error parsing {filename}: {e}")

    df = pd.DataFrame(data)
    print(f"Parsed {len(df)} files")
    for song, label in SONG_TO_LABEL.items():
        count = len(df[df['song'] == song]) if len(df) > 0 else 0
        print(f"  {label}: {song:20s} ({count} files)")

    if return_mappings:
        return df, SONG_TO_LABEL, LABEL_TO_SONG
    return df


# ==================== 音频处理 ====================

def remove_silence(audio, sr, top_db=30, frame_length=2048, hop_length=512, min_duration=0.1, verbose=False):
    """VAD: 移除音频首尾静音"""
    audio_abs = np.abs(audio)
    num_frames = 1 + (len(audio_abs) - frame_length) // hop_length
    energy = np.array([np.sum(audio_abs[i * hop_length:i * hop_length + frame_length] ** 2) for i in range(num_frames)])
    energy_db = 10 * np.log10(np.maximum(energy, 1e-10))
    voiced_frames = energy_db > (np.max(energy_db) - top_db)

    if not np.any(voiced_frames):
        return audio

    first_voiced = np.argmax(voiced_frames)
    last_voiced = len(voiced_frames) - np.argmax(voiced_frames[::-1]) - 1
    start_sample = max(0, first_voiced * hop_length)
    end_sample = min(len(audio), (last_voiced + 1) * hop_length + frame_length)

    if (end_sample - start_sample) / sr < min_duration:
        return audio
    return audio[start_sample:end_sample]


def preprocess_audio(audio, sr, remove_dc=True, normalize=True):
    """音频预处理：去直流+归一化"""
    audio = audio.copy()
    if remove_dc:
        audio = audio - np.mean(audio)
    if normalize:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
    return audio


# ==================== 数据增强 ====================
def _pyworld_analysis(audio, sr, frame_period=5.0):
    """PyWorld 分析 (提取 F0, SP, AP)"""
    audio = np.asarray(audio, dtype=np.float64)
    f0, t = pw.dio(audio, sr, frame_period=frame_period)
    f0 = pw.stonemask(audio, f0, t, sr)
    sp = pw.cheaptrick(audio, f0, t, sr)
    ap = pw.d4c(audio, f0, t, sr)
    return f0, sp, ap, frame_period

def pitch_shift_pyworld(audio, sr, n_steps):
    f0, sp, ap, fp = _pyworld_analysis(audio, sr)
    f0_shifted = f0 * (2 ** (n_steps / 12.0))
    return pw.synthesize(f0_shifted, sp, ap, sr, fp)

def time_stretch_pyworld(audio, sr, speed_ratio):
    f0, sp, ap, fp = _pyworld_analysis(audio, sr)
    n_frames = max(2, int(len(f0) / speed_ratio))
    idx_old, idx_new = np.arange(len(f0)), np.linspace(0, len(f0) - 1, n_frames)
    f0_s = np.interp(idx_new, idx_old, f0)
    sp_s = np.array([np.interp(idx_new, idx_old, sp[:, i]) for i in range(sp.shape[1])]).T
    ap_s = np.array([np.interp(idx_new, idx_old, ap[:, i]) for i in range(ap.shape[1])]).T
    return pw.synthesize(f0_s, sp_s, ap_s, sr, fp)

def add_gaussian_noise(audio, noise_level=0.005):
    noise = np.random.normal(0, np.std(audio) * noise_level, len(audio))
    return np.clip(audio + noise, -1.0, 1.0)


def augment_single_audio(audio, sr, augmentation_type, param):
    """Apply the specified augmentation method to a single audio"""
    try:
        if augmentation_type == 'pitch_shift':
            return pitch_shift_pyworld(audio, sr, n_steps=param)
        elif augmentation_type == 'time_stretch':
            return time_stretch_pyworld(audio, sr, speed_ratio=param)
        elif augmentation_type == 'add_noise':
            return add_gaussian_noise(audio, noise_level=param)
        else:
            raise ValueError(f"Unknown augmentation type: {augmentation_type}")
    except Exception as e:
        print(f"    Augmentation failed ({augmentation_type}, param={param}): {e}")
        return None


def _augment_worker(row_dict, strategies, output_dir, verbose=False):
    """Single file multiple strategy augmentation task (for parallel calling)"""
    file_path = row_dict['file_path']
    filename = row_dict['file_id']
    results = []
    try:
        audio, sr = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        audio = remove_silence(audio, sr, top_db=30, verbose=False)
        audio = preprocess_audio(audio, sr, remove_dc=True, normalize=True)

        for aug_type, aug_param in strategies:
            augmented_audio = augment_single_audio(audio, sr, aug_type, aug_param)
            if augmented_audio is None:
                continue

            base_name = filename.replace('.wav', '')
            aug_suffix = f"_aug_{aug_type}_{aug_param}".replace('.', 'p')
            new_filename = f"{base_name}{aug_suffix}.wav"
            new_filepath = os.path.join(output_dir, new_filename)

            sf.write(new_filepath, augmented_audio, sr)

            results.append({
                'file_id': new_filename,
                'file_path': new_filepath,
                'participant': row_dict['participant'],
                'type': row_dict['type'],
                'interpretation': row_dict['interpretation'],
                'song': row_dict['song'],
                'label': row_dict['label'],
                'augmentation': f"{aug_type}_{aug_param}"
            })

            if verbose:
                print(f"  Generated: {aug_type}({aug_param}) -> {new_filename}")
    except Exception as e:
        print(f"  Error: {filename} - {e}")

    return results


def augment_dataset(df, output_dir, target_total=4000, verbose=True, n_jobs=1, skip_existing=True):
    """Batch data augmentation: expand the original dataset to the target number"""
    print("\n" + "="*60)
    print("Starting Data Augmentation")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    if skip_existing:
        existing_aug_files = glob.glob(os.path.join(output_dir, '*_aug_*.wav'))
        if len(existing_aug_files) > 0:
            print(f"\n✓ Found {len(existing_aug_files)} existing augmented audio files")

            augmented_data = []

            for idx, row in df.iterrows():
                augmented_data.append({
                    'file_id': row['file_id'],
                    'file_path': row['file_path'],
                    'participant': row['participant'],
                    'type': row['type'],
                    'interpretation': row['interpretation'],
                    'song': row['song'],
                    'label': row['label'],
                    'augmentation': 'original'
                })

            for aug_file in existing_aug_files:
                filename = os.path.basename(aug_file)
                parts = filename.replace('.wav', '').split('_')

                if len(parts) >= 4:
                    original_filename = '_'.join(parts[:4]) + '.wav'
                    original_row = df[df['file_id'] == original_filename]

                    if len(original_row) > 0:
                        original_row = original_row.iloc[0]

                        if '_aug_' in filename:
                            aug_info = filename.split('_aug_')[1].replace('.wav', '')
                        else:
                            aug_info = 'unknown'

                        augmented_data.append({
                            'file_id': filename,
                            'file_path': aug_file,
                            'participant': original_row['participant'],
                            'type': original_row['type'],
                            'interpretation': original_row['interpretation'],
                            'song': original_row['song'],
                            'label': original_row['label'],
                            'augmentation': aug_info
                        })

            augmented_df = pd.DataFrame(augmented_data)
            current_total = len(augmented_df)

            if current_total >= target_total:
                print(f"Current samples ({current_total}) >= target ({target_total}), skipping generation")
                return augmented_df
            else:
                print(f"Current samples {current_total} < target {target_total}, will generate {target_total - current_total} more")
        else:
            print("No existing augmented files detected, will generate from scratch")
    else:
        print("skip_existing=False, will regenerate all augmented files")

    original_count = len(df)
    augmentations_needed = target_total - original_count

    print(f"Original samples: {original_count}")
    print(f"Target total: {target_total}")
    print(f"Augmentations needed: {augmentations_needed}")

    if augmentations_needed <= 0:
        print("Sample count already meets target, no augmentation needed")
        return df

    augmentations_per_sample = int(np.ceil(augmentations_needed / original_count))

    print(f"Generating {augmentations_per_sample} augmented versions per original sample")

    augmentation_strategies = [
        # 两个 0.5 音高偏移（±0.5 半音）
        ('pitch_shift', -0.5),
        ('pitch_shift', 0.5),
        # 一个倍速（略快）
        ('time_stretch', 1.1),
        # 一个白噪声
        ('add_noise', 0.005),
    ]

    if augmentations_per_sample > len(augmentation_strategies) and verbose:
        print(f"Note: Need {augmentations_per_sample} augmentations, will cycle through {len(augmentation_strategies)} strategies")

    augmented_data = []

    for idx, row in df.iterrows():
        augmented_data.append({
            'file_id': row['file_id'],
            'file_path': row['file_path'],
            'participant': row['participant'],
            'type': row['type'],
            'interpretation': row['interpretation'],
            'song': row['song'],
            'label': row['label'],
            'augmentation': 'original'
        })

    print("\nStarting to generate augmented samples...")

    tasks = []
    remaining = augmentations_needed
    for _, row in df.iterrows():
        if remaining <= 0:
            break
        num_for_this = min(augmentations_per_sample, remaining)
        strategies = [augmentation_strategies[i % len(augmentation_strategies)] for i in range(num_for_this)]
        tasks.append((row.to_dict(), strategies))
        remaining -= num_for_this

    if n_jobs == 1:
        results_lists = []
        for row_dict, strategies in tqdm(tasks, desc="Augmentation progress"):
            results_lists.append(_augment_worker(row_dict, strategies, output_dir, verbose=verbose))
    else:
        print(f"Using parallel computation: n_jobs={n_jobs}")
        results_lists = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_augment_worker)(row_dict, strategies, output_dir, verbose=False)
            for row_dict, strategies in tqdm(tasks, desc="Scheduling tasks")
        )

    for res in results_lists:
        augmented_data.extend(res)

    generated_count = sum(len(res) for res in results_lists)

    augmented_df = pd.DataFrame(augmented_data)

    print("\n" + "="*60)
    print(f"Data augmentation completed: Original {original_count} -> Generated {generated_count}, Total {len(augmented_df)}")
    print(f"Output directory: {output_dir}")

    if verbose:
        print("\nAugmentation type distribution:")
        aug_counts = augmented_df['augmentation'].value_counts()
        for aug_type, count in aug_counts.items():
            print(f"  {aug_type}: {count} samples")

    return augmented_df



# ==========================================
# 4. Mel 频谱图提取 (torchaudio)
# ==========================================
def extract_mel_spectrogram_torchaudio(file_path, target_frames=128, target_freq_bins=128, verbose=False):
    """
    使用 torchaudio 提取 Mel 频谱图
    """
    try:
        audio, sr = sf.read(file_path)

        # convert to mono audio
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        waveform = torch.FloatTensor(audio).unsqueeze(0)

        energy = torch.sum(waveform ** 2, dim=0)
        threshold = torch.max(energy) * 0.01

        voiced_indices = torch.where(energy > threshold)[0]

        # remove the silence at the beginning and end of the audio
        if len(voiced_indices) > 0:
            start_idx = voiced_indices[0].item()
            end_idx = voiced_indices[-1].item() + 1
            waveform = waveform[:, start_idx:end_idx]

        if waveform.shape[1] < sr * 0.1:
            if verbose:
                print(f"  警告: {file_path} 音频太短")
            return None

        # normalize the waveform
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

        mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            n_mels=target_freq_bins,
            f_min=50.0,
            f_max=8000.0,
            power=2.0,
            normalized=False
        )

        # convert to mel spectrogram
        mel_spec = mel_spectrogram_transform(waveform)
        mel_spec = mel_spec.squeeze(0)

        # convert to db
        db_transform = T.AmplitudeToDB(top_db=80.0)
        mel_spec_db = db_transform(mel_spec)

        # normalize the mel spectrogram
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

        # resize the mel spectrogram to the target size
        if mel_spec_db.shape[1] != target_frames or mel_spec_db.shape[0] != target_freq_bins:
            mel_spec_db = mel_spec_db.unsqueeze(0).unsqueeze(0)
            mel_spec_db = torch.nn.functional.interpolate(
                mel_spec_db,
                size=(target_freq_bins, target_frames),
                mode='bilinear',
                align_corners=False
            )
            mel_spec_db = mel_spec_db.squeeze(0).squeeze(0)

        spectrogram = mel_spec_db.cpu().numpy().copy()
        return spectrogram

    except Exception as e:
        if verbose:
            print(f"Error extracting Mel spectrogram from {file_path}: {e}")
        return None

def extract_f0_contour_pyworld(file_path, target_frames=32, verbose=False):
    """提取 F0 轮廓"""
    try:
        audio, sr = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = np.asarray(audio, dtype=np.float64)

        # VAD
        energy = audio ** 2
        voiced = energy > (energy.max() * 0.01)
        if np.any(voiced):
            idx = np.where(voiced)[0]
            audio = audio[idx[0]:idx[-1] + 1]
        if len(audio) < sr * 0.1:
            return None

        _f0, t = pw.dio(audio, sr)
        f0 = pw.stonemask(audio, _f0, t, sr)
        valid = f0 > 1e-6
        if not np.any(valid):
            return None

        log_f0 = np.log2(np.maximum(f0, 1e-6) / 100.0)
        orig_idx, target_idx = np.linspace(0, 1, len(log_f0)), np.linspace(0, 1, target_frames)
        log_f0_interp = np.interp(target_idx, orig_idx[valid], log_f0[valid])
        log_f0_interp = (log_f0_interp - log_f0_interp.mean()) / (log_f0_interp.std() + 1e-6)
        return log_f0_interp.astype(np.float32)
    except Exception:
        return None


def split_by_participant(df, train_ratio=0.7, val_ratio=0.15, seed=None):
    """按参与者划分数据集 (LOGO)"""
    seed = seed or RANDOM_SEED
    participants = np.array(sorted(df['participant'].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(participants)

    n_total = len(participants)
    n_train = max(1, int(round(n_total * train_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))
    if n_train + n_val >= n_total:
        n_train, n_val = max(1, n_total - 2), 1

    train_p = participants[:n_train]
    val_p = participants[n_train:n_train + n_val]
    test_p = participants[n_train + n_val:]
    if len(test_p) == 0:
        test_p, val_p = val_p[-1:], val_p[:-1]

    def idx_for(parts):
        return np.where(df['participant'].isin(parts).values)[0]

    return {
        'train': idx_for(train_p),
        'val': idx_for(val_p),
        'test': idx_for(test_p),
        'train_parts': train_p,
        'val_parts': val_p,
        'test_parts': test_p,
    }


#  6.unified dataset class, supports Mel spectrogram and optional F0 contour
class UnifiedAudioDataset(Dataset):
    def __init__(self, mel_spectrograms, labels, f0_contours=None,
                 apply_spec_augment=False, freq_mask_param=15, time_mask_param=35):
        self.mel_spectrograms = torch.tensor(mel_spectrograms, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.f0_contours = torch.tensor(f0_contours, dtype=torch.float32) if f0_contours is not None else None
        self.apply_spec_augment = apply_spec_augment

        if apply_spec_augment:
            self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
            self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel_spec = self.mel_spectrograms[idx].unsqueeze(0)  # (1, 128, 512)
        label = self.labels[idx]

        if self.apply_spec_augment:
            mel_spec = self.freq_masking(mel_spec)
            mel_spec = self.freq_masking(mel_spec)
            mel_spec = self.time_masking(mel_spec)
            mel_spec = self.time_masking(mel_spec)

        if self.f0_contours is not None:
            f0_seq = self.f0_contours[idx].unsqueeze(0)  # (1, 32)
            return mel_spec, f0_seq, label

        return mel_spec, label


# ==========================================
# 7. model definition
# ==========================================

# ----------------------
# 7.1 temporal attention mechanism
# ----------------------
class TemporalAttention(nn.Module):
    def __init__(self, input_dim=256, attention_hidden_size=128):
        super().__init__()
        self.attention_fc1 = nn.Linear(input_dim, attention_hidden_size)
        self.attention_fc2 = nn.Linear(attention_hidden_size, 1)

    def forward(self, x):
        attn_scores = torch.tanh(self.attention_fc1(x))
        attn_scores = self.attention_fc2(attn_scores)
        attn_weights = F.softmax(attn_scores, dim=1)
        weighted_output = torch.sum(x * attn_weights, dim=1)
        return weighted_output


# ----------------------
# 7.2 CRNN 主干与特征提取器 (专家 A - 轻量化版 50w参数)
# ----------------------
class CRNNBackbone(nn.Module):
    """
    轻量化 CNN+GRU 主干 (LightCRNN 架构)
    - 参数量: ~50w (原版 ~130w)
    - CNN 通道: [24, 48, 96, 192] (减半)
    - RNN 隐藏层: 80 (微调)
    - 输出维度: 160 (80*2)
    """
    def __init__(self, input_height=128, input_width=512,
                 rnn_hidden_size=80,      # 修改点：从 96 降为 80
                 rnn_num_layers=2,
                 dropout_rate=0.5,
                 use_attention=True,
                 attention_hidden_size=64): # 修改点：对应减小
        super().__init__()
        self.use_attention = use_attention

        # === CNN 部分：通道数减半 [24, 48, 96, 192] ===

        # Block 1: 1 -> 24
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )

        # Block 2: 24 -> 48
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )

        # Block 3: 48 -> 96
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )

        # Block 4: 96 -> 192
        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )

        # === 关键层：RNN 输入维度 ===
        # 对应最后一个 Conv 的通道数
        self.rnn_input_size = 192

        # === RNN 部分 ===
        self.gru = nn.GRU(
            input_size=self.rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=dropout_rate if rnn_num_layers > 1 else 0,
            bidirectional=True
        )

        self.gru_output_size = rnn_hidden_size * 2 # 80 * 2 = 160

        if use_attention:
            self.attention = TemporalAttention(
                input_dim=self.gru_output_size,
                attention_hidden_size=attention_hidden_size
            )

        self.output_dim = self.gru_output_size  # 160

    def forward_features(self, x):
        # x: (B, 1, 128, 512)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x shape: (B, 192, 8, T)

        # === 频率平均池化 ===
        x = torch.mean(x, dim=2) # (B, 192, T)
        x = x.permute(0, 2, 1)   # (B, T, 192)

        self.gru.flatten_parameters()
        x, _ = self.gru(x)

        if self.use_attention:
            x = self.attention(x)
        else:
            x = torch.mean(x, dim=1)

        return x  # (batch, 160)


class CRNNFeatureExtractor(CRNNBackbone):
    """
    CRNN 特征提取器 (时间序列专家) - LightCRNN 架构

    输入: (batch, 1, 128, 512)
    输出: (batch, 160) - 160 维特征向量 (80*2 双向GRU)
    参数量: ~50w (轻量化)
    """
    def __init__(self, input_height=128, input_width=512,
                 rnn_hidden_size=80, rnn_num_layers=2, dropout_rate=0.5,
                 use_attention=True, attention_hidden_size=64):
        super().__init__(input_height=input_height,
                         input_width=input_width,
                         rnn_hidden_size=rnn_hidden_size,
                         rnn_num_layers=rnn_num_layers,
                         dropout_rate=dropout_rate,
                         use_attention=use_attention,
                         attention_hidden_size=attention_hidden_size)

    def forward(self, x):
        return self.forward_features(x)


# ----------------------
# 7.3 MobileNetV3-Small 特征提取器 (576维)
# ----------------------
class MobileNetV3FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            self.backbone, _ = get_or_download_pretrained_mobilenetv3()
        else:
            self.backbone = timm.create_model(
                'mobilenetv3_small_100',
                pretrained=False,
                num_classes=0,
                in_chans=1
            )

        self.output_dim = 576
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feat_map = self.backbone.forward_features(x)
        x = self.global_pool(feat_map)
        x = x.flatten(1)
        return x


# ----------------------
# 7.4 F0 特征提取器
# ----------------------
class TinyF0Expert(nn.Module):
    def __init__(self, output_dim=32):
        super().__init__()

        self.output_dim = output_dim

        # 使用一维卷积捕捉旋律局部模式
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # 时间降采样 32 -> 16

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # 16 -> 8

        # 全局平均池化聚合旋律
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (batch, 1, 32)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)  # (batch, 16, 16)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)  # (batch, 32, 8)

        x = self.global_avg_pool(x)  # (batch, 32, 1)
        x = x.view(x.size(0), -1)    # (batch, 32)
        return x


# ----------------------
# 7.5 融合分类器
# ----------------------
class FusionClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_classes=8, dropout_rate=0.5):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


# ----------------------
# 7.6 特征融合集成模型
# ----------------------
class FeatureFusionEnsemble(nn.Module):
    def __init__(self, num_classes=8, freeze_experts=True, dropout_rate_fusion=0.5,
                 use_attention=True, use_f0=True, f0_weight=0.1):
        super().__init__()

        self.use_f0 = use_f0
        self.f0_weight = f0_weight  # F0 特征注入权重
        # 仅对指定类别注入 F0（默认 Married 和 Feeling）
        self.f0_target_classes = [SONG_TO_LABEL.get('Married', None), SONG_TO_LABEL.get('Feeling', None)]
        self.f0_target_classes = [c for c in self.f0_target_classes if c is not None]

        # 专家 A: CRNN (时间序列专家) - LightCRNN 架构
        self.crnn_expert = CRNNFeatureExtractor(
            input_height=128,
            input_width=512,
            rnn_hidden_size=80,  # LightCRNN: 80 -> 160维输出 (~50w参数)
            rnn_num_layers=2,
            dropout_rate=dropout_rate_fusion,
            use_attention=use_attention,
            attention_hidden_size=64  # 对应调整
        )

        self.mobilenet_expert = MobileNetV3FeatureExtractor(pretrained=True)
        if use_f0:
            self.f0_expert = TinyF0Expert(output_dim=32)
            self.f0_head = nn.Sequential(
                nn.BatchNorm1d(self.f0_expert.output_dim),
                nn.Dropout(dropout_rate_fusion),
                nn.Linear(self.f0_expert.output_dim, num_classes)
            )
        else:
            self.f0_expert = None
            self.f0_head = None

        self.fusion_classifier = FusionClassifier(
            input_dim=self.crnn_expert.output_dim + self.mobilenet_expert.output_dim,
            hidden_dim=128,
            num_classes=num_classes,
            dropout_rate=dropout_rate_fusion
        )

        self.freeze_experts = freeze_experts

    def freeze_expert_parameters(self):
        for param in self.crnn_expert.parameters():
            param.requires_grad = False
        for param in self.mobilenet_expert.parameters():
            param.requires_grad = False
        print("✓ 已冻结专家模型参数")

    def unfreeze_expert_parameters(self):
        for param in self.crnn_expert.parameters():
            param.requires_grad = True
        for param in self.mobilenet_expert.parameters():
            param.requires_grad = True
        print("✓ 已解冻专家模型参数")

    def load_expert_weights(self, crnn_path=None, mobilenet_path=None, device=None):
        if device is None:
            device = DEVICE
        if crnn_path and os.path.exists(crnn_path):
            print(f"加载 CRNN 权重: {crnn_path}")
            checkpoint = torch.load(crnn_path, map_location=device, weights_only=False)
            state_dict = checkpoint['model_state_dict']
            feature_extractor_dict = {}
            for key, value in state_dict.items():
                if not any(x in key for x in ['fc1', 'fc2', 'bn_fc1', 'bn_fc2']):
                    feature_extractor_dict[key] = value
            self.crnn_expert.load_state_dict(feature_extractor_dict, strict=False)
            print(f"  ✓ CRNN 加载完成")

        if mobilenet_path and os.path.exists(mobilenet_path):
            print(f"加载 MobileNetV3 权重: {mobilenet_path}")
            checkpoint = torch.load(mobilenet_path, map_location=device, weights_only=False)
            state_dict = checkpoint['model_state_dict']
            feature_extractor_dict = {}
            for key, value in state_dict.items():
                if key.startswith('backbone.'):
                    feature_extractor_dict[key] = value
            self.mobilenet_expert.load_state_dict(feature_extractor_dict, strict=False)
            print(f"  ✓ MobileNetV3 加载完成")

    def forward(self, x_mel, x_f0=None):
        crnn_features = self.crnn_expert(x_mel)
        mobilenet_features = self.mobilenet_expert(x_mel)
        fused_base = torch.cat([crnn_features, mobilenet_features], dim=1)
        base_logits = self.fusion_classifier(fused_base)

        if self.use_f0 and x_f0 is not None and self.f0_expert is not None and self.f0_head is not None and len(self.f0_target_classes) > 0:
            f0_features = self.f0_expert(x_f0)
            f0_logits = self.f0_head(f0_features)
            mask = torch.zeros_like(f0_logits)
            for c in self.f0_target_classes:
                mask[:, c] = 1.0
            f0_logits = f0_logits * mask
            output = base_logits + self.f0_weight * f0_logits
        else:
            output = base_logits

        return output


# ----------------------
# 7.7 CRNN 分类器 (LightCRNN)
# ----------------------
class CRNNClassifier(CRNNBackbone):
    def __init__(self, num_classes=8, input_height=128, input_width=512,
                 rnn_hidden_size=80,    # 默认改为 80
                 rnn_num_layers=2,
                 dropout_rate=0.5,      # 既然模型小了，Dropout可以稍微降低一点（0.7 -> 0.5）
                 use_attention=True,
                 attention_hidden_size=64):
        super().__init__(input_height=input_height,
                         input_width=input_width,
                         rnn_hidden_size=rnn_hidden_size,
                         rnn_num_layers=rnn_num_layers,
                         dropout_rate=dropout_rate,
                         use_attention=use_attention,
                         attention_hidden_size=attention_hidden_size)

        # 分类头 (输入维度: 160)
        self.fc1 = nn.Linear(self.gru_output_size, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.forward_features(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


# ----------------------
# 7.8 MobileNetV3 分类器
# ----------------------
class AudioMobileNetV3(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()

        if pretrained:
            # 使用缓存管理的预训练权重
            self.backbone, _ = get_or_download_pretrained_mobilenetv3()
        else:
            # 随机初始化
            self.backbone = timm.create_model(
                'mobilenetv3_small_100',
                pretrained=False,
                num_classes=0,
                in_chans=1
            )

        self.num_features = 576
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.BatchNorm1d(self.num_features),
            nn.Dropout(0.5),
            nn.Linear(self.num_features, num_classes)
        )

    def forward(self, x):
        feat_map = self.backbone.forward_features(x)
        features = self.global_pool(feat_map).flatten(1)
        return self.head(features)




# ==========================================
# 8. 频谱图批量提取
# ==========================================
def extract_spectrograms_from_dataframe(df, save_path=None, target_frames=128, target_freq_bins=128,
                                       verbose=False, n_jobs=1):
    """从 DataFrame 批量提取频谱图数据"""
    print(f"\n开始提取 {len(df)} 个文件的频谱图...")
    print(f"目标尺寸: ({target_freq_bins}, {target_frames})")

    def _worker(row_dict):
        file_path = row_dict['file_path']
        label = row_dict['label']
        idx = row_dict['idx']

        spec = extract_mel_spectrogram_torchaudio(file_path, target_frames, target_freq_bins, verbose=verbose)
        if spec is not None:
            return (spec, label, idx)
        return None

    tasks = []
    for idx, row in df.iterrows():
        tasks.append({
            'file_path': row['file_path'],
            'label': row['label'],
            'idx': idx
        })

    if n_jobs == 1:
        results = []
        for task in tqdm(tasks, desc="提取频谱图"):
            result = _worker(task)
            if result is not None:
                results.append(result)
    else:
        print(f"使用并行计算: n_jobs={n_jobs}")
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_worker)(task) for task in tqdm(tasks, desc="提取频谱图")
        )
        results = [r for r in results if r is not None]

    if len(results) == 0:
        print("错误: 没有成功提取任何频谱图!")
        return None, None, None

    spectrograms = []
    labels = []
    valid_indices = []

    for spec, label, idx in results:
        spectrograms.append(spec)
        labels.append(label)
        valid_indices.append(idx)

    spectrograms = np.array(spectrograms)
    labels = np.array(labels)
    valid_df = df.loc[valid_indices].reset_index(drop=True)

    print(f"\n频谱图提取完成!")
    print(f"  成功: {len(spectrograms)} 个文件")
    print(f"  失败: {len(df) - len(spectrograms)} 个文件")
    print(f"  频谱图形状: {spectrograms.shape}")

    if save_path:
        np.savez(save_path,
                 spectrograms=spectrograms,
                 labels=labels,
                 file_ids=valid_df['file_id'].values)
        print(f"频谱图已保存到: {save_path}")

    return spectrograms, labels, valid_df


def extract_dual_spectrograms_from_dataframe(df, save_path=None, mel_target_frames=512,
                                            mel_target_freq_bins=128, f0_target_frames=32,
                                            verbose=False, n_jobs=1):
    """
    从 DataFrame 批量提取 Mel 频谱图与 F0 序列

    返回:
        mel_spectrograms: numpy array (n_samples, 128, 512)
        f0_contours: numpy array (n_samples, 32)
        labels: numpy array (n_samples,)
        valid_df: 成功提取的文件对应的 DataFrame
    """
    print(f"\n开始提取 {len(df)} 个文件的 Mel + F0 特征...")
    print(f"  Mel 目标尺寸: ({mel_target_freq_bins}, {mel_target_frames})")
    print(f"  F0 目标长度: {f0_target_frames} (仅 Married/Feeling 提取真实值，其他为零)")

    def _worker(row_dict):
        file_path = row_dict['file_path']
        label = row_dict['label']
        idx = row_dict['idx']
        song = row_dict.get('song', None)

        # 提取 Mel 频谱图
        mel_spec = extract_mel_spectrogram_torchaudio(
            file_path, mel_target_frames, mel_target_freq_bins, verbose=verbose
        )

        if mel_spec is None:
            return None

        # 仅对 Married(5) 和 Feeling(7) 提取真实 F0
        married_label = SONG_TO_LABEL.get('Married', -1)
        feeling_label = SONG_TO_LABEL.get('Feeling', -1)

        if label in [married_label, feeling_label]:
            # 提取真实 F0 序列
            f0_seq = extract_f0_contour_pyworld(
                file_path, target_frames=f0_target_frames, verbose=verbose
            )
            if f0_seq is None:
                return None
        else:
            # 其他歌曲：使用全零占位符
            f0_seq = np.zeros(f0_target_frames, dtype=np.float32)

        return (mel_spec, f0_seq, label, idx)

    # 准备任务
    tasks = []
    for idx, row in df.iterrows():
        tasks.append({
            'file_path': row['file_path'],
            'label': row['label'],
            'song': row.get('song', None),
            'idx': idx
        })

    # 并行/串行执行
    if n_jobs == 1:
        results = []
        for task in tqdm(tasks, desc="提取双特征"):
            result = _worker(task)
            if result is not None:
                results.append(result)
    else:
        print(f"使用并行计算: n_jobs={n_jobs}")
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_worker)(task) for task in tqdm(tasks, desc="提取双特征")
        )
        results = [r for r in results if r is not None]

    if len(results) == 0:
        print("错误: 没有成功提取任何频谱图!")
        return None, None, None, None

    # 整理结果
    mel_spectrograms = []
    f0_contours = []
    labels = []
    valid_indices = []

    for mel_spec, f0_seq, label, idx in results:
        mel_spectrograms.append(mel_spec)
        f0_contours.append(f0_seq)
        labels.append(label)
        valid_indices.append(idx)

    mel_spectrograms = np.array(mel_spectrograms)
    f0_contours = np.array(f0_contours)
    labels = np.array(labels)
    valid_df = df.loc[valid_indices].reset_index(drop=True)

    print(f"\n双特征提取完成!")
    print(f"  成功: {len(mel_spectrograms)} 个文件")
    print(f"  失败: {len(df) - len(mel_spectrograms)} 个文件")
    print(f"  Mel 频谱图形状: {mel_spectrograms.shape}")
    print(f"  F0 轮廓形状: {f0_contours.shape}")

    # 统计真实 F0 提取数量
    married_label = SONG_TO_LABEL.get('Married', -1)
    feeling_label = SONG_TO_LABEL.get('Feeling', -1)
    target_count = np.sum((labels == married_label) | (labels == feeling_label))
    print(f"  F0 提取: {target_count} 个目标歌曲 (Married + Feeling), "
          f"{len(labels) - target_count} 个使用零占位符")

    # 保存
    if save_path:
        np.savez(save_path,
                 mel_spectrograms=mel_spectrograms,
                 f0_contours=f0_contours,
                 labels=labels,
                 file_ids=valid_df['file_id'].values)
        print(f"双特征已保存到: {save_path}")

    return mel_spectrograms, f0_contours, labels, valid_df


# ==========================================
# 9. This is a utility function to clear the cache
# ==========================================
def clear_cache(device=None, verbose=False):
    """
    Clear Python GC and CUDA cache.
    默认静默；仅在 verbose=True 时打印（且只打印一次），避免训练循环污染输出。

    Args:
        device: Target device, uses global DEVICE if None
        verbose: 是否打印一次性提示
    """
    import gc
    gc.collect()

    # Use global DEVICE if device is not specified
    if device is None:
        device = DEVICE

    use_cuda = (device == 'cuda' and torch.cuda.is_available())

    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # 仅在 verbose=True 时提示，且只提示一次
    if verbose and os.environ.get("CACHE_CLEAR_INFO_PRINTED") != "1":
        print("[Cache] cleared (GC + CUDA cache)")
        os.environ["CACHE_CLEAR_INFO_PRINTED"] = "1"


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ==========================================
# 10. 训练函数
# ==========================================
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001,
                device=None, early_stopping_patience=10, model_name='model',
                class_weights=None, freeze_experts=False):
    if device is None:
        device = DEVICE

    print(f"\n{'='*60}")
    print(f"训练 {model_name}")
    print(f"{'='*60}")

    clear_cache(device=device, verbose=False)

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    model = model.to(device)

    if freeze_experts and hasattr(model, 'freeze_expert_parameters'):
        model.freeze_expert_parameters()

    params_to_optimize = model.parameters()
    if freeze_experts and hasattr(model, 'fusion_classifier'):
        trainable_params = []
        trainable_params.extend(list(model.fusion_classifier.parameters()))
        if getattr(model, 'use_f0', False):
            if model.f0_expert: trainable_params.extend(list(model.f0_expert.parameters()))
            if model.f0_head: trainable_params.extend(list(model.f0_head.parameters()))
        params_to_optimize = trainable_params

    total_p, train_p = count_parameters(model)
    print(f"模型参数: 总计 {total_p:,}, 可训练 {train_p:,}")

    weight_tensor = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(params_to_optimize, lr=learning_rate, weight_decay=CONFIG.training.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=CONFIG.training.scheduler_mode,
        factor=CONFIG.training.scheduler_factor,
        patience=CONFIG.training.scheduler_patience
    )

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            if len(batch) == 3:
                x, extra, y = batch
                x, extra, y = x.to(device), extra.to(device), y.to(device)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                extra = None

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                if extra is not None and getattr(model, 'use_f0', False):
                     outputs = model(x, extra)
                else:
                     outputs = model(x)
                loss = criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    x, extra, y = batch
                    x, extra, y = x.to(device), extra.to(device), y.to(device)
                else:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    extra = None

                if extra is not None and getattr(model, 'use_f0', False):
                     outputs = model(x, extra)
                else:
                     outputs = model(x)

                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_acc)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or is_best:
            best_marker = " ✓ [BEST]" if is_best else ""
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train: {train_loss:.4f}, {train_acc:.2f}% | "
                  f"Val: {val_loss:.4f}, {val_acc:.2f}%{best_marker}")

        clear_cache(device=device, verbose=False)

        if patience_counter >= early_stopping_patience:
            print(f"\n早停触发！")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n恢复最佳模型 (验证: {best_val_acc:.2f}%)")

    return model, history




def evaluate_model(model, test_loader, label_to_song, device='cuda'):
    """评估模型"""
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    model = model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            # 自动检测是否有 F0 特征
            if len(batch) == 3:
                # (mel_spec, f0_seq, label) 格式
                mel_spec, f0_seq, labels = batch
                mel_spec = mel_spec.to(device)
                f0_seq = f0_seq.to(device)
                labels = labels.to(device)

                # 仅当模型支持 F0 时才传递
                if getattr(model, 'use_f0', False):
                    outputs = model(mel_spec, f0_seq)
                else:
                    outputs = model(mel_spec)
            else:
                # (mel_spec, label) 格式
                mel_spec, labels = batch
                mel_spec = mel_spec.to(device)
                labels = labels.to(device)
                outputs = model(mel_spec)

            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    test_acc = 100.0 * accuracy_score(all_labels, all_predictions)

    print("\n" + "="*60)
    print("测试集评估结果")
    print("="*60)
    print(f"测试准确率: {test_acc:.2f}%")

    target_names = [label_to_song[i] for i in sorted(label_to_song.keys())]
    print("\n分类报告:")
    print(classification_report(all_labels, all_predictions,
                               target_names=target_names, zero_division=0))

    print("混淆矩阵:")
    print(confusion_matrix(all_labels, all_predictions))

    return test_acc, all_predictions, all_labels


def plot_training_history(history, save_path='training_history.png'):
    """绘制训练曲线"""
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=4)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2, marker='s', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2, marker='o', markersize=4)
    axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_path}")
    plt.close()


# ==========================================
# 11. 主流程：互补特征融合集成
# ==========================================
def run_feature_fusion_ensemble_pipeline(config: Config = None,
                                          data_dir=None, augment=True,
                                          batch_size=None, num_epochs_experts=None, num_epochs_fusion=None,
                                          learning_rate=None, n_jobs=None, force_regenerate=False,
                                          skip_expert_training=False,
                                          crnn_model_path='crnn_model_mel_torchaudio.pth',
                                          mobilenet_model_path='mobilenetv3_model_mel_torchaudio.pth',
                                          dropout_rate_fusion=None,
                                          dropout_rate_expert=None,
                                          use_attention=None,
                                          use_f0=False):
    """
    互补特征融合集成完整流程

    新策略：先划分数据集，只对训练集进行增强到 5 倍，验证/测试集保持原始

    参数:
        config: Config 对象（推荐使用，可以统一管理所有参数）
        data_dir: 数据目录（如果为 None，从 config 或全局配置获取）
        augment: 是否使用数据增强（仅对训练集，自动增强到 5 倍）
        batch_size: 批次大小（如果为 None，从 config 获取）
        num_epochs_experts: 专家模型训练轮数（如果为 None，从 config 获取）
        num_epochs_fusion: 融合分类器训练轮数（如果为 None，从 config 获取）
        learning_rate: 学习率（如果为 None，从 config 获取）
        n_jobs: 并行进程数（如果为 None，从 config 获取）
        force_regenerate: 是否强制重新生成频谱图
        skip_expert_training: 是否跳过专家训练（使用预训练权重）
        crnn_model_path: CRNN 模型权重路径
        mobilenet_model_path: MobileNetV3-Small 模型权重路径
        dropout_rate_fusion: 融合分类器的 dropout 率（如果为 None，从 config 获取）
        dropout_rate_expert: 专家模型的 dropout 率（如果为 None，从 config 获取）
        use_attention: 是否使用注意力机制（如果为 None，从 config 获取）
        use_f0: 是否使用 F0 特征
    """
    # 参数处理
    config = config or CONFIG
    data_dir = data_dir or config.data.data_dir
    batch_size = batch_size or config.training.batch_size
    num_epochs_experts = num_epochs_experts or config.training.num_epochs_experts
    num_epochs_fusion = num_epochs_fusion or config.training.num_epochs_fusion
    learning_rate = learning_rate or config.training.learning_rate
    n_jobs = n_jobs or config.data.n_jobs or max(1, os.cpu_count() or 1)
    dropout_rate_fusion = dropout_rate_fusion or config.model.fusion_dropout_rate
    dropout_rate_expert = dropout_rate_expert or config.model.crnn_dropout_rate
    use_attention = use_attention if use_attention is not None else config.model.crnn_use_attention

    log_section("Feature Fusion Ensemble Pipeline")
    config.print_config()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clear_cache(device=device, verbose=False)

    # Step 1: 加载数据
    log_section("Step 1: Load Data")
    files = glob.glob(os.path.join(data_dir, '**/*.wav'), recursive=True)
    df_original = parse_filenames(files)

    # Step 2: 划分数据集
    log_section("Step 2: Split by Participant")
    splits = split_by_participant(df_original, train_ratio=0.7, val_ratio=0.15, seed=RANDOM_SEED)
    df_train_original = df_original.iloc[splits['train']].reset_index(drop=True)
    df_val_original = df_original.iloc[splits['val']].reset_index(drop=True)
    df_test_original = df_original.iloc[splits['test']].reset_index(drop=True)
    print(f"Train: {len(df_train_original)}, Val: {len(df_val_original)}, Test: {len(df_test_original)}")

    # Step 3: 数据增强
    log_section("Step 3: Data Augmentation")
    if augment:
        augmented_output_dir = r"C:\Users\guson\Desktop\ml_project\Data\augmented"
        train_target_total = len(df_train_original) * 5
        df_train_augmented = augment_dataset(df=df_train_original, output_dir=augmented_output_dir,
                                             target_total=train_target_total, verbose=False, n_jobs=n_jobs)
        print(f"Train augmented: {len(df_train_original)} -> {len(df_train_augmented)} ({len(df_train_augmented) / len(df_train_original):.1f}x)")
    else:
        df_train_augmented = df_train_original

    # Step 4: 提取特征
    log_section("Step 4: Feature Extraction")
    print("[1/3] Train features...")
    def _extract_features(df, name):
        print(f"[{name}]...")
        if use_f0:
            specs, f0s, labels, _ = extract_dual_spectrograms_from_dataframe(
                df=df, save_path=None, mel_target_frames=512, mel_target_freq_bins=128,
                f0_target_frames=32, verbose=False, n_jobs=n_jobs)
        else:
            specs, labels, _ = extract_spectrograms_from_dataframe(
                df=df, save_path=None, target_frames=512, target_freq_bins=128, verbose=False, n_jobs=n_jobs)
            f0s = None
        return specs, f0s, labels

    train_spectrograms, train_f0_contours, train_labels = _extract_features(df_train_augmented, "Train")
    print("[2/3] Val features...")
    val_spectrograms, val_f0_contours, val_labels = _extract_features(df_val_original, "Val")
    print("[3/3] Test features...")
    test_spectrograms, test_f0_contours, test_labels = _extract_features(df_test_original, "Test")

    if train_spectrograms is None or val_spectrograms is None or test_spectrograms is None:
        print("Error: Feature extraction failed")
        return
    print(f"Features: Train={train_spectrograms.shape}, Val={val_spectrograms.shape}, Test={test_spectrograms.shape}")

    # Step 5: DataLoader
    log_section("Step 5: Create DataLoader")
    train_dataset = UnifiedAudioDataset(train_spectrograms, train_labels, f0_contours=train_f0_contours,
                                        apply_spec_augment=True, freq_mask_param=15, time_mask_param=35)
    val_dataset = UnifiedAudioDataset(val_spectrograms, val_labels, f0_contours=val_f0_contours, apply_spec_augment=False)
    test_dataset = UnifiedAudioDataset(test_spectrograms, test_labels, f0_contours=test_f0_contours, apply_spec_augment=False)

    loader_kwargs = dict(batch_size=batch_size, num_workers=config.training.num_workers,
                        pin_memory=config.training.pin_memory if torch.cuda.is_available() else False)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    print(f"DataLoader: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    num_classes = len(np.unique(train_labels))
    class_weights = None

    # Step 6: 训练专家模型
    early_stop = config.training.early_stopping_patience or 10
    if not skip_expert_training:
        log_section("Step 6: Train Expert Models")

        # MobileNetV3
        mobilenet_model = AudioMobileNetV3(num_classes=num_classes, pretrained=config.model.mobilenet_pretrained)
        mobilenet_model, mobilenet_history = train_model(
            mobilenet_model, train_loader, val_loader, num_epochs_experts, learning_rate,
            device, early_stop, 'MobileNetV3', class_weights)
        torch.save({'model_state_dict': mobilenet_model.state_dict(), 'num_classes': num_classes}, mobilenet_model_path)
        mobilenet_test_acc, _, _ = evaluate_model(mobilenet_model, test_loader, LABEL_TO_SONG, device)

        # CRNN
        crnn_model = CRNNClassifier(
            num_classes, config.audio.n_mels, config.audio.max_length,
            config.model.crnn_rnn_hidden_size, config.model.crnn_rnn_num_layers,
            dropout_rate_expert, use_attention, config.model.crnn_attention_hidden_size)
        crnn_model, crnn_history = train_model(
            crnn_model, train_loader, val_loader, num_epochs_experts, learning_rate,
            device, early_stop, 'CRNN', class_weights)
        torch.save({'model_state_dict': crnn_model.state_dict(), 'num_classes': num_classes}, crnn_model_path)
        crnn_test_acc, _, _ = evaluate_model(crnn_model, test_loader, LABEL_TO_SONG, device)
        print(f"Expert Acc: MobileNet={mobilenet_test_acc:.2f}%, CRNN={crnn_test_acc:.2f}%")

    # Step 7: 融合模型
    log_section("Step 7: Fusion Ensemble")
    ensemble_model = FeatureFusionEnsemble(
        num_classes, config.model.freeze_experts, dropout_rate_fusion, use_attention, use_f0)
    ensemble_model.load_expert_weights(crnn_model_path, mobilenet_model_path, device)

    total_p, train_p = count_parameters(ensemble_model)
    print(f"Ensemble: Total={total_p:,}, Trainable={train_p:,}")

    ensemble_model, fusion_history = train_model(
        ensemble_model, train_loader, val_loader, num_epochs_fusion, learning_rate * 2,
        device, config.training.early_stopping_patience or 5, 'Fusion', class_weights, config.model.freeze_experts)

    # Step 8: 评估
    log_section("Step 8: Evaluation")
    ensemble_test_acc, predictions, true_labels = evaluate_model(ensemble_model, test_loader, LABEL_TO_SONG, device)

    # Step 9: 保存
    ensemble_save_path = 'feature_fusion_ensemble_model.pth'
    torch.save({'model_state_dict': ensemble_model.state_dict(), 'num_classes': num_classes,
                'test_acc': ensemble_test_acc, 'history': fusion_history}, ensemble_save_path)
    print(f"Model saved: {ensemble_save_path}")

    # 绘制训练曲线
    plot_training_history(fusion_history, save_path='training_history_fusion_ensemble.png')

    print(f"\n{'='*60}\nDone! Test Acc: {ensemble_test_acc:.2f}%\n{'='*60}")
    return ensemble_model, fusion_history, ensemble_test_acc


# ==================== 入口 ====================
if __name__ == '__main__':
    run_feature_fusion_ensemble_pipeline(config=CONFIG, augment=True, use_f0=True)

