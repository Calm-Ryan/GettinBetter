# 音声認識システム

## 概要
高性能な音声認識システムで、特に日本語音声の認識に最適化されています。WebRTCVADとRNNoiseによる高度な音声処理、Faster-Whisperによる高速な音声認識、ONNXランタイムによる推論の最適化を特徴としています。

## 主な機能
- 高精度な日本語音声認識
- ノイズに強い音声処理
- 高速な処理と低レイテンシ
- スケーラブルなバッチ処理

## クイックスタート

### インストール
```bash
pip install -r requirements.txt
```

### 基本的な使用方法
```python
from src.audio import AudioProcessor, WhisperTranscriber, TranscriptionProcessor
import yaml

# 設定の読み込み
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# コンポーネントの初期化
audio_processor = AudioProcessor(config['audio_processor'])
transcriber = WhisperTranscriber(config['whisper'])
processor = TranscriptionProcessor(config['transcription'])

# 音声認識の実行
audio_data = audio_processor.preprocess('audio.wav')
segments = audio_processor.split_audio(audio_data)
results = transcriber.batch_transcribe(segments)
output = processor.format_output(results)
```

## システム要件
- Python 3.8以上
- CUDA対応GPU（推奨）
- 8GB以上のRAM
- 8GB以上のGPU VRAM（推奨）

## ドキュメント
詳細なドキュメントは`docs`ディレクトリを参照してください：
- [システムアーキテクチャ](docs/architecture.md)
- [APIリファレンス](docs/api.md)
- [パフォーマンスと最適化](docs/performance.md)

## ライセンス
MIT License
