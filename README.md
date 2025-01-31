# GettinBetter - 商談分析・評価システム

## 概要
GettinBetterは、商談の音声データや書き起こしテキストを解析し、営業担当者へのフィードバックや成功確率のスコアリングを提供する分析システムです。音声認識（ASR）、自然言語処理（NLP）、機械学習を組み合わせて、商談の質的・量的な評価を自動化します。

### 主な機能
- 商談音声の高精度な文字起こし（Whisper ASR）
- テキスト解析による重要トピックの抽出
- センチメント分析による顧客満足度評価
- 機械学習モデルによる成約確率予測
- 分析結果のダッシュボード表示

## 開発状況
現在はフェーズ0/1（MVP構築段階）にあり、以下の機能を実装中です：
- ✅ 音声入力・管理システム
- ✅ Whisperによる音声認識
- ✅ 基本的なテキスト解析
- ✅ ストレージシステム（MinIO/PostgreSQL）
- 🚧 機械学習モデル開発
- 🚧 分析ダッシュボード

## システムアーキテクチャ
システムは以下の主要コンポーネントで構成されています：

```
[音声入力・保管] → [ASR・前処理] → [NLP解析] → [ML評価] → [結果表示]
```

詳細なアーキテクチャ説明は[こちら](docs/architecture.md)を参照してください。

## 技術スタック
- **音声処理**: Whisper ASR, WebRTCVAD, RNNoise
- **テキスト解析**: Transformers, SudachiPy
- **機械学習**: XGBoost, scikit-learn
- **ストレージ**: MinIO, PostgreSQL
- **バックエンド**: FastAPI, Python 3.8+
- **可視化**: Jupyter, Plotly

## セットアップ

### システム要件
- Python 3.8以上
- CUDA対応GPU（推奨: VRAM 8GB以上）
- RAM 16GB以上
- ストレージ容量 100GB以上（音声データ保存用）

### インストール
```bash
# 依存パッケージのインストール
pip install -r requirements.txt

# 設定ファイルの準備
cp config/config.yaml.example config/config.yaml
# config.yamlを環境に合わせて編集

# データベースの初期化
psql -f src/storage/migrations/init.sql
```

### 基本的な使用方法
```python
from src.audio import AudioProcessor, WhisperTranscriber
from src.text import TextAnalyzer
from src.storage import StorageFacade
import yaml

# 設定の読み込み
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# コンポーネントの初期化
storage = StorageFacade(config['storage'])
audio_processor = AudioProcessor(config['audio'])
transcriber = WhisperTranscriber(config['whisper'])
analyzer = TextAnalyzer(config['text'])

# 音声ファイルの解析
audio_file = 'meeting.wav'
storage.save_audio(audio_file)
text = transcriber.transcribe(audio_file)
analysis = analyzer.analyze(text)
storage.save_analysis(analysis)
```

## ドキュメント
- [システムアーキテクチャ](docs/architecture.md)
- [APIリファレンス](docs/api.md)
- [パフォーマンス最適化](docs/performance.md)
- [テスト仕様](docs/test.md)

## 開発ロードマップ
1. **フェーズ0/1 (現在)**: MVP構築
   - 基本的な音声認識・テキスト解析パイプライン
   - ストレージシステムの整備
   - 簡易的な分析機能

2. **フェーズ2**: 機能拡張
   - 機械学習モデルの精度向上
   - LLMによる要約・フィードバック生成
   - ダッシュボードの本格実装

3. **フェーズ3**: 高度化
   - リアルタイム解析
   - カスタムLLMの導入
   - 大規模データ処理対応

## ライセンス
MIT License - 詳細は[LICENSE](LICENSE)ファイルを参照してください。
