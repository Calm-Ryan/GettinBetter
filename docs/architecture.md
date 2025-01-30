# システムアーキテクチャ

## システム概要
音声認識システムは、高精度な日本語音声認識を実現するために、最新の技術を組み合わせた3層アーキテクチャを採用しています。

## アーキテクチャ図

### クラス図
```mermaid
classDiagram
    class IAudioProcessor {
        <<interface>>
        +preprocess(file_path: str) -> AudioData
        +split_audio(audio: AudioData) -> List[AudioSegment]
    }

    class AudioProcessor {
        -sample_rate: int
        -max_duration: int
        -vad: WebRTCVAD
        -denoiser: RNNoise
        +preprocess(file_path: str) -> AudioData
        +split_audio(audio: AudioData) -> List[AudioSegment]
        -remove_noise_rnnoise(audio: np.ndarray) -> np.ndarray
        -detect_speech_vad(audio: np.ndarray) -> List[Tuple]
    }

    class AudioData {
        +raw_audio: np.ndarray
        +sample_rate: int
        +duration: float
        +channels: int
        +file_metadata: Dict
        +vad_segments: List[Tuple]
    }

    class AudioSegment {
        +audio: np.ndarray
        +start_time: float
        +end_time: float
        +segment_id: str
        +is_speech: bool
    }

    class ITranscriber {
        <<interface>>
        +transcribe(audio: AudioData) -> TranscriptionResult
        +batch_transcribe(segments: List[AudioSegment]) -> List[TranscriptionResult]
    }

    class WhisperTranscriber {
        -model: WhisperModel
        -device: str
        -batch_size: int
        -use_onnx: bool
        +transcribe(audio: AudioData) -> TranscriptionResult
        +batch_transcribe(segments: List[AudioSegment]) -> List[TranscriptionResult]
        -load_model()
        -convert_to_onnx()
    }

    class TranscriptionResult {
        +text: str
        +start_time: float
        +end_time: float
        +confidence: float
        +speaker_id: Optional[str]
        +segment_id: str
        +words: List[Dict]
    }

    class TranscriptionProcessor {
        +merge_results(results: List[TranscriptionResult]) -> List[TranscriptionResult]
        +filter_low_confidence(results: List[TranscriptionResult]) -> List[TranscriptionResult]
        +format_output(results: List[TranscriptionResult]) -> Dict
    }

    IAudioProcessor <|.. AudioProcessor
    AudioProcessor --> AudioData
    AudioProcessor --> AudioSegment
    ITranscriber <|.. WhisperTranscriber
    WhisperTranscriber --> TranscriptionResult
    TranscriptionProcessor --> TranscriptionResult
```

### シーケンス図
```mermaid
sequenceDiagram
    participant C as Controller
    participant AP as AudioProcessor
    participant WT as WhisperTranscriber
    participant TP as TranscriptionProcessor
    participant DB as Database

    C->>AP: preprocess(file_path)
    activate AP
    Note over AP: WebRTCVADとRNNoiseによる処理
    AP-->>C: audio_data
    deactivate AP

    C->>AP: split_audio(audio_data)
    activate AP
    Note over AP: VADベースのセグメント分割
    AP-->>C: audio_segments
    deactivate AP

    C->>WT: batch_transcribe(audio_segments)
    activate WT
    Note over WT: Faster-Whisperによる認識
    WT-->>WT: load_model()
    loop For each batch
        WT-->>WT: prepare_audio()
        WT-->>WT: run_inference()
    end
    WT-->>C: transcription_results
    deactivate WT

    C->>TP: merge_results(transcription_results)
    activate TP
    TP-->>C: merged_results
    deactivate TP

    C->>TP: filter_low_confidence(merged_results)
    activate TP
    TP-->>C: filtered_results
    deactivate TP

    C->>TP: format_output(filtered_results)
    activate TP
    TP-->>C: formatted_output
    deactivate TP

    C->>DB: save_transcription(formatted_output)
    activate DB
    DB-->>C: success
    deactivate DB
```

## コンポーネント詳細

### 1. 音声処理層（AudioProcessor）
音声の前処理と分割を担当する層です。

#### 主要機能
- 音声ファイルの読み込みと正規化
- WebRTCVADによる音声区間検出
- RNNoiseによるノイズ除去
- 音声セグメントへの分割

#### 技術選定理由
- WebRTCVAD: 高精度なリアルタイム音声検出が可能
- RNNoise: 深層学習ベースの効率的なノイズ除去

### 2. 認識層（WhisperTranscriber）
音声認識を実行する中核層です。

#### 主要機能
- Faster-Whisperモデルの管理
- バッチ処理による効率的な認識
- ONNXランタイムによる最適化
- 日本語特化の設定

#### 技術選定理由
- Faster-Whisper: オリジナルWhisperの2-3倍の処理速度
- ONNX: 推論の高速化と最適化

### 3. 後処理層（TranscriptionProcessor）
認識結果の整形と最適化を担当する層です。

#### 主要機能
- 認識結果のマージ
- 低信頼度結果のフィルタリング
- 出力フォーマットの整形
- 日本語テキストの正規化

## データフロー

1. 入力フェーズ
   - 音声ファイルの読み込み
   - メタデータの抽出
   - 形式の正規化

2. 前処理フェーズ
   - ノイズ除去
   - 音声区間検出
   - セグメント分割

3. 認識フェーズ
   - モデルのロード
   - バッチ処理による認識
   - 信頼度スコアの計算

4. 後処理フェーズ
   - 結果のマージ
   - フィルタリング
   - フォーマット変換

## エラーハンドリング

### 実装方針
- 各層で専用の例外クラスを定義
- 適切なエラーメッセージとログ出力
- リトライ機構の実装
- グレースフルデグラデーション

### エラー種別
1. 入力エラー
   - ファイル不存在
   - フォーマット不正
   - メタデータ不正

2. 処理エラー
   - メモリ不足
   - GPU関連エラー
   - モデルロードエラー

3. システムエラー
   - リソース枯渇
   - 並列処理エラー
   - I/Oエラー

## 設定管理

### 設定ファイル構成
- config.yaml: メインの設定ファイル
- logging.yaml: ログ設定ファイル

### 主要設定項目
- 音声処理パラメータ
- モデル設定
- 最適化オプション
- エラーハンドリング設定

## 拡張性

### 拡張ポイント
1. 音声処理
   - 新しい前処理アルゴリズムの追加
   - 異なるVADの実装

2. 認識エンジン
   - 新しいモデルの追加
   - 異なる推論エンジンの実装

3. 後処理
   - カスタム出力フォーマット
   - 追加の後処理ステップ

### インターフェース
- IAudioProcessor
- ITranscriber
- カスタムイベントハンドラ