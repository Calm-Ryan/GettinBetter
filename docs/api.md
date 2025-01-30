# APIリファレンス

## AudioProcessor

### クラス概要
音声ファイルの前処理と分割を担当するクラスです。WebRTCVADとRNNoiseを使用して、高品質な音声処理を実現します。

### メソッド

#### `__init__(config: Dict[str, Any])`
AudioProcessorクラスのインスタンスを初期化します。

##### パラメータ
- `config`: 設定パラメータを含む辞書
  - `sample_rate`: サンプリングレート（Hz）
  - `max_duration`: 最大セグメント長（秒）
  - `vad_mode`: VADの感度（0-3）
  - `vad_frame_ms`: VADフレーム長（ms）
  - `min_speech_duration`: 最小音声区間長（秒）

##### 使用例
```python
config = {
    'sample_rate': 16000,
    'vad_mode': 3,
    'min_speech_duration': 0.3
}
processor = AudioProcessor(config)
```

#### `preprocess(file_path: str) -> AudioData`
音声ファイルの前処理を行います。

##### パラメータ
- `file_path`: 処理する音声ファイルのパス

##### 戻り値
- `AudioData`: 前処理済み音声データ

##### 例外
- `AudioProcessError`: 音声ファイルの読み込みや処理に失敗した場合

##### 使用例
```python
try:
    audio_data = processor.preprocess('audio.wav')
except AudioProcessError as e:
    print(f"処理エラー: {e}")
```

#### `split_audio(audio: AudioData) -> List[AudioSegment]`
音声を適切な長さのセグメントに分割します。

##### パラメータ
- `audio`: 分割する音声データ（AudioDataオブジェクト）

##### 戻り値
- `List[AudioSegment]`: 分割された音声セグメントのリスト

##### 使用例
```python
segments = processor.split_audio(audio_data)
print(f"セグメント数: {len(segments)}")
```

## WhisperTranscriber

### クラス概要
Faster-Whisperモデルによる音声認識を実行するクラスです。ONNXランタイムによる最適化をサポートしています。

### メソッド

#### `__init__(config: Dict[str, Any])`
WhisperTranscriberクラスのインスタンスを初期化します。

##### パラメータ
- `config`: 設定パラメータを含む辞書
  - `model_name`: モデル名
  - `device`: 実行デバイス
  - `compute_type`: 計算精度
  - `batch_size`: バッチサイズ
  - `language`: 言語設定
  - `beam_size`: ビームサイズ
  - `word_timestamps`: 単語レベルのタイムスタンプ有効化
  - `use_onnx`: ONNXランタイム使用フラグ

##### 使用例
```python
config = {
    'model_name': 'large-v3',
    'device': 'cuda',
    'language': 'ja',
    'use_onnx': True
}
transcriber = WhisperTranscriber(config)
```

#### `batch_transcribe(segments: List[AudioSegment]) -> List[TranscriptionResult]`
音声セグメントのバッチ処理による書き起こしを実行します。

##### パラメータ
- `segments`: 音声セグメントのリスト

##### 戻り値
- `List[TranscriptionResult]`: 認識結果のリスト

##### 例外
- `TranscriptionError`: 認識処理に失敗した場合
- `ModelError`: モデル関連のエラーが発生した場合

##### 使用例
```python
try:
    results = transcriber.batch_transcribe(segments)
    for result in results:
        print(f"認識テキスト: {result.text}")
        print(f"信頼度: {result.confidence}")
except TranscriptionError as e:
    print(f"認識エラー: {e}")
```

## TranscriptionProcessor

### クラス概要
認識結果の後処理を担当するクラスです。結果のマージ、フィルタリング、フォーマット変換を行います。

### メソッド

#### `__init__(config: Dict[str, Any])`
TranscriptionProcessorクラスのインスタンスを初期化します。

##### パラメータ
- `config`: 設定パラメータを含む辞書
  - `confidence_threshold`: 信頼度閾値
  - `min_segment_duration`: 最小セグメント長
  - `max_merge_interval`: 最大マージ間隔

##### 使用例
```python
config = {
    'confidence_threshold': 0.6,
    'min_segment_duration': 1.0
}
processor = TranscriptionProcessor(config)
```

#### `merge_results(results: List[TranscriptionResult]) -> List[TranscriptionResult]`
連続する認識結果をマージします。

##### パラメータ
- `results`: マージ対象の認識結果リスト

##### 戻り値
- `List[TranscriptionResult]`: マージ後の認識結果リスト

##### 使用例
```python
merged_results = processor.merge_results(results)
```

#### `filter_low_confidence(results: List[TranscriptionResult], threshold: Optional[float] = None) -> List[TranscriptionResult]`
低信頼度の結果をフィルタリングします。

##### パラメータ
- `results`: フィルタリング対象の認識結果リスト
- `threshold`: 信頼度閾値（省略可）

##### 戻り値
- `List[TranscriptionResult]`: フィルタリング後の認識結果リスト

##### 使用例
```python
filtered_results = processor.filter_low_confidence(results, threshold=0.7)
```

#### `format_output(results: List[TranscriptionResult]) -> Dict[str, Any]`
認識結果を指定された形式に整形します。

##### パラメータ
- `results`: 整形対象の認識結果リスト

##### 戻り値
- `Dict[str, Any]`: 整形された認識結果

##### 使用例
```python
output = processor.format_output(filtered_results)
print(f"総セグメント数: {output['metadata']['total_segments']}")
print(f"平均信頼度: {output['metadata']['average_confidence']}")
```

## データクラス

### AudioData
前処理済み音声データを保持するデータクラスです。

#### 属性
- `raw_audio`: 音声データ（numpy.ndarray）
- `sample_rate`: サンプリングレート
- `duration`: 音声の長さ（秒）
- `channels`: チャンネル数
- `file_metadata`: ファイルのメタデータ
- `vad_segments`: VADで検出した音声区間

### AudioSegment
分割された音声セグメントを表現するデータクラスです。

#### 属性
- `audio`: セグメントの音声データ
- `start_time`: 開始時間
- `end_time`: 終了時間
- `segment_id`: セグメントID
- `is_speech`: 音声区間判定結果

### TranscriptionResult
音声認識結果を格納するデータクラスです。

#### 属性
- `text`: 認識されたテキスト
- `start_time`: 開始時間
- `end_time`: 終了時間
- `confidence`: 信頼度
- `segment_id`: セグメントID
- `speaker_id`: 話者ID（オプション）
- `words`: 単語レベルのタイムスタンプ情報