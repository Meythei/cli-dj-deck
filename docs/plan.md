# 実装計画: cli-dj-deck 実オーディオ化

指示書: [TASK_real-audio.md](TASK_real-audio.md)。判断の詳細は [decisions.md](decisions.md)。

## 0. 指示書との矛盾・動かしてみないと決められない点

着手前の調査(Python 3.13 の仮想環境に依存を入れて実際に叩いた結果)で気づいた点。どれも致命的ではないので、
右の案で進める。3章の原則を変えるものはない。

| # | 点 | 進め方 |
|---|---|---|
| 1 | 指示書4章の「48000 サンプルを 128/124 倍速にして 46500、計算上の約46452」は計算違い。48000×124/128 = 46500 ちょうど。ただし 49548.39→49548、96877.9→96878 のように、出力長は `入力長 / factor` の丸めとは限らない | 長さを揃える処理はそのまま必須として入れる |
| 2 | `py` ランチャーに 3.12/3.13 が入っていなかった | uv で CPython 3.13.14 を入れ、`py -V:Astral/CPython3.13.14 -m venv .venv` で作り直した。README には python.org 版と uv 版の両方を書く |
| 3 | `python -m clidj render ...` はパッケージ化が前提だが、現状はリポジトリ直下のフラットなモジュール | フェーズAのあとに `clidj/` パッケージへ移動する(中身を変えない移動だけのコミット)。`python app.py` は互換のため薄いラッパーとして残す |
| 4 | 波形ピークを「拍基準の解像度で保存」すると、`regrid()` のたびに音声ファイルを読み直すことになる | 解析では時間基準の包絡線(約5.3ms刻み)を保存し、読み込み時に有効なグリッド(overrides 込み)で拍基準 16/拍 に変換する。UI から見える解像度は拍基準のまま |
| 5 | ラウドネス目標が「スニペットの」なのか曲単位なのか曖昧。スニペット単位で揃えるとブレイクや riser がドロップと同じ音量になる | 曲の統合ラウドネスから求めたゲインをスニペットに持たせる(バッファには焼き込まず、エンジンで掛ける) |
| 6 | 「予定サンプルちょうどに開始」と「プツッと鳴らない」は、ボイスの開始・停止に数msのランプを掛けると両立の定義が要る | ランプはその予定サンプルから始め、最初のサンプルから 0 でない値にする(`(k+1)/N`)。テストでは「予定サンプルの1つ前が0で、予定サンプルが非0」を開始位置の定義にする |
| 7 | `mmap_mode="r"` のバッファをコールバックで読むと、ページフォルト(=ディスクI/O)が起き得る | エンジンのコマンド受信スレッドで読み込み時に全ページを一度触ってから deque に積む。キャッシュファイルは内容キーで名前を決めて上書きしない(Windows ではマップ中のファイルを消せないため) |
| 8 | マスターのリミッタ(`pedalboard.BrickwallLimiter`)は先読み分(5ms=240サンプル)出力を遅らせる。オフラインレンダリングでサンプル位置を検証するとずれる | エンジンが遅延量を実測して公開し、オフライン書き出しはその分を詰める。リアルタイムの表示補正では出力レイテンシに加算する |
| 9 | フェーズ C(準備・保留・BPM切替)はフェーズ D のエンジンより前だが、準備待ちの保留は本来エンジンへの送信と絡む | C では `Session` に保留と BPM 切替の制御を実装し、ビジュアルのレーンに適用する。D で適用先をエンジンコマンドに差し替える |
| 10 | `pedalboard.Limiter` は JUCE の Limiter で、-10dB から常に圧縮しメイクアップゲインも掛かる(DJのマスター向きではない) | `BrickwallLimiter`(ルックアヘッド、メイクアップなし)を使い、最後に `np.clip(-1, 1)` を安全策として掛ける |
| 11 | 開発機には WASAPI の出力デバイスがあるが、開発者(Claude)は音を聴けない | 実デバイスではストリームが開けること・コールバックが回ること・アンダーラン数を数値で確認する。聴感で判断するものは `docs/listening-checklist.md` に回す |
| 12 | 10分間のアンダーランゼロ負荷テストを毎回の `pytest` に入れると遅すぎる | `slow` マーカーを付けて既定ではスキップし、短縮版(数秒)を通常テストに入れる。10分版は手元で1回実行して結果を decisions.md に書く |
| 13 | `--demo` のフェイクライブラリには音声ファイルがない | デモ曲の音声を決定的に合成してキャッシュに書き出し、`--demo` でも実際に音が出るようにする(テスト・試聴用WAVの素材も同じ合成器を使う) |
| 14 | 既存テストのうち Scheduler のゲインオートメーションのテストは、オートメーションがエンジンに移るので対象が変わる | 同じ意図(開始・中間・終了のゲイン、同じレーンの上書きで警告)をエンジンのテストとして書き直し、コミットメッセージに理由を書く |

## 全体の形(最終形)

```
UI プロセス                                              エンジンプロセス(spawn)
┌──────────────────────────────────────┐   Queue    ┌──────────────────────────────────┐
│ Textual UI ── Interpreter             │ ─────────▶ │ コマンド受信スレッド              │
│                 │                     │ EngineCmd  │  (バッファを mmap+prefault)       │
│              Session ── Scheduler     │            │        │ deque(atomic)            │
│   Library/Prep  │   (先読みで Thunk 評価)│            │        ▼                          │
│   (ProcessPool) └─ EngineClient       │ ◀───────── │ Engine.process(block) ← Backend   │
│                     (状態読み出し)     │ 共有メモリ │  TempoMap / LaneVoice / Mixer     │
└──────────────────────────────────────┘ (seqlock)  └──────────────────────────────────┘
                 ▲ レンダリング済みバッファは .npy ファイル経由(コピーしない)
```

- 時刻の正はエンジンのサンプルカウンタ、音楽的位置の正は拍。`TempoMap` が拍 ↔ サンプルを変換する
- `--no-audio` と `render` は同じ `Engine` をプロセス内で使う(`--no-audio` は音声のミックスを省略して状態だけ進める)。タイミングの実装を1つにするため

## フェーズA: 既存の問題の修正(フラットなモジュールのまま)

各項目で回帰テストを先に書き、落ちることを確認してから直す。

| 項目 | 変更 | テスト |
|---|---|---|
| 1 ルーラー/波形 | `LanesView` に `WaveAxis(left, width)` を導入し、`beat_to_col` / `col_to_beat` を1か所に。ルーラーと全レーンの波形は同じ列範囲・同じ中央 `width // 2` を使う | 描画結果の文字列で ▼ の列と波形のプレイヘッド列、小節線の列が一致 |
| 2 開始拍 | `Scheduler` が発火中イベントの予定拍(`firing_beat`)を公開。`Lane.start_snippet(snippet, start_beat)`、xf の開始・終了拍もこれを使う | デモセットで L3 が 64.0、L4 が 128.0 ちょうどに開始 |
| 3 スニペット名 | 代入文の右辺が `snip(...)` 呼び出しで `name=` 未指定なら変数名を名前にする | `kick = snip(...)` → `name == "kick"` |
| 4 after 負数 | `after(n<0)` をエラー。`at(n<1)` もエラー | キューに残らず error ログ |
| 5 start 前の at(1) | 停止中で指定位置 == 現在位置なら過去扱いしない。`poll` は `fire_at <= limit` 方式にし、開始時に位置0のイベントを発火 | `at(1, L1<<kick)` → `start()` → 1ティック後に L1 が拍0から再生 |
| 6 every 下限 | `every` の最小間隔 0.25 小節(1拍)、1ポーリングあたりの発火数上限(256)で打ち切ってエラーログ | `every(0.001, ...)` はエラー、上限超過でも固まらない |
| 7 load_set 再帰 | ネスト深さ上限 8 | 自己 load_set でエラー1件、再帰エラーにならない |
| 8 TRACKS 幅 | 列幅の合計+パディングがペインに収まるよう調整 | `run_test` で DataTable の仮想幅 ≤ 表示幅 |
| 9 LANES 高さ | レーン1本 = 情報1行 + 波形2行。ペインは `height: auto` にしてコンソールに残りを渡す。小節頭の列を薄く色付け | 描画行数 = ヘッダ + ルーラー + 3×4 |
| 10 ログ | 予約発火は `[009.1] L2 << bass`、停止中の即時は `(immediate)`、`now()` は `(now)`。「<<」→「now playing」の順。`at/after/every` 予約時に `queued #id @ bar N` | ログ順と表記のテスト |

コミット: 修正の意味単位で 3〜4 コミット。

## パッケージ化(フェーズAとBの間)

`clidj/` に移動: `transport, library, snippets, lanes, scheduler, interpreter` と `ui/app.py, ui/app.tcss`。
`clidj/__main__.py` に CLI(`python -m clidj [--set] [--demo] [--no-audio] [--device] [--list-devices]`、`python -m clidj render ...`)。
テストの import だけを書き換える。

## フェーズB: ライブラリと解析

| モジュール | 役割・主なインターフェース |
|---|---|
| `clidj/config.py` | `Paths`(config/data/cache。`platformdirs`、環境変数 `CLIDJ_HOME` で丸ごと差し替え可能=テスト用)、`Config.load()`(`config.toml` を `tomllib` で読み、既定値とマージ) |
| `clidj/library.py` | `Track`(num, track_id, path, title, artist, duration, analysis)、`track_id_for(path)`(サイズ+先頭/末尾 64KiB の blake2b)、`Library.scan()`(フォルダ走査、番号は初出順で永続化)、`resolve(num/title/id)`。`DEMO_LIBRARY` は残す |
| `clidj/analysis.py` | `analyze_file(path) -> TrackAnalysis`(bpm, first_beat_sec, key, lufs, envelope, duration)。BPM: librosa のオンセット包絡+`beat_track` で粗推定 → 位相ヒストグラムで BPM と位相を精密化。キー: `chroma_cqt` + キープロファイル照合 → Camelot。ラウドネス: BS.1770 を scipy で自前実装 |
| `clidj/overrides.py` | `overrides.json`(bpm/first_beat/key)。`effective_grid(track)` は overrides を優先 |
| `clidj/workers.py` | `WorkerPool`(`ProcessPoolExecutor`、spawn。テストではインライン実行に差し替え) |
| コマンド | `scan()` / `regrid(n, bpm=, offset_ms=)` / `setkey(n, "8A")` |

テスト: 合成クリックトラック(124BPM, 最初の拍 0.37s)で BPM ±0.1・位置 ±10ms、合成和音でキー、1kHz サインで LUFS、ID が移動に強いこと、番号の安定性、overrides の優先。

## フェーズC: スニペットのレンダリング

| モジュール | 役割・主なインターフェース |
|---|---|
| `clidj/render.py` | `RenderSpec`(track_id, path, grid, start_beat, length_beats, loop, set_bpm, sr, renderer_version) → `cache_key()`。`render_snippet(spec) -> np.ndarray`: 余白1拍付きで読み込み(`AudioFile.resampled_to`)→ `time_stretch`(factor==1 はバイパス)→ 余白除去と長さ合わせ → 非ループはフェード、ループは末尾を先頭にクロスフェード。`RenderCache`(`<cache>/renders/<key>.npy`、書き込みは一時ファイル→rename) |
| `clidj/prep.py` | `PrepManager.request(snippet, bpm)`、`state(snippet, bpm)` → `none/pending/ready/failed`、`poll()` で完了を回収してコールバック |
| `clidj/session.py` | インタプリタとエンジン(C ではビジュアルのレーン)の間の制御。`play(lane, snippet, beat)` は準備未完了なら保留して警告し、完了後の最初の区切りで再予約。`request_bpm(x)` は対象スニペットを x でレンダリングし、揃ったら次の小節頭に切替を予約 |
| コマンド | `prep()`、SNIPS タブの状態列 |

テスト: 124→128BPM で長さが計算値ちょうど・クリック位置 ±3ms、同 BPM でサンプル一致、ループ継ぎ目と端の差分閾値、キャッシュキーの変化、保留と BPM 切替の順序。

## フェーズD: オーディオエンジン(オフライン)

| モジュール | 役割・主なインターフェース |
|---|---|
| `clidj/tempo.py` | `TempoMap`: `beat_to_sample(b) -> float`、`sample_to_beat(s)`、`set_tempo(beat, bpm)`、イベントのサンプル = `floor(exact + 0.5)` |
| `clidj/engine/commands.py` | `EngineCommand` 群(dataclass、Python オブジェクトを含まない): `RegisterBuffer`, `Play`, `Stop`, `SetParam`, `Automate`, `CancelAutomation`, `SetTempo`, `TransportStart/Stop` |
| `clidj/engine/core.py` | `Engine.submit(cmd)`、`Engine.process_into(out)`: 次のコマンド/自動化の境界でブロックを分割。`LaneVoice` は拍からバッファ位置を計算(ループは `local % length`)。置き換え・停止・BPM切替時はフェードアウト中のボイスに移す |
| `clidj/engine/mixer.py` | `ParamSmoother`(数msの線形ランプ)、`Isolator3`(LR4 クロスオーバー 250Hz / 2.5kHz、`scipy.signal.sosfilt` を状態付きで回し、帯域ごとのゲインを掛ける)、`MasterBus`(ゲイン → BrickwallLimiter → clip) |
| `clidj/engine/status.py` | 状態のレイアウト(float64 配列、seqlock)。共有メモリとプロセス内配列の両方で同じ読み書きコード |
| `clidj/offline.py` | `python -m clidj render set.djs --bars N -o out.wav [--script cmds.txt] [--demo]`。スクリプトは `BAR[.BEAT]: コマンド` 行 |

テスト: 180000 サンプルちょうど、BPM 変更後のテンポマップ、xf 中間 ±0.01、ジッパー、4レーンでピーク ≤1.0、lo=0 で -30dB 以下、ボイス置換でクリックなし。

## フェーズE: リアルタイム化

| モジュール | 役割・主なインターフェース |
|---|---|
| `clidj/engine/backends.py` | `OfflineBackend`、`NullBackend`(実時間ペースのスレッド、締め切り超過をアンダーランとして数える)、`SoundDeviceBackend`(WASAPI 優先、`auto_convert`) |
| `clidj/engine/host.py` | エンジンプロセスの main(SIGINT 無視、親プロセスの死活監視)、`EngineClient`(Queue 送信、状態読み出し、プロセス死亡の検出)、`LocalEngineClient`(同一プロセス、`--no-audio`/render 用) |
| `clidj/scheduler.py` | `poll(head_beats + lookahead_beats)`: 先読み窓に入ったイベントだけ Thunk を評価し、予定拍付きで送る |
| `scripts/measure_lookahead.py` | IPC 遅延と UI ティックの揺れを実測 |
| `scripts/load_test.py` / `tests/test_load.py` | 4レーン+EQ+リミッタの NullBackend 負荷テスト(`slow`) |

テスト: NullBackend のエンジンプロセス起動・送信・状態・終了、先読み窓、遅延到着のカウント、エンジン強制終了で UI 側が落ちない。

## フェーズF: UI 統合

- 波形は解析済み包絡線から。TRACKS に BPM/Key/解析状態、SNIPS に準備状態
- ステータス行: デバイス、SR、ブロック、レイテンシ、アンダーラン、コールバック負荷(50%超で警告色)
- 先読み・準備待ち・遅延到着をレーン表示にも出す
- プレイヘッドは「今聞こえている拍」(エンジン位置 − 出力レイテンシ − リミッタ遅延)
- `scripts/render_listening.py` と `docs/listening-checklist.md`

## テスト方針(共通)

- 音源はテスト内で合成し `tmp_path` に書く。ユーザー設定・キャッシュは `CLIDJ_HOME=tmp_path` で隔離
- デバイスを開くテストはない(`SoundDeviceBackend` は手動確認のみ)。全テストはデバイスなしで通る
- 各フェーズの終わりで `pytest` 全通過を確認してからコミット
