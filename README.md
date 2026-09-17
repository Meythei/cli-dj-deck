# cli-dj (snippet-driven live-coding DJ)

ghostty 風にペイン分割されたターミナル上の DJ ツール。「曲を2枚かけて手でつなぐ」のではなく、事前に曲から切り出した
短いスニペット(DJ でいう tools)を、共通の時計(Transport)の上で **コードとして** 展開していきます。
FoxDot / TidalCycles のようなライブコーディング環境に近い操作感です。

- 自分の音声ファイルのライブラリを解析します(BPM・ビートグリッド・キー・ラウドネス・波形。バックグラウンド)
- スニペットは再生時ではなく **事前に** 現在の BPM へタイムストレッチしてキャッシュします
- 別プロセスのオーディオエンジンが、予約された拍に対応するサンプルちょうどで鳴らします(ブロック境界ではなく)
- コマンドは小節頭などの区切りにクオンタイズされるので、打つのが遅れても音楽的には揃います
- 同じエンジンでセットを WAV に書き出せます(音を聴かずに検証するための仕組みでもあります)

設計の経緯と判断は [docs/decisions.md](docs/decisions.md)、実装計画は [docs/plan.md](docs/plan.md)、
耳で確認する項目は [docs/listening-checklist.md](docs/listening-checklist.md) にあります。

## セットアップ(Windows)

Python **3.13** が必要です(numpy 2.5 / librosa 1.0 が 3.12 以上を要求するため。3.13 を選んだ理由は decisions.md の D1)。

```bash
py -3.13 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m clidj --list-devices
```

- python.org 版の 3.13 がなく uv で入れた場合は `py -V:Astral/CPython3.13.14 -m venv .venv`
- 依存(すべて Windows 向け wheel あり): textual, numpy, scipy, pedalboard(読み込み・タイムストレッチ・リミッタ)、
  librosa(解析)、sounddevice(PortAudio / WASAPI)、mutagen(タグ)、platformdirs
- `--list-devices` で出力デバイスが表示されればオーディオの準備はできています(`*` が各 API の既定。
  既定では WASAPI の既定出力を使います)
- Linux / macOS でもテストは通ります(PortAudio が必要)。動作確認は Windows 11 で行っています

## はじめての演奏(デモライブラリ)

音声ファイルがなくても、合成したデモ曲 8 曲で試せます。

```bash
.venv\Scripts\python -m clidj --demo --set sets/demo.djs
```

1. 起動すると AUDIO 行にデバイス名とレイテンシが出ます。デモ曲の音声の合成とスニペットのレンダリングが裏で走り、
   終わるまで L1 は `waiting for render: kick` と表示されます(初回は数十秒。2回目からはキャッシュ済み)
2. コンソールで `start()` と打つと、8小節ごとにレーンが加わり、33小節目から L1 → L4 にクロスフェードします
3. 演奏中に `L2.eq(lo=0)`、`L2.gain(0.5)`、`at(21, L3.stop())`、`bpm(132)` などを打ってみてください
4. `quit()` か Ctrl+C で終了します(オーディオエンジンのプロセスも一緒に終了します)

## 自分のライブラリを使う

1. 一度起動すると設定ファイルの雛形ができます: `%LOCALAPPDATA%\cli-dj\config.toml`
2. `[library]` の `folders` に音声ファイルのフォルダを書きます(WAV / FLAC / MP3 / AIFF / OGG。サブフォルダも対象)

   ```toml
   [library]
   folders = ["D:/Music/DJ"]
   ```
3. `python -m clidj` で起動して `scan()`。見つかった曲が TRACKS に並び、裏のワーカープロセス(既定 2)で解析が進みます。
   進捗は LIBRARY の枠の下に `analyzing 12/120` のように出ます。UI は止まりません
4. `✓` になった曲から `snip(番号, ...)` で切り出せます。番号は最初に見つけた順に振られ、あとから曲を足してもずれません
   (曲の識別は中身のハッシュなので、ファイルを移動・改名しても解析結果と補正は残ります)

TRACKS の記号: `·` 未解析 / `…` 解析中 / `✓` 済 / `✗` 失敗(`scan(retry=True)` で再試行) / `?` ファイルが見つからない。

### 解析結果の補正

自動解析は外れる前提です。BPM・グリッド・キーが違っていたら補正します。補正は `%LOCALAPPDATA%\cli-dj\overrides.json`
に保存され、自動解析より常に優先されます。グリッドを変えると、その曲のスニペットは自動でレンダリングし直されます。

```python
tracks()                     # 番号・BPM・キー・状態の一覧
regrid(3, bpm=127.98)        # BPM を指定
regrid(3, offset_ms=12)      # 拍0(グリッド)を 12ms 後ろへ。負で前へ。何度でも足し込める
regrid(3, first_beat=0.382)  # 拍0の位置を秒で直接指定
regrid(3, reset=True)        # BPM とグリッドの補正を消して自動解析に戻す
setkey(3, "8A")              # キー(Camelot 表記)
```

小節の1拍目(ダウンビート)は推定していません。`bar=1` が小節の頭に来ない曲は `offset_ms` で1拍単位にずらしてください。

## 起動オプション

```bash
.venv\Scripts\python -m clidj [オプション]
```

`python app.py` でも同じものが起動します。

| オプション | 意味 |
|---|---|
| `--demo` | 自分のライブラリの代わりにデモ用の8曲を使う |
| `--set PATH` | 起動時にセットファイルを実行する(Transport は自動では始まらない) |
| `--no-audio` | ビジュアルのみ。オーディオデバイスもスニペットのレンダリングも使わない |
| `--list-devices` | 出力デバイスの一覧を表示して終了 |
| `--device 20` / `--device "USB Audio"` | 出力デバイスを番号か名前の一部で指定(設定ファイルの `[audio] device` より優先) |
| `--null-audio` | デバイスなしでオーディオエンジンを実時間で動かす(タイミングや負荷の確認用。音は出ない) |

デバイスが開けなかった場合は、エラーを表示したうえで音なしのエンジンで動き続けます。

## 画面構成

```
+-- LIBRARY ------------+-- CONSOLE ----------------------------------------------+
| [TRACKS] [SNIPS]      | ログ                                                     |
| #  ✓ Title   BPM Key  |----------------------------------------------------------|
|                       | QUEUE  #3 @033.1 xf(L1, L4, bars=8)                      |
|                       | > 入力欄(↑↓で履歴)                                        |
+-----------------------+----------------------------------------------------------+
| TRANSPORT  017.3  ● RUN  128.0 BPM  4/4  q:bar  -> bpm(132) @ bar 21             |
| AUDIO      xrun 0  late 0  cpu 12% (max 38%)  lat 27ms  48k/512  Speakers [WASAPI] |
|     ·   |   ·   ·   ·   |   ·   ·  ▼·   |   ·   ·   ·   |   ·   ·   ·   |          |
| L1  kick         drums  8A   gain ████ eq▂██ loop 1/8   next @033.1: xf(L1, L4)   |
|     (2行のズーム波形。ルーラーと同じ列・同じ中央)                                   |
| L2  bass         bass   8A   gain ████ eq███ loop 1/8   waiting for render: bass  |
| ...                                                                              |
+----------------------------------------------------------------------------------+
```

- **LIBRARY**: TRACKS(曲の一覧と解析状態)と SNIPS(定義済みスニペットと準備状態 `· … ✓ ✗`)。枠の下に解析・レンダリングの進捗
- **CONSOLE**: ログ、予約中イベント(QUEUE、発火順に最大5件)、コマンド入力欄
- **TRANSPORT**: 今 **聞こえている** 位置(小節.拍)、再生状態、BPM、クオンタイズ。BPM の変更待ちがあれば黄色で表示
- **AUDIO**: 左から問題の数(`xrun` = アンダーラン、`late` = 予定のサンプルに間に合わなかったコマンド。赤、直後は反転表示)、
  コールバック負荷(直近の平均がブロック時間の50%以上で黄色、最大が100%を超えたら最大も黄色)、総レイテンシ
  (出力 + リミッタの先読み)、フォーマット、デバイス。`--no-audio` では `off`、エンジンが落ちたら `ENGINE DOWN`
- **レーン**: 1本 = 情報1行 + 波形2行。情報行はスニペット名・ロール・キー・ゲイン・EQ(lo/mid/hi)・ループ位置と、
  そのレーンに関する注記(次の予約、`waiting for render`、`LATE +12ms`、クロスフェードの進み具合)。
  波形は解析したピークを拍単位で描き、プレイヘッド(▼)を中央に固定、1文字 ≈ 1/4拍。ルーラーと全レーンの波形は同じ列の
  対応を使うので、▼・小節線・キックが縦に揃います。キーが合わない・ボーカルが重なる・BPM が大きく離れているレーンは名前が黄色になります

## コマンド言語

右上の入力欄に打って Enter。Python の式として構文解析しますが、`eval` / `exec` は使わず、許可した構文と関数・メソッドだけを
自前で評価します。

| 書き方 | 意味 |
|---|---|
| `start()` / `stop()` | Transport を開始 / 一時停止(`stop()` は位置を保ち、`start()` で同じ位置から再開) |
| `bpm(128)` | テンポ変更。先に全スニペットを新しい BPM でレンダリングし、揃ってから次の小節頭で切り替え(停止中は揃った時点) |
| `quant("bar")` | 既定のクオンタイズ(`beat` / `bar` / `phrase`(8小節) / `none`) |
| `kick = snip(1, cue=2, bars=8, loop=True, role="drums")` | スニペット定義(曲は番号・タイトルどちらでも。名前は変数名になる) |
| `hook = snip("Glass Horizon", bar=33, bars=4, role="vocal")` | 小節番号で切り出す例(`bars` は 0.25 小節 = 1拍単位で指定可) |
| `L1 << kick` / `L1.play(kick)` | 次の区切りで L1 に kick を流し始める |
| `L1.stop()` | 次の区切りで L1 を止める |
| `L1.gain(0.5)` / `L1.mute()` / `L1.unmute()` | 音量(0〜1、即時、5ms で平滑化) |
| `L1.eq(lo=0, mid=1, hi=0.8)` | 3バンドのアイソレータ(0〜1、0 でその帯域をカット。即時、5ms で平滑化) |
| `xf(L1, L2, bars=8)` | 次の区切りから 8小節かけて L1 → L2 に直線でクロスフェードし、終わったら L1 を止める |
| `at(33, 式)` | 33小節目の頭で式を実行 |
| `after(4, 式)` | 次の小節頭から数えて 4小節後に式を実行(負はエラー) |
| `every(8, 式)` | 次の小節頭から 8小節ごとに式を実行(最小 0.25 小節 = 1拍。`cancel` で停止) |
| `now(式)` | クオンタイズせず即実行(エンジンが受け取った次のブロック) |
| `queue()` | 予約中イベントをログに出す |
| `cancel(3)` / `cancel()` | ID 3 の予約(またはクロスフェード)を取り消す / 全部取り消す |
| `snips()` / `tracks()` | スニペット一覧 / ライブラリ一覧 |
| `prep()` | 定義済みの全スニペットを現在の BPM でレンダリングし、完了をログに出す(失敗分も再試行) |
| `scan()` / `scan(retry=True)` | ライブラリのフォルダを再スキャンし、新しい曲(と失敗した曲)を解析 |
| `regrid(3, bpm=..., offset_ms=..., first_beat=..., reset=...)` / `setkey(3, "8A")` | 解析結果の補正(上記) |
| `load_set("demo")` | `sets/demo.djs` を読み込んで実行(ネストは最大8段) |
| `clear()` / `help()` / `quit()` | ログのクリア / コマンド一覧 / 終了(Ctrl+C でも終了) |

`at` / `after` / `every` / `now` は特殊形式で、中の式(第2引数)は書いた時点では評価されず、その時が来てから評価されます。
`at(33, xf(L1, L2, 8))` と書いても、その場でクロスフェードが始まることはありません。予約の中のコマンドはクオンタイズを
二重にかけず、予約した拍そのもので実行されます(`at(33, L1 << kick)` は 34小節目ではなく 33小節目)。
`start()` の前に書いた `at(1, ...)` は `start()` の瞬間に実行されます。

ログの表記:

| 表記 | 意味 |
|---|---|
| `at(9, L2 << bass) queued #1 @ bar 9` | 予約した(何小節目に発火するか) |
| `[009.1] L2 << bass` | 予約が発火した(9小節目の頭で実行) |
| `L1 << kick queued #3 @ bar 5` | 通常のコマンドが次の区切りに回された |
| `L1 << kick (immediate)` | Transport 停止中なので即時に実行 |
| `L1 << kick (now)` | `now(...)` または `quant("none")` による即時実行 |

安全のための上限: 1ティックで発火するイベントは最大256件。属性はメソッド呼び出しの形でしか使えません(`L1.gain(0.5)` は可、
`L1.gain` を値として使うのは不可)。`snip()` の `bar=` / `bars=` は素材の曲の 4/4 小節で、Transport の拍子とは独立です。

## タイミングの仕組み(知っておくと困らないこと)

- **時刻の正はオーディオエンジンのサンプルカウンタ** です。UI は描画のたびにエンジンの状態(共有メモリ)を読むだけで、
  音のタイミングは UI の描画やタイピングの遅れの影響を受けません
- **予約は先読みして送られます**: 予約(`at` / `after` / `every`、クオンタイズされたコマンド)は、予定の拍の **先読み時間
  (既定 200ms)前** に評価されてエンジンに送られ、エンジンがその拍のサンプルちょうどで実行します。そのため、先読み時間より
  後に変数を再代入しても、すでに評価された予約には反映されません(例: 33小節目の 200ms 前以降に `kick = snip(...)` を
  打ち直しても、`at(33, L1 << kick)` は古い `kick` を鳴らします)
- **区切りの直前に打ったコマンド**: 次の区切りまで 60ms を切っているときは、その次の区切りに回ります(エンジンに間に合わせるため)
- **準備が済んでいないスニペット** は無音で鳴り始めることはありません。コマンドは保留になり(レーンに `waiting for render`)、
  レンダリング完了後の最初の区切りで鳴ります。本番前に `prep()` しておくと保留を避けられます
- **BPM 変更** は全スニペットのレンダリングが揃ってから次の小節頭で切り替わり、各レーンはスニペット内の拍位置を保ったまま
  新しいバッファに乗り換えます(5ms のクロスフェード)
- **遅れて届いたコマンド**(UI が長く止まった場合など)は次のブロックで適用され、拍のグリッドには乗ったまま最初の数msが欠けます。
  AUDIO 行の `late` とレーンの `LATE +Nms` で分かります
- 画面のプレイヘッドは「今スピーカーから聞こえている位置」(エンジンの位置 − 出力レイテンシ − リミッタの先読み 5ms)です

## セットファイル

拡張子 `.djs`。中身はコマンド言語そのもの(`#` から行末はコメント)。`sets/` に置いて `load_set("名前")`(拡張子なし)で
読み込むか、`--set sets/名前.djs` で起動時に読み込みます。`sets/demo.djs`:

```python
bpm(128)

kick  = snip(1, cue=2, bars=8, loop=True, role="drums")
bass  = snip(8, bar=17, bars=8, loop=True, role="bass")
hook  = snip(3, bar=33, bars=4, loop=True, role="vocal")
riser = snip(5, bar=25, bars=8, role="fx")
drop  = snip(5, cue=3, bars=8, loop=True, role="drums")

L1 << kick
at(9,  L2 << bass)
at(17, L3 << hook)
at(25, L4 << riser)
at(33, L4 << drop)          # riser が終わる小節で drop に差し替え
at(33, xf(L1, L4, bars=8))  # 同じ小節の予約は書いた順に実行される
```

`snip(番号, ...)` の番号はライブラリごとに違うので、自分のライブラリ用のセットでは番号かタイトルを自分の TRACKS に合わせてください。

## セットを WAV に書き出す(オフラインレンダリング)

実時間を待たずにセットを最後まで流して WAV(48kHz / 24bit)に書き出します。スニペットの準備は同期的に済ませてから始めるので
結果は毎回同じで、ファイルの N サンプル目が Transport の N サンプル目です。

```bash
.venv\Scripts\python -m clidj render sets/demo.djs --bars 40 -o renders/demo.wav --demo
```

| オプション | 意味 |
|---|---|
| `--bars N` | 何小節書き出すか(既定 32) |
| `-o PATH` | 出力先(既定 `renders/<セット名>.wav`。`renders/` は git 管理外) |
| `--script FILE` | 演奏中に REPL から打つコマンドを模擬する(下記) |
| `--demo` | デモライブラリを使う |
| `--no-start` | セットを読んだ後に自動で `start()` しない |
| `-v` | コンソールのログを表示する |

`--script` のファイルは1行1コマンドで、「その時点で打った」ことにする位置を `小節[.拍]:` で書きます。その位置に達した最初の
ブロックの境目で実行され、ふつうに打った場合と同じくクオンタイズされます。

```text
2.3: L2 << bass        # 2小節目の3拍目に打つ -> 3小節目の頭から鳴る
9:   xf(L1, L2, bars=8)
17:  bpm(130)
```

終了コードは、エラー・エンジンのエラーがなければ 0 です。

## 設定ファイル

`%LOCALAPPDATA%\cli-dj\config.toml`(初回起動時にコメント付きで作られます)。

| キー | 既定値 | 意味 |
|---|---|---|
| `[library] folders` | `[]` | スキャンするフォルダ |
| `[library] bpm_min` / `bpm_max` | 88 / 176 | 推定した BPM をこの範囲にオクターブで折りたたむ(ドラムンベースの 87 → 174 など) |
| `[library] workers` | 2 | 解析・レンダリングのワーカープロセス数 |
| `[audio] samplerate` / `blocksize` | 48000 / 512 | |
| `[audio] device` | 既定出力 | 番号か名前の一部 |
| `[audio] hostapi` | `"WASAPI"` | 既定デバイスを選ぶときに優先する API |
| `[audio] lookahead_ms` | 200 | 予約を先読みして送る時間(実測の根拠は decisions.md D13) |
| `[mix] loudness_target_lufs` | -14 | スニペットの音量を曲の統合ラウドネスから揃える目標(持ち上げは最大 +12dB) |
| `[mix] limiter_ceiling_db` | -1 | マスターリミッタの天井 |
| `[mix] master_gain` | 1.0 | |

### ファイルの置き場所

| 場所 | 中身 |
|---|---|
| `%LOCALAPPDATA%\cli-dj\config.toml`, `overrides.json` | 設定、解析の補正(人が編集・バックアップするもの) |
| `%LOCALAPPDATA%\cli-dj\library.json`, `analysis\` | 曲の一覧と番号、解析結果(波形の包絡線を含む) |
| `%LOCALAPPDATA%\cli-dj\logs\` | ワーカーとエンジンのログ(子プロセスの出力はここに出て、画面を壊さない) |
| `%LOCALAPPDATA%\cli-dj\Cache\renders\`, `demo-audio\` | レンダリング済みスニペット(`.npy`)、合成したデモ曲。消しても作り直せる |

環境変数 `CLIDJ_HOME` を指定すると、全部をそのフォルダの下(`config/` `data/` `cache/`)にまとめます。

## 構成

```
UI プロセス                                          エンジンプロセス(spawn)
Textual UI ── Interpreter ── Session ── Scheduler     コマンドスレッド(バッファの mmap と先読み)
                              │  ├─ Library / 解析ワーカー    │ deque
                              │  ├─ PrepManager / レンダリングワーカー
                              │  └─ EngineClient ──Queue──▶  オーディオコールバック: Engine.process()
                              │                  ◀──共有メモリ── 状態(seqlock)
```

| モジュール | 役割 |
|---|---|
| `clidj/cli.py`, `__main__.py` | `python -m clidj` と `render` サブコマンド |
| `clidj/ui/app.py` | Textual の TUI(Textual を import するのはここだけ) |
| `clidj/interpreter.py` | ast ベースの安全なインタプリタ(特殊形式 `at` / `after` / `every` / `now` の Thunk) |
| `clidj/session.py` | インタプリタの裏側: ライブラリ・ジョブ・準備・スケジューラ・エンジンへの送信、UI 用の状態(Transport とレーンはエンジンの状態の写し) |
| `clidj/scheduler.py` | 予約イベントのキュー(先読みつきで発火) |
| `clidj/transport.py`, `clidj/tempo.py` | 小節・拍・クオンタイズの計算 / 拍 ↔ サンプルのテンポマップ |
| `clidj/library.py`, `clidj/analysis.py`, `clidj/workers.py` | ライブラリ(安定 ID、番号、補正)/ BPM・グリッド・キー・ラウドネス・波形の解析 / ワーカープロセス |
| `clidj/snippets.py`, `clidj/render.py`, `clidj/prep.py` | スニペットの切り出し / ストレッチと長さ合わせ・キャッシュ / 準備状態と保留 |
| `clidj/engine/core.py`, `dsp.py`, `commands.py`, `status.py` | サンプル単位のエンジン、アイソレータ EQ・リミッタ・平滑化、コマンド、共有メモリの状態 |
| `clidj/engine/host.py`, `backends.py`, `client.py` | エンジンプロセスとクライアント、SoundDevice / Null バックエンド、プロセス内クライアント(`--no-audio` と render) |
| `clidj/offline.py`, `clidj/synth.py` | オフラインレンダリング / テスト・デモ用の決定的な合成音 |
| `scripts/` | `load_test.py`(10分の負荷テスト)、`measure_lookahead.py`(先読みの実測)、`render_listening.py`(聴感チェック用 WAV) |

## テスト

```bash
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```

- 全テストはオーディオデバイスなしで通ります(実デバイスを開くテストはありません。リアルタイムのテストは NullBackend)。所要約1分半
- 音源はすべてテスト内で合成し、ユーザーの設定・キャッシュには触れません(`tests/conftest.py` が `CLIDJ_HOME` を一時ディレクトリにする)
- 負荷テストは既定で5秒。10分版: `set CLIDJ_LOAD_SECONDS=600` してから `pytest tests/test_load.py`、
  または `python scripts/load_test.py --seconds 600`(`--device` で実デバイス、マスターゲイン 0 で無音)

## 既知の制限

- テンポが途中で変わる曲には対応していません(ビートグリッドは「BPM + 最初の拍」の1本)。ダウンビートも推定しません
- タイムストレッチは Rubber Band の R2 エンジン(短い FFT 窓)です。R3 は拍位置がずれるため使っていません。音質は
  [docs/listening-checklist.md](docs/listening-checklist.md) で耳での確認が必要です(開発時には未確認)
- キー推定・BPM 推定の精度は合成音でしか確認していません(BPM ±0.02、最初の拍 ±5.4ms)。実際の曲では補正が必要になることがあります
- NullBackend の 10分負荷テストは、開発機で外部要因の約1.3秒の間に 3回のアンダーランが出ています(実デバイスでは 10分間 0)。decisions.md D12
- エンジンのプロセスが落ちた場合は cli-dj の再起動が必要です(UI は落ちず `ENGINE DOWN` を表示します)
- レンダリングキャッシュは自動では掃除しません(`Cache\renders` を消せば作り直されます)
- ヘッドホン用のキュー出力、MIDI コントローラ、ステム分離は未対応。レーン数は 4(`clidj/lanes.py` の `LANE_NAMES`)
- 再生位置を移動するコマンド(シーク)はありません。`stop()` は一時停止です
