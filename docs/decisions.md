# 設計判断の記録

指示書 13章に従い「決めたこと / 検討した他の案 / 理由」を追記していく。新しいものほど下。

---

## D1. Python のバージョン: 3.13

- **決めたこと**: CPython 3.13(開発機では 3.13.14)。`.venv` は `py -V:Astral/CPython3.13.14 -m venv .venv` で作り直した
- **他の案**: 3.12(3.12.13)
- **理由**: 2026-09-16 時点の PyPI で、使う依存すべてに 3.13 の Windows 向け wheel があることを確認し、実際に 3.13 の venv へ入れて import・`time_stretch`・デバイス列挙まで動かした
  - numpy 2.5.3 / scipy 1.18.1 / librosa 1.0.0 は `requires_python >=3.12`
  - pedalboard 0.9.25、numba 0.67.0、llvmlite 0.49.0、msgpack 1.2.2 に cp313 win_amd64 wheel あり
  - soxr 1.1.0 は cp313 専用 wheel はないが abi3 wheel で入る
  - sounddevice 0.5.6、soundfile 0.14.0 は PortAudio/libsndfile 同梱の `py3-none-win_amd64`
  - 3.13 では `multiprocessing.shared_memory.SharedMemory(track=False)` が使えるので、子プロセスが共有メモリにアタッチしたときの resource_tracker の誤警告・誤解放(Linux/macOS)を避けられる
  - 3.12 より 3.13 を選ばない理由は見つからなかった
- 開発機の `py` ランチャーには 3.10/3.11 しかなかったため、uv(`uv python install 3.13`)で入れた。uv が PEP 514 でレジストリ登録するので `py -V:Astral/CPython3.13.14` で呼べる。python.org の 3.13 を入れた環境では `py -3.13` でよい
