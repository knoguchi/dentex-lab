# dentex-lab — 歯科X線の疾患検出を PyTorch で学ぶ

[English version](README.md)

パノラマ歯科X線画像から **う蝕（Caries）、深いう蝕（Deep Caries）、根尖病変（Periapical Lesion）、埋伏歯（Impacted）** を検出する。
バウンディングボックスによる物体検出。COCO 学習済みの Faster R-CNN（torchvision）をファインチューニングする。
学習用の最小構成：スクリプト3本、アプリフレームワークなし。

## データ

[DENTEX](https://huggingface.co/datasets/ibrahimhamamci/DENTEX)（MICCAI 2023 チャレンジ、
CC BY-NC-SA 4.0）。完全ラベル付きのサブセットのみ使用。

| 分割  | 画像数 | ボックス数 |
|-------|--------|-----------|
| train | 705    | 3529（Caries 2189、Impacted 604、Deep Caries 578、Periapical 158） |
| val   | 50     | 182   |

学習用 zip は約 11 GB（大半はラベルなし画像）。`scripts/download_data.py` は zip を HTTP Range
リクエストで読み、ラベル付きサブセット（約 2.9 GB）だけを展開する。

```sh
uv sync
uv run python scripts/download_data.py            # 学習サブセット -> data/raw/training_data/
# 検証セット（小さい）は別途取得:
uv run python -c "from huggingface_hub import hf_hub_download as d; [d('ibrahimhamamci/DENTEX', f'DENTEX/{f}', repo_type='dataset', local_dir='data/raw') for f in ['validation_data.zip','validation_triple.json']]"
(cd data/raw && unzip -q DENTEX/validation_data.zip -x '*/.ipynb_checkpoints/*')
```

## 学習

```sh
uv run python train.py --epochs 10            # Apple M 系（MPS）で 1 エポック約 5〜6 分
uv run python train.py --epochs 1 --limit 40  # 動作確認用
```

検証 mAP が最良のチェックポイントを `checkpoints/fasterrcnn_dentex.pt` に保存する。

## 推論

```sh
uv run python infer.py data/raw/validation_data/quadrant_enumeration_disease/xrays/val_0.png --out result.png --thresh 0.5
```

検出結果を標準出力に表示し、ボックスを描いた画像を書き出す。

## ファイル構成

- `dental_data.py` — DENTEX の COCO 形式 JSON を `(画像テンソル, {boxes, labels})` に変換する `Dataset`。
- `train.py` — モデル構築、学習ループ、mAP 評価（torchmetrics）。
- `infer.py` — チェックポイント読み込み、1 枚推論、ボックス描画。
- `scripts/download_data.py` — リモート zip から必要なファイルだけ取り出すダウンローダ。

## 理解しておきたいポイント

- **ラベル 0 は背景。** torchvision の検出ヘッドはクラス 0 を背景用に予約している。DENTEX の id は +1 してずらす。
- **ボックスは `[x1, y1, x2, y2]`。** torchvision はこの形式だが、COCO JSON は `[x, y, w, h]`。変換が必要。
- **train モードは損失を、eval モードは予測を返す。** `model(images, targets)` の呼び出しは同じ。
- **損失は 4 つ**：RPN の objectness と box 回帰（候補領域）、分類器と box 回帰（最終ヘッド）。
- **リサイズ**：Dataset で長辺 1333 に縮小する。その後モデル内部の `GeneralizedRCNNTransform` が
  min_size 800 に再リサイズし、ImageNet の統計で正規化する。
- **データ拡張**：左右反転のみ。パノラマX線を反転すると左右の象限が入れ替わるが、診断ラベルには影響しない。
- **評価指標**：mAP@[.5:.95]（COCO 方式）。少数クラス（Periapical Lesion、158 ボックス）は最も低くなる。

## 初回実行の結果（10 エポック、MPS、約 50 分）

| epoch | train loss | val mAP@[.5:.95] | mAP@.5 | Impacted | Caries | Periapical | Deep Caries |
|------:|-----------:|-----------------:|-------:|---------:|-------:|-----------:|------------:|
| 0 | 0.699 | 0.172 | 0.276 | 0.45 | 0.18 | 0.01 | 0.05 |
| 3 | 0.359 | 0.280 | 0.448 | 0.52 | 0.19 | 0.17 | 0.25 |
| **5** | 0.291 | **0.316** | **0.481** | 0.55 | 0.25 | 0.21 | 0.26 |
| 9 | 0.166 | 0.293 | 0.436 | 0.55 | 0.25 | 0.14 | 0.23 |

エポック 5 以降、train loss は下がり続けるが val mAP は頭打ち／低下：705 枚での典型的な過学習。
次に試す案：データ拡張の強化（明るさ・コントラスト、スケールジッタ）、学習率を下げる、
バックボーンの凍結層を増やす、Caries と Deep Caries を統合（モデルが常に混同する）。
