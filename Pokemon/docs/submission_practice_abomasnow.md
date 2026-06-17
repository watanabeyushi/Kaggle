# Mega Abomasnow ex 提出練習

Simulation 部門で **初めて submission.tar.gz を出す** ための手順書です。

## 提出物

| ファイル | 中身 |
|----------|------|
| `submission.tar.gz` | 圧縮ファイル1つを My Submissions にアップロード |

tar.gz の中身:

| パス | 取得元 |
|------|--------|
| `main.py` | [agents/mega_abomasnow_ex.py](../agents/mega_abomasnow_ex.py) |
| `cg/` | Kaggle Input **cg-lib** 内の `cg` フォルダ |
| `deck.csv` | Kaggle Input **Mega Abomasnow ex デッキ dataset** |

## Step 1: Kaggle で Notebook を用意

1. コンペページ → **Code** → **New Notebook**
2. **Add Input** で次の3つを追加

| Input | 目的 |
|-------|------|
| **cg-lib** | ゲーム API（`cg-lib/cg/`） |
| **Mega Abomasnow ex deck** | `deck.csv`（60枚） |
| **本リポジトリ（Pokemon）** | `agents/mega_abomasnow_ex.py` |

3. このリポジトリの [notebooks/03_submission_practice_mega_abomasnow.ipynb](../notebooks/03_submission_practice_mega_abomasnow.ipynb) を Upload するか、同内容の Notebook を作成

## Step 2: ノートブック実行

1. 03 ノートブックを **上から順に全セル実行**
2. セル2で Input パスが表示され `OK — all inputs found` になること
3. セル4で `OK — ready to submit submission.tar.gz` になること

## Step 3: tar.gz をダウンロード

1. Notebook 右側 **Output** タブ
2. **`submission.tar.gz`** をダウンロード

## Step 4: My Submissions にアップロード

1. コンペページ → **My Submissions**
2. **Upload** → ダウンロードした `submission.tar.gz`
3. Validation Episode の結果を待つ

## 成功 / 失敗

| 症状 | 原因 | 対処 |
|------|------|------|
| step 1 で `ModuleNotFoundError: No module named 'cg'` | `.py` 単体提出、または tar に `cg/` なし | 03 ノートブックで tar.gz を作り直す |
| `deck.csv not found` | デッキ Input 未追加 | Mega Abomasnow ex 用 dataset を Add |
| `mega_abomasnow_ex.py not found` | リポジトリ Input 未追加 | Pokemon リポジトリを Add |
| Validation が進み対戦ログが出る | 成功 | — |

## ローカル参照（Kaggle 本番では Input を使用）

| ファイル | パス |
|----------|------|
| エージェント | `Pokemon/agents/mega_abomasnow_ex.py` |
| deck.csv（参考） | `Pokemon/input/raw/decks/mega-abomasnow-ex/deck.csv` |
| cg-lib（参考） | `Pokemon/input/raw/cg-lib/cg/` |

## 関連

- [competition_notes.md](competition_notes.md) — 提出形式・トラブルシュート
- [02_package_submission.ipynb](../notebooks/02_package_submission.ipynb) — 他デッキ用の汎用パッケージ
