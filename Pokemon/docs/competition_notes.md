# Competition Notes

## API 概要

- **`cg.game`**: 実バトル (`battle_start`, `battle_select`, `battle_finish`)
- **`cg.api`**: 型定義・Search API (`search_begin`, `search_step`, `search_end`)

## データ

- Kaggle Input: `cg-lib` データセット
- ローカル: `input/raw/cg-lib/` に配置

## サンプルコード

- `sample_code/reinforcement-learning-and-mcts-sample-code.ipynb`
  - Transformer + MCTS + Search API
  - Encoder 24 単語 / Decoder は合法手ごと

## 調査メモ

- `help(search_begin)` / `inspect.signature()` で API を確認
- 1 手目の `obs` を JSON ダンプしてスキーマを把握

## 提出形式

Kaggle には **`submission.tar.gz`** を提出する（`.py` 単体不可）。

| ファイル | 内容 |
|----------|------|
| `main.py` | エージェント（`agents/*.py` をリネーム） |
| `cg/` | cg-lib Input 内の cg フォルダ |
| `deck.csv` | デッキ Input |

- ローカル: `python scripts/package_submission.py --agent mega-abomasnow-ex`
- Kaggle 練習: [notebooks/03_submission_practice_mega_abomasnow.ipynb](../notebooks/03_submission_practice_mega_abomasnow.ipynb)
- 手順書: [submission_practice_abomasnow.md](submission_practice_abomasnow.md)

## トラブルシュート

### `ModuleNotFoundError: No module named 'cg'`

`.py` のみ提出している。上記 3 ファイルを tar.gz に同梱して再提出。

## リンク

- （コンペ URL を追記）
