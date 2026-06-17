# Pokemon TCG (Kaggle)

Pokémon TCG バトルコンペ用の作業リポジトリ。

## ディレクトリ構成

```
Pokemon/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ input/raw/          # ローカル用データ（cg-lib 等）
├─ sample_code/        # コンペ公式・参考サンプル
├─ notebooks/          # EDA・読解用ノートブック
├─ src/                # 共通設定・ユーティリティ
├─ agents/             # 提出用エージェント
├─ outputs/            # 提出物・ログ（git 管理外）
└─ docs/               # メモ・調査ノート
```

## セットアップ

```bash
cd Pokemon
pip install -r requirements.txt
```

Kaggle 上では `cg-lib` を Input に追加してください。ローカルでは `input/raw/cg-lib` を配置し、`src/config.py` のパスを参照します。

## クイックスタート

### 初回提出練習（Mega Abomasnow ex）

1. Kaggle Notebook の Input に **cg-lib / Abomasnow deck / 本リポジトリ** を Add
2. [notebooks/03_submission_practice_mega_abomasnow.ipynb](notebooks/03_submission_practice_mega_abomasnow.ipynb) を実行
3. Output の `submission.tar.gz` を **My Submissions** にアップロード

手順詳細: [docs/submission_practice_abomasnow.md](docs/submission_practice_abomasnow.md)

### その他

1. `notebooks/00_data_check.ipynb` — 環境・データの確認
2. `notebooks/01_sample_code_reading.ipynb` — サンプルコードの読解
3. `sample_code/reinforcement-learning-and-mcts-sample-code.ipynb` — RL + MCTS 公式サンプル
4. `agents/baseline_agent.py` — 提出用ベースライン
5. `agents/*.py` — ルールベースサンプル（ipynb の `main.py` をそのまま py 化）

### Kaggle 提出用エージェント（sample_code から複製）

各 ipynb の `%%writefile main.py` 内容をそのまま `.py` にしたファイルです。

**重要:** `.py` 単体では提出できません。`submission.tar.gz` に以下 3 つを同梱してください（ipynb 末尾セルと同じ）。

| ファイル | 内容 |
|----------|------|
| `main.py` | エージェント本体（agents/*.py をリネーム） |
| `cg/` | cg-lib データセット内の `cg` フォルダ |
| `deck.csv` | デッキ用 Input データセットの deck.csv |

```bash
python scripts/package_submission.py --agent mega-abomasnow-ex
# → outputs/submissions/mega-abomasnow-ex-submission.tar.gz を Kaggle に提出
```

| デッキ | agents/*.py | 元 ipynb |
|--------|-------------|----------|
| Mega Abomasnow ex | `mega_abomasnow_ex.py` | `mega-abomasnow-ex-deck.ipynb` |
| Dragapult ex | `dragapult_ex.py` | `dragapult-ex-deck.ipynb` |
| Iono's | `iono_s.py` | `iono-s-deck.ipynb` |
| Mega Lucario ex | `mega_lucario_ex.py` | `mega-lucario-ex-deck.ipynb` |

## 参考

- ファイル構成: [docs/project_structure.md](docs/project_structure.md)
- 提出練習: [docs/submission_practice_abomasnow.md](docs/submission_practice_abomasnow.md)
- ユキノオー戦略: [docs/mega_abomasnow_strategy.md](docs/mega_abomasnow_strategy.md)
- コンペメモ: [docs/competition_notes.md](docs/competition_notes.md)
- アイデア: [docs/ideas.md](docs/ideas.md)
