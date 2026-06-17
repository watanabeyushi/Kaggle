# プロジェクト構成

`Pokemon/` ディレクトリのファイル・フォルダ構成と役割。

## ツリー

```
Pokemon/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ input/
│  └─ raw/
│     └─ .gitkeep              # ローカルデータ置き場（git 管理外）
├─ sample_code/
│  ├─ reinforcement-learning-and-mcts-sample-code.ipynb
│  ├─ mega-abomasnow-ex-deck.ipynb
│  ├─ dragapult-ex-deck.ipynb
│  ├─ iono-s-deck.ipynb
│  └─ mega-lucario-ex-deck.ipynb
├─ notebooks/
│  ├─ 00_data_check.ipynb
│  ├─ 01_sample_code_reading.ipynb
│  ├─ 02_package_submission.ipynb
│  └─ 03_submission_practice_mega_abomasnow.ipynb
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  └─ utils.py
├─ agents/
│  ├─ baseline_agent.py
│  ├─ mega_abomasnow_ex.py       # 提出用 main.py（ipynb から複製）
│  ├─ dragapult_ex.py
│  ├─ iono_s.py
│  └─ mega_lucario_ex.py
├─ outputs/
│  ├─ submissions/
│  │  └─ .gitkeep              # 提出ファイル（git 管理外）
│  └─ logs/
│     └─ .gitkeep              # 実行ログ（git 管理外）
└─ docs/
   ├─ project_structure.md     # 本ファイル
   ├─ submission_practice_abomasnow.md  # ユキノオー提出練習
   ├─ mega_abomasnow_strategy.md        # ユキノオー Agent 戦略調査
   ├─ competition_notes.md
   └─ ideas.md
```

## ルート

| ファイル | 役割 |
|----------|------|
| `README.md` | プロジェクト概要・セットアップ・クイックスタート |
| `requirements.txt` | Python 依存パッケージ |
| `.gitignore` | データ・出力・キャッシュ等の除外設定 |

## `input/raw/`

ローカル開発用の入力データ置き場。

| 配置例 | 説明 |
|--------|------|
| `input/raw/cg-lib/` | コンペ API ライブラリ（Kaggle Input と同等） |

`.gitignore` により中身はリポジトリに含めない。`.gitkeep` のみコミットしてディレクトリを維持する。

Kaggle 実行時は `/kaggle/input/**/cg-lib` から自動検出（`src/utils.py`）。

## `sample_code/`

コンペ公式・参考用のサンプルコード。原則 **編集せず参照用** として保持。

| ファイル | 説明 |
|----------|------|
| `reinforcement-learning-and-mcts-sample-code.ipynb` | RL + MCTS + Transformer + Search API |
| `mega-abomasnow-ex-deck.ipynb` 等 | デッキ別ルールベースエージェント（末尾で submission.tar.gz 作成） |

## `notebooks/`

探索・検証・読解用ノートブック。番号順に実行する想定。

| ファイル | 説明 |
|----------|------|
| `00_data_check.ipynb` | 環境確認、`cg-lib` の有無、カード/わざマスタ件数 |
| `01_sample_code_reading.ipynb` | サンプルコードの構成・Encoder/Decoder メモ |
| `02_package_submission.ipynb` | 汎用 tar.gz 作成 |
| `03_submission_practice_mega_abomasnow.ipynb` | Mega Abomasnow ex 初回提出練習 |

## `src/`

ノートブック・エージェントから共通利用する Python モジュール。

| ファイル | 説明 |
|----------|------|
| `config.py` | パス定数（`INPUT_DIR`, `CG_LIB_DIR`, `OUTPUT_DIR` 等）、`SAMPLE_DECK` |
| `utils.py` | `cg-lib` 検出・`sys.path` 設定（`setup_cg_lib`, `cg_lib_status`） |

## `agents/`

提出・対戦用エージェント実装。

| ファイル | 説明 |
|----------|------|
| `baseline_agent.py` | ランダム合法手エージェント（スタブ） |
| `mega_abomasnow_ex.py` 等 | 公式ルールベースサンプルの `main.py` をそのまま py 化。Kaggle 提出時は `main.py` にリネーム |

## `outputs/`

実行結果の出力先。中身は git 管理外。

| ディレクトリ | 説明 |
|--------------|------|
| `submissions/` | 提出用ファイル |
| `logs/` | 学習・評価ログ |

## `docs/`

調査メモ・設計ノート。

| ファイル | 説明 |
|----------|------|
| `project_structure.md` | 本ファイル（ディレクトリ構成） |
| `competition_notes.md` | コンペ・API の調査メモ |
| `ideas.md` | 改善アイデア・TODO |

## データの流れ（概要）

```
input/raw/cg-lib  ──→  src/utils.setup_cg_lib()
                              │
         sample_code/  ──→  notebooks/（読解・検証）
                              │
                         agents/（提出用）
                              │
                         outputs/（成果物）
```

## 関連ドキュメント

- [competition_notes.md](competition_notes.md) — API・コンペ情報
- [ideas.md](ideas.md) — タスク・アイデア一覧
- [../README.md](../README.md) — セットアップ手順
