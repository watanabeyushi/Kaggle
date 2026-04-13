# 特徴エンジニアリング チュートリアル

## 1. 導入

### 1.1 特徴エンジニアリングとは

特徴エンジニアリングは、機械学習モデルの性能を向上させるために、生データからより有用な特徴量を作成するプロセスです。適切な特徴エンジニアリングにより、モデルの予測精度を大幅に改善できる可能性があります。

### 1.2 現在のコードの問題点

現在のノートブックでは、以下の2つの特徴のみを使用しています：

- `date_count`: ユーザーごとの取引回数
- `average_unit_price_sum`: ユーザーごとの平均単価の合計

これらの特徴だけでは、顧客離脱（churn）予測に必要な情報が不足している可能性があります。時系列パターン、行動の変化、統計的特徴など、より多くの情報を抽出することで、モデルの性能を向上させることができます。

### 1.3 本チュートリアルの目的

本チュートリアルでは、以下の特徴エンジニアリング手法を実装します：

1. **時系列特徴**: 日付関連の特徴量
2. **統計的特徴**: 価格データの統計量
3. **時間窓特徴**: 最近の行動パターン
4. **トレンド特徴**: 行動の変化を捉える特徴
5. **周期性特徴**: 曜日・月などの周期性
6. **インタラクション特徴**: 特徴同士の組み合わせ
7. **欠損値・異常値の処理**: データの品質向上

---

## 2. 各特徴カテゴリの説明と実装

### 2.1 時系列特徴（日付関連）

時系列特徴は、ユーザーの行動の時間的なパターンを捉えるために重要です。

#### 実装例

```python
import pandas as pd
import numpy as np

# 日付をdatetime型に変換
data['date'] = pd.to_datetime(data['date'])

# 基準日を設定（最新の日付または特定の日付）
reference_date = data['date'].max()

# ユーザーごとの時系列特徴を作成
features = pd.DataFrame()
features['user_id'] = data['user_id'].unique()

# 最初の取引日
features['first_date'] = data.groupby('user_id')['date'].min().values

# 最後の取引日
features['last_date'] = data.groupby('user_id')['date'].max().values

# 取引期間（日数）
features['date_range'] = (features['last_date'] - features['first_date']).dt.days

# 最後の取引からの経過日数
features['days_since_last_transaction'] = (reference_date - features['last_date']).dt.days

# 最初の取引からの経過日数
features['days_since_first_transaction'] = (reference_date - features['first_date']).dt.days

# 取引間隔の平均（取引回数が1回の場合は0）
transaction_count = data.groupby('user_id').size().values
features['avg_days_between_transactions'] = features['date_range'] / (transaction_count - 1)
features['avg_days_between_transactions'] = features['avg_days_between_transactions'].fillna(0)
```

#### 特徴の説明

- `first_date`: ユーザーが最初に取引を行った日付
- `last_date`: ユーザーが最後に取引を行った日付
- `date_range`: 最初と最後の取引の間隔（日数）
- `days_since_last_transaction`: 最後の取引からの経過日数（離脱の兆候を捉える）
- `days_since_first_transaction`: 最初の取引からの経過日数（顧客の歴史）
- `avg_days_between_transactions`: 取引間隔の平均（利用頻度の指標）

---

### 2.2 統計的特徴（average_unit_price関連）

価格データから統計量を抽出することで、ユーザーの消費パターンをより詳しく理解できます。

#### 実装例

```python
# 現在は合計のみ → より多くの統計量を追加
price_stats = data.groupby('user_id')['average_unit_price'].agg([
    'mean',      # 平均
    'median',    # 中央値
    'std',       # 標準偏差
    'min',       # 最小値
    'max',       # 最大値
    'count'      # 取引回数
]).reset_index()

price_stats.columns = ['user_id', 'price_mean', 'price_median', 'price_std', 
                       'price_min', 'price_max', 'transaction_count']

# 追加の統計特徴
price_stats['price_range'] = price_stats['price_max'] - price_stats['price_min']
price_stats['price_cv'] = price_stats['price_std'] / (price_stats['price_mean'] + 1e-6)  # 変動係数

# 特徴をマージ
features = pd.merge(features, price_stats, on='user_id', how='left')
```

#### 特徴の説明

- `price_mean`: 平均単価の平均値
- `price_median`: 平均単価の中央値（外れ値の影響を受けにくい）
- `price_std`: 平均単価の標準偏差（価格のばらつき）
- `price_min`: 最小の平均単価
- `price_max`: 最大の平均単価
- `price_range`: 価格の範囲（最大値 - 最小値）
- `price_cv`: 変動係数（標準偏差 / 平均値、相対的なばらつき）

---

### 2.3 時間窓特徴（最近の行動パターン）

最近の行動パターンは、離脱予測において特に重要です。直近の期間での行動を分析します。

#### 実装例

```python
# 直近30日、60日、90日の特徴を作成
for window in [30, 60, 90]:
    # 指定期間内のデータを抽出
    recent_data = data[data['date'] >= (reference_date - pd.Timedelta(days=window))]
    
    # 取引回数
    recent_transactions = recent_data.groupby('user_id').size().reset_index()
    recent_transactions.columns = ['user_id', f'transactions_last_{window}d']
    
    # 価格の合計
    recent_price_sum = recent_data.groupby('user_id')['average_unit_price'].sum().reset_index()
    recent_price_sum.columns = ['user_id', f'price_sum_last_{window}d']
    
    # 価格の平均
    recent_price_mean = recent_data.groupby('user_id')['average_unit_price'].mean().reset_index()
    recent_price_mean.columns = ['user_id', f'price_mean_last_{window}d']
    
    # 特徴をマージ
    features = pd.merge(features, recent_transactions, on='user_id', how='left')
    features = pd.merge(features, recent_price_sum, on='user_id', how='left')
    features = pd.merge(features, recent_price_mean, on='user_id', how='left')
```

#### 特徴の説明

- `transactions_last_{window}d`: 直近N日間の取引回数
- `price_sum_last_{window}d`: 直近N日間の価格の合計
- `price_mean_last_{window}d`: 直近N日間の価格の平均

これらの特徴により、ユーザーの最近の行動パターンを捉えることができます。

---

### 2.4 トレンド特徴（行動の変化）

ユーザーの行動がどのように変化しているかを捉える特徴です。前半期間と後半期間を比較します。

#### 実装例

```python
# データを前半と後半に分割
mid_date = data['date'].quantile(0.5)

first_half = data[data['date'] <= mid_date]
second_half = data[data['date'] > mid_date]

# 前半期間の特徴
first_half_features = pd.DataFrame()
first_half_features['user_id'] = data['user_id'].unique()
first_half_features['transaction_count_first_half'] = first_half.groupby('user_id').size().values
first_half_features['price_mean_first_half'] = first_half.groupby('user_id')['average_unit_price'].mean().values

# 後半期間の特徴
second_half_features = pd.DataFrame()
second_half_features['user_id'] = data['user_id'].unique()
second_half_features['transaction_count_second_half'] = second_half.groupby('user_id').size().values
second_half_features['price_mean_second_half'] = second_half.groupby('user_id')['average_unit_price'].mean().values

# トレンド特徴の計算
trend_features = pd.merge(first_half_features, second_half_features, on='user_id', how='outer')
trend_features = trend_features.fillna(0)

# 取引回数のトレンド（後半 / 前半）
trend_features['transaction_trend'] = trend_features['transaction_count_second_half'] / (trend_features['transaction_count_first_half'] + 1)

# 価格のトレンド（後半 / 前半）
trend_features['price_trend'] = trend_features['price_mean_second_half'] / (trend_features['price_mean_first_half'] + 1)

# 特徴をマージ
features = pd.merge(features, trend_features[['user_id', 'transaction_trend', 'price_trend']], on='user_id', how='left')
```

#### 特徴の説明

- `transaction_trend`: 取引回数の変化率（後半 / 前半）
  - 1より大きい: 取引が増加
  - 1より小さい: 取引が減少（離脱の兆候）
- `price_trend`: 価格の変化率（後半 / 前半）
  - 1より大きい: 単価が上昇
  - 1より小さい: 単価が下降

---

### 2.5 周期性特徴（曜日・月など）

ユーザーの行動に周期性があるかを捉える特徴です。

#### 実装例

```python
# 曜日、月、日を抽出
data['weekday'] = data['date'].dt.dayofweek  # 0=月曜日, 6=日曜日
data['month'] = data['date'].dt.month
data['day_of_month'] = data['date'].dt.day

# 最も活動的な曜日
most_active_weekday = data.groupby('user_id')['weekday'].agg(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else -1
).reset_index()
most_active_weekday.columns = ['user_id', 'most_active_weekday']

# 活動した曜日の種類数
unique_weekdays = data.groupby('user_id')['weekday'].nunique().reset_index()
unique_weekdays.columns = ['user_id', 'unique_weekdays']

# 活動した月の種類数
unique_months = data.groupby('user_id')['month'].nunique().reset_index()
unique_months.columns = ['user_id', 'unique_months']

# 特徴をマージ
features = pd.merge(features, most_active_weekday, on='user_id', how='left')
features = pd.merge(features, unique_weekdays, on='user_id', how='left')
features = pd.merge(features, unique_months, on='user_id', how='left')
```

#### 特徴の説明

- `most_active_weekday`: 最も活動的な曜日（0-6）
- `unique_weekdays`: 活動した曜日の種類数（多様性の指標）
- `unique_months`: 活動した月の種類数（継続性の指標）

---

### 2.6 インタラクション特徴

既存の特徴を組み合わせて、新しい特徴を作成します。

#### 実装例

```python
# 総価値（取引回数 × 平均価格）
features['total_value'] = features['transaction_count'] * features['price_mean']

# 1日あたりの価値
features['value_per_day'] = features['total_value'] / (features['date_range'] + 1)

# 取引頻度（1日あたりの取引回数）
features['transaction_frequency'] = features['transaction_count'] / (features['date_range'] + 1)

# 価格のばらつきと平均の比率
features['price_std_mean_ratio'] = features['price_std'] / (features['price_mean'] + 1e-6)

# 最近の活動率（直近30日の取引回数 / 全体の取引回数）
features['recent_activity_ratio'] = features['transactions_last_30d'] / (features['transaction_count'] + 1)
```

#### 特徴の説明

- `total_value`: 総価値（取引回数 × 平均価格）
- `value_per_day`: 1日あたりの価値（顧客価値の指標）
- `transaction_frequency`: 取引頻度（1日あたりの取引回数）
- `price_std_mean_ratio`: 価格のばらつきと平均の比率
- `recent_activity_ratio`: 最近の活動率（直近30日の取引回数 / 全体の取引回数）

---

### 2.7 欠損値・異常値の処理

データの品質を向上させるために、欠損値と異常値を処理します。

#### 実装例

```python
# 欠損値の補完（数値列は0で補完、カテゴリ列は最頻値で補完）
numeric_columns = features.select_dtypes(include=[np.number]).columns
features[numeric_columns] = features[numeric_columns].fillna(0)

# 無限大の処理
features = features.replace([np.inf, -np.inf], 0)

# 外れ値のクリッピング（オプション）
# 各数値列の1%点と99%点でクリッピング
for col in numeric_columns:
    if col != 'user_id':  # user_idは除外
        q1 = features[col].quantile(0.01)
        q99 = features[col].quantile(0.99)
        features[col] = features[col].clip(lower=q1, upper=q99)
```

#### 処理の説明

- **欠損値の補完**: 数値列は0で補完（または平均値、中央値など）
- **無限大の処理**: 0に置き換え（除算などで発生する可能性がある）
- **外れ値のクリッピング**: 極端な値を1%点と99%点でクリッピング（オプション）

---

## 3. 完全な実装例

以下は、上記のすべての特徴エンジニアリングを統合した完全なコード例です。

```python
import pandas as pd
import numpy as np

# データの読み込み
data = pd.read_csv("../input/japan-ai-cup/data.csv")
data['date'] = pd.to_datetime(data['date'])

# 基準日を設定
reference_date = data['date'].max()

# 基本特徴の作成
features = pd.DataFrame()
features['user_id'] = data['user_id'].unique()

# ===== 1. 基本統計特徴 =====
agg_dict = {
    'date': ['count', 'min', 'max'],
    'average_unit_price': ['sum', 'mean', 'median', 'std', 'min', 'max']
}
basic_features = data.groupby('user_id').agg(agg_dict)
basic_features.columns = ['_'.join(col).strip() for col in basic_features.columns.values]
basic_features = basic_features.reset_index()
basic_features.columns = ['user_id', 'transaction_count', 'date_min', 'date_max',
                          'price_sum', 'price_mean', 'price_median', 'price_std',
                          'price_min', 'price_max']

# 時系列特徴
basic_features['date_range'] = (pd.to_datetime(basic_features['date_max']) - 
                                pd.to_datetime(basic_features['date_min'])).dt.days
basic_features['days_since_last'] = (reference_date - pd.to_datetime(basic_features['date_max'])).dt.days
basic_features['days_since_first'] = (reference_date - pd.to_datetime(basic_features['date_min'])).dt.days
basic_features['avg_days_between_transactions'] = basic_features['date_range'] / (basic_features['transaction_count'] - 1)
basic_features['avg_days_between_transactions'] = basic_features['avg_days_between_transactions'].fillna(0)

# 価格関連の追加特徴
basic_features['price_range'] = basic_features['price_max'] - basic_features['price_min']
basic_features['price_cv'] = basic_features['price_std'] / (basic_features['price_mean'] + 1e-6)

features = pd.merge(features, basic_features, on='user_id', how='left')

# ===== 2. 時間窓特徴 =====
for window in [30, 60, 90]:
    recent = data[data['date'] >= (reference_date - pd.Timedelta(days=window))]
    if len(recent) > 0:
        recent_features = recent.groupby('user_id').agg({
            'date': 'count',
            'average_unit_price': ['sum', 'mean']
        })
        recent_features.columns = [f'last_{window}d_' + '_'.join(col).strip() 
                                   for col in recent_features.columns.values]
        recent_features = recent_features.reset_index()
        recent_features.columns = ['user_id', f'transactions_last_{window}d',
                                   f'price_sum_last_{window}d', f'price_mean_last_{window}d']
        features = pd.merge(features, recent_features, on='user_id', how='left')

# ===== 3. トレンド特徴 =====
mid_date = data['date'].quantile(0.5)
first_half = data[data['date'] <= mid_date]
second_half = data[data['date'] > mid_date]

first_half_features = first_half.groupby('user_id').agg({
    'date': 'count',
    'average_unit_price': 'mean'
}).reset_index()
first_half_features.columns = ['user_id', 'transaction_count_first_half', 'price_mean_first_half']

second_half_features = second_half.groupby('user_id').agg({
    'date': 'count',
    'average_unit_price': 'mean'
}).reset_index()
second_half_features.columns = ['user_id', 'transaction_count_second_half', 'price_mean_second_half']

trend_features = pd.merge(first_half_features, second_half_features, on='user_id', how='outer')
trend_features = trend_features.fillna(0)
trend_features['transaction_trend'] = trend_features['transaction_count_second_half'] / (trend_features['transaction_count_first_half'] + 1)
trend_features['price_trend'] = trend_features['price_mean_second_half'] / (trend_features['price_mean_first_half'] + 1)

features = pd.merge(features, trend_features[['user_id', 'transaction_trend', 'price_trend']], 
                    on='user_id', how='left')

# ===== 4. 周期性特徴 =====
data['weekday'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month

most_active_weekday = data.groupby('user_id')['weekday'].agg(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else -1
).reset_index()
most_active_weekday.columns = ['user_id', 'most_active_weekday']

unique_weekdays = data.groupby('user_id')['weekday'].nunique().reset_index()
unique_weekdays.columns = ['user_id', 'unique_weekdays']

unique_months = data.groupby('user_id')['month'].nunique().reset_index()
unique_months.columns = ['user_id', 'unique_months']

features = pd.merge(features, most_active_weekday, on='user_id', how='left')
features = pd.merge(features, unique_weekdays, on='user_id', how='left')
features = pd.merge(features, unique_months, on='user_id', how='left')

# ===== 5. インタラクション特徴 =====
features['total_value'] = features['transaction_count'] * features['price_mean']
features['value_per_day'] = features['total_value'] / (features['date_range'] + 1)
features['transaction_frequency'] = features['transaction_count'] / (features['date_range'] + 1)
features['price_std_mean_ratio'] = features['price_std'] / (features['price_mean'] + 1e-6)

if 'transactions_last_30d' in features.columns:
    features['recent_activity_ratio'] = features['transactions_last_30d'] / (features['transaction_count'] + 1)

# ===== 6. 欠損値・異常値の処理 =====
numeric_columns = features.select_dtypes(include=[np.number]).columns
features[numeric_columns] = features[numeric_columns].fillna(0)
features = features.replace([np.inf, -np.inf], 0)

# 外れ値のクリッピング（オプション）
for col in numeric_columns:
    if col != 'user_id':
        q1 = features[col].quantile(0.01)
        q99 = features[col].quantile(0.99)
        features[col] = features[col].clip(lower=q1, upper=q99)

# 最終的な特徴の確認
print(f"特徴数: {len(features.columns) - 1}")  # user_idを除く
print(f"特徴名: {list(features.columns)}")
features.head()
```

---

## 4. 使用方法

### 4.1 既存のノートブックへの統合

上記の完全な実装例を、ノートブックのCell 13の代わりに使用してください。

```python
# 既存のコード（Cell 13）を以下のコードに置き換え
# ... 完全な実装例のコードをここに貼り付け ...
```

### 4.2 学習データとテストデータの準備

特徴エンジニアリング後、既存のコード（Cell 15-18）と同じように学習データとテストデータを準備します。

```python
# 学習データの準備
X_train = pd.merge(train_flag, features, on="user_id", how="left")
X_train = X_train.drop(["user_id", "churn"], axis=1)
y_train = train_flag["churn"].values

# テストデータの準備
X_test = pd.merge(sub, features, on="user_id", how="left")
X_test = X_test.drop(["user_id", "pred"], axis=1)
```

### 4.3 モデルの学習

既存のLightGBMのコード（Cell 19）をそのまま使用できます。特徴数が増えることで、モデルの性能が向上する可能性があります。

### 4.4 特徴の重要度の確認

モデル学習後、特徴の重要度を確認することで、どの特徴が重要かを把握できます。

```python
# 特徴の重要度を確認（最初のfoldのモデルを使用）
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': models[0].feature_importance(importance_type='gain')
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print(feature_importance.head(20))
```

---

## 5. まとめ

本チュートリアルでは、以下の特徴エンジニアリング手法を実装しました：

1. **時系列特徴**: 日付関連の特徴量で、ユーザーの行動の時間的パターンを捉える
2. **統計的特徴**: 価格データの統計量で、消費パターンを詳しく分析
3. **時間窓特徴**: 最近の行動パターンで、離脱の兆候を捉える
4. **トレンド特徴**: 行動の変化を捉える特徴で、離脱リスクを評価
5. **周期性特徴**: 曜日・月などの周期性で、行動パターンを理解
6. **インタラクション特徴**: 特徴同士の組み合わせで、より複雑な関係を捉える
7. **欠損値・異常値の処理**: データの品質を向上させる

これらの特徴を追加することで、モデルの予測性能が向上する可能性があります。特に、時系列特徴と時間窓特徴は、顧客離脱予測において非常に重要です。

### 次のステップ

- 特徴選択: 重要度の低い特徴を削除してモデルを簡素化
- ハイパーパラメータの調整: 特徴数が増えたため、LightGBMのパラメータを再調整
- クロスバリデーション: 各foldでの性能を確認し、過学習を防ぐ
- アンサンブル: 複数のモデルを組み合わせて性能を向上

