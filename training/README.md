# 学習コードについて

## はじめに
- `training`ディレクトリ内に移動してから実行してください．
- `training`ディレクトリ下に`.env`ファイルを作成し，wandbのAPIキーを設定してください．

## Singularity
- これらの学習コードはABCI上で実行することを想定して作成しました．
- `scripts/build_ms-swift_container.sh`を実行すると，`scripts/config.sh`に従ってコンテナをビルドします．

## データ
- `training/data`にデータを配置してください．
- MS-SwiftのDPO，SimPO用のデータセット形式に変換するには，`scripts/format_data.sh`を実行してください．
    - `DATASET_NAME`, `OUTPUT_NAME`を設定してから実行してください．

## 学習
- `scripts/config.sh`にファイル等の変数を設定してください．
- `configs/base.json`にハイパーパラメータを設定してください．
- 学習を開始するには，`scripts/run_DPO.sh`を実行してください．
