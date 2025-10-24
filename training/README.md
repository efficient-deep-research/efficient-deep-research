# 学習コードについて

## はじめに
- `training`ディレクトリ内に移動してから実行してください．
- `training`ディレクトリ下に`.env`ファイルを作成し，wandbのAPIキーを設定してください．
- 追記：`.env`にキーを設定してください

## Singularity
- これらの学習コードはABCI上で実行することを想定して作成しました．
- `scripts/build_ms-swift_container.sh`を実行すると，`scripts/config.sh`に従ってコンテナをビルドします．
- 追記：自分の都合で`HF_HOME`を使っていないので、使うなら`scripts/config.sh`にuncommetしてください

## データ
- `training/data`にデータを配置してください．
- MS-SwiftのDPO，SimPO用のデータセット形式に変換するには，`scripts/format_data.sh`を実行してください．
    - `DATASET_NAME`, `OUTPUT_NAME`を設定してから実行してください．

## 学習
- `scripts/config.sh`にファイル等の変数を設定してください．
- `configs/base.json`にハイパーパラメータを設定してください．
- 追記：`scripts/run_DPO.sh`に学習に関わる設定を移動しましたので、適時確認してください．
- 追記：array jobの実行の仕方は、`scripts/sweep_beta.sh` & `scripts/sweep_data.sh`に参照してください
- 学習を開始するには，`scripts/run_DPO.sh`を実行してください．

## マージとアップロード
- `scripts/run_DPO.sh`のモデルチェックポイントのパスを変えてから実行してください．
