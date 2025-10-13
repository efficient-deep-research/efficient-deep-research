#!/bin/bash
set -euxo pipefail

# nginxの一時ディレクトリを作成する
# 作成しない場合，nginxが起動時にキャッシュ用ディレクトリに書き込もうとするけど，
# Singularityイメージ内のvar/はRead-Onlyのため，エラーが出てしまう
mkdir -p ./nginx_cache/{client_temp,proxy_temp,fastcgi_temp,uwsgi_temp,scgi_temp}
mkdir -p ./nginx_run

singularity exec \
    --bind ./nginx.conf:/etc/nginx/nginx.conf \
    --bind ./nginx.conf:/etc/nginx/conf.d/nginx.conf \
    --bind ./.htpasswd:/etc/nginx/conf.d/.htpasswd \
    --bind ./nginx_cache:/var/cache/nginx \
    --bind ./nginx_run:/run \
    nginx_latest.sif \
    nginx -g "daemon off;"
