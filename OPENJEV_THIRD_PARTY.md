# Open-Jev 上游源码（third_party）

本仓库**不包含** `openjev_generative_rec/third_party/Open-Jev/` 的源码，
原因是本机该目录的文件权限异常、无法读取/打包，且它属于第三方项目。

- 上游提交：`bd4118882f733574a3250a4b65fe4d884130c08b`
- 同目录下 `UPSTREAM_COMMIT` 文件保存同一值
- 本仓库只包含 `openjev_generative_rec/pipeline/` 里自己写的适配层

恢复方式（在能正常访问该目录的机器上）：

    git clone <Open-Jev 上游地址> openjev_generative_rec/third_party/Open-Jev
    cd openjev_generative_rec/third_party/Open-Jev
    git checkout bd4118882f733574a3250a4b65fe4d884130c08b

本地恢复好之后，从 `.gitignore` 里删掉对应那一行，即可把源码一起提交。