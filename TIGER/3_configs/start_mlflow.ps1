#参数声明
param(
    [string]$Python = "D:/03_Environments/02_IDEs/miniforge/envs/rec-tiger/python.exe"
)

# 错误处理策略
$ErrorActionPreference = "Stop"

#计算项目根目录
$projectRoot = Split-Path -Parent $PSScriptRoot

#读取配置文件（保留原始文本）
$configText = Get-Content -LiteralPath (Join-Path $PSScriptRoot "mlflow.json") -Raw -Encoding UTF8

#去掉 JSON 里的 // 行注释
$configText = [regex]::Replace($configText, '"(?:\\.|[^"\\])*"|//[^\r\n]*', {
    param($match)
    if ($match.Value.StartsWith('//')) { '' } else { $match.Value }
})

#解析 JSON
$config = $configText | ConvertFrom-Json

#解析出数据库路径与 artifact 路径（转为绝对路径）
$databasePath = [IO.Path]::GetFullPath((Join-Path $projectRoot $config.backend_store_path))
$artifactPath = [IO.Path]::GetFullPath((Join-Path $projectRoot $config.artifact_root))

#解析 tracking URI 并校验
$endpoint = [uri]$config.tracking_uri
if ($endpoint.Scheme -ne "http" -or $endpoint.Host -ne "127.0.0.1" -or $endpoint.AbsolutePath -ne "/") {
    throw "本地启动脚本要求 tracking_uri 为 http://127.0.0.1:端口。"
}

#确保目录存在
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $databasePath), $artifactPath | Out-Null
$databaseUri = "sqlite:///" + $databasePath.Replace("\", "/")
$artifactUri = ([uri]$artifactPath).AbsoluteUri

& $Python -m mlflow server  `
    --backend-store-uri $databaseUri  `
    --serve-artifacts  `
    --artifacts-destination $artifactUri  `
    --host $endpoint.Host  `
    --port $endpoint.Port
exit $LASTEXITCODE
