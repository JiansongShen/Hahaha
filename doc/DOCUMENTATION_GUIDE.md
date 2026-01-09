# 文档生成指南

本文档介绍如何为项目生成和维护API文档。

## 本地生成文档

### 前提条件

- 安装 Doxygen (`sudo apt-get install doxygen` 或 `brew install doxygen`)
- 安装 Graphviz (`sudo apt-get install graphviz` 或 `brew install graphviz`) 用于生成图表

### 生成文档

```bash
# 使用配置文件生成文档
doxygen Doxyfile

# 或使用提供的脚本
./dev/build_docs.sh
```

生成的文档将位于 `doc/doxygen/html/` 目录中。

## CI/CD 集成

项目使用 GitHub Actions 自动构建文档：

- 每当推送到 `main`、`master` 或 `develop` 分支时，文档会自动构建
- 拉取请求也会构建文档以验证文档生成过程正常工作
- 文档作为 artifacts 保存，可在构建后下载

### 工作流文件

- `.github/workflows/ci-docs.yml` - 文档构建工作流
- 该工作流会在每次提交时验证文档生成是否正常工作

### 如果要部署到GitHub Pages

如果你想将文档部署到GitHub Pages，需要：

1. 进入仓库的 Settings -> Pages
2. 在 "Source" 部分选择 "GitHub Actions"
3. 然后可以使用以下更新的工作流配置：

```yaml
name: Build and Deploy Documentation

on:
  push:
    branches: [ "main", "master" ]

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  deploy-documentation:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Install Doxygen
      run: |
        sudo apt-get update
        sudo apt-get install -y doxygen graphviz

    - name: Generate Documentation
      run: |
        doxygen Doxyfile

    - name: Setup Pages
      uses: actions/configure-pages@v4

    - name: Upload artifact
      uses: actions/upload-pages-artifact@v3
      with:
        path: 'doc/doxygen/html'

    - name: Deploy to GitHub Pages
      id: deployment
      uses: actions/deploy-pages@v4
```

## 配置说明

主要配置文件是 `Doxyfile`，其中的关键设置包括：

- `INPUT` - 源代码目录
- `OUTPUT_DIRECTORY` - 输出目录
- `PROJECT_NAME` - 项目名称
- `GENERATE_HTML` - 启用HTML输出
- `GENERATE_LATEX` - 启用LaTeX输出（CI中已禁用以加快构建速度）

## 自定义样式

- `custom_style.css` - 自定义CSS样式，使文档更美观

## 故障排除

如果文档生成失败，请检查：

1. 所有依赖项是否已安装
2. Doxyfile 中的路径是否正确
3. 源代码是否有适当的注释
