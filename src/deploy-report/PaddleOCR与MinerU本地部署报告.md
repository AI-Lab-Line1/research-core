# PaddleOCR 与 MinerU 本地部署报告

## 一、部署背景

在 Windows 环境下完成光学字符识别（OCR）和 PDF 文档解析工具的本地化部署，选用 **PaddleOCR** 作为 OCR 引擎、**MinerU**（magic-pdf）作为 PDF 解析工具，底层依赖 PaddlePaddle GPU 版本以利用硬件加速。

---

## 二、软硬件环境

| 项目 | 规格 |
|------|------|
| 操作系统 | Microsoft Windows |
| GPU 加速 | 可用（CUDA 13.1） |
| Python 解释器 | 3.10 |
| 虚拟环境 | paddle_vl（conda 管理） |
| PaddlePaddle | 3.3.1（gpu 版） |
| PaddleOCR | 通过 pip 安装的最新版 |
| MinerU | 1.3.12（magic-pdf[full]） |
| 最终部署状态 | 成功 |

---

## 三、部署流程

### 3.1 创建隔离环境

```bash
conda create -n paddle_vl python=3.10
conda activate paddle_vl
```

### 3.2 安装 PaddlePaddle GPU 版

```bash
python -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

### 3.3 验证 GPU 加速

```bash
python -c "import paddle; print(paddle.is_compiled_with_cuda())"
# 输出：True
```

### 3.4 安装 PaddleOCR

```bash
pip install paddleocr
```

### 3.5 安装 MinerU

```bash
pip install magic-pdf[full]
```

### 3.6 验证 PaddleOCR 导入

```bash
python -c "from paddleocr import PaddleOCR; print('OK')"
```

### 3.7 验证 MinerU 导入

```bash
python -c "import magic_pdf; print('MinerU OK')"
```

---

## 四、使用方法

### 4.1 PaddleOCR 图片文字识别

```bash
python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(use_angle_cls=True, lang='ch'); result = ocr.ocr('图片路径.jpg', cls=True); print(result)"
```

### 4.2 MinerU PDF 文档解析

```bash
magic-pdf pdf-parse --pdf 文档.pdf --output 输出文件夹
```

---

## 五、验证结果

| 检查项 | 结果 |
|--------|------|
| 虚拟环境创建与激活 | 通过 |
| PaddlePaddle GPU 检测 | True |
| PaddleOCR 模块导入 | 正常 |
| MinerU 模块导入 | 正常 |
