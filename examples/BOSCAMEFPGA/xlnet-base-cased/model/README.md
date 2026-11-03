固定模型：xlnet/xlnet-base-cased，revision见manifest.json。
config.json直接来自hf-mirror，原始文件hash记录在manifest。
未将大checkpoint加入仓库。tools/fetch_assets.py默认仅下载config与SentencePiece；
显式--weights下载完整checkpoint，tools/prepare_weights.py转换W8A8矩阵并保留FP32词表。
完整模型运行需要按model_runtime/model.h为权重、24MiB scratch、hidden以及可选logits分配独立64字节对齐缓冲。
