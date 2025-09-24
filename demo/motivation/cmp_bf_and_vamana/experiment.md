# cmp with vamana

## 1 处理数据集

将数据集处理为 1K/10K/50K/100K/250K/500K/1M 的 fvecs

- input: 输入 fvecs 文件（需要是 1M）
- -o: 输出目录，默认是 ./
- -b: 输出文件前缀

```
# 生成当前目录输出（文件名前缀继承输入名），输出 1K/10K/50K/100K/250K/500K/1M
/home/danbai.wq/DiskANN/demo/motivation/cmp_bf_and_vamana/split_fvecs.py /path/to/sift_1M.fvecs

# 指定输出目录与前缀
/home/danbai.wq/DiskANN/demo/motivation/cmp_bf_and_vamana/split_fvecs.py /data/dataset/gist/gist_test/gist_base.fvecs -o /data/dataset/gist/gist_test -b gist
```

## 2 计算相应的 gt

已记录在脚本 `run_prepare_and_gt.sh` 内

```
# 将 base fvecs 转为 fbin
./apps/utils/fvecs_to_bin float /data/dataset/gist/gist_test/gist_1M.fvecs /data/dataset/gist/gist_test/gist_1M.fbin

# 将 query fvecs 转为 fbin
# ./apps/utils/fvecs_to_bin float /data/dataset/gist/gist_query.fvecs ./data/gist_query.fbin

# 计算 query 和 base 的 gt
./apps/utils/compute_groundtruth  --data_type float --dist_fn l2 --base_file  /data/dataset/gist/gist_test/gist_1K.fbin --query_file   /data/dataset/gist/gist_test/gist_query.fbin --gt_file ./data/gist_query_1K_gt10 --K 10
```

## 3 构建查询 memory vamana

脚本：

```bash
/home/danbai.wq/DiskANN/demo/motivation/cmp_bf_and_vamana/run_memory_vamana.sh
```

脚本内容：

```
./apps/build_memory_index  --data_type float --dist_fn l2 --data_path /data/dataset/{sift, gist}/{sift, gist}_test/{sift, gist}_{1K,10K,50K,100K,500K,1M}.fbin --index_path_prefix /data/dataset/{sift, gist}/{sift, gist}_test/index_{sift, gist}_{1K,10K,50K,100K,500K,1M}_R32_L50_A1.2 -R 32 -L 50 --alpha 1.2

./apps/search_memory_index  --data_type float --dist_fn l2 --index_path_prefix /data/dataset/{sift, gist}/{sift, gist}_test/index_{sift, gist}_{1K,10K,50K,100K,500K,1M}_R32_L50_A1.2 --query_file /data/dataset/{sift, gist}/{sift, gist}_test/{sift, gist}_query.fbin  --gt_file ./data/{sift, gist}_query_{1K,10K,50K,100K,500K,1M}_gt10 -K 10 -L 10 20 30 40 50 100 --result_path /data/dataset/{sift, gist}/{sift, gist}_test/res
```



## 4 运行暴搜获取暴搜性能

脚本：

```
# 使用默认数据根 /data/dataset
/home/danbai.wq/DiskANN/demo/motivation/cmp_bf_and_vamana/run_bf_simd.sh

# 可覆盖数据根
DATA_ROOT=/data/dataset \
/home/danbai.wq/DiskANN/demo/motivation/cmp_bf_and_vamana/run_bf_simd.sh
```

脚本内容：

```
# L2，输出QPS，不落盘结果
/home/danbai.wq/DiskANN/demo/motivation/cmp_bf_and_vamana/bf_simd_search /data/dataset/{sift, gist}/{sift, gist}_test/{sift, gist}_{1K,10K,50K,100K,500K,1M}.fbin /data/dataset/{sift, gist}/{sift, gist}_test/{sift, gist}_query.fbin -k 10 --metric l2

# 内积(IP)，保存每个查询的topK id到文本
/home/danbai.wq/DiskANN/demo/motivation/cmp_bf_and_vamana/bf_simd_search /path/to/base.fbin /path/to/query.fbin -k 100 --metric ip --save /path/to/out.txt
```

