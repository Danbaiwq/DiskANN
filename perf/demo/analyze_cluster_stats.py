#!/usr/bin/env python3
"""
簇向量数量分布分析工具
读取 cluster_stats.txt，生成直方图并计算统计信息
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

def analyze_cluster_stats(stats_file, output_dir=None):
    """分析簇统计数据"""
    if not os.path.exists(stats_file):
        print(f"Error: Statistics file does not exist {stats_file}")
        return False
    
    # 读取数据
    try:
        df = pd.read_csv(stats_file)
        vector_counts = df['vector_count'].values
    except Exception as e:
        print(f"Failed to read file: {e}")
        return False
    
    # 计算统计信息
    total_clusters = len(vector_counts)
    total_vectors = np.sum(vector_counts)
    mean_count = np.mean(vector_counts)
    std_count = np.std(vector_counts)
    variance = np.var(vector_counts)
    min_count = np.min(vector_counts)
    max_count = np.max(vector_counts)
    median_count = np.median(vector_counts)
    
    # 打印统计信息
    print("=" * 60)
    print("Cluster Vector Distribution Statistics")
    print("=" * 60)
    print(f"Total Clusters:        {total_clusters}")
    print(f"Total Vectors:         {total_vectors}")
    print(f"Mean Vectors/Cluster:  {mean_count:.2f}")
    print(f"Standard Deviation:    {std_count:.2f}")
    print(f"Variance:              {variance:.2f}")
    print(f"Minimum:               {min_count}")
    print(f"Maximum:               {max_count}")
    print(f"Median:                {median_count:.2f}")
    print(f"Coeff. of Variation:   {(std_count/mean_count)*100:.2f}%")
    print("=" * 60)
    
    # 生成直方图
    plt.figure(figsize=(16, 12))
    
    # 子图1: 直方图
    plt.subplot(2, 3, 1)
    plt.hist(vector_counts, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(mean_count, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_count:.1f}')
    plt.axvline(median_count, color='green', linestyle='--', linewidth=2, label=f'Median: {median_count:.1f}')
    plt.xlabel('Vectors per Cluster')
    plt.ylabel('Number of Clusters')
    plt.title('Cluster Vector Count Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 箱线图
    plt.subplot(2, 3, 2)
    plt.boxplot(vector_counts, vert=True)
    plt.ylabel('Vectors per Cluster')
    plt.title('Box Plot of Vector Counts')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 累积分布
    plt.subplot(2, 3, 3)
    sorted_counts = np.sort(vector_counts)
    cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    plt.plot(sorted_counts, cumulative, linewidth=2)
    plt.xlabel('Vectors per Cluster')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function')
    plt.grid(True, alpha=0.3)
    
    # 子图4: 簇ID vs 向量数量散点图（前100个簇）
    plt.subplot(2, 3, 4)
    show_clusters = min(100, total_clusters)
    plt.scatter(range(show_clusters), vector_counts[:show_clusters], alpha=0.6, s=20)
    plt.axhline(mean_count, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_count:.1f}')
    plt.xlabel(f'Cluster ID (First {show_clusters})')
    plt.ylabel('Vector Count')
    plt.title('Scatter Plot of Vector Counts')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图5: 所有簇ID的向量数量柱状图
    plt.subplot(2, 3, 5)
    cluster_ids = np.arange(total_clusters)
    plt.bar(cluster_ids, vector_counts, alpha=0.7, color='lightcoral', width=max(1, total_clusters//200))
    plt.axhline(mean_count, color='blue', linestyle='--', alpha=0.8, label=f'Mean: {mean_count:.1f}')
    plt.xlabel('Cluster ID')
    plt.ylabel('Vector Count')
    plt.title('Vector Count by Cluster ID (All Clusters)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图6: 前50个簇的详细柱状图
    plt.subplot(2, 3, 6)
    show_detail = min(50, total_clusters)
    plt.bar(range(show_detail), vector_counts[:show_detail], alpha=0.8, color='mediumseagreen')
    plt.axhline(mean_count, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_count:.1f}')
    plt.xlabel('Cluster ID')
    plt.ylabel('Vector Count')
    plt.title(f'Vector Count by Cluster ID (First {show_detail} Clusters)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    if output_dir is None:
        output_dir = os.path.dirname(stats_file)
    
    plot_file = os.path.join(output_dir, 'cluster_distribution_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Analysis plots saved to: {plot_file}")
    
    # 保存详细统计到文件
    stats_output = os.path.join(output_dir, 'cluster_analysis_summary.txt')
    with open(stats_output, 'w', encoding='utf-8') as f:
        f.write("Cluster Vector Distribution Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total Clusters: {total_clusters}\n")
        f.write(f"Total Vectors: {total_vectors}\n")
        f.write(f"Mean Vectors/Cluster: {mean_count:.4f}\n")
        f.write(f"Standard Deviation: {std_count:.4f}\n")
        f.write(f"Variance: {variance:.4f}\n")
        f.write(f"Minimum: {min_count}\n")
        f.write(f"Maximum: {max_count}\n")
        f.write(f"Median: {median_count:.4f}\n")
        f.write(f"Coeff. of Variation: {(std_count/mean_count)*100:.4f}%\n")
        f.write("\nPercentile Information:\n")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            percentile = np.percentile(vector_counts, p)
            f.write(f"  {p}th percentile: {percentile:.2f}\n")
    
    print(f"Detailed statistics saved to: {stats_output}")
    
    # 显示图表（如果环境支持）
    try:
        if os.environ.get('DISPLAY') and os.environ.get('PLOT_SHOW', '0') != '0':
            plt.show()
    except:
        pass
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Analyze cluster vector distribution')
    parser.add_argument('stats_file', nargs='?', default='cluster_stats.txt',
                       help='Cluster statistics file path (default: cluster_stats.txt)')
    parser.add_argument('-o', '--output', help='Output directory (default: same as input file)')
    
    args = parser.parse_args()
    
    # 如果没有提供完整路径，尝试在当前目录和常见路径查找
    stats_file = args.stats_file
    if not os.path.exists(stats_file):
        # 尝试在 /data/1/demo 目录查找
        alt_path = os.path.join('/data/1/demo', os.path.basename(stats_file))
        if os.path.exists(alt_path):
            stats_file = alt_path
        else:
            print(f"Error: Cannot find statistics file {args.stats_file}")
            print("Please ensure you have run the build script to generate cluster_stats.txt")
            return 1
    
    success = analyze_cluster_stats(stats_file, args.output)
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main()) 