#!/usr/bin/env python3
"""
Cluster Access Statistics Analysis Tool
Reads cluster_access_stats.txt and generates visualization and statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

def analyze_cluster_access(access_file, cluster_file=None, output_dir=None):
    """Analyze cluster access statistics"""
    if not os.path.exists(access_file):
        print(f"Error: Access statistics file does not exist {access_file}")
        return False
    
    # Read access data
    try:
        access_df = pd.read_csv(access_file)
        cluster_ids = access_df['cluster_id'].values
        access_counts = access_df['access_count'].values
    except Exception as e:
        print(f"Failed to read access file: {e}")
        return False
    
    # Read cluster size data if available
    cluster_sizes = None
    size_map = None
    aligned_sizes = None
    aligned_access = None
    if cluster_file and os.path.exists(cluster_file):
        try:
            cluster_df = pd.read_csv(cluster_file)
            # Expect columns: cluster_id, vector_count
            if 'cluster_id' in cluster_df.columns and 'vector_count' in cluster_df.columns:
                cluster_sizes = cluster_df['vector_count'].values
                size_map = dict(zip(cluster_df['cluster_id'].values, cluster_df['vector_count'].values))
                # Align by cluster_id to ensure same length for scatter/correlation
                merged = pd.merge(
                    access_df[['cluster_id', 'access_count']],
                    cluster_df[['cluster_id', 'vector_count']],
                    on='cluster_id', how='inner'
                )
                if not merged.empty:
                    aligned_access = merged['access_count'].values
                    aligned_sizes = merged['vector_count'].values
            print(f"Loaded cluster size data from {cluster_file}")
        except Exception as e:
            print(f"Warning: Failed to read cluster file: {e}")
    
    # Calculate statistics
    total_clusters = len(access_counts)
    total_accesses = np.sum(access_counts)
    mean_access = np.mean(access_counts)
    std_access = np.std(access_counts)
    variance_access = np.var(access_counts)
    min_access = np.min(access_counts)
    max_access = np.max(access_counts)
    median_access = np.median(access_counts)
    
    # Find hot and cold clusters (use positions, then map to actual IDs)
    non_zero_accesses = access_counts[access_counts > 0]
    zero_access_count = np.sum(access_counts == 0)
    order = np.argsort(access_counts)
    hot_pos = order[-10:][::-1]  # positions of top 10
    hot_ids = cluster_ids[hot_pos]
    
    # Print statistics
    print("=" * 60)
    print("Cluster Access Statistics")
    print("=" * 60)
    print(f"Total Clusters:        {total_clusters}")
    print(f"Total Accesses:        {total_accesses}")
    print(f"Mean Access Count:     {mean_access:.2f}")
    print(f"Standard Deviation:    {std_access:.2f}")
    print(f"Variance:              {variance_access:.2f}")
    print(f"Minimum:               {min_access}")
    print(f"Maximum:               {max_access}")
    print(f"Median:                {median_access:.2f}")
    print(f"Clusters Never Accessed: {zero_access_count} ({zero_access_count/total_clusters*100:.1f}%)")
    print(f"Clusters Accessed:     {len(non_zero_accesses)} ({len(non_zero_accesses)/total_clusters*100:.1f}%)")
    if len(non_zero_accesses) > 0:
        print(f"Coeff. of Variation:   {(std_access/mean_access)*100:.2f}%")
    print("=" * 60)
    
    print("\nTop 10 Most Accessed Clusters:")
    for i, pos in enumerate(hot_pos):
        cid = int(cluster_ids[pos])
        size_info = f" (size: {size_map.get(cid)})" if size_map is not None and cid in size_map else ""
        print(f"  {i+1}. Cluster {cid}: {int(access_counts[pos])} accesses{size_info}")
    
    # Generate a single plot: Top-50 clusters by access
    plt.figure(figsize=(16, 6))
    top_n = 50
    # Prefer aligned merge with sizes
    if cluster_file and aligned_access is not None and aligned_sizes is not None and len(aligned_access) > 0:
        merged = pd.merge(
            access_df[['cluster_id', 'access_count']],
            pd.DataFrame({'cluster_id': list(size_map.keys()), 'vector_count': list(size_map.values())}),
            on='cluster_id', how='inner'
        )
        merged.sort_values('access_count', ascending=False, inplace=True)
        top = merged.head(top_n)
        x_ids = top['cluster_id'].astype(int).values
        y_sizes = top['vector_count'].values
        y_access = top['access_count'].values
        x_pos = np.arange(len(x_ids))
        bars = plt.bar(x_pos, y_sizes, color='skyblue', edgecolor='black')
        for i, b in enumerate(bars):
            plt.text(b.get_x() + b.get_width()/2.0, b.get_height()*0.5, str(int(y_access[i])),
                     ha='center', va='center', fontsize=8, color='black')
        plt.xticks(x_pos, [str(cid) for cid in x_ids], rotation=45)
        plt.xlabel('Cluster ID')
        plt.ylabel('Cluster Size (Vector Count)')
        plt.title(f'Top-{len(x_ids)} Clusters: Bar=Size, Label=Access Count')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
    else:
        # Fallback: no size data, show access-only top-N
        order_desc = np.argsort(access_counts)[::-1]
        sel = order_desc[:top_n]
        x_ids = cluster_ids[sel]
        y_access = access_counts[sel]
        x_pos = np.arange(len(x_ids))
        bars = plt.bar(x_pos, y_access, color='orange', edgecolor='black')
        for i, b in enumerate(bars):
            plt.text(b.get_x() + b.get_width()/2.0, b.get_height()*0.5, str(int(y_access[i])),
                     ha='center', va='center', fontsize=8, color='black')
        plt.xticks(x_pos, [str(int(cid)) for cid in x_ids], rotation=45)
        plt.xlabel('Cluster ID')
        plt.ylabel('Access Count')
        plt.title(f'Top-{len(x_ids)} Clusters by Access (size unavailable)')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

    # Save plots
    if output_dir is None:
        output_dir = os.path.dirname(access_file)
 
    plot_file = os.path.join(output_dir, 'cluster_access_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Analysis plots saved to: {plot_file}")
    
    # Save detailed statistics
    stats_output = os.path.join(output_dir, 'cluster_access_summary.txt')
    with open(stats_output, 'w', encoding='utf-8') as f:
        f.write("Cluster Access Statistics Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total Clusters: {total_clusters}\n")
        f.write(f"Total Accesses: {total_accesses}\n")
        f.write(f"Mean Access Count: {mean_access:.4f}\n")
        f.write(f"Standard Deviation: {std_access:.4f}\n")
        f.write(f"Variance: {variance_access:.4f}\n")
        f.write(f"Minimum: {min_access}\n")
        f.write(f"Maximum: {max_access}\n")
        f.write(f"Median: {median_access:.4f}\n")
        f.write(f"Clusters Never Accessed: {zero_access_count} ({zero_access_count/total_clusters*100:.2f}%)\n")
        f.write(f"Clusters Accessed: {len(non_zero_accesses)} ({len(non_zero_accesses)/total_clusters*100:.2f}%)\n")
        if len(non_zero_accesses) > 0:
            f.write(f"Coeff. of Variation: {(std_access/mean_access)*100:.4f}%\n")
        
        f.write("\nPercentile Information:\n")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            percentile = np.percentile(access_counts, p)
            f.write(f"  {p}th percentile: {percentile:.2f}\n")
        
        f.write(f"\nTop 10 Most Accessed Clusters:\n")
        for i, pos in enumerate(hot_pos[:10]):
            cid = int(cluster_ids[pos])
            size_info = f" (size: {size_map.get(cid)})" if size_map is not None and cid in size_map else ""
            f.write(f"  {i+1}. Cluster {cid}: {int(access_counts[pos])} accesses{size_info}\n")
        
        if aligned_sizes is not None and aligned_access is not None and len(aligned_sizes) > 0:
            correlation = np.corrcoef(aligned_sizes, aligned_access)[0, 1]
            f.write(f"\nCorrelation with Cluster Size: {correlation:.4f}\n")
    
    print(f"Detailed statistics saved to: {stats_output}")
    
    # Display plots if environment supports
    try:
        if os.environ.get('DISPLAY') and os.environ.get('PLOT_SHOW', '0') != '0':
            plt.show()
    except:
        pass
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Analyze cluster access statistics')
    parser.add_argument('access_file', nargs='?', default='cluster_access_stats.txt',
                       help='Cluster access statistics file path (default: cluster_access_stats.txt)')
    parser.add_argument('-c', '--cluster-file', help='Cluster size statistics file (cluster_stats.txt)')
    parser.add_argument('-o', '--output', help='Output directory (default: same as input file)')
    
    args = parser.parse_args()
    
    # Try to find files in common locations
    access_file = args.access_file
    if not os.path.exists(access_file):
        alt_path = os.path.join('/data/1/demo', os.path.basename(access_file))
        if os.path.exists(alt_path):
            access_file = alt_path
        else:
            print(f"Error: Cannot find access statistics file {args.access_file}")
            print("Please ensure you have run the search script to generate cluster_access_stats.txt")
            return 1
    
    # Try to find cluster size file
    cluster_file = args.cluster_file
    if not cluster_file:
        # Try to find cluster_stats.txt in the same directory
        cluster_file = os.path.join(os.path.dirname(access_file), 'cluster_stats.txt')
        if not os.path.exists(cluster_file):
            cluster_file = None
    
    success = analyze_cluster_access(access_file, cluster_file, args.output)
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main()) 