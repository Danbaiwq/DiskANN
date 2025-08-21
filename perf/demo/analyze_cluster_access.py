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
    if cluster_file and os.path.exists(cluster_file):
        try:
            cluster_df = pd.read_csv(cluster_file)
            cluster_sizes = cluster_df['vector_count'].values
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
    
    # Find hot and cold clusters
    non_zero_accesses = access_counts[access_counts > 0]
    zero_access_count = np.sum(access_counts == 0)
    hot_clusters = np.argsort(access_counts)[-10:][::-1]  # Top 10 most accessed
    
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
    for i, cluster_id in enumerate(hot_clusters):
        size_info = f" (size: {cluster_sizes[cluster_id]})" if cluster_sizes is not None else ""
        print(f"  {i+1}. Cluster {cluster_id}: {access_counts[cluster_id]} accesses{size_info}")
    
    # Generate plots
    fig_rows = 3 if cluster_sizes is not None else 2
    plt.figure(figsize=(16, fig_rows * 4))
    
    # Plot 1: Access count histogram
    plt.subplot(fig_rows, 3, 1)
    plt.hist(access_counts, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    plt.axvline(mean_access, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_access:.1f}')
    plt.axvline(median_access, color='green', linestyle='--', linewidth=2, label=f'Median: {median_access:.1f}')
    plt.xlabel('Access Count per Cluster')
    plt.ylabel('Number of Clusters')
    plt.title('Distribution of Cluster Access Counts')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Box plot
    plt.subplot(fig_rows, 3, 2)
    plt.boxplot(access_counts, vert=True)
    plt.ylabel('Access Count per Cluster')
    plt.title('Box Plot of Access Counts')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative distribution
    plt.subplot(fig_rows, 3, 3)
    sorted_counts = np.sort(access_counts)
    cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    plt.plot(sorted_counts, cumulative, linewidth=2)
    plt.xlabel('Access Count per Cluster')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Access count by cluster ID (all clusters)
    plt.subplot(fig_rows, 3, 4)
    plt.bar(cluster_ids, access_counts, alpha=0.7, color='orange', width=max(1, total_clusters//200))
    plt.axhline(mean_access, color='blue', linestyle='--', alpha=0.8, label=f'Mean: {mean_access:.1f}')
    plt.xlabel('Cluster ID')
    plt.ylabel('Access Count')
    plt.title('Access Count by Cluster ID (All Clusters)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Top 50 clusters detailed view
    plt.subplot(fig_rows, 3, 5)
    show_detail = min(50, total_clusters)
    plt.bar(range(show_detail), access_counts[:show_detail], alpha=0.8, color='purple')
    plt.axhline(mean_access, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_access:.1f}')
    plt.xlabel('Cluster ID')
    plt.ylabel('Access Count')
    plt.title(f'Access Count by Cluster ID (First {show_detail} Clusters)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Hot clusters (top 20)
    plt.subplot(fig_rows, 3, 6)
    top_20 = min(20, len(hot_clusters))
    top_cluster_ids = hot_clusters[:top_20]
    top_access_counts = access_counts[top_cluster_ids]
    plt.bar(range(top_20), top_access_counts, alpha=0.8, color='red')
    plt.xlabel('Rank (Most Accessed)')
    plt.ylabel('Access Count')
    plt.title(f'Top {top_20} Most Accessed Clusters')
    plt.xticks(range(top_20), [f'C{cid}' for cid in top_cluster_ids], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 7-9: Correlation with cluster sizes (if available)
    if cluster_sizes is not None:
        # Plot 7: Access vs Size scatter
        plt.subplot(fig_rows, 3, 7)
        plt.scatter(cluster_sizes, access_counts, alpha=0.6, s=20)
        plt.xlabel('Cluster Size (Vector Count)')
        plt.ylabel('Access Count')
        plt.title('Access Count vs Cluster Size')
        plt.grid(True, alpha=0.3)
        
        # Calculate correlation
        correlation = np.corrcoef(cluster_sizes, access_counts)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        # Plot 8: Access rate (access/size) histogram
        plt.subplot(fig_rows, 3, 8)
        # Avoid division by zero
        access_rates = np.divide(access_counts, cluster_sizes, 
                               out=np.zeros_like(access_counts, dtype=float), 
                               where=cluster_sizes!=0)
        plt.hist(access_rates, bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Access Rate (Access/Size)')
        plt.ylabel('Number of Clusters')
        plt.title('Distribution of Access Rates')
        plt.grid(True, alpha=0.3)
        
        # Plot 9: Size vs Access rate scatter
        plt.subplot(fig_rows, 3, 9)
        plt.scatter(cluster_sizes, access_rates, alpha=0.6, s=20, color='brown')
        plt.xlabel('Cluster Size (Vector Count)')
        plt.ylabel('Access Rate (Access/Size)')
        plt.title('Cluster Size vs Access Rate')
        plt.grid(True, alpha=0.3)
    
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
        for i, cluster_id in enumerate(hot_clusters[:10]):
            size_info = f" (size: {cluster_sizes[cluster_id]})" if cluster_sizes is not None else ""
            f.write(f"  {i+1}. Cluster {cluster_id}: {access_counts[cluster_id]} accesses{size_info}\n")
        
        if cluster_sizes is not None:
            correlation = np.corrcoef(cluster_sizes, access_counts)[0, 1]
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