#!/usr/bin/env python3
"""
Analyze results across different dataset scales (100, 500, 1000 samples).
"""

import json
from pathlib import Path

def extract_results(log_file):
    """Extract best validation loss from training log."""
    with open(log_file, 'r') as f:
        for line in f:
            if 'Best validation loss:' in line:
                return float(line.split(':')[1].strip())
    return None

def main():
    print("\n" + "=" * 80)
    print("                跨规模实验结果分析")
    print("=" * 80)
    print()
    
    # Results
    results = {
        100: {
            'text_only': 0.1818,
            'text_audio': 0.1585,
        },
        500: {
            'text_only': 0.1321,
            'text_audio': 0.1212,
        },
        1000: {
            'text_only': None,
            'text_audio': None,
        }
    }
    
    # Load 1000-sample results
    text_only_1000 = extract_results('outputs/scale1000_text_only_log.txt')
    text_audio_1000 = extract_results('outputs/scale1000_text_audio_log.txt')
    
    if text_only_1000:
        results[1000]['text_only'] = text_only_1000
    if text_audio_1000:
        results[1000]['text_audio'] = text_audio_1000
    
    # Print results table
    print("┌─────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
    print("│  样本数量   │  纯文本基线  │  文本+情感   │  绝对降低    │  相对提升%   │")
    print("├─────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")
    
    for size in [100, 500, 1000]:
        text_only = results[size]['text_only']
        text_audio = results[size]['text_audio']
        
        if text_only and text_audio:
            abs_diff = text_only - text_audio
            rel_improvement = (abs_diff / text_only) * 100
            print(f"│  {size:4d} 样本  │    {text_only:.4f}    │    {text_audio:.4f}    │    {abs_diff:.4f}    │    {rel_improvement:5.2f}%    │")
        else:
            status = "训练中..." if size == 1000 else "未完成"
            print(f"│  {size:4d} 样本  │    {status:8s}  │")
    
    print("└─────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")
    print()
    
    # Trend analysis
    print("📊 性能趋势分析:")
    print()
    
    if results[1000]['text_only'] and results[1000]['text_audio']:
        print("1. 验证损失随数据量变化:")
        print()
        print("   纯文本基线:")
        for size in [100, 500, 1000]:
            if results[size]['text_only']:
                improvement_from_100 = ((results[100]['text_only'] - results[size]['text_only']) / results[100]['text_only']) * 100
                print(f"     {size:4d} 样本: {results[size]['text_only']:.4f} (相比100样本改进 {improvement_from_100:5.1f}%)")
        
        print()
        print("   文本+情感:")
        for size in [100, 500, 1000]:
            if results[size]['text_audio']:
                improvement_from_100 = ((results[100]['text_audio'] - results[size]['text_audio']) / results[100]['text_audio']) * 100
                print(f("     {size:4d} 样本: {results[size]['text_audio']:.4f} (相比100样本改进 {improvement_from_100:5.1f}%)")
        
        print()
        print("2. 情感嵌入的相对优势:")
        print()
        for size in [100, 500, 1000]:
            if results[size]['text_only'] and results[size]['text_audio']:
                rel_imp = ((results[size]['text_only'] - results[size]['text_audio']) / results[size]['text_only']) * 100
                print(f"     {size:4d} 样本: {rel_imp:5.2f}%")
        
        print()
        print("3. 关键洞察:")
        print()
        
        # Calculate trends
        rel_imp_100 = ((results[100]['text_only'] - results[100]['text_audio']) / results[100]['text_only']) * 100
        rel_imp_1000 = ((results[1000]['text_only'] - results[1000]['text_audio']) / results[1000]['text_only']) * 100
        
        if rel_imp_1000 > rel_imp_100:
            print(f"   ✅ 情感嵌入的相对优势随数据增加而增强!")
            print(f"      100样本: {rel_imp_100:.2f}% → 1000样本: {rel_imp_1000:.2f}%")
        elif rel_imp_1000 < rel_imp_100:
            print(f"   📉 情感嵌入的相对优势随数据增加而下降")
            print(f"      100样本: {rel_imp_100:.2f}% → 1000样本: {rel_imp_1000:.2f}%")
            print(f"   💡 建议: 微调情感编码器以充分利用更多数据")
        else:
            print(f"   ⚖️ 情感嵌入的相对优势保持稳定")
    
    print()
    print("=" * 80)
    
    # Save results
    output = {
        'results': results,
        'summary': {
            'scales': [100, 500, 1000],
            'improvements': {}
        }
    }
    
    for size in [100, 500, 1000]:
        if results[size]['text_only'] and results[size]['text_audio']:
            rel_imp = ((results[size]['text_only'] - results[size]['text_audio']) / results[size]['text_only']) * 100
            output['summary']['improvements'][size] = rel_imp
    
    with open('outputs/scale_analysis_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n✅ 结果已保存到: outputs/scale_analysis_results.json\n")

if __name__ == '__main__':
    main()
