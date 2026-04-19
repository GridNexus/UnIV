#!/usr/bin/env python3
"""
统计 frame_data 目录总大小
用法: python dir_size.py [目录路径]
"""

import os
import sys
from pathlib import Path


def format_size(size_bytes):
    """将字节转换为人类可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} EB"


def get_dir_size(path, show_progress=True):
    """
    遍历目录计算总大小
    返回: (总字节数, 文件数, 目录数, 错误数)
    """
    total_size = 0
    file_count = 0
    dir_count = 0
    error_count = 0
    errors = []

    path = Path(path)
    
    if not path.exists():
        print(f"❌ 错误: 路径 '{path}' 不存在")
        sys.exit(1)
    
    if not path.is_dir():
        print(f"❌ 错误: '{path}' 不是目录")
        sys.exit(1)

    print(f"🔍 正在扫描: {path.absolute()}")
    print("-" * 50)

    # 使用 rglob 递归遍历所有文件和目录
    try:
        for item in path.rglob('*'):
            try:
                if item.is_file():
                    size = item.stat().st_size
                    total_size += size
                    file_count += 1
                    
                    # 每1000个文件显示一次进度
                    if show_progress and file_count % 1000 == 0:
                        print(f"  已处理 {file_count} 个文件... 当前总计: {format_size(total_size)}")
                        
                elif item.is_dir():
                    dir_count += 1
                    
            except (OSError, PermissionError, FileNotFoundError) as e:
                error_count += 1
                if len(errors) < 5:  # 只记录前5个错误
                    errors.append(f"  ⚠️  {item}: {e}")
                continue
                
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断扫描")
        print(f"已统计部分结果:")
    
    return total_size, file_count, dir_count, error_count, errors


def main():
    # 默认目录
    target_dir = os.getenv("TARGET_DIR", "/path/to/frame_data")
    
    try:
        total_size, file_count, dir_count, error_count, errors = get_dir_size(target_dir)
        
        print("-" * 50)
        print(f"📊 统计结果:")
        print(f"   总大小: {format_size(total_size)} ({total_size:,} bytes)")
        print(f"   文件数: {file_count:,} 个")
        print(f"   目录数: {dir_count:,} 个")
        
        if error_count > 0:
            print(f"   错误数: {error_count} 个 (权限拒绝或文件丢失)")
            if errors:
                print(f"\n⚠️  部分错误详情:")
                for err in errors:
                    print(err)
                if error_count > 5:
                    print(f"   ... 还有 {error_count - 5} 个错误未显示")
        
        # 计算平均文件大小
        if file_count > 0:
            avg_size = total_size / file_count
            print(f"   平均大小: {format_size(avg_size)}")
            
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()