#!/usr/bin/env python3
"""
结果打包器

负责打包实验结果，生成可交付的 zip 文件
"""

import zipfile
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime


class ResultPackager:
    """实验结果打包器"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def package(self, output_path: Path, include_patterns: List[str]) -> bool:
        """
        打包结果

        Args:
            output_path: 输出 zip 文件路径
            include_patterns: 要包含的文件模式列表

        Returns:
            是否成功
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 收集要打包的文件
            files_to_pack = []
            for pattern in include_patterns:
                # 替换占位符
                pattern = pattern.format(output_dir=self.base_dir)

                # 支持通配符
                if '*' in pattern:
                    base_path = Path(pattern.split('*')[0])
                    if base_path.exists():
                        for file_path in Path(pattern).parent.parent.rglob(Path(pattern).name):
                            if file_path.is_file():
                                files_to_pack.append(file_path)
                else:
                    file_path = Path(pattern)
                    if file_path.exists() and file_path.is_file():
                        files_to_pack.append(file_path)

            if not files_to_pack:
                print("⚠️  没有找到要打包的文件")
                return False

            # 创建 zip 文件
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in files_to_pack:
                    # 计算相对路径
                    try:
                        arcname = file_path.relative_to(self.base_dir.parent)
                    except ValueError:
                        arcname = file_path.name

                    zipf.write(file_path, arcname)

            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ 打包完成: {output_path} ({file_size:.1f} MB)")
            print(f"   包含 {len(files_to_pack)} 个文件")
            return True

        except Exception as e:
            print(f"❌ 打包失败: {e}")
            return False

    def cleanup_temp_files(self):
        """清理临时文件（可选）"""
        # 可以在这里删除不需要保留的中间文件
        pass
