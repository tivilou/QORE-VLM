#!/usr/bin/env python3
"""
结果分析器

负责汇总实验结果，生成对比报告
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional


class ResultAnalyzer:
    """实验结果分析器"""

    def __init__(self, experiments_dir: Path):
        self.experiments_dir = experiments_dir

    def collect_results(self) -> Dict[str, Dict[str, Any]]:
        """收集所有实验结果"""
        results = {}

        for exp_dir in self.experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            status_file = exp_dir / "status.json"
            if not status_file.exists():
                continue

            try:
                with open(status_file, 'r') as f:
                    status = json.load(f)
                    results[exp_dir.name] = status
            except Exception as e:
                print(f"⚠️  无法读取 {exp_dir.name}: {e}")

        return results

    def generate_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """生成摘要"""
        total = len(results)
        success = sum(1 for r in results.values() if r.get('status') == 'success')
        failed = sum(1 for r in results.values() if r.get('status') == 'failed')
        timeout = sum(1 for r in results.values() if r.get('status') == 'timeout')
        error = sum(1 for r in results.values() if r.get('status') == 'error')

        # 提取指标
        metrics = {}
        for name, result in results.items():
            if result.get('status') == 'success' and 'metrics' in result:
                metrics[name] = result['metrics']

        # 找到最佳配置
        best_recall = None
        best_f1 = None
        if metrics:
            best_recall = max(metrics.keys(),
                            key=lambda k: metrics[k].get('recall', 0) if metrics[k].get('recall') is not None else 0)
            best_f1 = max(metrics.keys(),
                         key=lambda k: metrics[k].get('f1', 0) if metrics[k].get('f1') is not None else 0)

        return {
            'total_experiments': total,
            'success': success,
            'failed': failed,
            'timeout': timeout,
            'error': error,
            'metrics': metrics,
            'best_recall_config': best_recall,
            'best_f1_config': best_f1
        }

    def run_analysis_script(self, script_path: Path, results_dir: Path,
                           output_path: Path) -> bool:
        """运行自定义分析脚本"""
        if not script_path.exists():
            print(f"⚠️  分析脚本不存在: {script_path}")
            return False

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                'python', str(script_path),
                '--results_dir', str(results_dir),
                '--output', str(output_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                print(f"✅ 分析完成: {output_path}")
                return True
            else:
                print(f"❌ 分析失败: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ 运行分析脚本出错: {e}")
            return False

    def generate_markdown_report(self, summary: Dict[str, Any],
                                 output_path: Path):
        """生成 Markdown 报告"""
        lines = []
        lines.append("# 实验结果摘要\n")
        lines.append(f"**总实验数**: {summary['total_experiments']}\n")
        lines.append(f"**成功**: {summary['success']}\n")
        lines.append(f"**失败**: {summary['failed']}\n")
        lines.append(f"**超时**: {summary['timeout']}\n")
        lines.append(f"**错误**: {summary['error']}\n")

        if summary['metrics']:
            lines.append("\n## 结果对比\n")
            lines.append("\n| 配置 | Recall@5 | Precision@5 | 冗余度 | F1 | EM |")
            lines.append("\n|------|----------|-------------|--------|----|----|")

            for name, metrics in summary['metrics'].items():
                recall = metrics.get('recall', 0) or 0
                precision = metrics.get('precision', 0) or 0
                redundancy = metrics.get('redundancy', 0) or 0
                f1 = metrics.get('f1', 0) or 0
                em = metrics.get('em', 0) or 0

                lines.append(
                    f"\n| {name} | {recall:.4f} | {precision:.4f} | "
                    f"{redundancy:.4f} | {f1:.4f} | {em:.4f} |"
                )

            if summary['best_recall_config']:
                lines.append(f"\n\n**最佳 Recall**: {summary['best_recall_config']}")
            if summary['best_f1_config']:
                lines.append(f"\n**最佳 F1**: {summary['best_f1_config']}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(''.join(lines))

        print(f"✅ 报告生成: {output_path}")
