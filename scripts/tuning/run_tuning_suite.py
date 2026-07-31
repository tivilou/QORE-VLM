#!/usr/bin/env python3
"""
实验套件运行器 - 主控脚本

自动运行、监控、分析和打包实验结果

使用方法:
    python scripts/experiments/run_tuning_suite.py \\
        --config config/phase1_diagnosis.yaml \\
        --parallel 1 \\
        --retry 2

示例:
    # Phase 1 诊断
    python scripts/experiments/run_tuning_suite.py \\
        --config scripts/experiments/config/phase1_diagnosis.yaml

    # 快速测试（只跑 50 题）
    python scripts/experiments/run_tuning_suite.py \\
        --config scripts/experiments/config/phase1_diagnosis.yaml \\
        --override max_samples=50
"""

import argparse
import yaml
import json
import sys
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# This suite lives under ``scripts.tuning``.  Keep the imports aligned with the
# on-disk package so invoking it as ``python scripts/tuning/run_tuning_suite.py``
# works from the repository root.
from scripts.tuning.utils.experiment_runner import ExperimentRunner
from scripts.tuning.utils.result_analyzer import ResultAnalyzer
from scripts.tuning.utils.packager import ResultPackager


def _fmt_metric(value: Any, spec: str = ".4f") -> str:
    """格式化指标，缺失时显示 N/A。

    跑 --skip_generation 时 result.json 的 metrics 里没有 mean_f1/mean_em，
    experiment_runner._read_result 用 .get() 取值，于是 'f1' 这个键**存在但值是
    None**。这时 m.get('f1', 0) 不会走默认值 0（键在），返回 None，
    f"{None:.4f}" 直接 TypeError —— 实验本身已经成功了，却在打印摘要时炸掉。
    result_analyzer.py 那边早就用 `or 0` 防过，这里补齐。
    """
    if value is None:
        return "N/A"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


class TuningSuite:
    """实验套件控制器"""

    def __init__(self, config_path: Path, overrides: Dict[str, Any] = None):
        self.config_path = config_path
        self.config = self.load_config(config_path)

        # 应用覆盖参数
        if overrides:
            self.apply_overrides(overrides)

        self.base_dir = Path(self.config['output_dir'])
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = datetime.now()
        self.results = []

    def load_config(self, config_path: Path) -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def apply_overrides(self, overrides: Dict[str, Any]):
        """应用命令行覆盖参数"""
        for exp in self.config['experiments']:
            for key, value in overrides.items():
                if key in exp['args']:
                    exp['args'][key] = value
                    print(f"  覆盖参数: {exp['name']}.{key} = {value}")

    def check_environment(self) -> bool:
        """检查环境"""
        print("\n" + "="*70)
        print("环境检查")
        print("="*70)

        checks = self.config.get('execution', {}).get('pre_checks', [])

        for check in checks:
            if isinstance(check, dict):
                for check_type, check_value in check.items():
                    if check_type == 'check_gpu':
                        if not self._check_gpu():
                            return False
                    elif check_type == 'check_disk_space_gb':
                        if not self._check_disk_space(check_value):
                            return False
                    elif check_type == 'check_python_packages':
                        if not self._check_packages(check_value):
                            return False

        print("✅ 环境检查通过\n")
        return True

    def _check_gpu(self) -> bool:
        """检查 GPU"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
            if result.returncode == 0:
                print("✅ GPU 可用")
                return True
            else:
                print("❌ GPU 不可用")
                return False
        except Exception as e:
            print(f"⚠️  无法检查 GPU: {e}")
            return True  # 不阻止运行

    def _check_disk_space(self, required_gb: int) -> bool:
        """检查磁盘空间"""
        try:
            stat = shutil.disk_usage(Path.cwd())
            free_gb = stat.free / (1024**3)
            if free_gb >= required_gb:
                print(f"✅ 磁盘空间充足: {free_gb:.1f} GB 可用")
                return True
            else:
                print(f"❌ 磁盘空间不足: {free_gb:.1f} GB 可用, 需要 {required_gb} GB")
                return False
        except Exception as e:
            print(f"⚠️  无法检查磁盘空间: {e}")
            return True

    def _check_packages(self, packages: List[str]) -> bool:
        """检查 Python 包"""
        all_ok = True
        for pkg in packages:
            try:
                __import__(pkg)
                print(f"✅ {pkg} 已安装")
            except ImportError:
                print(f"❌ {pkg} 未安装")
                all_ok = False
        return all_ok

    def run_experiments(self) -> List[Dict[str, Any]]:
        """运行所有实验"""
        print("="*70)
        print(f"开始实验: {self.config['name']}")
        print("="*70)
        print(f"实验数量: {len(self.config['experiments'])}")
        print(f"输出目录: {self.base_dir}\n")

        experiments = self.config['experiments']
        parallel = self.config.get('execution', {}).get('parallel', 1)
        retry = self.config.get('execution', {}).get('retry_on_failure', 0)
        timeout = self.config.get('execution', {}).get('timeout_minutes', 30)

        results = []

        if parallel == 1:
            # 串行运行
            for i, exp_config in enumerate(experiments, 1):
                print(f"\n[{i}/{len(experiments)}] 运行: {exp_config['name']}")
                print(f"  描述: {exp_config.get('description', 'N/A')}")
                print(f"  预计耗时: {exp_config.get('expected_time_minutes', '?')} 分钟")

                runner = ExperimentRunner(exp_config, self.base_dir)
                success, info = runner.run(timeout_minutes=timeout, retry=retry)

                if success:
                    print(f"  ✅ 成功 (耗时: {info['elapsed_seconds']:.0f}s)")
                    if 'metrics' in info:
                        m = info['metrics']
<<<<<<< Updated upstream
                        print(f"     Recall: {_fmt_metric(m.get('recall'))}, "
                              f"F1: {_fmt_metric(m.get('f1'))}, "
                              f"冗余: {_fmt_metric(m.get('redundancy'))}")
=======
                        def format_metric(value: Any) -> str:
                            """Format optional metrics returned by an evaluator."""
                            return f"{value:.4f}" if isinstance(value, (int, float)) else "N/A"

                        print(f"     Recall: {format_metric(m.get('recall'))}, "
                              f"F1: {format_metric(m.get('f1'))}, "
                              f"冗余: {format_metric(m.get('redundancy'))}")
>>>>>>> Stashed changes
                else:
                    print(f"  ❌ 失败: {info.get('error', 'Unknown error')}")

                results.append({
                    'experiment': exp_config['name'],
                    'success': success,
                    'info': info
                })
        else:
            # 并行运行
            print(f"并行运行 {parallel} 个实验\n")

            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {}
                for exp_config in experiments:
                    runner = ExperimentRunner(exp_config, self.base_dir)
                    future = executor.submit(runner.run, timeout, retry)
                    futures[future] = exp_config['name']

                for i, future in enumerate(as_completed(futures), 1):
                    exp_name = futures[future]
                    try:
                        success, info = future.result()
                        status = "✅" if success else "❌"
                        print(f"[{i}/{len(experiments)}] {status} {exp_name}")

                        results.append({
                            'experiment': exp_name,
                            'success': success,
                            'info': info
                        })
                    except Exception as e:
                        print(f"[{i}/{len(experiments)}] ❌ {exp_name}: {e}")
                        results.append({
                            'experiment': exp_name,
                            'success': False,
                            'info': {'error': str(e)}
                        })

        return results

    def post_process(self) -> bool:
        """后处理：分析和打包"""
        print("\n" + "="*70)
        print("后处理")
        print("="*70)

        post_config = self.config.get('post_process', {})

        # 分析结果
        analyze_config = post_config.get('analyze', {})
        if analyze_config.get('enabled', True):
            print("\n📊 分析结果...")
            analyzer = ResultAnalyzer(self.base_dir / "experiments")

            # 收集结果
            all_results = analyzer.collect_results()
            summary = analyzer.generate_summary(all_results)

            # 保存摘要
            summary_file = self.base_dir / "run_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"  ✅ 摘要保存: {summary_file}")

            # 生成 Markdown 报告
            report_file = self.base_dir / "analysis" / "quick_summary.md"
            analyzer.generate_markdown_report(summary, report_file)

            # 运行自定义分析脚本
            script_path = analyze_config.get('script')
            if script_path:
                script_path = Path(script_path)
                args = analyze_config.get('args', {})
                results_dir = Path(str(args.get('results_dir', '')).format(
                    output_dir=self.base_dir))
                output_path = Path(str(args.get('output', '')).format(
                    output_dir=self.base_dir))

                analyzer.run_analysis_script(script_path, results_dir, output_path)

        # 打包结果
        package_config = post_config.get('package', {})
        if package_config.get('enabled', True):
            print("\n📦 打包结果...")
            packager = ResultPackager(self.base_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_pattern = package_config.get('output', '')
            output_path = Path(str(output_pattern).format(
                output_dir=self.base_dir,
                timestamp=timestamp
            ))

            include_patterns = package_config.get('include', [])
            packager.package(output_path, include_patterns)

        return True

    def generate_final_report(self):
        """生成最终报告"""
        end_time = datetime.now()
        duration = end_time - self.start_time

        print("\n" + "="*70)
        print("实验完成")
        print("="*70)
        print(f"名称: {self.config['name']}")
        print(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总耗时: {duration}")
        print(f"输出目录: {self.base_dir}")

        # 统计
        success_count = sum(1 for r in self.results if r['success'])
        total_count = len(self.results)
        print(f"\n实验统计:")
        print(f"  总数: {total_count}")
        print(f"  成功: {success_count}")
        print(f"  失败: {total_count - success_count}")

        # 查看分析报告
        analysis_file = self.base_dir / "analysis" / "analysis.md"
        if analysis_file.exists():
            print(f"\n📊 分析报告: {analysis_file}")

        # 查看打包文件
        package_dir = self.base_dir / "package"
        if package_dir.exists():
            packages = list(package_dir.glob("*.zip"))
            if packages:
                print(f"\n📦 打包文件: {packages[0]}")

        print("\n" + "="*70)

    def run(self):
        """运行完整流程"""
        try:
            # 环境检查
            if not self.check_environment():
                print("\n❌ 环境检查失败，退出")
                return 1

            # 运行实验
            self.results = self.run_experiments()

            # 后处理
            self.post_process()

            # 最终报告
            self.generate_final_report()

            # 检查是否有失败
            if all(r['success'] for r in self.results):
                return 0
            else:
                return 1

        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断")
            return 130
        except Exception as e:
            print(f"\n\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()
            return 1


def parse_overrides(override_args: List[str]) -> Dict[str, Any]:
    """解析覆盖参数"""
    overrides = {}
    for arg in override_args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # 尝试转换类型
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
            overrides[key] = value
    return overrides


def main():
    parser = argparse.ArgumentParser(
        description='运行实验套件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='配置文件路径'
    )

    parser.add_argument(
        '--override',
        nargs='*',
        default=[],
        help='覆盖参数，格式: key=value (例如: max_samples=50)'
    )

    args = parser.parse_args()

    if not args.config.exists():
        print(f"❌ 配置文件不存在: {args.config}")
        return 1

    # 解析覆盖参数
    overrides = parse_overrides(args.override)

    # 运行实验套件
    suite = TuningSuite(args.config, overrides)
    return suite.run()


if __name__ == '__main__':
    sys.exit(main())
