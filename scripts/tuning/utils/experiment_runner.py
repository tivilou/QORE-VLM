#!/usr/bin/env python3
"""
实验运行器

负责运行单个实验，监控进度，处理失败
"""

import subprocess
import time
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime


class ExperimentRunner:
    """单个实验运行器"""

    def __init__(self, experiment_config: Dict[str, Any], base_dir: Path):
        self.config = experiment_config
        self.name = experiment_config['name']
        self.base_dir = base_dir
        self.log_dir = base_dir / "experiments" / self.name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.stdout_log = self.log_dir / "stdout.log"
        self.stderr_log = self.log_dir / "stderr.log"
        self.status_file = self.log_dir / "status.json"

    def build_command(self) -> str:
        """构建命令行"""
        cmd = self.config['command']
        args = self.config.get('args', {})

        # 替换占位符
        for key, value in args.items():
            if isinstance(value, str):
                value = value.format(output_dir=self.base_dir)
                args[key] = value

        # 构建参数
        for key, value in args.items():
            if isinstance(value, bool):
                if value:
                    cmd += f" --{key}"
            else:
                cmd += f" --{key} {value}"

        return cmd

    def run(self, timeout_minutes: int = 30, retry: int = 0) -> Tuple[bool, Dict[str, Any]]:
        """
        运行实验

        Returns:
            (success, result_info)
        """
        for attempt in range(retry + 1):
            if attempt > 0:
                print(f"  [重试 {attempt}/{retry}] {self.name}")

            success, info = self._run_once(timeout_minutes)

            if success:
                return True, info

            # 失败后等待一会再重试
            if attempt < retry:
                time.sleep(5)

        return False, info

    def _run_once(self, timeout_minutes: int) -> Tuple[bool, Dict[str, Any]]:
        """执行一次实验"""
        cmd = self.build_command()

        # 记录状态
        status = {
            'name': self.name,
            'command': cmd,
            'start_time': datetime.now().isoformat(),
            'status': 'running'
        }
        self._save_status(status)

        start_time = time.time()

        try:
            with open(self.stdout_log, 'w') as stdout_f, \
                 open(self.stderr_log, 'w') as stderr_f:

                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    cwd=Path.cwd()
                )

                # 等待完成或超时
                try:
                    returncode = process.wait(timeout=timeout_minutes * 60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    elapsed = time.time() - start_time
                    status.update({
                        'status': 'timeout',
                        'end_time': datetime.now().isoformat(),
                        'elapsed_seconds': elapsed,
                        'error': f'Timeout after {timeout_minutes} minutes'
                    })
                    self._save_status(status)
                    return False, status

                elapsed = time.time() - start_time

                if returncode == 0:
                    # 成功
                    status.update({
                        'status': 'success',
                        'end_time': datetime.now().isoformat(),
                        'elapsed_seconds': elapsed,
                        'returncode': returncode
                    })

                    # 尝试读取结果
                    result = self._read_result()
                    if result:
                        status['metrics'] = result

                    self._save_status(status)
                    return True, status
                else:
                    # 失败
                    with open(self.stderr_log, 'r') as f:
                        error_msg = f.read()[-500:]  # 最后 500 字符

                    status.update({
                        'status': 'failed',
                        'end_time': datetime.now().isoformat(),
                        'elapsed_seconds': elapsed,
                        'returncode': returncode,
                        'error': error_msg
                    })
                    self._save_status(status)
                    return False, status

        except Exception as e:
            elapsed = time.time() - start_time
            status.update({
                'status': 'error',
                'end_time': datetime.now().isoformat(),
                'elapsed_seconds': elapsed,
                'error': str(e)
            })
            self._save_status(status)
            return False, status

    def _read_result(self) -> Optional[Dict[str, Any]]:
        """读取实验结果"""
        result_file = self.log_dir / "result.json"
        if not result_file.exists():
            return None

        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                if 'metrics' in data:
                    return {
                        'recall': data['metrics'].get('mean_recall'),
                        'precision': data['metrics'].get('mean_precision'),
                        'redundancy': data['metrics'].get('mean_redundancy'),
                        'f1': data['metrics'].get('mean_f1'),
                        'em': data['metrics'].get('mean_em')
                    }
        except Exception as e:
            print(f"  ⚠️  无法读取结果: {e}")

        return None

    def _save_status(self, status: Dict[str, Any]):
        """保存状态"""
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)

    def get_status(self) -> Optional[Dict[str, Any]]:
        """获取当前状态"""
        if not self.status_file.exists():
            return None

        try:
            with open(self.status_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None
