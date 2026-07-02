# -*- coding: utf-8 -*-
"""把多次运行的 results_run_*.txt 做平均，并画 Test Accuracy vs Time 的均值±STD 阴影。

特点：
- 兼容 Python2/Python3（不使用类型注解 / f-string）。
- 以 (Epoch, Model) 为对齐键：每个 epoch 对多次 run 取 mean/std。
- 横轴用每个 epoch 的累计 Time(s) 的均值（不同 run 的时间略有差异时也能对齐）。

输入文件格式（TSV，制表符分隔）示例（你的 results_run_1.txt 就是这种结构）：
Epoch\tModel\tTrainLoss\tTrainAcc\tTestLoss\tTestAcc\tTime(s)\tParamCount

用法示例：
  python plot_mean_results.py \
    --inputs \
      /home/ywang/Article/vgg_gradmax/outputs_compare/vgg/20260109_1740_seed42_lr0.05/results_run_1.txt \
      /home/ywang/Article/vgg_gradmax/outputs_compare/vgg/20260109_2045_seed17_lr0.05/results_run_1.txt \
      /home/ywang/Article/vgg_gradmax/outputs_compare/vgg/20260110_105252_run01_seed1501552845_lr0.05/results_run_1.txt \
    --outdir /home/ywang/Article/vgg_gradmax/outputs_compare/vgg/avg_plot \
    --models c a b d e

输出：
- <outdir>/test_accuracy_vs_time_mean.png
- <outdir>/test_accuracy_vs_time_mean.tsv（聚合后的均值和标准差）
"""

from __future__ import print_function, division

import os
import csv
import argparse

import numpy as np

# 为了在无显示环境（服务器）下保存图
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def read_results_tsv(path):
    """读取单个 results_run_*.txt，返回 dict: model -> {epoch -> (time, test_acc)}"""
    data = {}
    with open(path, 'r') as f:
        # 兼容 Windows/Unix 换行
        reader = csv.DictReader(f, delimiter='\t')
        required = ['Epoch', 'Model', 'TestAcc', 'Time(s)']
        for k in required:
            if k not in reader.fieldnames:
                raise ValueError('文件缺少列: %s (实际列: %s) -> %s' % (k, reader.fieldnames, path))

        for row in reader:
            epoch = _safe_int(row.get('Epoch'))
            model = row.get('Model')
            test_acc = _safe_float(row.get('TestAcc'))
            t = _safe_float(row.get('Time(s)'))

            if epoch is None or model is None:
                continue

            if model not in data:
                data[model] = {}
            # 同一个 (model, epoch) 如果出现多次，以最后一次为准
            data[model][epoch] = (t, test_acc)
    return data


def aggregate_runs(run_dicts, models=None, use_epoch_intersection=True):
    """聚合多次 run。

    Args:
      run_dicts: list of dict(model -> dict(epoch -> (time, testacc)))
      models: None 或 list[str]
      use_epoch_intersection: True 则对齐使用各 run 都存在的 epoch 交集。

    Returns:
      agg: dict model -> dict with keys:
        'epochs', 'mean_time', 'mean_acc', 'std_acc', 'n'
    """
    # 收集所有模型名
    all_models = set()
    for rd in run_dicts:
        all_models.update(rd.keys())

    if models is not None and len(models) > 0:
        use_models = [m for m in models if m in all_models]
    else:
        use_models = sorted(list(all_models))

    agg = {}
    for m in use_models:
        # 每个 run 的 epoch 集合
        epoch_sets = []
        for rd in run_dicts:
            if m in rd:
                epoch_sets.append(set(rd[m].keys()))
            else:
                epoch_sets.append(set())

        if use_epoch_intersection:
            common_epochs = None
            for s in epoch_sets:
                if common_epochs is None:
                    common_epochs = set(s)
                else:
                    common_epochs = common_epochs.intersection(s)
            if common_epochs is None:
                common_epochs = set()
            epochs = sorted(list(common_epochs))
        else:
            # union（会出现缺失 epoch -> NaN）
            epochs = sorted(list(set().union(*epoch_sets)))

        if len(epochs) == 0:
            continue

        mean_time = []
        mean_acc = []
        std_acc = []
        n_list = []

        for ep in epochs:
            times = []
            accs = []
            for rd in run_dicts:
                if m in rd and ep in rd[m]:
                    t, a = rd[m][ep]
                    if not np.isnan(t):
                        times.append(t)
                    if not np.isnan(a):
                        accs.append(a)

            if len(accs) == 0:
                # 这个 epoch 没数据（union 模式下可能发生）
                mean_time.append(np.nan)
                mean_acc.append(np.nan)
                std_acc.append(np.nan)
                n_list.append(0)
                continue

            # time：如果缺失就用 NaN
            if len(times) == 0:
                mt = np.nan
            else:
                mt = float(np.mean(times))

            ma = float(np.mean(accs))
            # std：样本标准差更合理（ddof=1），但 n=1 时退化为 0
            if len(accs) >= 2:
                sa = float(np.std(accs, ddof=1))
            else:
                sa = 0.0

            mean_time.append(mt)
            mean_acc.append(ma)
            std_acc.append(sa)
            n_list.append(len(accs))

        agg[m] = {
            'epochs': np.array(epochs, dtype=np.int64),
            'mean_time': np.array(mean_time, dtype=np.float64),
            'mean_acc': np.array(mean_acc, dtype=np.float64),
            'std_acc': np.array(std_acc, dtype=np.float64),
            'n': np.array(n_list, dtype=np.int64),
        }

    return agg


def save_agg_tsv(agg, out_path):
    """把聚合结果保存成 TSV，方便你之后再画别的图。"""
    # 展平成行：model, epoch, mean_time, mean_acc, std_acc, n
    with open(out_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Model', 'Epoch', 'MeanTime(s)', 'MeanTestAcc', 'StdTestAcc', 'N'])
        for model in sorted(agg.keys()):
            a = agg[model]
            for i in range(len(a['epochs'])):
                writer.writerow([
                    model,
                    int(a['epochs'][i]),
                    ('%.6f' % a['mean_time'][i]) if not np.isnan(a['mean_time'][i]) else 'nan',
                    ('%.6f' % a['mean_acc'][i]) if not np.isnan(a['mean_acc'][i]) else 'nan',
                    ('%.6f' % a['std_acc'][i]) if not np.isnan(a['std_acc'][i]) else 'nan',
                    int(a['n'][i]),
                ])


def plot_mean_std(agg, out_png, title='Test Accuracy vs Time', xlabel='Time (s)', ylabel='Test Accuracy'):
    """画均值曲线 + STD 阴影。"""
    # 你之前的配色（可按需扩展）
    color_map = {
        'c': 'green',
        'a': 'blue',
        'b': 'orange',
        'd': 'brown',
        'e': 'deeppink',
        'f': 'purple',
    }

    plt.figure()

    # 为了保证图例顺序更稳定，按 model 名排序
    for model in sorted(agg.keys()):
        a = agg[model]
        x = a['mean_time']
        y = a['mean_acc']
        s = a['std_acc']

        # 去掉 NaN 点（如果使用 union 对齐，可能出现）
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(s)
        x = x[mask]
        y = y[mask]
        s = s[mask]
        if x.size == 0:
            continue

        color = color_map.get(model, None)
        label = 'Model %s' % model

        plt.plot(x, y, label=label, color=color)
        plt.fill_between(x, y - s, y + s, color=color, alpha=0.2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    outdir = os.path.dirname(out_png)
    if outdir and (not os.path.exists(outdir)):
        os.makedirs(outdir)

    plt.savefig(out_png)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True,
                   help='多个 results_run_*.txt 路径（同一配置的多次运行）')
    p.add_argument('--outdir', required=True, help='输出目录')
    p.add_argument('--models', nargs='*', default=None,
                   help='只画这些模型，例如: --models c a b d e （不填则画文件里出现的全部模型）')
    p.add_argument('--title', default='Test Accuracy vs Time', help='图标题')
    p.add_argument('--use_epoch_intersection', action='store_true',
                   help='只使用所有 run 都存在的 epoch（推荐）。默认也是 True。')
    p.add_argument('--use_epoch_union', action='store_true',
                   help='使用 epoch 并集（允许某些 run 缺失 epoch，会出现 NaN 点）。')
    return p.parse_args()


def main():
    args = parse_args()

    # 默认：使用交集（更稳）
    use_intersection = True
    if args.use_epoch_union:
        use_intersection = False
    if args.use_epoch_intersection:
        use_intersection = True

    run_dicts = []
    for path in args.inputs:
        if not os.path.isfile(path):
            raise IOError('找不到输入文件: %s' % path)
        run_dicts.append(read_results_tsv(path))

    agg = aggregate_runs(run_dicts, models=args.models, use_epoch_intersection=use_intersection)
    if len(agg) == 0:
        raise ValueError('没有可画的数据：请检查 --models 是否写错，或输入文件是否包含对应模型。')

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    out_png = os.path.join(args.outdir, 'test_accuracy_vs_time_mean.png')
    out_tsv = os.path.join(args.outdir, 'test_accuracy_vs_time_mean.tsv')

    save_agg_tsv(agg, out_tsv)
    plot_mean_std(agg, out_png, title=args.title)

    print('已保存:')
    print('  图:  %s' % out_png)
    print('  表:  %s' % out_tsv)


if __name__ == '__main__':
    main()
