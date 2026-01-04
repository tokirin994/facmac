from collections import defaultdict
import logging
import numpy as np
import os
import pandas as pd


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])
        
        # 用于保存CSV和绘图
        self.args = None
        self.model_base_path = None

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            item = "{:.4f}".format(np.mean([x[1] if isinstance(x[1], float) else x[1].item() for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)

    def setup_model_path(self, args, model_base_path=None):
        """设置统计文件保存路径，用于后续保存CSV和图表"""
        self.args = args
        if model_base_path is None:
            # 使用unique_token作为基础路径，不包含具体的步数目录
            if hasattr(args, 'unique_token'):
                self.model_base_path = os.path.join(args.local_results_path, "models", args.unique_token)
            else:
                self.model_base_path = os.path.join(args.local_results_path, "models", "default")
        else:
            # 如果传入的是具体步数目录，则提取父目录（unique_token目录）
            if model_base_path and os.path.basename(model_base_path).isdigit():
                # 如果最后一级目录是数字（步数），则使用父目录
                self.model_base_path = os.path.dirname(model_base_path)
            else:
                self.model_base_path = model_base_path

    def save_stats_to_csv(self, save_path=None):
        """将训练统计数据保存到CSV文件（包含训练和测试数据）"""
        if save_path is None:
            if self.model_base_path is None:
                self.console_logger.warning("Model base path not set, cannot save CSV")
                return
            save_path = os.path.join(self.model_base_path, "training_stats.csv")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 过滤：排除episode字段（会重新计算），但保留所有其他数据（包括test_前缀的数据）
        training_stats = {k: v for k, v in self.stats.items() 
                         if k != 'episode'}
        
        if len(training_stats) == 0:
            self.console_logger.warning("No training statistics to save")
            return
        
        # 获取episode_limit（环境最大步长）
        episode_limit = 50  # 默认值
        if self.args is not None:
            # 尝试从多个位置获取episode_limit
            if hasattr(self.args, 'episode_limit'):
                episode_limit = self.args.episode_limit
            elif hasattr(self.args, 'env_args'):
                env_args = self.args.env_args
                if isinstance(env_args, dict):
                    episode_limit = env_args.get('episode_limit', 50)
                elif hasattr(env_args, 'episode_limit'):
                    episode_limit = env_args.episode_limit
        
        # 准备数据：将训练统计数据转换为DataFrame
        # 格式：每个统计项一列，时间步作为索引
        all_timesteps = set()
        for key, values in training_stats.items():
            for t, _ in values:
                all_timesteps.add(t)
        
        all_timesteps = sorted(list(all_timesteps))
        
        # 可选：过滤时间步，只保留满足runner_log_interval的时间步
        # 这样可以避免"整千多五十"的时间步（如2050, 4050等）
        if self.args is not None and hasattr(self.args, 'runner_log_interval'):
            interval = self.args.runner_log_interval
            if interval > 0:
                # 只保留满足间隔的时间步（如2000, 4000, 6000...）
                filtered_timesteps = [t for t in all_timesteps if t % interval == 0]
                if len(filtered_timesteps) > 0:
                    all_timesteps = filtered_timesteps
                    self.console_logger.info(f"Filtered timesteps to multiples of {interval}: {len(filtered_timesteps)} timesteps")
        
        # 创建DataFrame
        data_dict = {}
        for key, values in training_stats.items():
            # 创建时间步到值的映射
            value_dict = {t: v for t, v in values}
            # 为每个时间步填充值（如果该时间步没有记录，则为NaN）
            data_dict[key] = [value_dict.get(t, np.nan) for t in all_timesteps]
        
        # 计算episode：episode = timestep / episode_limit - 1
        episodes = [int(t / episode_limit) - 1 for t in all_timesteps]
        data_dict['episode'] = episodes
        
        df = pd.DataFrame(data_dict, index=all_timesteps)
        df.index.name = 'timestep'
        
        # 重新排列列，将episode放在前面
        cols = ['episode'] + [col for col in df.columns if col != 'episode']
        df = df[cols]
        
        # 保存到CSV
        df.to_csv(save_path)
        self.console_logger.info(f"Training statistics saved to {save_path} (including test data, episode calculated from timestep)")
        return save_path

    def plot_training_curves(self, save_path=None, plot_keys=None):
        """绘制训练曲线图并保存（使用episode作为x轴）"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
        except ImportError:
            self.console_logger.warning("matplotlib not available, cannot plot training curves")
            return
        
        if save_path is None:
            if self.model_base_path is None:
                self.console_logger.warning("Model base path not set, cannot save plots")
                return
            save_path = os.path.join(self.model_base_path, "training_curves.png")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 获取episode_limit（环境最大步长）
        episode_limit = 50  # 默认值
        if self.args is not None:
            # 尝试从多个位置获取episode_limit
            if hasattr(self.args, 'episode_limit'):
                episode_limit = self.args.episode_limit
            elif hasattr(self.args, 'env_args'):
                env_args = self.args.env_args
                if isinstance(env_args, dict):
                    episode_limit = env_args.get('episode_limit', 50)
                elif hasattr(env_args, 'episode_limit'):
                    episode_limit = env_args.episode_limit
        
        # 如果没有指定要绘制的键，则绘制所有统计项（包括测试数据，排除episode）
        if plot_keys is None:
            all_keys = [k for k in self.stats.keys() 
                       if k != "episode" and len(self.stats[k]) > 0]
        else:
            all_keys = [k for k in plot_keys if k in self.stats and len(self.stats[k]) > 0]
        
        if len(all_keys) == 0:
            self.console_logger.warning("No statistics to plot")
            return
        
        # 配对mean和std统计项
        # 找出所有mean和std的配对
        mean_keys = [k for k in all_keys if k.endswith('_mean')]
        std_keys = [k for k in all_keys if k.endswith('_std')]
        
        # 创建配对：{base_name: (mean_key, std_key)}
        paired_stats = {}
        unpaired_keys = []
        
        for key in all_keys:
            if key.endswith('_mean'):
                base_name = key[:-5]  # 移除'_mean'
                std_key = base_name + '_std'
                if std_key in all_keys:
                    paired_stats[base_name] = (key, std_key)
                else:
                    unpaired_keys.append(key)
            elif key.endswith('_std'):
                # std键会在mean键处理时配对，这里跳过
                continue
            else:
                # 既不是mean也不是std的键
                unpaired_keys.append(key)
        
        # 计算子图布局：配对的数量 + 未配对的数量
        n_plots = len(paired_stats) + len(unpaired_keys)
        if n_plots == 0:
            self.console_logger.warning("No statistics to plot after pairing")
            return
        
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()
        
        plot_idx = 0
        
        # 绘制配对的mean/std统计项（曲线+阴影）
        for base_name, (mean_key, std_key) in sorted(paired_stats.items()):
            if plot_idx >= len(axes):
                break
            ax = axes[plot_idx]
            
            mean_values = self.stats[mean_key]
            std_values = self.stats[std_key]
            
            if len(mean_values) == 0 or len(std_values) == 0:
                continue
            
            # 获取时间步并转换为episode
            mean_timesteps = [t for t, v in mean_values]
            std_timesteps = [t for t, v in std_values]
            
            # 找到共同的时间步
            common_timesteps = sorted(set(mean_timesteps) & set(std_timesteps))
            if len(common_timesteps) == 0:
                continue
            
            episodes = [int(t / episode_limit) - 1 for t in common_timesteps]
            
            # 创建时间步到值的映射
            mean_dict = {t: v for t, v in mean_values}
            std_dict = {t: v for t, v in std_values}
            
            mean_vals = [mean_dict[t] if isinstance(mean_dict[t], float) 
                        else mean_dict[t].item() if hasattr(mean_dict[t], 'item') 
                        else float(mean_dict[t]) for t in common_timesteps]
            std_vals = [std_dict[t] if isinstance(std_dict[t], float) 
                       else std_dict[t].item() if hasattr(std_dict[t], 'item') 
                       else float(std_dict[t]) for t in common_timesteps]
            
            # 绘制曲线和阴影
            ax.plot(episodes, mean_vals, linewidth=1.5, alpha=0.8, label=base_name.replace('_', ' ').title())
            ax.fill_between(episodes, 
                           [m - s for m, s in zip(mean_vals, std_vals)],
                           [m + s for m, s in zip(mean_vals, std_vals)],
                           alpha=0.3)
            
            ax.set_xlabel('Episode')
            ax.set_ylabel(base_name.replace('_', ' ').title())
            ax.set_title(base_name.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plot_idx += 1
        
        # 绘制未配对的统计项（普通曲线）
        for key in sorted(unpaired_keys):
            if plot_idx >= len(axes):
                break
            ax = axes[plot_idx]
            
            values = self.stats[key]
            if len(values) == 0:
                continue
            
            timesteps = [t for t, v in values]
            # 将时间步转换为episode：episode = timestep / episode_limit - 1
            episodes = [int(t / episode_limit) - 1 for t in timesteps]
            stats_values = [v if isinstance(v, float) else v.item() if hasattr(v, 'item') else float(v) for t, v in values]
            
            ax.plot(episodes, stats_values, linewidth=1.5, alpha=0.7)
            ax.set_xlabel('Episode')
            ax.set_ylabel(key)
            ax.set_title(key.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # 隐藏多余的子图
        for idx in range(plot_idx, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.console_logger.info(f"Training curves saved to {save_path} (x-axis: episode)")
        return save_path

    def finalize_training(self, model_save_path=None):
        """训练完成后的最终处理：保存CSV和绘制曲线"""
        # 如果传入了模型保存路径，提取父目录（unique_token目录）
        if model_save_path is not None:
            if os.path.isfile(model_save_path):
                # 如果是文件，获取其所在目录
                model_dir = os.path.dirname(model_save_path)
            else:
                model_dir = model_save_path
            
            # 如果最后一级目录是数字（步数），则使用父目录（unique_token目录）
            if os.path.basename(model_dir).isdigit():
                self.model_base_path = os.path.dirname(model_dir)
            else:
                self.model_base_path = model_dir
        
        # 确保使用unique_token目录作为统计文件保存路径
        if self.args is not None and hasattr(self.args, 'unique_token'):
            stats_path = os.path.join(self.args.local_results_path, "models", self.args.unique_token)
            self.model_base_path = stats_path
        
        # 检查是否需要保存CSV和绘图
        if self.args is None:
            self.console_logger.warning("Args not set, skipping CSV and plot saving")
            return
        
        # 保存CSV（总是保存）
        try:
            self.save_stats_to_csv()
        except Exception as e:
            self.console_logger.error(f"Failed to save CSV: {e}")
        
        # 根据参数决定是否绘制曲线
        if getattr(self.args, 'plot_training_curves', True):
            try:
                self.plot_training_curves()
            except Exception as e:
                self.console_logger.error(f"Failed to plot training curves: {e}")


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger