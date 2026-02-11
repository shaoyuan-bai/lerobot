"""
Task D 测试工具：测量推理各模块耗时

使用方法：
1. 在需要测量的函数上添加装饰器：
   
   from test_inference_timing import time_it
   
   @time_it('gripper_set_position')
   def set_position(self, position, blocking=False):
       ...
   
   @time_it('camera_read_frame')
   def read_frame(self):
       ...
   
   @time_it('arm_send_action')
   def send_action(self, action):
       ...

2. 运行推理（至少30秒，获得足够样本）

3. 在推理结束前调用：
   from test_inference_timing import print_timing_report
   print_timing_report()
"""

import time
import functools
from collections import defaultdict
from typing import Dict, List
import statistics

# 全局存储：{函数名: [耗时列表]}
_timing_data: Dict[str, List[float]] = defaultdict(list)

def time_it(name: str):
    """装饰器：记录函数执行时间"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            _timing_data[name].append(elapsed * 1000)  # 转为毫秒
            return result
        return wrapper
    return decorator

def print_timing_report():
    """打印性能报告"""
    if not _timing_data:
        print("❌ 没有收集到任何性能数据")
        return
    
    print("\n" + "=" * 80)
    print("⏱️  推理性能分析报告")
    print("=" * 80)
    
    # 计算统计
    stats = {}
    for name, timings in _timing_data.items():
        if not timings:
            continue
        stats[name] = {
            'count': len(timings),
            'mean': statistics.mean(timings),
            'median': statistics.median(timings),
            'min': min(timings),
            'max': max(timings),
            'stdev': statistics.stdev(timings) if len(timings) > 1 else 0,
            'total': sum(timings)
        }
    
    # 按平均耗时排序
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    print(f"\n📊 各模块统计（共 {len(sorted_stats)} 个模块）\n")
    print(f"{'模块名':<30} {'调用次数':>8} {'平均(ms)':>10} {'中位数(ms)':>12} {'最小(ms)':>10} {'最大(ms)':>10} {'标准差(ms)':>12}")
    print("-" * 100)
    
    for name, stat in sorted_stats:
        print(f"{name:<30} {stat['count']:>8} {stat['mean']:>10.2f} {stat['median']:>12.2f} "
              f"{stat['min']:>10.2f} {stat['max']:>10.2f} {stat['stdev']:>12.2f}")
    
    # 找出最慢模块
    print("\n" + "=" * 80)
    print("🐌 最慢模块分析：")
    print("=" * 80)
    
    if sorted_stats:
        slowest_name, slowest_stat = sorted_stats[0]
        print(f"\n最慢模块: {slowest_name}")
        print(f"  平均耗时: {slowest_stat['mean']:.2f} ms")
        print(f"  最大耗时: {slowest_stat['max']:.2f} ms")
        print(f"  调用次数: {slowest_stat['count']}")
        print(f"  总耗时占比: {slowest_stat['total'] / sum(s['total'] for s in stats.values()) * 100:.1f}%")
    
    # 检查夹爪socket耗时
    gripper_modules = [name for name in _timing_data.keys() if 'gripper' in name.lower()]
    if gripper_modules:
        print("\n" + "=" * 80)
        print("🤏 夹爪模块详细分析：")
        print("=" * 80)
        
        for name in gripper_modules:
            stat = stats[name]
            print(f"\n【{name}】")
            print(f"  平均耗时: {stat['mean']:.2f} ms")
            print(f"  中位数: {stat['median']:.2f} ms")
            print(f"  最大耗时: {stat['max']:.2f} ms (潜在卡顿)")
            print(f"  最小耗时: {stat['min']:.2f} ms")
            print(f"  标准差: {stat['stdev']:.2f} ms")
            
            if stat['max'] > 50:  # >50ms认为是卡顿
                print(f"  ⚠️  检测到卡顿：最大耗时 {stat['max']:.2f}ms > 50ms")
            
            if stat['mean'] > 10:  # >10ms认为较慢
                print(f"  ⚠️  平均耗时较高：{stat['mean']:.2f}ms")
    
    # 检查相机耗时
    camera_modules = [name for name in _timing_data.keys() if 'camera' in name.lower() or 'frame' in name.lower()]
    if camera_modules:
        print("\n" + "=" * 80)
        print("📷 相机模块详细分析：")
        print("=" * 80)
        
        for name in camera_modules:
            stat = stats[name]
            print(f"\n【{name}】")
            print(f"  平均耗时: {stat['mean']:.2f} ms")
            print(f"  帧率: ~{1000/stat['mean']:.1f} FPS")
            print(f"  最大耗时: {stat['max']:.2f} ms")
    
    # 检查机械臂耗时
    arm_modules = [name for name in _timing_data.keys() if 'arm' in name.lower() or 'rm65' in name.lower()]
    if arm_modules:
        print("\n" + "=" * 80)
        print("🦾 机械臂模块详细分析：")
        print("=" * 80)
        
        for name in arm_modules:
            stat = stats[name]
            print(f"\n【{name}】")
            print(f"  平均耗时: {stat['mean']:.2f} ms")
            print(f"  最大耗时: {stat['max']:.2f} ms")
    
    print("\n" + "=" * 80)
    print("📋 结论：")
    print("=" * 80)
    
    if sorted_stats:
        print(f"\n最慢模块是: {sorted_stats[0][0]} (平均 {sorted_stats[0][1]['mean']:.2f}ms)")
        
        # 判断瓶颈
        top3 = sorted_stats[:3]
        print("\n性能瓶颈 TOP3:")
        for i, (name, stat) in enumerate(top3, 1):
            print(f"  {i}. {name}: {stat['mean']:.2f}ms (最大 {stat['max']:.2f}ms)")
    
    print("\n💡 优化建议:")
    if gripper_modules and any(stats[name]['mean'] > 10 for name in gripper_modules):
        print("  - 夹爪耗时较高，考虑:")
        print("    1. 使用持久连接代替临时socket")
        print("    2. 降低夹爪控制频率")
        print("    3. 异步发送夹爪指令")
    
    if camera_modules:
        for name in camera_modules:
            if stats[name]['mean'] > 50:  # >50ms说明相机较慢
                print(f"  - {name} 较慢，考虑降低分辨率或帧率")

def get_timing_data() -> Dict[str, List[float]]:
    """获取原始数据（用于导出）"""
    return dict(_timing_data)

def clear_timing_data():
    """清空数据"""
    _timing_data.clear()

def save_timing_data(filepath: str = "timing_report.json"):
    """保存为JSON"""
    import json
    with open(filepath, 'w') as f:
        json.dump({k: v for k, v in _timing_data.items()}, f, indent=2)
    print(f"✅ 性能数据已保存到: {filepath}")

if __name__ == "__main__":
    print(__doc__)
    print("\n使用示例：")
    print("""
# 在 epg_gripper.py 中：
from test_inference_timing import time_it

@time_it('gripper_set_position')
def set_position(self, position, blocking=False):
    ...

# 在 rm65_follower.py 中：
@time_it('arm_send_action')
def send_action(self, action):
    ...

# 在相机代码中：
@time_it('camera_read_frame')
def read(self):
    ...

# 推理结束后：
from test_inference_timing import print_timing_report
print_timing_report()
    """)
