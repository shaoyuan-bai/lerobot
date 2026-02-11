"""
Task B 测试工具：确认推理时谁在写夹爪（检测双写冲突）

使用方法：
1. 在 rm65_follower.py 的 send_action() 中添加：
   from test_gripper_writer import log_gripper_write_lerobot
   log_gripper_write_lerobot(gripper_pos_normalized, gripper_pos_raw)

2. 如果存在其他夹爪控制程序，在那里也添加：
   from test_gripper_writer import log_gripper_write_external
   log_gripper_write_external(value, source_name="外部程序名")

3. 启动推理，抓取5秒日志

4. 运行分析：python test_gripper_writer.py --analyze
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path(__file__).parent / "gripper_write_log.jsonl"

def log_gripper_write_lerobot(gripper_pos_normalized: float, gripper_pos_raw: int):
    """在 rm65_follower.py 中调用"""
    log_entry = {
        "timestamp": time.time(),
        "datetime": datetime.now().isoformat(),
        "pid": os.getpid(),
        "source": "LeRobot_rm65_follower",
        "gripper_action_norm": float(gripper_pos_normalized),
        "gripper_action_raw": int(gripper_pos_raw),
        "sent_value_255": int(gripper_pos_raw)
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

def log_gripper_write_epg(position: float, position_raw: int):
    """在 epg_gripper.py 的 set_position() 中调用"""
    log_entry = {
        "timestamp": time.time(),
        "datetime": datetime.now().isoformat(),
        "pid": os.getpid(),
        "source": "EPGGripperClient_set_position",
        "position_0_100": float(position),
        "position_raw_255": int(position_raw)
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

def log_gripper_write_external(value, source_name: str = "External"):
    """在其他可能的夹爪控制程序中调用"""
    log_entry = {
        "timestamp": time.time(),
        "datetime": datetime.now().isoformat(),
        "pid": os.getpid(),
        "source": source_name,
        "value": value
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

def analyze_logs():
    """分析日志，检测双写冲突"""
    if not LOG_FILE.exists():
        print("❌ 日志文件不存在，请先运行推理并添加日志调用")
        return
    
    logs = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    
    if not logs:
        print("❌ 日志为空")
        return
    
    print(f"\n📊 共捕获 {len(logs)} 条夹爪写入记录\n")
    
    # 按来源分组
    sources = {}
    for log in logs:
        source = log["source"]
        if source not in sources:
            sources[source] = []
        sources[source].append(log)
    
    print("=" * 80)
    print("📝 各来源统计：")
    print("=" * 80)
    for source, entries in sources.items():
        print(f"\n【{source}】")
        print(f"  总写入次数: {len(entries)}")
        print(f"  PID: {entries[0]['pid']}")
        print(f"  时间范围: {entries[0]['datetime']} ~ {entries[-1]['datetime']}")
        
        # 显示前3条和后3条
        print(f"\n  前3条样本:")
        for entry in entries[:3]:
            print(f"    {entry['datetime']}: {entry}")
        
        if len(entries) > 6:
            print(f"  ... (省略 {len(entries) - 6} 条) ...")
        
        if len(entries) > 3:
            print(f"\n  后3条样本:")
            for entry in entries[-3:]:
                print(f"    {entry['datetime']}: {entry}")
    
    print("\n" + "=" * 80)
    print("🔍 冲突检测：")
    print("=" * 80)
    
    if len(sources) == 1:
        source_name = list(sources.keys())[0]
        print(f"✅ 只有 [{source_name}] 在写夹爪，无冲突")
    elif len(sources) == 0:
        print("❌ 没有捕获到任何写入")
    else:
        print(f"⚠️  检测到 {len(sources)} 个来源在写夹爪：")
        for source in sources.keys():
            print(f"    - {source}")
        print("\n⚠️  存在潜在双写冲突！")
        
        # 检查时间重叠
        print("\n⏱️  时间线分析（检查是否同时写入）：")
        all_logs_sorted = sorted(logs, key=lambda x: x["timestamp"])
        
        conflicts = []
        for i in range(len(all_logs_sorted) - 1):
            curr = all_logs_sorted[i]
            next_log = all_logs_sorted[i + 1]
            time_gap = next_log["timestamp"] - curr["timestamp"]
            
            if time_gap < 0.1 and curr["source"] != next_log["source"]:  # 100ms内不同来源
                conflicts.append((curr, next_log, time_gap))
        
        if conflicts:
            print(f"\n⚠️  发现 {len(conflicts)} 处时间冲突（<100ms内不同来源写入）：")
            for i, (log1, log2, gap) in enumerate(conflicts[:5], 1):  # 只显示前5个
                print(f"\n  冲突 {i}:")
                print(f"    {log1['source']}: {log1['datetime']} -> {log1}")
                print(f"    {log2['source']}: {log2['datetime']} (间隔 {gap*1000:.1f}ms) -> {log2}")
        else:
            print("  ✅ 未发现紧密时间冲突（可能是顺序交替写入）")
    
    print("\n" + "=" * 80)
    print("📋 结论（三选一）：")
    print("=" * 80)
    if len(sources) == 1:
        source_name = list(sources.keys())[0]
        if "LeRobot" in source_name or "rm65_follower" in source_name:
            print("✅ 只有 LeRobot 写")
        elif "External" in source_name:
            print("✅ 只有另一个程序写")
        else:
            print(f"✅ 只有 {source_name} 写")
    elif len(sources) > 1:
        print("⚠️  两边都写（冲突确认）")
    else:
        print("❓ 未检测到写入")
    
    print("\n💡 提示：如果只看到 EPGGripperClient_set_position，说明只有 LeRobot 在写（这是正常的）")
    print("        如果看到多个不同来源，说明存在双写冲突！")

if __name__ == "__main__":
    import sys
    
    if "--analyze" in sys.argv:
        analyze_logs()
    elif "--clear" in sys.argv:
        if LOG_FILE.exists():
            LOG_FILE.unlink()
            print(f"✅ 已清空日志: {LOG_FILE}")
    else:
        print(__doc__)
        print(f"\n当前日志文件: {LOG_FILE}")
        if LOG_FILE.exists():
            print(f"日志大小: {LOG_FILE.stat().st_size} bytes")
        else:
            print("日志文件尚未创建")
        print("\n使用方法:")
        print("  python test_gripper_writer.py          # 显示帮助")
        print("  python test_gripper_writer.py --analyze # 分析日志")
        print("  python test_gripper_writer.py --clear   # 清空日志")
