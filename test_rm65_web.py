#!/usr/bin/env python3
"""检查 RM65 的 Web 界面"""

import requests

def check_web_interface():
    """检查 Web 界面"""
    ip = "169.254.128.20"
    
    print("=" * 60)
    print("RM65 Web 界面检查")
    print("=" * 60)
    print()
    
    # 获取主页
    try:
        response = requests.get(f"http://{ip}:80/", timeout=5)
        print(f"状态码: {response.status_code}")
        print()
        print("HTML 内容:")
        print("-" * 60)
        print(response.text)
        print("-" * 60)
        
        # 保存到文件
        with open("rm65_web_interface.html", "w", encoding="utf-8") as f:
            f.write(response.text)
        print()
        print("✓ 已保存到 rm65_web_interface.html")
        
    except Exception as e:
        print(f"✗ 获取失败: {e}")
    
    print()
    print("=" * 60)
    print("请在浏览器中访问: http://169.254.128.20:80")
    print("查看是否有初始化或设置选项")
    print("=" * 60)

if __name__ == "__main__":
    check_web_interface()
