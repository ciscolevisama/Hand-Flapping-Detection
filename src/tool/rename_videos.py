import os
import re
from pathlib import Path

def clean_name(name: str) -> str:
    """
    清理文件名：
    - 空格 -> 下划线
    - 去掉 [ ] 括号
    - 去掉多余的连续下划线
    """
    name = name.replace(" ", "_")
    name = re.sub(r"[\[\]]", "", name)
    name = re.sub(r"_+", "_", name)  # 避免多个下划线连在一起
    return name

def rename_videos(folder="data/raw_videos"):
    folder = Path(folder)
    if not folder.exists():
        print(f"❌ 文件夹不存在: {folder}")
        return

    for file in folder.iterdir():
        if not file.is_file():
            continue
        new_name = clean_name(file.name)
        if new_name != file.name:
            new_path = file.with_name(new_name)
            print(f"🔄 {file.name}  →  {new_name}")
            file.rename(new_path)

    print("✅ 批量重命名完成")

if __name__ == "__main__":
    rename_videos("data/raw_videos")
