from pathlib import Path

def get_subfolder_names(folder_path):
    """
    返回指定文件夹中所有子文件夹的名称
    """
    path = Path(folder_path)

    if not path.exists():
        raise FileNotFoundError(f"路径不存在：{folder_path}")

    if not path.is_dir():
        raise NotADirectoryError(f"这不是一个文件夹：{folder_path}")

    subfolder_names = [
        item.name
        for item in path.iterdir()
        if item.is_dir()
    ]

    return subfolder_names


if __name__ == "__main__":
    folder = input("请输入文件夹路径：").strip()

    try:
        subfolders = get_subfolder_names(folder)

        print("子文件夹名称如下：")
        for name in subfolders:
            print(name)

    except Exception as e:
        print(f"出错了：{e}")