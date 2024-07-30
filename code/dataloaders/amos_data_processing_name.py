import os

def list_files_to_list_file(folder_path, output_file):
    # 获取文件夹中的所有文件名
    files = os.listdir(folder_path)
    
    # 过滤掉文件夹，只保留文件
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]

    files.sort()
    # 将文件名写入到 .list 文件中
    with open(output_file, 'w') as f:
        for file_name in files:
            f.write(f"{file_name}\n")

# 示例用法
folder_path = '/data_hdd/users/pengzelin/SSL4MIS/data/AbdomenMR/test'  # 替换为你的文件夹路径
output_file = '/data_hdd/users/pengzelin/SSL4MIS/data/AbdomenMR/test.list'  # 替换为你想要的输出文件名
list_files_to_list_file(folder_path, output_file)
