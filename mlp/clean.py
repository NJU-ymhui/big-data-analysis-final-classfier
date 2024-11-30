import pandas as pd


def is_valid_utf8(line):
    try:
        line.encode('utf-8')
    except UnicodeEncodeError:
        print("err")
        return False
    return True


input_file = 'papers_cleaned.csv'  # 输入文件路径
output_file = 'papers_cleaned.csv'  # 输出文件路径

# 读取CSV文件，忽略编码错误
with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

# 检查每行是否为有效的UTF-8字符串，并过滤掉非UTF-8的行
valid_lines = [line for line in lines if is_valid_utf8(line)]

# 将有效的行写入新文件
with open(output_file, 'w', encoding='utf-8-sig') as f:
    f.writelines(valid_lines)

print(f"文件已清理并保存为 {output_file}")
