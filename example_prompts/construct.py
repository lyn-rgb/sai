import os
import csv
import glob
from pathlib import Path

def generate_benchmark_csv():
    """生成基准测试CSV文件"""
    # 基础路径
    base_path = Path("/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/benchmark1/processed")
    
    # 子目录路径
    full_video_prompt_dir = base_path / "full_video_prompt"
    frames_dir = base_path / "frames"
    audio_dir = base_path / "audio"
    
    # 输出文件
    output_csv = "benchmark_new-v6.csv"
    
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # 获取所有txt文件并排序
    txt_files = sorted(glob.glob(os.path.join(full_video_prompt_dir, "*_full_caption.txt")))
    
    # 准备CSV数据
    csv_data = [["text_prompt", "ip_image_path", "ip_audio_path"]]
    
    for txt_file in txt_files:
        # 提取文件编号
        base_name = os.path.basename(txt_file)
        file_id = base_name.split("_")[0]  # 例如 "00001"
        
        # 读取文本内容
        with open(txt_file, 'r', encoding='utf-8') as f:
            text_content = f.read().strip()
        
        # 查找存在的图片文件（支持多种格式） 
        ########确定图片名字是对的
        image_path = ""
        for ext in image_extensions:
            potential_path = frames_dir / f"{file_id}_frame{ext}"
            if potential_path.exists():
                image_path = str(potential_path)
                break
        
        # 构建音频文件路径
        audio_path = audio_dir / f"{file_id}_audio.wav"
        audio_exists = audio_path.exists()
        audio_path_str = str(audio_path) if audio_exists else ""
        
        # 第一条记录：包含text prompt, 图片地址，音频地址
        if image_path and audio_exists:
            csv_data.append([text_content, image_path, str(audio_path)])
        
        # 第二条记录：包含text prompt, 图片地址
        if image_path:
            csv_data.append([text_content, image_path, ""])
        
        # 第三条记录：只包含text prompt
        csv_data.append([text_content, "", ""])
    
    # 写入CSV文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    
    print(f"CSV文件已生成: {output_csv}")
    print(f"总共处理了 {len(txt_files)} 个文本文件")
    print(f"生成了 {len(csv_data)-1} 条记录")

if __name__ == "__main__":
    generate_benchmark_csv()




# import os
# import csv
# import glob

# # 基础路径
# base_path = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/benchmark1/processed"

# # 子目录路径
# full_video_prompt_dir = os.path.join(base_path, "full_video_prompt")
# frames_dir = os.path.join(base_path, "frames")
# audio_dir = os.path.join(base_path, "audio")

# # 输出文件
# output_csv = "benchmark.csv"

# # 获取所有txt文件并排序
# txt_files = sorted(glob.glob(os.path.join(full_video_prompt_dir, "*_full_caption.txt")))

# # 准备CSV数据
# csv_data = [["text_prompt", "ip_image_path", "ip_audio_path"]]

# for txt_file in txt_files:
#     # 提取文件编号
#     base_name = os.path.basename(txt_file)
#     file_id = base_name.split("_")[0]  # 例如 "00001"
    
#     # 读取文本内容
#     with open(txt_file, 'r', encoding='utf-8') as f:
#         text_content = f.read().strip()
    
#     # 构建图片和音频文件路径
#     image_path = os.path.join(frames_dir, f"{file_id}_frame.jpg")
#     audio_path = os.path.join(audio_dir, f"{file_id}_audio.wav")
    
#     # 检查文件是否存在
#     image_exists = os.path.exists(image_path)
#     audio_exists = os.path.exists(audio_path)
    
#     # 第一条记录：包含text prompt, 图片地址，音频地址
#     if image_exists and audio_exists:
#         csv_data.append([text_content, image_path, audio_path])
    
#     # 第二条记录：包含text prompt, 图片地址
#     if image_exists:
#         csv_data.append([text_content, image_path, ""])
    
#     # 第三条记录：只包含text prompt
#     csv_data.append([text_content, "", ""])

# # 写入CSV文件
# with open(output_csv, 'w', newline='', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerows(csv_data)

# print(f"CSV文件已生成: {output_csv}")
# print(f"总共处理了 {len(txt_files)} 个文本文件")
# print(f"生成了 {len(csv_data)-1} 条记录")