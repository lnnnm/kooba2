from PIL import Image, ImageChops
import os
import math
import re

# 1. 配置：图片所在文件夹 & 输出文件名
input_folder = r"C:\Users\DELL\Desktop\myfig"  # ← 修改为你的文件夹路径
output_file  = 'myfig.png'    # ← 输出的大图名

# 2. 读取所有图片路径（按数字顺序）
exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
files = [fn for fn in os.listdir(input_folder) if fn.lower().endswith(exts)]

def numeric_key(fn):
    name = os.path.splitext(fn)[0]
    try:
        return int(name)
    except:
        m = re.search(r'\d+', name)
        return int(m.group()) if m else float('inf')

paths = [os.path.join(input_folder, fn) for fn in sorted(files, key=numeric_key)]
n_imgs = len(paths)
cols   = 2
rows   = math.ceil(n_imgs / cols)

# 3. 打开图片并「去除白边」
trimmed_imgs = []
for p in paths:
    img = Image.open(p)
    # 把接近白色的背景点当成纯白
    bg = Image.new(img.mode, img.size, (255,255,255))
    diff = ImageChops.difference(img, bg)
    # 扩一点边缓冲
    bbox = diff.getbbox()
    if bbox:
        # 这里可以微调裁切范围，例如向内/外各缩放 few pixels
        left, upper, right, lower = bbox
        # 上下左右再缩一点，让坐标轴 label 不被切掉
        pad = 5
        left   = max(left-pad,   0)
        upper  = max(upper-pad,  0)
        right  = min(right+pad,  img.width)
        lower  = min(lower+pad,  img.height)
        img = img.crop((left, upper, right, lower))
    trimmed_imgs.append(img)

# 4. 计算网格单元大小：取所有子图的最大尺寸
widths, heights = zip(*(im.size for im in trimmed_imgs))
cell_w, cell_h = max(widths), max(heights)

# 5. 创建白底大图
grid_img = Image.new('RGB', (cols*cell_w, rows*cell_h), (255,255,255))

# 6. 依次粘贴居中
for idx, im in enumerate(trimmed_imgs):
    row = idx // cols
    col = idx % cols
    x = col * cell_w + (cell_w - im.width)//2
    y = row * cell_h + (cell_h - im.height)//2
    grid_img.paste(im, (x, y))

# 7. 保存
grid_img.save(output_file)
print(f"Saved trimmed merged image to {output_file}")
