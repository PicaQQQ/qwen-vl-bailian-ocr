import os
import sys
import base64
from PIL import Image
import io
from openai import OpenAI
import time
import concurrent

BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"


#图片分辨率 最大 28*28*3000    注意 28*28*3600差不多是1920*1080 
MAX_PIXELS=28*28*1800

#并发数量
MAX_WORKERS=10

#设置为APIKEY
DASHSCOPE_API_KEY='xxx'	

#设置模型
#MODEL_NAME="qwen2.5-vl-72b-instruct"
#MODEL_NAME="qwen2.5-vl-3b-instruct"
MODEL_NAME="qwen2.5-vl-7b-instruct"

#识别提示词
#默认提示词
#PROMPT_TEXT="Read all the text in the image."
PROMPT_TEXT="将文件内容转为文字，输出为txt格式（不需要添加'''txt），如果存在上下层叠的文件，只转化最上层文件，如果存在多页文件，从左向右依次转化，生成的文字要符合英式英语格式，注意文件内容中数字为英文数字格式，注意区分数字部分的小数点和逗号，注意区分英镑符号和数字"



def ocr_image_with_api(base64_image, image_type):

    if image_type in ["jpg", "jpeg"]:
        mime_type = "jpeg"
    elif image_type == "png":
        mime_type = "png"
    elif image_type == "bmp":
        mime_type = "bmp"
    else:
        mime_type = "jpeg"
    
    image_data = f"data:image/{mime_type};base64,{base64_image}"
    
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=BASE_URL,
    )
    status="success"
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data},
                            "min_pixels": 28 * 28 * 4,      # 最小像素数
                            "max_pixels": MAX_PIXELS    # 最大像素数
                        },
                        {"type": "text", "text": PROMPT_TEXT},
                    ],
                }
            ],
        )
        result=completion.choices[0].message.content
    except Exception as e:
        status="failed"
        result=str(e)
    return status,result


def process_image_task(filename, directory):

    filepath = os.path.join(directory, filename)
    # 分离文件名和扩展名
    name_without_ext, ext = os.path.splitext(filename)
    # 获取图片类型（去掉点，并转为小写）
    image_type = ext[1:].lower()
    
    with Image.open(filepath) as img:
        width, height = img.size
        # 如果图片分辨率超过指定的最大值，则进行缩放
        if width * height > MAX_PIXELS:
            # 计算缩放因子：分辨率乘积按平方比例缩放
            scale_factor = (MAX_PIXELS / (width * height)) ** 0.5
            new_width = max(1, int(width * scale_factor))
            new_height = max(1, int(height * scale_factor))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 将处理后的图片保存到内存中，并进行base64编码
        buffered = io.BytesIO()
        # 指定保存的格式，对jpg/jpeg进行处理
        format = image_type.upper()
        if format in ['JPG', 'JPEG']:
            format = 'JPEG'
        elif format == 'PNG':
            format = 'PNG'
        elif format == 'BMP':
            format = 'BMP'
        else:
            format = 'PNG'
        
        img.save(buffered, format=format)
        encoded_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    print(f"正在处理 {filename} ...")
    status,ocr_text = ocr_image_with_api(encoded_data, image_type)
    if status=="success":
        output_file = os.path.join(directory, f"{name_without_ext}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(ocr_text)
        print(f"已完成 {filename} ...")
    else:
        print(f"失败 {filename} ...")
        print(f"错误信息 {ocr_text} ...")

    return "Over"



def process_images_with_ocr(directory,max_workers=MAX_WORKERS):


    image_filenames = []
    # 定义支持的图片后缀
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    
    # 遍历目录中的所有文件

    for filename in os.listdir(directory):
        if filename.lower().endswith(valid_extensions):
            image_filenames.append(filename)
 
    ocr_results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 为每个图片任务提交一个线程
        futures = {
            executor.submit(process_image_task, filename, directory): filename 
            for filename in image_filenames
        }
        # 等待所有任务完成，并收集结果
        for future in concurrent.futures.as_completed(futures):
            res_st = future.result()
            ocr_results[filename] = res_st
    return ocr_results


if __name__ == "__main__":

    start_time = time.time()  
    image_directory = sys.argv[1]
    results = process_images_with_ocr(image_directory)
    end_time = time.time()  
    elapsed_time = end_time - start_time
    print("用时：",elapsed_time)



