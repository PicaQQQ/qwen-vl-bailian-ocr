import os
import sys
import base64
import io
from openai import OpenAI
import time
import concurrent

BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"




#并发数量
MAX_WORKERS=2

#设置为APIKEY
DASHSCOPE_API_KEY='xxx'	




#设置模型
#MODEL_NAME="qwen-turbo-latest"
#效果很一般 但也翻译了 √
#模型调用-输入：¥0.0003/千Token
#模型调用-输出：¥0.0006/千Token


#MODEL_NAME="qwen-plus-2025-01-25"
#效果良好 √√
#模型调用-输入：¥0.0008/千Token
#模型调用-输出：¥0.002/千Token


#MODEL_NAME="qwen-max-2025-01-25"
#效果很好 √√√
#模型调用-输入：¥0.0024/千Token
#模型调用-输出：¥0.0096/千Token


#MODEL_NAME="qwen2.5-32b-instruct"
#效果很好 √√√
#模型调用-输入：¥0.002/千Token
#模型调用-输出：¥0.006/千Token

#MODEL_NAME="deepseek-v3"
#效果很好 √√√
#模型调用-输入：¥0.002/千Token
#模型调用-输出：¥0.008/千Token





#识别提示词


PROMPT_TEXT="请将文本翻译成中文，要求输出的文本中每段都有中英文对照，英文原文在前，中文译文在后，要求翻译全部文本，要求输出中只包含原文和译文"

PROMPT_TEXT_COMBINE="请将下面文本翻译成中文，要求输出的文本中每段都有中英文对照，英文原文在前，中文译文在后，要求翻译全部文本，要求输出中只包含原文和译文:\n\n"

def trans_with_txt_data(my_content):


    
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=BASE_URL,
    )

    com_prompt = PROMPT_TEXT_COMBINE+my_content
    status="success"
    try:
        #方式一
        '''
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {'role': 'user', 'content': com_prompt}],
        )
        '''
        #方式二
        
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': my_content},
                {'role': 'user', 'content': PROMPT_TEXT}],
        )
        
        result=completion.choices[0].message.content
    except Exception as e:
        status="failed"
        result=str(e)
    return status,result


def process_txt_task(filename, directory):

    filepath = os.path.join(directory, filename)
    # 分离文件名和扩展名
    name_without_ext, ext = os.path.splitext(filename)

    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    output_directory=directory+"_output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"文件夹 '{output_directory}' 已创建。")



    
    print(f"正在处理 {filename} ...")
    status,ocr_text = trans_with_txt_data(content)
    if status=="success":
        output_file = os.path.join(output_directory, f"{name_without_ext}-translate.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(ocr_text)
        print(f"已完成 {filename} ...")
    else:
        print(f"失败 {filename} ...")
        print(f"错误信息 {ocr_text} ...")

    return "Over"



def process_txt(directory,max_workers=MAX_WORKERS):


    txt_filenames = []
    # 定义支持的图片后缀
    valid_extensions = ('.txt')
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.lower().endswith(valid_extensions):
            txt_filenames.append(filename)
 
    ocr_results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_txt_task, filename, directory): filename 
            for filename in txt_filenames
        }
        # 等待所有任务完成，并收集结果
        for future in concurrent.futures.as_completed(futures):
            res_st = future.result()
            ocr_results[filename] = res_st
    return ocr_results


if __name__ == "__main__":
    print("本次调用模型："+MODEL_NAME)
    start_time = time.time()  
    txt_directory = sys.argv[1]
    results = process_txt(txt_directory)
    end_time = time.time()  
    elapsed_time = end_time - start_time
    print("用时：",elapsed_time)



