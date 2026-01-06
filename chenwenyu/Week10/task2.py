from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

def extract_text_from_image(image_path):
    # 加载模型
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "../../../models/Qwen/Qwen2-VL-7B-Instruct",
        dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("../../../models/Qwen/Qwen2-VL-7B-Instruct")
    
    image = Image.open(image_path)
    
    # 使用明确的文字提取指令
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "请提取图片中的所有文字内容和数学符号，按原样输出，不要添加任何解释。"}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = generated_ids[0][len(inputs.input_ids[0]):]
    output_text = processor.decode(generated_ids_trimmed, skip_special_tokens=True)
    
    return output_text

def save_text_as_markdown(text, filepath):
    """
    将文本保存为Markdown文件
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # 添加Markdown标题（可选）
            f.write("# 从图片中提取的文字\n\n")
            f.write(text)

        # 添加图像引用（可选）
        if filepath.endswith('.md'):
            # 保存图像到同一目录
            import os
            image_filename = os.path.splitext(os.path.basename(filepath))[0] + "_source.png"
            # 这里可以添加代码将原始图片复制到同一目录，但需要原始图片路径

        print(f"✓ 文本已保存到: {filepath}")
    except Exception as e:
        print(f"✗ 保存文件时出错: {e}")

def extract_and_save(image_path, output_md_path):
    """
    提取文字并保存的完整流程
    """
    print(f"正在处理图片: {image_path}")
    
    # 提取文字
    text_content = extract_text_from_image(image_path)
    print("\n提取的文字内容:")
    print("-" * 50)
    print(text_content)
    print("-" * 50)
    
    # 保存为Markdown
    save_text_as_markdown(text_content, output_md_path)
    
    return text_content

if __name__ == "__main__":
    # 使用示例
    text_content = extract_text_from_image("./pics/test16.png")
    print("提取的文字:", text_content)

    # 保存为Markdown文件
    save_text_as_markdown(text_content, "./pics/test16.md")

    # 或者使用更简洁的方式
    # text_content = extract_and_save("./pics/test16.png", "./pics/test16.md")
