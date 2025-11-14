from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
import os
import base64
from io import BytesIO
from PIL import Image
import logging
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


# 配置
class Config:
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "your-api-key-here")
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    MODEL = "qwen3-vl-plus"


app.config.from_object(Config)


def init_openai_client():
    """初始化OpenAI客户端"""
    return OpenAI(
        api_key=app.config['DASHSCOPE_API_KEY'],
        base_url=app.config['BASE_URL']
    )


def process_image_file(image_file):
    """处理上传的图片文件"""
    try:
        # 打开图片
        image = Image.open(image_file)

        # 转换为RGB模式（如果需要）
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 调整图片大小（可选，避免图片太大）
        max_size = (1024, 1024)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)

        # 将图片转换为base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        logger.error(f"图片处理错误: {str(e)}")
        raise e


@app.route('/')
def index():
    """前端页面"""
    return render_template('index.html')


@app.route('/api/extract-text', methods=['POST'])
def extract_text():
    """提取图片文字API"""
    try:
        data = request.json
        image_data = data.get('image')
        question = data.get('question', '请提取图片中的文字内容')
        enable_thinking = data.get('enable_thinking', False)

        if not image_data:
            return jsonify({'error': '未提供图片数据'}), 400

        # 初始化客户端
        client = init_openai_client()

        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data
                        },
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        # 额外参数
        extra_body = {
            'enable_thinking': enable_thinking,
            "thinking_budget": 81920
        }

        # 创建聊天完成请求
        completion = client.chat.completions.create(
            model=app.config['MODEL'],
            messages=messages,
            stream=True,
            extra_body=extra_body
        )

        # 处理流式响应
        reasoning_content = ""
        answer_content = ""
        is_answering = False

        def generate():
            nonlocal reasoning_content, answer_content, is_answering

            for chunk in completion:
                if not chunk.choices:
                    # 处理使用量信息
                    if hasattr(chunk, 'usage'):
                        usage_data = {
                            'usage': {
                                'prompt_tokens': getattr(chunk.usage, 'prompt_tokens', 0),
                                'completion_tokens': getattr(chunk.usage, 'completion_tokens', 0),
                                'total_tokens': getattr(chunk.usage, 'total_tokens', 0)
                            },
                            'status': 'completed'
                        }
                        yield f"data: {json.dumps(usage_data)}\n\n"
                    continue

                delta = chunk.choices[0].delta

                # 处理思考过程
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content
                    response_data = {
                        'reasoning_content': reasoning_content,
                        'status': 'thinking'
                    }
                    yield f"data: {json.dumps(response_data)}\n\n"
                elif delta.content:
                    # 开始回复
                    if not is_answering:
                        is_answering = True

                    answer_content += delta.content
                    response_data = {
                        'answer_content': answer_content,
                        'status': 'answering'
                    }
                    yield f"data: {json.dumps(response_data)}\n\n"

            # 完成
            response_data = {
                'status': 'completed',
                'reasoning_content': reasoning_content,
                'answer_content': answer_content
            }
            yield f"data: {json.dumps(response_data)}\n\n"

        return app.response_class(generate(), mimetype='text/plain')

    except Exception as e:
        logger.error(f"API错误: {str(e)}")
        error_response = {
            'error': str(e),
            'status': 'error'
        }
        return jsonify(error_response), 500


@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    """上传图片API"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': '未找到图片文件'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': '未选择文件'}), 400

        # 处理图片
        image_data_url = process_image_file(image_file)

        return jsonify({
            'success': True,
            'image_data': image_data_url,
            'message': '图片上传成功'
        })

    except Exception as e:
        logger.error(f"图片上传错误: {str(e)}")
        return jsonify({'error': f'图片处理失败: {str(e)}'}), 500


@app.route('/api/process-url', methods=['POST'])
def process_url():
    """处理图片URL"""
    try:
        data = request.json
        image_url = data.get('url')

        if not image_url:
            return jsonify({'error': '未提供图片URL'}), 400

        return jsonify({
            'success': True,
            'image_data': image_url,
            'message': 'URL处理成功'
        })

    except Exception as e:
        logger.error(f"URL处理错误: {str(e)}")
        return jsonify({'error': f'URL处理失败: {str(e)}'}), 500


if __name__ == '__main__':
    # 检查API密钥
    if app.config['DASHSCOPE_API_KEY'] == "your-api-key-here":
        print("警告: 请设置DASHSCOPE_API_KEY环境变量")

    app.run(debug=True, host='0.0.0.0', port=5000)