from flask import Flask, request, jsonify, send_file, session, redirect
from flask_cors import CORS
import csv
import os
import uuid
import subprocess
import pandas as pd
import sys
import threading
from werkzeug.utils import secure_filename

# 添加utils目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from model_train import train_model

app = Flask(__name__, static_folder='.')
app.secret_key = 'your_secret_key'  # 用于session加密
CORS(app)  # 启用跨域支持

# 上传文件保存目录
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 模拟用户数据库
users = {
    "admin": "123456",
    "user1": "password123"
}

# 训练任务状态
training_tasks = {}

# 训练线程字典
training_threads = {}

@app.route('/')
def index():
    return send_file('login.html')

@app.route('/login')
def login_page():
    return send_file('login.html')

@app.route('/upload')
def upload_page():
    if 'username' in session:
        return send_file('upload.html')
    return redirect('/login')

@app.route('/dataset')
def dataset_page():
    if 'username' in session:
        return send_file('dataset.html')
    return redirect('/login')

@app.route('/check')
def check_page():
    if 'username' in session:
        return send_file('check.html')
    return redirect('/login')

@app.route('/api/check_dataset/<train_id>', methods=['GET'])
def check_dataset(train_id):
    if 'username' not in session:
        return jsonify({
            "success": False,
            "message": "未登录"
        }), 401
    
    if train_id not in training_tasks:
        return jsonify({
            "success": False,
            "message": "训练任务不存在"
        }), 404
    
    # 获取分页参数
    page = int(request.args.get('page', 1))
    page_size = 10  # 每页显示10条数据
    
    task = training_tasks[train_id]
    data = task['data']
    
    # 计算分页
    total_rows = len(data)
    total_pages = (total_rows + page_size - 1) // page_size
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    
    # 对数据进行校验
    checked_data = []
    errors = []
    warnings = []
    
    for i, item in enumerate(data[start_idx:end_idx]):
        idx = start_idx + i
        check_result = check_data_item(item, idx)
        checked_data.append(check_result)
        
        if check_result['status'] == 'error':
            errors.append(f"第 {idx+1} 行: {check_result['message']}")
        elif check_result['status'] == 'warning':
            warnings.append(f"第 {idx+1} 行: {check_result['message']}")
    
    # 检查整个数据集的问题
    all_errors = []
    all_warnings = []
    
    # 检查所有数据项
    for i, item in enumerate(data):
        check_result = check_data_item(item, i)
        if check_result['status'] == 'error':
            all_errors.append(f"第 {i+1} 行: {check_result['message']}")
        elif check_result['status'] == 'warning':
            all_warnings.append(f"第 {i+1} 行: {check_result['message']}")
    
    # 检查数据集大小
    if total_rows < 10:
        all_warnings.append(f"数据集只有 {total_rows} 条记录，建议至少提供10条以上的数据以获得更好的训练效果")
    
    # 生成校验摘要
    check_summary = {
        "totalRows": total_rows,
        "errors": all_errors,
        "warnings": all_warnings
    }
    
    if len(all_errors) > 0:
        check_summary["status"] = "error"
        check_summary["canTrain"] = False
    elif len(all_warnings) > 0:
        check_summary["status"] = "warning"
        check_summary["canTrain"] = True
    else:
        check_summary["status"] = "success"
        check_summary["canTrain"] = True
    
    return jsonify({
        "success": True,
        "fileName": task['filename'],
        "totalRows": total_rows,
        "currentPage": page,
        "totalPages": total_pages,
        "data": checked_data,
        "checkSummary": check_summary
    })

# 检查单个数据项
def check_data_item(item, idx):
    result = {
        "text": item['text'],
        "result": item['result'],
        "status": "success"
    }
    
    # 检查text字段
    if not item['text'] or len(item['text'].strip()) == 0:
        result['status'] = 'error'
        result['message'] = 'text字段不能为空'
        return result
    
    # 检查result字段
    if not item['result'] or len(item['result'].strip()) == 0:
        result['status'] = 'error'
        result['message'] = 'result字段不能为空'
        return result
    
    # 检查text长度
    if len(item['text']) < 5:
        result['status'] = 'warning'
        result['message'] = 'text字段内容过短，建议提供更详细的输入'
        return result
    
    # 检查result长度
    if len(item['result']) < 5:
        result['status'] = 'warning'
        result['message'] = 'result字段内容过短，建议提供更详细的输出'
        return result
    
    # 检查text和result是否完全相同
    if item['text'] == item['result']:
        result['status'] = 'warning'
        result['message'] = 'text和result内容完全相同，可能不利于模型学习'
        return result
    
    return result

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if username in users and users[username] == password:
        session['username'] = username
        # 创建用户专属的上传和checkpoint目录
        user_upload_dir = os.path.join(UPLOAD_FOLDER, username)
        user_checkpoint_dir = os.path.join(UPLOAD_FOLDER, username, 'checkpoints')
        os.makedirs(user_upload_dir, exist_ok=True)
        os.makedirs(user_checkpoint_dir, exist_ok=True)
        
        return jsonify({
            "success": True,
            "message": "登录成功"
        })
    else:
        return jsonify({
            "success": False,
            "message": "用户名或密码错误"
        }), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    return jsonify({"success": True})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'username' not in session:
        return jsonify({
            "success": False,
            "message": "未登录"
        }), 401
    
    if 'file' not in request.files:
        return jsonify({
            "success": False,
            "message": "没有文件"
        })
    
    file = request.files['file']
    model = request.form.get('model', 'glm4')  # 获取选择的模型，默认为glm4
    
    if file.filename == '':
        return jsonify({
            "success": False,
            "message": "未选择文件"
        })
    
    if not file.filename.endswith('.csv'):
        return jsonify({
            "success": False,
            "message": "只支持CSV文件"
        })
    
    # 生成唯一的训练ID
    train_id = str(uuid.uuid4())
    
    # 保存文件
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, f"{train_id}_{filename}")
    file.save(file_path)
    
    # 验证CSV文件格式
    try:
        df = pd.read_csv(file_path)
        if 'text' not in df.columns or 'result' not in df.columns:
            os.remove(file_path)
            return jsonify({
                "success": False,
                "message": "CSV文件必须包含'text'和'result'两列"
            })
    except Exception as e:
        os.remove(file_path)
        return jsonify({
            "success": False,
            "message": f"CSV文件格式错误: {str(e)}"
        })
    
    # 保存训练任务信息
    training_tasks[train_id] = {
        "status": "uploaded",
        "file": file_path,
        "filename": filename,
        "model": model,
        "user": session['username'],
        "start_time": pd.Timestamp.now().isoformat(),
        "data": df.to_dict('records')  # 保存CSV数据
    }
    
    return jsonify({
        "success": True,
        "message": "文件上传成功，数据校验通过",
        "trainId": train_id
    })

# 训练模型的后台线程函数
def train_model_thread(train_id):
    """在后台线程中执行模型训练"""
    try:
        task = training_tasks[train_id]
        training_tasks[train_id]['status'] = 'training'
        
        # 获取任务信息
        csv_path = task['file']
        username = task['user']
        model_type = task['model']
        
        # 调用模型训练函数
        checkpoint_dir = train_model(csv_path, username, model_type)
        
        if checkpoint_dir:
            training_tasks[train_id]['status'] = 'completed'
            training_tasks[train_id]['checkpoint_dir'] = checkpoint_dir
            training_tasks[train_id]['end_time'] = pd.Timestamp.now().isoformat()
        else:
            training_tasks[train_id]['status'] = 'failed'
            training_tasks[train_id]['error'] = '训练过程中出现错误'
    except Exception as e:
        training_tasks[train_id]['status'] = 'failed'
        training_tasks[train_id]['error'] = str(e)
    finally:
        # 清理线程引用
        if train_id in training_threads:
            del training_threads[train_id]

@app.route('/api/train/<train_id>', methods=['POST'])
def start_training(train_id):
    """启动模型训练"""
    if 'username' not in session:
        return jsonify({
            "success": False,
            "message": "未登录"
        }), 401
    
    if train_id not in training_tasks:
        return jsonify({
            "success": False,
            "message": "训练任务不存在"
        }), 404
    
    task = training_tasks[train_id]
    
    # 检查任务是否属于当前用户
    if task['user'] != session['username']:
        return jsonify({
            "success": False,
            "message": "无权访问此训练任务"
        }), 403
    
    # 检查任务状态
    if task['status'] == 'training':
        return jsonify({
            "success": False,
            "message": "训练任务已在进行中"
        })
    
    if task['status'] == 'completed':
        return jsonify({
            "success": False,
            "message": "训练任务已完成"
        })
    
    # 启动训练线程
    train_thread = threading.Thread(target=train_model_thread, args=(train_id,))
    train_thread.daemon = True  # 设置为守护线程，主线程结束时自动结束
    train_thread.start()
    
    # 保存线程引用
    training_threads[train_id] = train_thread
    
    return jsonify({
        "success": True,
        "message": "训练任务已启动",
        "trainId": train_id
    })

@app.route('/api/train_status/<train_id>', methods=['GET'])
def get_training_status(train_id):
    """获取训练任务状态"""
    if 'username' not in session:
        return jsonify({
            "success": False,
            "message": "未登录"
        }), 401
    
    if train_id not in training_tasks:
        return jsonify({
            "success": False,
            "message": "训练任务不存在"
        }), 404
    
    task = training_tasks[train_id]
    
    # 检查任务是否属于当前用户
    if task['user'] != session['username']:
        return jsonify({
            "success": False,
            "message": "无权访问此训练任务"
        }), 403
    
    # 构建状态响应
    status_info = {
        "trainId": train_id,
        "status": task['status'],
        "model": task['model'],
        "filename": task['filename'],
        "startTime": task['start_time']
    }
    
    # 添加额外信息
    if task['status'] == 'completed':
        status_info['endTime'] = task.get('end_time')
        status_info['checkpointDir'] = task.get('checkpoint_dir')
    
    if task['status'] == 'failed':
        status_info['error'] = task.get('error', '未知错误')
    
    return jsonify({
        "success": True,
        "data": status_info
    })

@app.route('/api/dataset/<train_id>', methods=['GET'])
def get_dataset(train_id):
    if 'username' not in session:
        return jsonify({
            "success": False,
            "message": "未登录"
        }), 401
    
    if train_id not in training_tasks:
        return jsonify({
            "success": False,
            "message": "训练任务不存在"
        }), 404
    
    # 获取分页参数
    page = int(request.args.get('page', 1))
    page_size = 10  # 每页显示10条数据
    
    task = training_tasks[train_id]
    data = task['data']
    
    # 计算分页
    total_rows = len(data)
    total_pages = (total_rows + page_size - 1) // page_size
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    
    return jsonify({
        "success": True,
        "fileName": task['filename'],
        "totalRows": total_rows,
        "currentPage": page,
        "totalPages": total_pages,
        "data": data[start_idx:end_idx]
    })

@app.route('/api/start_training_task/<train_id>', methods=['POST'])
def start_training_task(train_id):
    if 'username' not in session:
        return jsonify({
            "success": False,
            "message": "未登录"
        }), 401
    
    if train_id not in training_tasks:
        return jsonify({
            "success": False,
            "message": "训练任务不存在"
        }), 404
    
    task = training_tasks[train_id]
    
    # 更新任务状态
    task['status'] = 'training'
    
    # 这里可以调用train.py进行模型训练
    # subprocess.Popen(['python', 'train.py', task['file'], train_id, task['model']])
    
    return jsonify({
        "success": True,
        "message": "训练已开始"
    })

@app.route('/api/cancel_training/<train_id>', methods=['POST'])
def cancel_training(train_id):
    if 'username' not in session:
        return jsonify({
            "success": False,
            "message": "未登录"
        }), 401
    
    if train_id not in training_tasks:
        return jsonify({
            "success": False,
            "message": "训练任务不存在"
        }), 404
    
    task = training_tasks[train_id]
    
    # 删除上传的文件
    if os.path.exists(task['file']):
        os.remove(task['file'])
    
    # 删除任务
    del training_tasks[train_id]
    
    return jsonify({
        "success": True,
        "message": "训练已取消"
    })

@app.route('/api/training_status/<train_id>', methods=['GET'])
def training_status(train_id):
    if train_id not in training_tasks:
        return jsonify({
            "success": False,
            "message": "训练任务不存在"
        }), 404
    
    return jsonify({
        "success": True,
        "status": training_tasks[train_id]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)