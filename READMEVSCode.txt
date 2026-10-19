# 1. 打开 VS Code → 文件 → 打开文件夹 → 选 D:\Projects\resume-agent.v2
# 2. Ctrl+` 开终端，应该直接在项目目录下（提示符显示 PS D:\Projects\resume-agent.v2>）

# 3. 放行执行策略（只跑一次）
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# 输入 Y 回车

# 4. 安装依赖
pip install -r requirements.txt

# 5. 语法检查（无输出=OK）
python -m compileall app evaluation frontend tests

# 6. 单元测试（20 passed）
python -m pytest tests -v

# 7. 跑量化指标
python -m evaluation.run_eval

# 8. 启动后端
python -m uvicorn app.main:app --reload --port 8000

现在访问：
- http://127.0.0.1:8000/ → 根路由
- http://127.0.0.1:8000/health → 看系统状态
- http://127.0.0.1:8000/docs → 自动生成的 API 文档（ 这个最有用 ，可以在页面上直接测试所有接口）

# 9. 另开终端启动前端
python -m streamlit run frontend/streamlit_app.py

