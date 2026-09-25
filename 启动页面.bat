@echo off
title AgenticRAG QA - 启动页面
cd /d C:\Users\YHL\PycharmProjects\Desktop\agentic-rag-qa
echo.
echo ========================================
echo   正在启动问答页面，请稍候...
echo   浏览器会自动打开 http://localhost:8501
echo   关闭这个窗口 = 关闭页面
echo ========================================
echo.
D:\anaconda3\envs\raggpu\python.exe -m streamlit run src/ui/app.py
pause
