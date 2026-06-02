"""
PyInstaller 打包脚本
任务要求：使用 pyinstaller 将 Streamlit 应用打包成一个独立的 exe 可执行文件
"""

import os
import sys
import subprocess
import shutil


def clean_build_dirs():
    """清理构建目录"""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"已清理: {dir_name}")


def create_streamlit_runner():
    """创建 Streamlit 运行器脚本"""
    runner_code = '''
import streamlit.web.cli as stcli
import sys
import os

# 设置环境变量
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_SERVER_PORT'] = '8501'

if __name__ == '__main__':
    sys.argv = ['streamlit', 'run', 'app.py', '--server.headless=true']
    sys.exit(stcli.main())
'''
    with open('run_streamlit.py', 'w', encoding='utf-8') as f:
        f.write(runner_code)
    print("已创建: run_streamlit.py")


def build_exe():
    """使用 PyInstaller 打包 exe"""
    print("=" * 60)
    print("开始打包 RAG-QA-System 为可执行文件")
    print("=" * 60)

    # 清理旧构建
    clean_build_dirs()

    # 创建 Streamlit 运行器
    create_streamlit_runner()

    # PyInstaller 命令
    cmd = [
        'pyinstaller',
        '--name=RAG-QA-System',
        '--onefile',
        '--windowed',
        '--icon=NONE',
        # 添加数据文件
        '--add-data', 'app.py;.',
        '--add-data', 'config.py;.',
        '--add-data', 'rag_pipeline.py;.',
        '--add-data', 'document_loader.py;.',
        '--add-data', 'embeddings.py;.',
        '--add-data', 'utils;utils',
        '--add-data', 'docs;docs',
        # 隐藏导入
        '--hidden-import', 'streamlit',
        '--hidden-import', 'langchain',
        '--hidden-import', 'langchain_community',
        '--hidden-import', 'chromadb',
        '--hidden-import', 'ollama',
        '--hidden-import', 'pypdf',
        '--hidden-import', 'docx2txt',
        '--hidden-import', 'tiktoken',
        # 排除不需要的模块以减小体积
        '--exclude-module', 'matplotlib',
        '--exclude-module', 'numpy.random._examples',
        # 主脚本
        'run_streamlit.py'
    ]

    print("\n执行命令:")
    print(' '.join(cmd))
    print()

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ 打包成功！")
        print(f"可执行文件位于: dist/RAG-QA-System.exe")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 打包失败: {e}")
        return False
    finally:
        # 清理临时文件
        if os.path.exists('run_streamlit.py'):
            os.remove('run_streamlit.py')
            print("已清理临时文件: run_streamlit.py")


def create_simple_launcher():
    """
    创建简单的启动器（替代方案）
    由于 Streamlit 的特殊性，直接打包较复杂，
    这里提供一个使用 Python 解释器运行的启动器
    """
    launcher_code = '''@echo off
chcp 65001 >nul
echo ==========================================
echo    RAG-QA-System 智能问答系统
echo ==========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.9+
    pause
    exit /b 1
)

REM 检查虚拟环境
if exist "venv\\Scripts\\activate.bat" (
    echo [信息] 激活虚拟环境...
    call venv\\Scripts\\activate.bat
) else (
    echo [警告] 未找到虚拟环境，使用系统 Python
)

REM 检查依赖
echo [信息] 检查依赖...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo [信息] 安装依赖...
    pip install -r requirements.txt
)

REM 启动应用
echo [信息] 启动 RAG-QA-System...
echo.
streamlit run app.py

pause
'''

    with open('启动RAG系统.bat', 'w', encoding='utf-8') as f:
        f.write(launcher_code)
    print("已创建: 启动RAG系统.bat")


def main():
    """主函数"""
    print("RAG-QA-System 打包工具")
    print("=" * 60)
    print()
    print("说明：")
    print("1. 由于 Streamlit 的特殊架构，完全打包为独立 exe 较为复杂")
    print("2. 推荐方案：使用启动脚本 + Python 环境")
    print("3. 如需纯独立 exe，请确保已安装所有依赖并运行 build_exe()")
    print()

    # 创建简单的启动器
    create_simple_launcher()

    print()
    print("选项:")
    print("1. 创建启动脚本（推荐，需要目标电脑有 Python）")
    print("2. 尝试打包为独立 exe（实验性，可能较大）")
    print("3. 退出")
    print()

    choice = input("请选择 (1-3): ").strip()

    if choice == '1':
        print("\n✅ 已创建启动脚本: 启动RAG系统.bat")
        print("将此脚本与项目文件一起复制到目标电脑即可运行")
    elif choice == '2':
        build_exe()
    else:
        print("退出")


if __name__ == "__main__":
    main()
