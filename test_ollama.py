"""
Ollama API 测试脚本
任务要求：编写一个测试脚本，验证 Ollama API 能正常返回结果
"""

import requests
import sys
from config import OLLAMA_BASE_URL, DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL


def test_ollama_connection():
    """测试 Ollama 服务是否运行"""
    print("=" * 50)
    print("测试 Ollama 连接")
    print("=" * 50)

    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama 服务运行正常")
            print(f"   地址: {OLLAMA_BASE_URL}")
            print(f"\n已安装的模型:")
            for model in models:
                print(f"   - {model.get('name', 'unknown')}")
            return True
        else:
            print(f"❌ Ollama 服务返回错误: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到 Ollama 服务")
        print(f"   请确保 Ollama 已安装并运行: ollama serve")
        return False
    except Exception as e:
        print(f"❌ 连接测试失败: {e}")
        return False


def test_llm_generation():
    """测试大语言模型生成"""
    print("\n" + "=" * 50)
    print(f"测试 LLM 生成 (模型: {DEFAULT_LLM_MODEL})")
    print("=" * 50)

    test_prompt = "你好，请用一句话介绍自然语言处理技术。"

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": DEFAULT_LLM_MODEL,
                "prompt": test_prompt,
                "stream": False,
            },
            timeout=60,
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ LLM 生成测试成功")
            print(f"   提示: {test_prompt}")
            print(f"   回答: {result.get('response', '无响应')[:100]}...")
            return True
        else:
            print(f"❌ LLM 生成失败: {response.status_code}")
            print(f"   请确保模型已下载: ollama pull {DEFAULT_LLM_MODEL}")
            return False
    except Exception as e:
        print(f"❌ LLM 测试失败: {e}")
        return False


def test_embedding():
    """测试嵌入模型"""
    print("\n" + "=" * 50)
    print(f"测试嵌入模型 (模型: {DEFAULT_EMBEDDING_MODEL})")
    print("=" * 50)

    test_text = "自然语言处理是人工智能的重要分支。"

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={
                "model": DEFAULT_EMBEDDING_MODEL,
                "prompt": test_text,
            },
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            embedding = result.get("embedding", [])
            print(f"✅ 嵌入模型测试成功")
            print(f"   文本: {test_text}")
            print(f"   向量维度: {len(embedding)}")
            print(f"   向量前5个值: {embedding[:5]}")
            return True
        else:
            print(f"❌ 嵌入模型测试失败: {response.status_code}")
            print(f"   请确保模型已下载: ollama pull {DEFAULT_EMBEDDING_MODEL}")
            return False
    except Exception as e:
        print(f"❌ 嵌入测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n🔍 Ollama API 测试脚本")
    print("本脚本将测试 Ollama 服务的连接、LLM 生成和嵌入功能\n")

    results = []

    # 测试连接
    results.append(("Ollama 连接", test_ollama_connection()))

    # 测试 LLM
    results.append(("LLM 生成", test_llm_generation()))

    # 测试嵌入
    results.append(("嵌入模型", test_embedding()))

    # 总结
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)

    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 所有测试通过！Ollama API 工作正常。")
        return 0
    else:
        print("\n⚠️ 部分测试失败，请检查 Ollama 安装和模型下载。")
        print("\n常用命令:")
        print("  启动服务: ollama serve")
        print(f"  下载 LLM: ollama pull {DEFAULT_LLM_MODEL}")
        print(f"  下载嵌入: ollama pull {DEFAULT_EMBEDDING_MODEL}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
