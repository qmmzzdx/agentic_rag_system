"""
UI组件模块，包含所有Streamlit UI渲染逻辑
"""
import streamlit as st
from datetime import datetime
from typing import Tuple, List
import logging
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStoreService
from utils.chat_history import ChatHistoryManager
from langchain.schema import Document

logger = logging.getLogger(__name__)


class UIComponents:
    """UI组件类，封装了所有Streamlit UI渲染逻辑"""

    # 1.渲染模型选择组件
    @staticmethod
    def render_model_selection(available_models: List[str], current_model: str,
                               embedding_models: List[str], current_embedding_model: str) -> Tuple[str, str]:
        """
        渲染模型选择组件
        参数:
            available_models - 可用模型列表
            current_model - 当前选中的模型
            embedding_models - 可用嵌入模型列表
            current_embedding_model - 当前选中的嵌入模型

        返回:
            (用户选择的模型, 用户选择的嵌入模型)
        """
        st.sidebar.header("⚙️ 系统设置")

        # 获取当前模型的索引
        model_index = available_models.index(
            current_model) if current_model in available_models else 0
        embedding_index = embedding_models.index(
            current_embedding_model) if current_embedding_model in embedding_models else 0

        new_model = st.sidebar.selectbox(
            "🤖 选择模型",
            options=available_models,
            index=model_index,
            help="选择要使用的语言模型"
        )

        new_embedding_model = st.sidebar.selectbox(
            "📐 嵌入模型",
            options=embedding_models,
            index=embedding_index,
            help="选择用于文档嵌入的模型"
        )

        return new_model, new_embedding_model

    # 2. 渲染RAG设置组件
    @staticmethod
    def render_rag_settings(rag_enabled: bool, similarity_threshold: float,
                            default_threshold: float) -> Tuple[bool, float]:
        """
        渲染RAG设置组件
        参数:
            rag_enabled - 是否启用RAG
            similarity_threshold - 相似度阈值
            default_threshold - 默认相似度阈值

        返回:
            (是否启用RAG, 相似度阈值)
        """
        st.sidebar.subheader("🔍 RAG设置")

        new_rag_enabled = st.sidebar.checkbox(
            "📚 启用RAG文档检索",
            value=rag_enabled,
            help="启用检索增强生成功能，使用上传的文档增强回答"
        )

        new_similarity_threshold = st.sidebar.slider(
            "🎯 相似度阈值",
            min_value=0.0,
            max_value=1.0,
            value=similarity_threshold,
            step=0.05,
            help="调整检索相似度阈值，值越高要求匹配度越精确"
        )

        # 重置相似度阈值按钮
        if st.sidebar.button("🔄 重置相似度阈值", use_container_width=True):
            new_similarity_threshold = default_threshold
            st.toast("已重置相似度阈值", icon="🔄")

        return new_rag_enabled, new_similarity_threshold

    # 3. 渲染聊天统计信息
    @staticmethod
    def render_chat_stats(chat_history):
        """
        渲染聊天统计信息组件
        参数:
            chat_history - 聊天历史管理器
        """
        st.sidebar.header("💬 对话历史")
        stats = chat_history.get_stats()
        st.sidebar.info(
            f"💬 总对话数: {stats['total_messages']} | 👤 用户消息: {stats['user_messages']}"
        )

        # 导出历史按钮
        if st.sidebar.button("📥 导出对话历史", use_container_width=True):
            csv = chat_history.export_to_csv()
            if csv:
                st.sidebar.download_button(
                    label="💾 下载CSV文件",
                    data=csv,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        # 清空对话按钮
        if st.sidebar.button("✨ 清空对话", use_container_width=True):
            chat_history.clear_history()
            st.toast("🗑️ 对话已清空", icon="✅")
            st.rerun()

    @staticmethod
    def render_vector_store_status(vector_store: VectorStoreService, doc_count: int = 0):
        """显示向量存储状态"""
        if vector_store.vector_store:
            try:
                doc_count = len(vector_store.vector_store.docstore._dict)
                st.sidebar.success(f"✅ 向量索引已加载 ({doc_count} 个文档块)")
            except AttributeError:
                st.sidebar.success("✅ 向量索引已加载")
        else:
            st.sidebar.warning("⚠️ 当前无可用文档索引")

    # 4. 渲染文档上传组件
    @staticmethod
    def render_document_upload(
        document_processor: DocumentProcessor,
        vector_store: VectorStoreService,
        processed_documents: List[str]
    ) -> Tuple[int, VectorStoreService]:
        """
        渲染文档上传组件
        参数:
            document_processor - 文档处理器
            vector_store - 向量存储服务
            processed_documents - 已处理的文档列表

        返回:
            (新添加的文档块数量, 更新后的向量存储服务)
        """
        # 新添加的文档块数量
        new_doc_count = 0

        # 展开面板状态：当没有已处理文档时展开
        with st.expander("📁 上传RAG文档", expanded=not bool(processed_documents)):
            # 文件上传器
            st.info("📤 请上传PDF或TXT文件")
            uploaded_files = st.file_uploader(
                "📄 上传文档",
                type=["pdf", "txt"],
                accept_multiple_files=True
            )

            new_docs = []
            # 处理上传的文件
            if uploaded_files and st.button("⚙️ 处理文档", key="process_docs"):
                with st.spinner("🔧 正在处理文档..."):
                    for uploaded_file in uploaded_files:
                        if uploaded_file.name not in processed_documents:
                            try:
                                # 处理文件内容
                                result = document_processor.process_file(
                                    uploaded_file)

                                if isinstance(result, list):
                                    # 对于PDF文档块，直接使用
                                    new_docs.extend(result)
                                    new_doc_count += len(result)
                                else:
                                    # 对于TXT文档，创建Document对象
                                    doc = Document(
                                        page_content=result,
                                        metadata={"source": uploaded_file.name}
                                    )
                                    new_docs.append(doc)
                                    new_doc_count += 1

                                processed_documents.append(uploaded_file.name)
                                st.success(f"✅ 已处理: {uploaded_file.name}")
                            except Exception as e:
                                st.error(
                                    f"❌ 处理失败: {uploaded_file.name} - {str(e)}")
                        else:
                            st.warning(f"⚠️ 已存在: {uploaded_file.name}")

                # 构建向量索引
                if new_docs:
                    with st.spinner("🧩 正在构建向量索引..."):
                        try:
                            # 如果已有向量存储，添加到现有索引
                            if vector_store.vector_store:
                                vector_store.vector_store.add_documents(
                                    new_docs)
                                st.success(f"成功添加 {len(new_docs)} 个文档块到现有索引")
                            else:
                                # 否则创建新索引
                                vector_store.vector_store = vector_store.create_vector_store(
                                    new_docs)
                                st.success("成功创建新文档索引")
                        except Exception as e:
                            st.error(f"❌ 构建索引失败: {str(e)}")
                            return new_doc_count, vector_store

                    # 保存向量索引
                    if vector_store.vector_store:
                        with st.spinner("💾 保存文档索引..."):
                            try:
                                vector_store._save_vector_store(
                                    vector_store.vector_store)
                                st.success("✅ 文档索引保存成功！")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ 保存索引失败: {str(e)}")
                    else:
                        st.error("❌ 向量索引创建失败")

            # 显示已处理文档列表
            if processed_documents:
                st.subheader("📋 已处理文档")
                for doc in processed_documents:
                    st.markdown(f"- 📄 {doc}")

                # 清除文档按钮
                if st.button("🗑️ 清除所有文档", key="clear_docs"):
                    with st.spinner("🧹 正在清除向量索引..."):
                        try:
                            vector_store.clear_index()
                            processed_documents.clear()
                            st.success("✅ 所有文档已清除")
                        except Exception as e:
                            st.error(f"❌ 清除索引失败: {str(e)}")
                    st.rerun()
            return new_doc_count, vector_store

    # 5. 渲染聊天历史
    @staticmethod
    def render_chat_history(chat_history: ChatHistoryManager):
        """
        渲染聊天历史组件
        参数:
            chat_history - 聊天历史管理器
        """
        # 处理不同消息类型
        role_handlers = {
            "assistant_think": UIComponents.render_assistant_think,
            "retrieved_doc": UIComponents.render_retrieved_doc,
        }

        for message in chat_history.history:
            role = message.get('role', '')
            content = message.get('content', '')

            # 使用字典映射处理不同角色的消息
            handler = role_handlers.get(role)
            if handler:
                handler(content)
            else:
                # 渲染默认消息
                UIComponents.render_default_message(role, content)

    @staticmethod
    def render_assistant_think(content: str):
        """渲染助手的推理过程"""
        with st.expander("💡 查看推理过程 <think> ... </think>"):
            st.markdown(content)

    @staticmethod
    def render_retrieved_doc(content: str):
        """渲染召回的文档块"""
        with st.expander(f"🔍 查看本次召回的文档块", expanded=False):
            if isinstance(content, list):
                for idx, doc in enumerate(content, 1):
                    st.markdown(f"📄 **文档块{idx}:**\n{doc}")
            else:
                st.markdown(content)

    @staticmethod
    def render_default_message(role: str, content: str):
        """渲染默认消息"""
        with st.chat_message(role):
            st.write(content)
