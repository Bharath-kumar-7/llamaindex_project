from llama_index.core import PromptTemplate
import config

from modules.llm_interface import create_llm


def generate_initial_facts(index):

    llm = create_llm()

    prompt = PromptTemplate(config.INITIAL_FACTS_TEMPLATE)

    query_engine = index.as_query_engine(
        similarity_top_k=config.SIMILARITY_TOP_K, llm=llm, text_qa_template=prompt
    )

    response = query_engine.query("Provide three interesting facts about this person")

    return response.response


def answer_user_query(index, query):

    llm = create_llm()

    prompt = PromptTemplate(config.USER_QUESTION_TEMPLATE)

    query_engine = index.as_query_engine(
        similarity_top_k=config.SIMILARITY_TOP_K, llm=llm, text_qa_template=prompt
    )

    response = query_engine.query(query)

    return response.response
