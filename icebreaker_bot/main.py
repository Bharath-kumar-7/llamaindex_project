from modules.data_extraction import extract_linkedin_profile
from modules.data_processing import split_profile_data, create_vector_database
from modules.query_engine import generate_initial_facts, answer_user_query


def chatbot_interface(index):

    print("\nAsk questions about the person.")
    print("Type 'exit' to quit.\n")

    while True:

        query = input("You: ")

        if query.lower() in ["exit", "quit"]:
            print("Bot: Goodbye!")
            break

        response = answer_user_query(index, query)

        print("Bot:", response)


def process_linkedin():

    profile_data = extract_linkedin_profile(mock=True)

    nodes = split_profile_data(profile_data)

    index = create_vector_database(nodes)

    facts = generate_initial_facts(index)

    print("\nThree interesting facts:\n")
    print(facts)

    chatbot_interface(index)


if __name__ == "__main__":
    process_linkedin()
