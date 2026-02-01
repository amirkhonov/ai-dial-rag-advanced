from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


SYSTEM_PROMPT = """You are a helpful RAG-powered assistant that helps users with microwave usage questions.

You will receive context from a microwave manual followed by a user's question.

INSTRUCTIONS:
1. Use ONLY the provided RAG context to answer the user's question
2. If the question is not related to microwave usage, politely decline to answer
3. If the context doesn't contain enough information to answer the question, say so honestly
4. Be concise and helpful in your responses
5. Do not make up information that is not in the provided context
6. Only answer questions that are relevant to the conversation history and provided context
"""

USER_PROMPT = """RAG Context:
{context}

User Question:
{question}
"""


def main():
    # Create embeddings client with 'text-embedding-3-small-1' model
    embeddings_client = DialEmbeddingsClient(
        deployment_name='text-embedding-3-small-1',
        api_key=API_KEY
    )
    
    # Create chat completion client
    chat_client = DialChatCompletionClient(
        deployment_name='gpt-4o',
        api_key=API_KEY
    )
    
    # Create text processor with DB config
    db_config = {
        'host': 'localhost',
        'port': 5433,
        'database': 'vectordb',
        'user': 'postgres',
        'password': 'postgres'
    }
    text_processor = TextProcessor(embeddings_client, db_config)
    
    # Process the microwave manual
    import os
    manual_path = os.path.join(os.path.dirname(__file__), 'embeddings', 'microwave_manual.txt')
    print("Processing microwave manual...")
    text_processor.process_text_file(
        file_path=manual_path,
        chunk_size=512,
        overlap=50,
        dimensions=1536,
        truncate=True
    )
    print("Manual processed and stored in vector database!\n")
    
    # Initialize conversation
    conversation = Conversation()
    system_message = Message(Role.SYSTEM, SYSTEM_PROMPT)
    conversation.add_message(system_message)
    
    print("🤖 Microwave Assistant Ready!")
    print("Ask me anything about microwave usage. Type 'exit' to quit.\n")
    
    # Console chat loop
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        
        # Step 1: Retrieval - search for relevant context
        context_chunks = text_processor.search(
            query=user_input,
            search_mode=SearchMode.COSINE_DISTANCE,
            top_k=3,
            min_score=0.5,
            dimensions=1536
        )
        
        # Step 2: Augmentation - combine context with user question
        context_text = "\n\n".join(context_chunks)
        augmented_prompt = USER_PROMPT.format(
            context=context_text,
            question=user_input
        )
        
        # Add user message to conversation
        user_message = Message(Role.USER, augmented_prompt)
        conversation.add_message(user_message)
        
        # Step 3: Generation - get LLM response
        try:
            ai_response = chat_client.get_completion(conversation.get_messages())
            conversation.add_message(ai_response)
            
            print(f"\n🤖 Assistant: {ai_response.content}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    # PAY ATTENTION THAT YOU NEED TO RUN Postgres DB ON THE 5433 WITH PGVECTOR EXTENSION!
    # RUN docker-compose.yml
    main()