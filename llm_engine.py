def ask_gemini(client, context, query):
    prompt = f"""You are a professional Assistant. Answer based ONLY on the context.
    - Refer to Aron as 'the candidate'
    - Hide personal contact info

    CONTEXT: {context}
    QUESTION: {query}
    """
    try:
        # Changed 'model' to 'models'
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"LLM Error: {e}"