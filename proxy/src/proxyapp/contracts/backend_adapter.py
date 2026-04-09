def build_backend_chat_request_from_responses(request_data: dict, model_path: str) -> dict:
    input_data = request_data.get("input", "")
    instructions = request_data.get("instructions")
    tools = request_data.get("tools", [])
    tool_choice = request_data.get("tool_choice")
    max_output_tokens = request_data.get("max_output_tokens")
    temperature = request_data.get("temperature")
    top_p = request_data.get("top_p")

    messages = []
    if instructions:
        messages.append({"role": "system", "content": instructions})

    if isinstance(input_data, str):
        messages.append({"role": "user", "content": input_data})
    elif isinstance(input_data, list):
        for item in input_data:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                messages.append({"role": role, "content": content})

    chat_request = {
        "model": model_path,
        "messages": messages,
        "stream": True,
    }

    if tools:
        chat_request["tools"] = tools
    if tool_choice is not None:
        chat_request["tool_choice"] = tool_choice
    if max_output_tokens is not None:
        chat_request["max_tokens"] = max_output_tokens
    if temperature is not None:
        chat_request["temperature"] = temperature
    if top_p is not None:
        chat_request["top_p"] = top_p

    return chat_request
