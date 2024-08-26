def get_prompt(path: str) -> str:
    '''
    Get the prompt from prompt.txt file.
    '''
    
    with open(path, 'r') as file:
        prompt = file.read()
        
    return prompt
    
    
