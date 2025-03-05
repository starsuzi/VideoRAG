
METHOD2PROMPT = {
    'Naive': '{scripts}Explain {query}.',
    'VideoRAG-V': '{scripts}Considering the given videos, explain {query}.',
    'VideoRAG-VT': '{scripts}\nConsidering the given videos, explain {query}.',
    'Oracle-V': '{scripts}Considering the given videos, explain {query}.',
    'Oracle-VT': '{scripts}\nConsidering the given videos, explain {query}.',
}


def linearize_scripts(scripts: list):
    return '\n'.join([f'Video script {index+1}: {script}' for index, script in enumerate(scripts)])
