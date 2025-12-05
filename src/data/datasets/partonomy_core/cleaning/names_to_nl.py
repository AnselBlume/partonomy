import os
import json
from vllm import LLM, SamplingParams
import logging

logger = logging.getLogger(__name__)

CONCEPT_TO_NL_PROMPT = '''
    Transform the following identifier into a natural language phrase.

    Guidelines:
    - The identifier consists of a category and a concept separated by '--' (e.g. 'airplanes--agricultural').
    - Convert this into a natural language phrase that is easy to read but still uniquely identifiable.

    Examples:
    - airplanes--agricultural --> agricultural airplane
    - office supplies--dry wet erase board --> dry or wet erase board
    - geometry--circle --> circle
    - geography--sinkhole --> sinkhole
    - geography--sea --> sea
    - office supplies--keyboard --> computer keyboard
    - boats--polar research vessel --> polar research boat
    - garden--hand weeder --> hand weeder for gardening
    - garden--post hole digger --> garden post hole digger

    Provide only the final phrase as your output.
    Here is the identifier: '{identifier}'
'''

IMPROVE_MAPPING_PROMPT = '''
    You are a helpful assistant that improves a mapping of identifiers to natural language phrases.
    These phrases should sound natural but you should try to keep them uniquely identifiable.

    Here is the current mapping:
    {mapping}

    Please output the improved mapping in JSON format:
'''

def transform_identifiers(identifiers: list[str], llm: LLM, sampling_params: SamplingParams) -> list[str]:
    '''
    Transforms a list of identifiers (each in the format 'category--concept')
    into natural language phrases.

    The prompt instructs the LLM to:
    - Parse the identifier (a category and a concept separated by '--')
    - Convert it into a natural language phrase that is easy to read but still uniquely identifiable.

    The function batches the requests to VLLM.
    '''
    conversations = []
    for identifier in identifiers:
        # Validate identifier format.
        try:
            category, concept = identifier.split('--', 1)
        except ValueError:
            raise ValueError(f'Invalid identifier format (expected "category--concept"): {identifier}')

        conversation = [
            {
                'role': 'system',
                'content': 'You are a helpful assistant that transforms structured identifiers into natural language phrases.'
            },
            {
                'role': 'user',
                'content': CONCEPT_TO_NL_PROMPT.format(identifier=identifier)
            }
        ]
        conversations.append(conversation)

    outputs = llm.chat(conversations, sampling_params)
    results = [output.outputs[0].text.strip().lower() for output in outputs]

    return results

def convert_identifiers_to_natural_language(identifiers: list[str], llm: LLM, sampling_params: SamplingParams) -> list[str]:
    # Batch process the identifiers.
    transformed_phrases = transform_identifiers(identifiers, llm, sampling_params)
    for original, transformed in zip(identifiers, transformed_phrases):
        logger.debug(f'Original: {original}')
        logger.debug(f'Transformed: {transformed}\n')

    return transformed_phrases

def improve_json_mapping(mapping: dict, llm: LLM, sampling_params: SamplingParams) -> str:
    conversation = [{
        'role': 'system',
        'content': IMPROVE_MAPPING_PROMPT.format(mapping=mapping)
    }]

    output = llm.chat(conversation, sampling_params)
    results = output[0].outputs[0].text.strip().lower()

    return results

if __name__ == '__main__':
    import coloredlogs
    coloredlogs.install(level='DEBUG')

    improve_mapping = True
    mapping_path = 'concepts_names_to_nl.json'
    improved_mapping_path = 'concepts_names_to_nl_improved.txt'

    # Initialize the LLM with desired configuration
    model = 'microsoft/phi-4'
    gpu_memory_utilization = .95
    temperature = 0.0
    max_tokens = 100

    # Collect conept names
    masks_dir = '/shared/nas2/blume5/fa24/concept_downloading/data/image_annotations/24-11-18/annotations/merged_annotations/masks'
    dir_names = os.listdir(masks_dir)
    concept_names = [n for n in dir_names if os.path.isdir(os.path.join(masks_dir, n)) and '--part:' not in n]

    # Convert
    llm = LLM(model=model, gpu_memory_utilization=gpu_memory_utilization)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    if improve_mapping:
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)

        max_tokens = 100000
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

        new_concept_str = improve_json_mapping(mapping, llm, sampling_params)
        with open(improved_mapping_path, 'w') as f: # Don't try to parse JSON in case the LLM returns a non-JSON string
            f.write(new_concept_str)

    else: # Convert to natural language
        new_concept_names = convert_identifiers_to_natural_language(concept_names, llm, sampling_params)

        # Save the mapping
        mapping = {concept : nl for concept, nl in zip(concept_names, new_concept_names)}

        with open('concepts_names_to_nl.json', 'w') as f:
            json.dump(mapping, f, indent=4, sort_keys=True)