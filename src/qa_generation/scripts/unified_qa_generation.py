# %%
from jsonargparse import ArgumentParser
import json
import os
import sys

sys.path = [os.path.realpath(f'{__file__}/../../..')] + sys.path

from data.datasets.build_dataset_descriptors import ObjectPartMatcher
from qa_generation import QAGenerator, QAGeneratorConfig,QuestionType, ConceptGraph
from qa_generation.answer_mutation import AnswerMutator, AnswerMutatorConfig, PartSampler, PartSamplerConfig, ObjectSampler, ObjectSamplerConfig
from data.utils import load_yaml, load_json
from data.part_dataset_descriptor import PartDatasetDescriptor, clean_dataset_descriptor
from data.datasets import \
    init_paco_lvis, init_pascal_part, init_partimagenet, partonomy_to_concept_graph, \
    paco_lvis_to_concept_graph, pascal_part_to_concept_graph, partimagenet_to_concept_graph, \
    generate_coco_part_dataset_descriptor, generate_partonomy_dataset_descriptor
from qa_generation.part_comparison import PartComparator, PartComparatorConfig
import logging

logger = logging.getLogger(__name__)

def _dump_descriptor(descriptor: PartDatasetDescriptor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(descriptor.to_dict(), f, sort_keys=False, indent=4)

def parse_args(cl_args: list[str], config_str: str = None):
    parser = ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['partonomy', 'paco_lvis', 'pascal_part', 'partimagenet'])
    parser.add_argument('--output_dir', type=str, default='/shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_descriptors')
    parser.add_argument('--modified_image_output_dir', type=str, default=None)
    parser.add_argument('--base_dataset_dir', type=str, default='/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset')

    parser.add_argument('--regenerate_descriptor', type=bool, default=False, help='Regenerate the descriptor from the dataset instead of loading from disk')
    parser.add_argument('--regenerate_graph', type=bool, default=False, help='Regenerate the graph from the dataset instead of loading from disk')
    parser.add_argument('--save_cleaned_descriptor', type=bool, default=False, help='Overwrite the descriptor on disk with the cleaned descriptor')
    parser.add_argument('--generate_qa_pairs', type=bool, default=True, help='Whether to generate QA pairs')

    parser.add_argument('--qa_generator_config', type=QAGeneratorConfig, default=QAGeneratorConfig())
    parser.add_argument('--part_sampler_config', type=PartSamplerConfig, default=PartSamplerConfig())
    parser.add_argument('--object_sampler_config', type=ObjectSamplerConfig, default=ObjectSamplerConfig())
    parser.add_argument('--answer_mutator_config', type=AnswerMutatorConfig, default=AnswerMutatorConfig())

    parser.add_argument('--part_comparator.config', type=PartComparatorConfig, default=PartComparatorConfig())
    parser.add_argument('--part_comparator.cache_to_disk', type=bool, default=True)
    parser.add_argument('--part_comparator.cache_file_template', type=str, default='{dataset}-part_comparator_cache.json')

    if config_str:
        args = parser.parse_string(config_str)
    else:
        args = parser.parse_args(cl_args, config_str)

    return args, parser

def main(cl_args: list[str] = None, config_str: str = None):
    # Load the concept graph (or data file with a class-part relation)
    concept_graph = None
    object_part_matcher = ObjectPartMatcher() # For COCO datasets

    args, parser = parse_args(cl_args, config_str)

    object_name_map = None
    part_name_map = None
    for dataset in args.datasets:
        print(f">> Processing [ {dataset} ] ")
        '''
        For a class, there can be multiple images
        - A PartDatasetInstance should correspond to a single image - and thus single image path
        - A PartDatasetInstance may have multiple segmentation masks (single image + multiple masks)
        - A part-level label (e.g., "spraying_rig") may have multiple segmentation masks
        '''
        if dataset == "paco_lvis":
            class_map, img_ids, img_dir, coco_api = init_paco_lvis(args.base_dataset_dir)
            graph_path = os.path.join(args.base_dataset_dir, "vlpart", "paco", "graph.yaml")
            descriptor_path = os.path.join(args.output_dir, "paco_lvis", "descriptor.json")
            object_part_matcher.class_map = class_map
            object_part_matcher.coco_api = coco_api

            if not os.path.exists(graph_path) or args.regenerate_graph:
                concept_graph = paco_lvis_to_concept_graph(args.base_dataset_dir, class_map)
            else:
                concept_graph = load_yaml(graph_path)

            if os.path.exists(descriptor_path) and not args.regenerate_descriptor:
                part_dataset_descriptor = PartDatasetDescriptor.from_dict(load_json(descriptor_path))
            else:
                part_dataset_descriptor = generate_coco_part_dataset_descriptor(
                    dataset,
                    img_ids,
                    img_dir,
                    coco_api,
                    object_part_matcher,
                    modified_image_output_dir=args.modified_image_output_dir
                )
                _dump_descriptor(part_dataset_descriptor, descriptor_path)

            part_name_map = lambda x: x.replace('_', ' ') # Map underscores to spaces

        elif dataset == "pascal_part":
            class_map, img_ids, img_dir, coco_api = init_pascal_part(args.base_dataset_dir)
            graph_path = os.path.join('/shared/nas2/jk100/partonomy_private/data', "pascal_part", "graph.yaml")
            descriptor_path = os.path.join(args.output_dir, "pascal_part", "descriptor.json")
            object_part_matcher.class_map = class_map
            object_part_matcher.coco_api = coco_api

            if not os.path.exists(graph_path) or args.regenerate_graph:
                concept_graph = pascal_part_to_concept_graph(args.base_dataset_dir, class_map)
            else:
                concept_graph = load_yaml(graph_path)

            if os.path.exists(descriptor_path) and not args.regenerate_descriptor:
                part_dataset_descriptor = PartDatasetDescriptor.from_dict(load_json(descriptor_path))
            else:
                part_dataset_descriptor = generate_coco_part_dataset_descriptor(
                    dataset,
                    img_ids,
                    img_dir,
                    coco_api,
                    object_part_matcher,
                    modified_image_output_dir=args.modified_image_output_dir
                )
                _dump_descriptor(part_dataset_descriptor, descriptor_path)

        elif dataset == "partimagenet":
            class_map, img_ids, img_dir, coco_api = init_partimagenet(args.base_dataset_dir)
            graph_path = os.path.join(args.base_dataset_dir, "partimagenet", "graph.yaml")
            descriptor_path = os.path.join(args.output_dir, "partimagenet", "descriptor.json")
            object_part_matcher.class_map = class_map
            object_part_matcher.coco_api = coco_api

            if not os.path.exists(graph_path) or args.regenerate_graph:
                concept_graph = partimagenet_to_concept_graph(args.base_dataset_dir, class_map)
            else:
                concept_graph = load_yaml(graph_path)

            if os.path.exists(descriptor_path) and not args.regenerate_descriptor:
                part_dataset_descriptor = PartDatasetDescriptor.from_dict(load_json(descriptor_path))
            else:
                part_dataset_descriptor = generate_coco_part_dataset_descriptor(
                    dataset,
                    img_ids,
                    img_dir,
                    coco_api,
                    object_part_matcher,
                    modified_image_output_dir=args.modified_image_output_dir
                )
                _dump_descriptor(part_dataset_descriptor, descriptor_path)

            part_name_map = lambda x: 'tire' if x == 'tier' else x # Fix typo in dataset

        elif dataset == "partonomy":
            # Update the base dataset directory to the newest version of the Partonomy dataset
            # TODO uncomment the following
            # partonomy_base_dataset_dir = '/shared/nas2/blume5/fa24/concept_downloading/data/image_annotations'
            # logger.info(f'Overriding path at args.base_dataset_dir to Partonomy dataset at: {partonomy_base_dataset_dir}')

            # concept_graph = partonomy_to_concept_graph(partonomy_base_dataset_dir)
            # descriptor_path = os.path.join(args.output_dir, "partonomy", "descriptor.json")

            # if os.path.exists(descriptor_path) and not args.regenerate_descriptor:
            #     part_dataset_descriptor = PartDatasetDescriptor.from_dict(load_json(descriptor_path))
            # else:
            #     part_dataset_descriptor = generate_partonomy_dataset_descriptor(
            #         partonomy_base_dataset_dir,
            #         dataset,
            #         concept_graph
            #     )
            #     _dump_descriptor(part_dataset_descriptor, descriptor_path)

            descriptor_path = '/work/hdd/beig/blume5/partonomy/data/partonomy_descriptors/partonomy/partonomy-descriptor.json'
            graph_path = '/work/hdd/beig/blume5/partonomy/data/partonomy_descriptors/partonomy/graph.yaml'

            part_dataset_descriptor = PartDatasetDescriptor.from_dict(load_json(descriptor_path))
            concept_graph = load_yaml(graph_path)

            # Clean up object names from category--object to natural language
            object_name_map_path = os.path.join(args.output_dir, "partonomy", "object_names_to_nl.json")
            if os.path.exists(object_name_map_path):
                logger.info(f'Loading object name map from {object_name_map_path}')
                name_map_dict = load_json(object_name_map_path)
                object_name_map = lambda x: name_map_dict.get(x, x)

        else:
            raise ValueError(f"No {dataset} in the DATASET_NAMES.")

        # load concept graph
        concept_graph = ConceptGraph(instance_graph=concept_graph['instance_graph'], part_graph=concept_graph['component_graph'])

        # Set the part and instance graph of the concept graph for cleaning
        part_dataset_descriptor.part_graph = concept_graph.part_graph
        part_dataset_descriptor.instance_graph = concept_graph.instance_graph

        clean_dataset_descriptor(
            part_dataset_descriptor,
            object_name_map=object_name_map,
            part_name_map=part_name_map,
            deduce_part_graph=True
        )

        if args.save_cleaned_descriptor:
            cleaned_descriptor_path = descriptor_path.replace('.json', '_cleaned.json')
            logger.info(f'Saving cleaned descriptor to {cleaned_descriptor_path}')
            _dump_descriptor(part_dataset_descriptor, cleaned_descriptor_path)

        concept_graph = ConceptGraph( # Update ConceptGraph with cleaned instance/part graphs
            instance_graph=part_dataset_descriptor.instance_graph,
            part_graph=part_dataset_descriptor.part_graph
        )

        if not args.generate_qa_pairs:
            continue

        # Generate QA Pairs
        part_comparator = PartComparator(config=args.part_comparator.config)

        if args.part_comparator.config.cache_comparisons and args.part_comparator.cache_to_disk:
            cache_file = args.part_comparator.cache_file_template.format(dataset=dataset)

            if os.path.exists(cache_file):
                logger.info(f'Loading part comparator cache from {os.path.realpath(cache_file)}')
                part_comparator.load_cache(cache_file)

        part_sampler = PartSampler(part_dataset_descriptor.instances, part_dataset_descriptor.part_graph, config=args.part_sampler_config)
        object_sampler = ObjectSampler(part_dataset_descriptor.instances, part_dataset_descriptor.instance_graph, config=args.object_sampler_config)
        mutator = AnswerMutator(part_sampler, object_sampler, config=args.answer_mutator_config)

        generator = QAGenerator(concept_graph, part_dataset_descriptor, mutator, part_comparator)

        # Save once after precaching
        if args.part_comparator.config.cache_comparisons and args.part_comparator.cache_to_disk:
            logger.info(f'Saving part comparator cache to {cache_file}')
            part_comparator.save_cache(cache_file)

        output_path = os.path.join(args.output_dir, dataset, f'{dataset}_qa_pairs.json')

        qa_pairs = generator.generate_qa_pairs(question_types=[
            # QuestionType.IDENTIFICATION,
            QuestionType.IDENTIFICATION_WITH_LABEL,
            # QuestionType.POSITIVE,
            QuestionType.POSITIVE_WITH_LABEL,
            # QuestionType.NEGATIVE,
            # QuestionType.NEGATIVE_WITH_LABEL,
            # QuestionType.DIFFERENCE,
            # QuestionType.DIFFERENCE_WITH_LABEL,
            # QuestionType.WHOLE_TO_PART,
            # QuestionType.PART_TO_WHOLE
        ], save_intermediate=True, json_save_path=output_path)

        # Save again after QAPair generation
        if args.part_comparator.config.cache_comparisons and args.part_comparator.cache_to_disk:
            logger.info(f'Saving part comparator cache to {cache_file}')
            part_comparator.save_cache(cache_file)

if __name__ == '__main__':
    import coloredlogs
    coloredlogs.install(level='INFO')

    main(config_str='''
        # datasets: [pascal_part, partimagenet, partonomy, paco_lvis]
        datasets: [partonomy]

        regenerate_descriptor: false
        regenerate_graph: false
        save_cleaned_descriptor: false
        generate_qa_pairs: true

        output_dir: /work/hdd/beig/blume5/partonomy/data/partonomy_descriptors
        modified_image_output_dir: /shared/nas2/blume5/sp25/partonomy/partonomy_private/data/modified_images
        base_dataset_dir: /shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset

        qa_generator_config:
            comparator_batch_size: 100

        part_comparator:
            config:
                # strategy: EXACT
                strategy: LLM

                # llm_model: microsoft/Phi-4-mini-instruct
                llm_model: microsoft/phi-4

                vllm_gpu_memory_utilization: .95
                vllm_num_gpus: 2

        part_sampler_config:
            predict_closest_parts: true
    ''')